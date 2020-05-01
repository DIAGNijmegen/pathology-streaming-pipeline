import time
import numpy as np
import torch
import torch.distributed as dist
import dataclasses
import pathlib

from streaming.torch_utils.diagnostics import check_params_distributed
from streaming.torch_utils.trainer import Trainer, TrainerOptions

if '1.6' in torch.__version__: # type:ignore
    from torch.cuda.amp import autocast

@dataclasses.dataclass
class CheckpointedTrainerOptions(TrainerOptions):
    batch_size: int = 16
    drop_last: bool = True
    checkpointed_net: torch.nn.Module = torch.nn.Sequential()
    mixedprecision: bool = False
    gather_batch_on_one_gpu: bool = False

class CheckpointedTrainer(Trainer):
    def __init__(self, options: CheckpointedTrainerOptions):
        self.batch_size = options.batch_size
        self.drop_last = options.drop_last
        self.checkpointed_net = options.checkpointed_net
        self.batch_overal_count = 0
        self.mixedprecision = options.mixedprecision
        self.gather_batch_on_one_gpu = options.gather_batch_on_one_gpu
        super().__init__(options)

    def sync_networks_distributed_if_needed(self, check=True):
        if self.distributed:
            self.sync_network_distributed(self.net, check)
            self.sync_network_distributed(self.checkpointed_net, check)

    def prepare_network_for_training(self):
        self.checkpointed_net.train()
        super().prepare_network_for_training()
        self.prepare_batchnorm_for_evaluation(self.checkpointed_net)

    def turn_batchnorm_off_for_network(self, network):
        for module in network.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):  # type:ignore
                module.eval()

    def prepare_network_for_evaluation(self):
        super().prepare_network_for_evaluation()
        self.checkpointed_net.eval()

    def reset_batch_stats(self):
        self.batch_images = []
        self.batch_predictions = []
        self.batch_labels = []
        self.running_batch_loss = 0

    def reset_epoch_stats(self):
        super().reset_epoch_stats()
        self.reset_batch_stats()

    def checkpoint_image_forward(self, x):
        with torch.no_grad():
            if self.mixedprecision:
                with autocast():
                    output = self.checkpointed_net.forward(x.cuda())
            else:
                output = self.checkpointed_net.forward(x.cuda())
        return output

    def accumulate_output(self, x, y, output):
        if self.dtype != torch.uint8:
            self.batch_images.append(x[None].cpu())
        else:
            self.batch_images.append(x[None])
        self.batch_labels.append(y.cpu())
        self.batch_predictions.append(output)  # save memory when doing bs > 0

    def backward_image(self, x, gradient):
        out = self.checkpointed_net.forward(x.cuda())
        out.backward(gradient[None])

    def backward_batch(self, loss, fmap):
        if self.should_finish_batch_on_gpu():
            if self.mixedprecision:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            fmap_grad = fmap.grad.data
            fmap_grad.detach()
        else:
            fmap_grad = fmap.new_empty(fmap.shape)
            dist.broadcast(fmap_grad, src=0)
            fmap_grad = fmap_grad[self.gpu_rank * self.batch_size:self.gpu_rank * self.batch_size + self.batch_size]
        del loss, fmap
        self.backward_batch_checkpointed(fmap_grad)
        del fmap_grad
       
    def backward_batch_checkpointed(self, fmap_grad):
        for i, x in enumerate(self.batch_images):
            self.backward_image(x, fmap_grad[i])

    def distribute_gradients_if_needed(self):
        if self.distributed:
            for _, param in self.checkpointed_net.named_parameters():
                if param.grad is None: continue
                else: dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            for _, param in self.net.named_parameters():
                if param.grad is None: continue
                else: dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def accumulate_batch(self):
        labels = torch.stack(self.batch_labels).cuda()
        fmap = torch.cat(self.batch_predictions, 0)  # type:ignore

        # trying memory management
        del self.batch_predictions, self.batch_labels
        self.batch_predictions = []  # side effect of function! Memory reasons

        if self.gather_batch_on_one_gpu: 
            labels, fmap = self.gather_batch(fmap, labels)

        return labels, fmap

    def gather_batch(self, fmap, labels):
        if self.distributed:
            gathered_output = [fmap.new_empty(fmap.shape) for i in range(self.n_gpus)]
            gathered_labels = [labels.new_empty(labels.shape).fill_(-1) for i in range(self.n_gpus)]
            dist.all_gather(gathered_output, fmap)
            dist.all_gather(gathered_labels, labels)

            # trying memory management
            del fmap, labels

            fmap = torch.cat(gathered_output, 0)  # type:ignore
            labels = torch.cat(gathered_labels, 0).view(-1).cpu()  # type:ignore

        return labels, fmap

    def should_finish_batch_on_gpu(self):
        return not self.distributed or (self.gpu_rank == 0 and self.gather_batch_on_one_gpu)

    def evaluate_on_accumulated_batch(self):
        labels, fmap = self.accumulate_batch()
        if self.should_finish_batch_on_gpu():
            fmap.requires_grad = torch.is_grad_enabled()  # type: ignore
            loss, accuracy, predictions = self.evaluate_batch(fmap, labels)
            self.save_batch_stats(loss, accuracy, predictions, labels.cpu().numpy())
            return loss, accuracy, fmap
        else:
            return None, None, fmap

    def step_optimizer_if_needed(self):
        if self.accumulated_batches == self.accumulate_over_n_batches:
            super().step_optimizer_if_needed()
            self.batch_overal_count += 1
            self.check_networks_stats()
            self.accumulated_batches = 0

    def check_networks_stats(self):
        if self.distributed: check_params_distributed(self.checkpointed_net, self.n_gpus, self.gpu_rank)
        if self.distributed: check_params_distributed(self.net, self.n_gpus, self.gpu_rank)

    def train_on_accumulated_batch(self):
        loss, accuracy, fmap = self.evaluate_on_accumulated_batch()
        if self.should_finish_batch_on_gpu():
            loss = loss / self.accumulate_over_n_batches / self.n_gpus
        self.backward_batch(loss, fmap)
        self.accumulated_batches += 1
        self.step_optimizer_if_needed()
        if self.should_finish_batch_on_gpu():
            return loss.detach() * self.accumulate_over_n_batches, accuracy
        else:
            return None, None

    def train_batch_if_needed(self):
        number_of_image_trained = len(self.batch_images)
        if number_of_image_trained % self.batch_size == 0:
            self.train_checkpointed_batch()

    def evaluate_batch_if_needed(self):
        number_of_image_trained = len(self.batch_images)
        if number_of_image_trained % self.batch_size == 0:
            self.evaluate_checkpointed_batch()

    def train_checkpointed_batch(self):
        loss, accuracy = self.train_on_accumulated_batch()
        self.batch_callback(self, self.batches_evaluated, loss, accuracy)
        self.reset_batch_stats()

    def evaluate_checkpointed_batch(self):
        loss, accuracy, _ = self.evaluate_on_accumulated_batch()
        if loss is not None: loss.detach().cpu()
        self.batch_callback(self, self.batches_evaluated, loss, accuracy)
        del loss, accuracy, _
        self.reset_batch_stats()

    def train_full_dataloader(self, batch_callback):
        self.batch_callback = batch_callback
        self.optimizer.zero_grad()

        start_time = time.time()
        first_delay = True

        for x, y in self.dataloader:
            if first_delay: first_delay = False
            else: self.check_dataloading_speed(start_time, time.time())

            for image, label in zip(x, y):
                output = self.checkpoint_image_forward(image[None])
                self.accumulate_output(image, label, output)
                self.train_batch_if_needed()

            start_time = time.time()

        if not self.drop_last and len(self.batch_images) > 0:
            self.train_checkpointed_batch()

    def evaluate_full_dataloader(self, batch_callback):
        self.batch_callback = batch_callback

        for x, y in self.dataloader:
            for image, label in zip(x, y):
                output = self.checkpoint_image_forward(image[None])
                self.accumulate_output(image, label, output)
                self.evaluate_batch_if_needed()

    def check_dataloading_speed(self, start_time, stop_time, threshold=1.0):
        if stop_time - start_time > threshold and not hasattr(self, 'warned_dataloader'):
            print(f'Data loading takes:{stop_time - start_time:.2f} seconds')
            print(f'Try adding more RAM or more workers')
            self.warned_dataloader = True

    def save_checkpoint(self, name, epoch, additional={}):
        state = {
            'checkpoint': epoch,
            'state_dict_net': self.net.state_dict(),
            'state_dict_checkpointed': self.checkpointed_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        state.update(additional)
        print('Saving', epoch, self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'))

        try:
            torch.save(state, self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'))
            torch.save(state, self.save_dir / pathlib.Path('checkpoint_' + name + '_last'))
        except Exception as e:
            print('WARNING: Network not stored', e)

    def load_state_dict(self, state):
        try: self.optimizer.load_state_dict(state['optimizer'])
        except: print('WARNING: Optimizer not restored')
        self.net.load_state_dict(state['state_dict_net'])
        self.checkpointed_net.load_state_dict(state['state_dict_checkpointed'])

class CheckpointedMultiClassTrainer(CheckpointedTrainer):
    def accuracy_with_predictions(self, predictions, labels):
        equal = np.equal(np.round(torch.sigmoid(predictions)), labels.numpy() == 1)
        equal_c = np.sum(equal, axis=1)
        correct = (equal_c == labels.shape[1]).sum()
        return correct / len(predictions)
