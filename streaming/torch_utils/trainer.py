import os
import typing

import dataclasses
import numpy as np
import torch
import torch.distributed as dist
import pathlib

try:
    from torch.cuda.amp import autocast  # pylint: disable=import-error,no-name-in-module
    from torch.cuda.amp import GradScaler  # pylint: disable=import-error,no-name-in-module
except ModuleNotFoundError:
    pass

@dataclasses.dataclass
class TrainerOptions:
    net: torch.nn.Module = None  # type:ignore
    dataloader: torch.utils.data.DataLoader = None  # type:ignore
    optimizer: torch.optim.Optimizer = None  # type:ignore
    criterion: torch.nn.CrossEntropyLoss = None  # type:ignore
    save_dir: pathlib.Path = None  # type:ignore
    freeze: list = dataclasses.field(default_factory=list)
    accumulate_over_n_batches: int = 1
    distributed: bool = False
    gpu_rank: int = 0
    n_gpus: int = 1
    test_time_bn: bool = False
    dtype: torch.dtype = torch.float32
    mixedprecision: bool = False
    multilabel: bool = False 
    regression: bool = False

class Trainer():
    images_evaluated: int = 0
    accumulated_batches: int = 0
    accumulated_loss: float = 0.0
    accumulated_accuracy: float = 0.0

    all_predictions = []
    all_labels = []

    def __init__(self, options: TrainerOptions):
        self.net = options.net
        self.dataloader = options.dataloader
        self.optimizer = options.optimizer
        self.criterion = options.criterion
        self.save_dir = options.save_dir
        self.freeze = options.freeze
        self.accumulate_over_n_batches = options.accumulate_over_n_batches
        self.distributed = options.distributed
        self.gpu_rank = options.gpu_rank
        self.n_gpus = options.n_gpus
        self.test_time_bn = options.test_time_bn
        self.dtype = options.dtype
        if self.distributed: self.config_distributed(self.n_gpus, self.gpu_rank)
        self.mixedprecision = options.mixedprecision
        if self.mixedprecision:
            self.grad_scaler = GradScaler(init_scale=8192, growth_interval=4)
        self.multilabel = options.multilabel
        self.regression = options.regression
        self.reset_epoch_stats()

    def config_distributed(self, n_gpus, gpu_rank=None):
        self.sync_networks_distributed_if_needed()
        self.n_gpus = torch.cuda.device_count() if n_gpus is None else n_gpus
        assert gpu_rank is not None
        self.gpu_rank = gpu_rank

    def sync_networks_distributed_if_needed(self, check=True):
        if self.distributed: self.sync_network_distributed(self.net, check)

    def sync_network_distributed(self, net, check=True):
        for _, param in net.named_parameters():
            dist.broadcast(param.data, 0)

        for mod in net.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                dist.broadcast(mod.running_mean, 0)
                dist.broadcast(mod.running_var, 0)

    def prepare_network_for_training(self):
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        self.net.train()
        for mod in self.freeze:
            mod.eval()

    def prepare_network_for_evaluation(self):
        torch.set_grad_enabled(False)
        self.net.eval()
        self.prepare_batchnorm_for_evaluation(self.net)

    def prepare_batchnorm_for_evaluation(self, net):
        for mod in net.modules():
            if isinstance(mod, torch.nn.BatchNorm2d): 
                if self.test_time_bn: mod.train()
                else: mod.eval()

    def reset_epoch_stats(self):
        self.accumulated_loss = 0
        self.accumulated_accuracy = 0
        self.batches_evaluated = 0
        self.images_evaluated = 0
        self.accumulated_batches = 0

        self.all_predictions = []
        self.all_labels = []

    def save_batch_stats(self, loss, accuracy, predictions, labels):
        self.accumulated_loss += float(loss) * len(labels)
        self.accumulated_accuracy += accuracy * len(labels)
        self.batches_evaluated += 1
        self.images_evaluated += len(labels)

        self.all_predictions.append(predictions)
        self.all_labels.append(labels.copy())  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189 | fix RuntimeError: received 0 items of ancdata

    def stack_epoch_predictions(self):
        self.all_predictions, self.all_labels = self.epoch_predictions_and_labels(gather=True)

    def correct_loss_for_multigpu(self):
        self.accumulated_loss = 0.0
        self.accumulated_accuracy = 0.0
        for pred, label in zip(self.all_predictions, self.all_labels):
            self.accumulated_loss += float(self.criterion(pred[None], label[None]))
            self.accumulated_accuracy += self.accuracy_with_predictions(pred[None], label)
        self.images_evaluated = len(self.all_predictions)

    def epoch_predictions_and_labels(self, gather=False):
        preds, labels = [], []
        if len(self.all_predictions) > 0:
            preds = np.vstack(self.all_predictions)
            labels = np.vstack(self.all_labels)

            if self.distributed and gather:
                preds = list(self.gather(preds))
                labels = list(self.gather(labels))

            preds = torch.from_numpy(np.array(preds))
            labels = torch.from_numpy(np.array(labels).astype(self.all_labels[0].dtype))

            # reshape to correct shapes
            if len(self.all_labels[0].shape) == 1: labels = labels.flatten()
            # labels = labels.view(-1, self.all_labels[0].shape[0])
            if len(self.all_predictions[0].shape) == 1: preds = preds.flatten()
            # preds = preds.view(-1, self.all_predictions[0].shape[0])
            return preds.float(), labels
        else:
            return torch.FloatTensor(), torch.LongTensor()

    def gather(self, results):
        results = torch.tensor(results, dtype=torch.float32).cuda()
        tensor_list = [results.new_empty(results.shape) for _ in range(self.n_gpus)]
        dist.all_gather(tensor_list, results)
        cpu_list = [tensor.cpu().numpy() for tensor in tensor_list]
        return np.concatenate(cpu_list, axis=0)

    def average_epoch_loss(self):
        if self.images_evaluated == 0: return -1
        return self.accumulated_loss / self.images_evaluated

    def average_epoch_accuracy(self):
        if self.images_evaluated == 0: return -1
        return self.accumulated_accuracy / self.images_evaluated

    def train_epoch(self, batch_callback) -> typing.Tuple[np.array, np.array]:
        self.sync_networks_distributed_if_needed()
        self.prepare_network_for_training()
        self.reset_epoch_stats()
        self.train_full_dataloader(batch_callback)
        self.stack_epoch_predictions()
        if self.distributed: self.correct_loss_for_multigpu()
        return self.all_predictions, self.all_labels

    def validation_epoch(self, batch_callback):
        self.prepare_network_for_evaluation()
        self.reset_epoch_stats()
        self.evaluate_full_dataloader(batch_callback)
        self.stack_epoch_predictions()
        if self.distributed: self.correct_loss_for_multigpu()
        return self.all_predictions, self.all_labels

    def train_full_dataloader(self, batch_callback):
        for x, y in self.dataloader:
            loss, accuracy, predictions = self.train_on_batch(x, y)
            self.save_batch_stats(loss, accuracy, predictions, y.cpu().numpy())
            batch_callback(self, self.batches_evaluated, loss, accuracy)

    def evaluate_full_dataloader(self, batch_callback):
        for x, y in self.dataloader:
            loss, accuracy, predictions = self.forward_batch(x, y)
            self.save_batch_stats(loss, accuracy, predictions, y.cpu().numpy())
            batch_callback(self, self.batches_evaluated, loss, accuracy)

    def forward_batch(self, x, y):
        if self.mixedprecision:
            with autocast():
                output, loss = self.forward_batch_with_loss(x, y)
        else:
            output, loss = self.forward_batch_with_loss(x, y)
        output = output.detach().cpu()
        accuracy = self.accuracy_with_predictions(output, y.cpu())
        # NOTE: removed a `del output` here, could cause memory issues
        return loss, accuracy, output.numpy()

    def forward_batch_with_loss(self, x, y):
        output = self.net.forward(x.cuda())
        label = y.cuda()
        loss = self.criterion(output, label)
        return output, loss

    def train_on_batch(self, x, y):
        loss, accuracy, predictions = self.forward_batch(x, y)
        full_loss = float(loss)
        loss = loss / self.accumulate_over_n_batches / self.n_gpus
        if self.mixedprecision: self.grad_scaler.scale(loss).backward()
        else: loss.backward()
        self.accumulated_batches += 1
        self.step_optimizer_if_needed()
        return full_loss, accuracy, predictions

    def step_optimizer_if_needed(self):
        if self.accumulated_batches == self.accumulate_over_n_batches:
            self.distribute_gradients_if_needed()
            if self.mixedprecision:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                # prohibit scales larger than 65536, training crashes, 
                # maybe due to gradient accumulation?
                if self.grad_scaler.get_scale() > 65536.0:
                    self.grad_scaler.update(new_scale=65536.0)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_batches = 0

    def distribute_gradients_if_needed(self):
        if self.distributed:
            for _, param in self.net.named_parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def save_checkpoint(self, name, epoch, additional={}):
        state = {
            'checkpoint': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        state.update(additional)
        print('Saving', 'checkpoint_' + name + '_' + str(epoch) + '_network')
        try:
            torch.save(state, self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'))
            torch.save(state, self.save_dir / pathlib.Path('checkpoint_' + name + '_last'))
        except Exception as e:
            print('WARNING: Network not stored', e)

    def checkpoint_available_for_name(self, name, epoch=-1):
        if epoch > -1:
            print(self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'))
            print(os.path.isfile(self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network')))
            return os.path.isfile(self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'))
        else:
            return os.path.isfile(self.save_dir / pathlib.Path('checkpoint_' + name + '_last'))

    def load_network_checkpoint(self, name):
        state = torch.load(self.save_dir / pathlib.Path('checkpoint_' + name))
        self.load_state_dict(state)

    def load_checkpoint(self, name, epoch=-1):
        if epoch > -1:
            state = torch.load(self.save_dir / pathlib.Path('checkpoint_' + name + '_' + str(epoch) + '_network'),
                               map_location=lambda storage, loc: storage)
        else:
            state = torch.load(self.save_dir / pathlib.Path('checkpoint_' + name + '_last'),
                               map_location=lambda storage, loc: storage)
        return state

    def load_state_dict(self, state):
        try: self.optimizer.load_state_dict(state['optimizer'])
        except KeyError: print('WARNING: Optimizer not restored')
        self.net.load_state_dict(state['state_dict'])

    def load_checkpoint_if_available(self, name, epoch=-1):
        if self.checkpoint_available_for_name(name, epoch):
            state = self.load_checkpoint(name, epoch)
            self.load_state_dict(state)
            return True, state
        return False, None

    def accuracy_with_predictions(self, predictions, labels):
        if self.regression:
            return 0
        if self.multilabel:
            equal = np.equal(np.round(torch.sigmoid(predictions.float())), labels.numpy() == 1)
            equal_c = np.sum(equal, axis=1)
            equal = (equal_c == labels.shape[1]).sum()
        elif predictions.shape[1] == 1:
            equal = np.equal(np.round(torch.sigmoid(predictions.float())), labels)
        else:
            equal = np.equal(np.argmax(torch.softmax(predictions.float(), dim=1), axis=1), labels)
        return float(equal.sum()) / float(predictions.shape[0])

