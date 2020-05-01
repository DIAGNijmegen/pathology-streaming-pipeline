import argparse
import dataclasses
import pathlib
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.hub
import torch.utils
import torch.utils.data
import torchvision

from .tissue_dataset import TissueDataset
from .torch_utils.samplers import (DistributedWeightedRandomSampler,
                                  OrderedDistributedSampler)
from .torch_utils.streaming_trainer import (StreamingCheckpointedTrainer,
                                           StreamingTrainerOptions)
from .torch_utils.utils import count_parameters, progress_bar


@dataclasses.dataclass
class ExperimentOptions:
    """ REQUIRED """
    name: str = ''  # The name of the current experiment, used for saving checkpoints
    num_classes: int = 1  # The number of classes in the task

    train_csv: str = ''  # The filenames (without extension) and labels of train set
    val_csv: str = ''  # The filenames (without extension) and labels of validation or test set
    data_dir: str = ''  # The directory where the images reside
    filetype: str = '.jpg'  #  The file-extension of the images
    save_dir: str = ''  # Where to save the checkpoints

    """ NOT REQUIRED """
    train_set_size: int = -1  # Sometimes you want to test on smaller train-set you can limit the n-images here

    # pretrain options
    pretrained: bool = True  # Whether to use ImageNet weights

    # train options
    image_size: int = 16384  # Effective input size of the network
    tile_size: int = 5120  # The input/tile size of the streaming-part of the network
    epochs: int = 50  # How many epochs to train
    lr: float = 1e-4  # Learning rate
    batch_size: int = 16  # Effective mini-batch size
    multilabel: bool = False
    regression: bool = False

    validation: bool = True  # Whether to run on validation set
    validation_interval: int = 1  # How many times to run on validation set, after n train epochs
    epoch_multiply: int = 1  # This will increase the size of one train epoch by reusing train images

    # speed
    variable_input_shapes: bool = False  # When the images vary a lot with size, this helps with speed
    mixedprecision: bool = True  # Paper is trained with full precision, but this is way faster
    normalize_on_gpu: bool = True  # Helps with RAM usage of dataloaders
    num_workers: int = 1  # Number of dataloader workers

    # model options
    resnet: bool = True  # Only resnet is tested so far
    mobilenet: bool = False  # Experimental
    train_streaming_layers: bool = True  # Whether to backpropagate of streaming-part of network
    train_all_layers: bool = False  # Whether to finetune whole network, or only last block

    # save and logging options
    resuming: bool = True  # Will restart from the last checkpoint with same experiment-name
    resume_name: str = ''  # Restart from another experiment with this name
    resume_epoch: int = -1  # Restart from specific epoch
    save: bool = True  # Save checkpoints
    progressbar: bool = True  # Show the progressbar

    # evaluation options
    weight_averaging: bool = False  # average weights over 5 epochs around picked epoch
    only_eval: bool = False  # Only do one evaluation epoch

    local_rank: int = 0  # Do not touch, used by PyTorch when training distributed
    accumulate_batch: int = -1  # Do not touch, is calculated automatically

    def configure_parser_with_options(self):
        """ Create an argparser based on the attributes """
        parser = argparse.ArgumentParser(description='MultiGPU streaming')
        for name, default in dataclasses.asdict(self).items():
            argname = '--' + name
            tp = type(default)
            if tp is bool:
                if default == True:
                    argname = '--no_' + name
                    parser.add_argument(argname, action='store_false', dest=name)
                else: parser.add_argument(argname, action='store_true')
            else:
                parser.add_argument(argname, default=default, type=tp)
        return parser

    def parser_to_options(self, parsed_args: dict):
        """ Parse an argparser """
        for name, value in parsed_args.items():
            self.__setattr__(name, value)

def fix_seed():
    torch.manual_seed(0)
    np.random.seed(0)  # type:ignore

class Experiment(object):
    validator: StreamingCheckpointedTrainer
    validation_dataset: torch.utils.data.Dataset
    validation_loader: torch.utils.data.DataLoader
    trainer: StreamingCheckpointedTrainer
    train_dataset: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    train_sampler: torch.utils.data.DistributedSampler
    settings: ExperimentOptions
    batch_size: int
    distributed: bool
    world_size: int
    verbose: bool

    def __init__(self, settings: ExperimentOptions, distributed, world_size):
        self.settings = settings
        self.verbose = (self.settings.local_rank == 0)
        self._configure_batch_size(world_size)

        self.world_size = world_size
        self.distributed = distributed

        if self.settings.mixedprecision and not self.settings.train_all_layers:
            self.settings.train_streaming_layers = False

        torch.cuda.set_device(int(self.settings.local_rank))
        torch.backends.cudnn.benchmark = True  # type:ignore

    def _configure_batch_size(self, world_size):
        if self.settings.accumulate_batch == -1:
            self.settings.accumulate_batch = int(self.settings.batch_size / world_size)
            self.settings.batch_size = 1
        else:
            self.settings.batch_size = int(self.settings.batch_size / world_size) 
            self.settings.accumulate_batch = self.settings.accumulate_batch

        if self.verbose:
            print(f'Per GPU batch-size: {self.settings.batch_size}, ' + 
                  f'accumulate over batch: {self.settings.accumulate_batch}')

        assert self.settings.batch_size > 0
        assert self.settings.accumulate_batch > 0

    def run_experiment(self):
        self.configure_experiment()
        if self.settings.only_eval: self.eval_epoch(0)
        else: self.train_epochs()

    def configure_experiment(self):
        if self.distributed:
            self._test_distributed()
        self._configure_dataloaders()
        self._configure_model()
        self._configure_optimizer()
        self._configure_loss()
        self._configure_trainers()
        self._resume_if_needed()
        self._sync_distributed_if_needed()
        self._enable_mixed_precision_if_needed()
        self._log_details(self.net)
        if self.settings.variable_input_shapes:
            self._configure_tile_delta()

    def _test_distributed(self):
        if self.verbose: print('Test distributed')
        results = torch.FloatTensor([0])  # type:ignore
        results = results.cuda()
        tensor_list = [results.new_empty(results.shape) for i in range(self.world_size)]
        dist.all_gather(tensor_list, results)
        if self.verbose: print('Succeeded distributed communication')

    def train_epochs(self):
        epochs_to_train = np.arange(self.start_at_epoch, self.settings.epochs)
        for e in epochs_to_train:
            self.train_epoch(e, self.trainer)
            if self.settings.validation and e % self.settings.validation_interval == 0: 
                self.eval_epoch(e)

    def train_epoch(self, e, trainer):
        self.epoch = e
        preds, gt = trainer.train_epoch(self._train_batch_callback)
        if self.verbose: self.log_train_metrics(preds, gt, e)
        if self.distributed: self.train_sampler.set_epoch(int(e + 10))
        if self.settings.mixedprecision and e == 0: self.trainer.grad_scaler.set_growth_interval(20)

    def log_train_metrics(self, preds, gt, e):
        print('Train loop, example predictions.\nLabels:\n', gt.flatten()[:10],
              '\nPredictions\n', preds[:10])

        avg_acc = self.trainer.average_epoch_accuracy()
        avg_loss = self.trainer.average_epoch_loss()
        print('Train loop, accuracy', avg_acc, 'loss', avg_loss)

        for i, param_group in enumerate(self.optimizer.param_groups):  # type:ignore
            print('Current lr:', param_group['lr'])

    def log_eval_metrics(self, preds, gt, e):
        avg_acc = self.validator.average_epoch_accuracy()
        avg_loss = self.validator.average_epoch_loss()
        print('Evaluation loop, accuracy', avg_acc, 'loss', avg_loss)

    def eval_epoch(self, e):
        preds, gt = self.validator.validation_epoch(self._eval_batch_callback)

        if self.verbose:
            if self.settings.only_eval: str_e = self.settings.resume_epoch
            else: str_e = str(e)

            path = pathlib.Path(self.settings.save_dir)
            np.save(str(path / pathlib.Path(f'{self.settings.name}_eval_preds_{str_e}')), preds)
            np.save(str(path / pathlib.Path(f'{self.settings.name}_eval_gt_{str_e}')), gt)

            self.log_eval_metrics(preds, gt, e)
            self.save_if_needed(e)

    def save_if_needed(self, e):
        if self.settings.save and not self.settings.only_eval:
            try:
                self.trainer.save_checkpoint(self.settings.name, e)
            except Exception as error:
                print(f'Saving model failed {error}')

    def _configure_dataloaders(self):
        self.train_dataset = self._get_dataset(validation=False, csv_file=self.settings.train_csv)
        self.train_loader, self.train_sampler = self._get_dataloader(self.train_dataset, shuffle=True)
        self.validation_dataset = self._get_dataset(validation=True, csv_file=self.settings.val_csv)
        self.validation_loader, _ = self._get_dataloader(self.validation_dataset, shuffle=False)

    def _get_dataloader(self, dataset: torch.utils.data.Dataset, shuffle=True):
        batch_size, num_workers = 1, self.settings.num_workers
        sampler = None
        if self.distributed:
            if not shuffle:
                sampler = OrderedDistributedSampler(dataset, num_replicas=torch.cuda.device_count(),
                                                    rank=self.settings.local_rank, batch_size=1)
            else:
                # only works for binary problems; (e.g. cancer yes/no)
                # sampler = self._get_weighted_sampler_with_dataset(dataset)
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, 
                                                                          num_replicas=torch.cuda.device_count(),
                                                                          rank=self.settings.local_rank)

            shuffle = False

        # TODO: maybe disable automatic batching?
        # https://pytorch.org/docs/stable/data.html
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=False)

        return loader, sampler

    # a distributed weighted sampler with a binary task.
    def _get_weighted_sampler_with_dataset(self, dataset):
        total_pos = np.sum(dataset.labels)
        total_neg = len(dataset.labels) - total_pos
        weights = np.array(dataset.labels, dtype=np.float32)
        weights[weights==0] = total_pos / total_neg
        weights[weights==1] = 1
        sampler = DistributedWeightedRandomSampler(weights,
                                                   num_samples=int(total_neg*2),
                                                   replacement=True,
                                                   labels=dataset.labels)
        return sampler

    def _get_dataset(self, validation, csv_file):
        limit = self.settings.train_set_size
        if validation: limit = -1
        return TissueDataset(img_size=self.settings.image_size,
                             img_dir=self.settings.data_dir,
                             filetype=self.settings.filetype,
                             csv_fname=csv_file,
                             validation=validation,
                             augmentations=(not validation),
                             limit_size=limit,
                             variable_input_shapes=self.settings.variable_input_shapes,
                             tile_size=self.settings.tile_size,
                             multiply_len=self.settings.epoch_multiply if not validation else 1,
                             num_classes=self.settings.num_classes,
                             regression=self.settings.regression)

    def _train_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            progress_bar(batches_evaluated, math.ceil(len(self.train_dataset) / self.world_size),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Train", avg_loss, avg_acc, loss))

    def _eval_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            progress_bar(batches_evaluated, math.ceil(len(self.validation_dataset) / self.world_size),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Val", avg_loss, avg_acc, loss))

    def _configure_optimizer(self):
        params = self._get_trainable_params()
        self.optimizer = torch.optim.SGD(params, lr=self.settings.lr, momentum=0.9)

    def _get_trainable_params(self):
        if self.settings.train_all_layers:
            params = list(self.stream_net.parameters()) + list(self.net.parameters())
        else:
            print('WARNING: optimizer only training last params of network!')
            if self.settings.mixedprecision:
                params = list(self.net.parameters())
                for param in self.stream_net.parameters(): param.requires_grad = False
            else:
                params = list(self.stream_net[-1].parameters()) + list(self.net.parameters())
                for param in self.stream_net[:-1].parameters(): param.requires_grad = False
        return params

    def _configure_trainers(self):
        options = StreamingTrainerOptions()
        options.dataloader = self.train_loader
        options.net = self.net
        options.optimizer = self.optimizer
        options.criterion = self.loss  # type:ignore
        options.save_dir = pathlib.Path(self.settings.save_dir)
        options.checkpointed_net = self.stream_net
        options.batch_size = self.settings.batch_size
        options.accumulate_over_n_batches = self.settings.accumulate_batch
        options.n_gpus = self.world_size
        options.gpu_rank = int(self.settings.local_rank)
        options.distributed = self.distributed
        options.freeze = self.freeze_layers
        options.tile_shape = (1, 3, self.settings.tile_size, self.settings.tile_size)
        options.dtype = torch.uint8  # not needed, but saves memory
        options.train_streaming_layers = self.settings.train_streaming_layers
        options.mixedprecision = self.settings.mixedprecision
        options.normalize_on_gpu = self.settings.normalize_on_gpu
        options.multilabel = self.settings.multilabel
        options.regression = self.settings.regression
        self.trainer = StreamingCheckpointedTrainer(options)

        self.validator = StreamingCheckpointedTrainer(options, sCNN=self.trainer.sCNN)
        self.validator.dataloader = self.validation_loader
        self.validator.accumulate_over_n_batches = 1

    def _configure_tile_delta(self):
        if isinstance(self.trainer, StreamingCheckpointedTrainer):
            delta = self.settings.tile_size - (self.trainer.sCNN.tile_gradient_lost.left
                                               + self.trainer.sCNN.tile_gradient_lost.right)
            delta = delta // self.trainer.sCNN.output_stride[-1]
            delta *= self.trainer.sCNN.output_stride[-1]
            # if delta < 3000:
            #     delta = (3000 // delta + 1) * delta
            print('DELTA', delta.item())
            self.train_dataset.tile_delta = delta.item()
            self.validation_dataset.tile_delta = delta.item()

    def _configure_loss(self):
        weight = None
        if self.settings.multilabel:
            self.loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        elif self.settings.regression:
            self.loss = torch.nn.SmoothL1Loss() 
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def _configure_model(self):
        if self.settings.mobilenet:
            self._configure_mobilenet()
        elif self.settings.resnet:
            net = self._configure_resnet()
            self.stream_net, self.net = self._split_model(net)
        self._freeze_bn_layers()

    def _configure_mobilenet(self):
        net = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=self.settings.pretrained)
        net.features[1].conv[0][0].stride = (2, 2)  # TODO: or maybe add averagepool after first conv
        net.classifier = torch.nn.Linear(1280, 1)
        torch.nn.init.normal_(net.classifier.weight, 0, 0.01)
        torch.nn.init.zeros_(net.classifier.bias)
        self.stream_net = torch.nn.Sequential(*net.features[0:13]).cuda()
        net.features = net.features[13:]
        self.net = net.cuda()

    def _configure_resnet(self):
        net = torchvision.models.resnet34(pretrained=self.settings.pretrained)
        net.fc = torch.nn.Linear(512, self.settings.num_classes)
        torch.nn.init.xavier_normal_(net.fc.weight)
        net.fc.bias.data.fill_(0)  # type:ignore
        net.avgpool = torch.nn.AdaptiveMaxPool2d(1)
        net.cuda()
        # net = net.type(self.dtype)
        return net

    def _freeze_bn_layers(self):
        self.freeze_layers = [l for l in self.stream_net.modules() if isinstance(l, torch.nn.BatchNorm2d)]
        self.freeze_layers += [l for l in self.net.modules() if isinstance(l, torch.nn.BatchNorm2d)]

    def _sync_distributed_if_needed(self):
        if self.distributed:
            self.trainer.sync_networks_distributed_if_needed()
            self.validator.sync_networks_distributed_if_needed()

    def _resume_if_needed(self):
        state = None
        resumed = False

        if self.settings.resuming:
            resumed, state = self._try_resuming_last_checkpoint(resumed)

        if not resumed and self.settings.resume_epoch > -1:
            name = self.settings.resume_name if self.settings.resume_name else self.settings.name

            if self.settings.weight_averaging:
                resumed, state = self._resume_with_averaging(name)
            else:
                resumed, state = self._resume_with_epoch(name)

            print('Did not reset optimizer, maybe using lr from checkpoint')
            assert self.trainer.net == self.validator.net  # type:ignore
            assert resumed

        self._calculate_starting_epoch(resumed, state)
        del state

    def _resume_with_epoch(self, name):
        resumed, state = self.trainer.load_checkpoint_if_available(name, self.settings.resume_epoch)
        resumed, state = self.validator.load_checkpoint_if_available(name, self.settings.resume_epoch)
        return resumed, state

    def _resume_with_averaging(self, resume_name, window=5, beta=0.1):
        param_dict = {}

        # sum all parameters
        checkpoint_range = np.arange(self.settings.resume_epoch - window // 2, 
                                     self.settings.resume_epoch + window // 2 + 1)
        for i in checkpoint_range:
            try:
                current_param_dict = dict(self.trainer.load_checkpoint(resume_name, i))
            except:
                print(f'Did not find {i}')
                return False, None

            if not param_dict:
                param_dict = current_param_dict
            else:
                for key in ['state_dict_net', 'state_dict_checkpointed']:
                    for name in current_param_dict[key]:
                        param_dict[key][name].data.add_(current_param_dict[key][name].data)

        for key in ['state_dict_net', 'state_dict_checkpointed']:
            for name in param_dict[key]:
                param_dict[key][name].data.mul_(1 / window)

        self.trainer.net.load_state_dict(param_dict['state_dict_net'])
        self.trainer.checkpointed_net.load_state_dict(param_dict['state_dict_checkpointed'])

        return True, param_dict

    def _try_resuming_last_checkpoint(self, resumed):
        name = self.settings.name
        resumed, state = self.trainer.load_checkpoint_if_available(name)
        resumed, state = self.validator.load_checkpoint_if_available(name)

        if resumed and self.verbose:
            print("WARNING: look out, learning rate from resumed optimizer is used!")
        return resumed, state

    def _reset_optimizer(self):
        # reset optimizer
        self._configure_optimizer()
        self.trainer.optimizer = self.optimizer

    def _calculate_starting_epoch(self, resumed, state):
        start_at_epoch = 0
        if resumed:
            start_at_epoch = state['checkpoint'] + 1
            if self.verbose: print(f'Resuming from epoch {start_at_epoch}')

        self.start_at_epoch = start_at_epoch

    def _split_model(self, net):
        if not self.settings.mixedprecision: 
            stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                             net.layer1, net.layer2, net.layer3,
                                             net.layer4[0])
            net.layer4 = net.layer4[1:]
        else:
            stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                             net.layer1, net.layer2, net.layer3)

        net.layer1 = torch.nn.Sequential()
        net.layer2 = torch.nn.Sequential()
        net.layer3 = torch.nn.Sequential()
        net.conv1 = torch.nn.Sequential()
        net.bn1 = torch.nn.Sequential()
        net.relu = torch.nn.Sequential()
        net.maxpool = torch.nn.Sequential()

        return stream_net, net

    def _enable_mixed_precision_if_needed(self):
        if self.settings.mixedprecision:
            if isinstance(self.trainer, StreamingCheckpointedTrainer):
                self.trainer.sCNN.dtype = torch.half
                self.trainer.mixedprecision = True
            if isinstance(self.validator, StreamingCheckpointedTrainer):
                self.validator.sCNN.dtype = torch.half
                self.validator.mixedprecision = True

    def _log_details(self, net):
        if self.verbose:
            print("PyTorch version", torch.__version__)  # type: ignore
            print("Running distributed:", self.distributed)
            print("CUDA memory allocated:", torch.cuda.memory_allocated())
            print("Number of parameters (stream):", count_parameters(self.stream_net))
            print("Number of parameters (final):", count_parameters(self.net))
            print("Len train_loader", len(self.train_loader), '*', self.world_size)
            print("Len val_loader", len(self.validation_loader), '*', self.world_size)
            print("Arguments:", self.settings)
            print()

if __name__ == "__main__":
    distributed = (torch.cuda.device_count() > 1)
    fix_seed()

    options = ExperimentOptions()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))

    print('Starting', args.local_rank, 'of n gpus:', torch.cuda.device_count())

    if distributed: torch.distributed.init_process_group(backend='nccl', init_method='env://')  # type:ignore
    exp = Experiment(options, distributed, torch.cuda.device_count())
    exp.run_experiment()
