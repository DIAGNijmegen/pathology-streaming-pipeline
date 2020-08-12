import pathlib
import math
import dataclasses
from pprint import pprint

import numpy as np
import torch
import torch.hub
import torch.utils
import torch.utils.data
import torch.distributed
import torchvision

from torch.utils.data.sampler import WeightedRandomSampler
from prefetch_generator import BackgroundGenerator

from streaming.tissue_dataset import TissueDataset
from streaming.torch_utils.samplers import OrderedDistributedSampler, DistributedWeightedRandomSampler
from streaming.torch_utils.streaming_trainer import \
    StreamingCheckpointedTrainer, StreamingTrainerOptions
from streaming.torch_utils.utils import count_parameters, progress_bar
from streaming.experiment_options import ExperimentOptions

from torch.utils.data import DataLoader

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Experiment():
    validator: StreamingCheckpointedTrainer
    validation_dataset: torch.utils.data.Dataset
    validation_loader: torch.utils.data.DataLoader

    trainer: StreamingCheckpointedTrainer
    train_dataset: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    train_sampler: torch.utils.data.DistributedSampler

    settings: ExperimentOptions

    distributed: bool
    world_size: int

    verbose: bool
    resumed: bool = False

    optimizer = None
    loss: torch.nn.Module
    epoch: int
    freeze_layers: list = []
    start_at_epoch: int = 0

    net: torch.nn.Module
    stream_net: torch.nn.Sequential

    def __init__(self, settings: ExperimentOptions, running_distributed, world_size):
        """Initialize an experiment, this is an utility class to get training
        quickly with this framework.

        This class is responsible for setting up the model, optimizer, loss,
        dataloaders and the trainer classes.

        Args:
            settings (ExperimentOptions): dataclass containing all the options
                for this experiment, parsed from cli arguments.
            distributed: whether we are gonna train on multiple gpus
            world_size: how many gpus are in the system
        """
        self.settings = settings
        self.verbose = (self.settings.local_rank == 0)
        self.world_size = world_size
        self.distributed = running_distributed

        # When training with mixed precision and only finetuning last layers, we
        # do not have to backpropagate the streaming layers
        if self.settings.mixedprecision and not self.settings.train_all_layers:
            self.settings.train_streaming_layers = False

        torch.cuda.set_device(int(self.settings.local_rank))
        torch.backends.cudnn.benchmark = True  # type:ignore

    def run_experiment(self):
        self.configure_experiment()
        if self.settings.only_eval: self.eval_epoch(0)
        else: self.train_and_eval_epochs()

    def configure_experiment(self):
        if self.distributed: self._test_distributed()
        self._configure_batch_size_per_gpu(self.world_size)
        self._configure_dataloaders()
        self._configure_model()
        self._configure_optimizer()
        self._configure_loss()
        self._configure_trainers()
        self._resume_if_needed()
        self._sync_distributed_if_needed()
        self._enable_mixed_precision_if_needed()
        self._log_details(self.net)
        if self.settings.variable_input_shapes: self._configure_tile_delta()

    def _test_distributed(self):
        if self.verbose: print('Test distributed')
        results = torch.FloatTensor([0])  # type:ignore
        results = results.cuda()
        tensor_list = [results.new_empty(results.shape) for _ in range(self.world_size)]
        torch.distributed.all_gather(tensor_list, results)
        if self.verbose: print('Succeeded distributed communication')

    def _configure_batch_size_per_gpu(self, world_size):
        """
        This functions calculates how we will devide the batch over multiple
        GPUs, and how many image gradients we are gonna accumulate before doing
        an optimizer step.
        """
        if self.settings.accumulate_batch == -1:
            self.settings.accumulate_batch = int(self.settings.batch_size / world_size)
            self.settings.batch_size = 1
        elif not self.settings.gather_batch_on_one_gpu:
            self.settings.batch_size = int(self.settings.batch_size / world_size)
            self.settings.accumulate_batch = self.settings.accumulate_batch
        else:
            self.settings.batch_size = int(self.settings.batch_size / world_size)

        if self.verbose:
            print(f'Per GPU batch-size: {self.settings.batch_size}, ' +
                  f'accumulate over batch: {self.settings.accumulate_batch}')

        assert self.settings.batch_size > 0
        assert self.settings.accumulate_batch > 0

    def train_and_eval_epochs(self):
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

        for _, param_group in enumerate(self.optimizer.param_groups):  # type:ignore
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
            self.save_predictions(preds, gt, str_e)
            self.log_eval_metrics(preds, gt, e)
            self.save_if_needed(e)

    def save_predictions(self, preds, gt, str_e):
        path = pathlib.Path(self.settings.save_dir)
        try:
            np.save(str(path / pathlib.Path(f'{self.settings.name}_eval_preds_{str_e}')), preds)
            np.save(str(path / pathlib.Path(f'{self.settings.name}_eval_gt_{str_e}')), gt)
        except Exception as exc:
            print(f'Predictions could not be written to disk: {exc}')

    def save_if_needed(self, e):
        if self.settings.save and not self.settings.only_eval:
            self.trainer.save_checkpoint(self.settings.name, e)

    def _configure_dataloaders(self):
        self.train_dataset = self._get_dataset(validation=False, csv_file=self.settings.train_csv)
        self.train_loader, self.train_sampler = self._get_dataloader(self.train_dataset, shuffle=True)
        self.validation_dataset = self._get_dataset(validation=True, csv_file=self.settings.val_csv)
        self.validation_loader, _ = self._get_dataloader(self.validation_dataset, shuffle=False)

    def _get_dataloader(self, dataset: torch.utils.data.Dataset, shuffle=True):
        batch_size, num_workers = 1, self.settings.num_workers
        sampler = None

        if self.settings.weighted_sampler:
            if shuffle:
                sampler = self.weighted_sampler(dataset)
                shuffle = False

        if self.distributed:
            if shuffle:
                shuffle = False
                sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                          num_replicas=torch.cuda.device_count(),
                                                                          rank=self.settings.local_rank)
            else:
                if self.distributed:
                    sampler = OrderedDistributedSampler(dataset, num_replicas=torch.cuda.device_count(),
                                                        rank=self.settings.local_rank, batch_size=1)

        # TODO: maybe disable automatic batching?
        # https://pytorch.org/docs/stable/data.html
        # pin memory True saves GPU memory?
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            pin_memory=False)

        return loader, sampler

    def weighted_sampler(self, dataset):
        labels = np.array([int(label) for _, label in dataset.images])
        total_pos = np.sum(labels)
        total_neg = len(labels) - total_pos
        weights = np.array(labels, dtype=np.float32)
        weights[labels==0] = total_pos / total_neg
        weights[labels==1] = 1
        if self.distributed:
            sampler = DistributedWeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
        else:
            sampler = WeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
        return sampler

    def _get_dataset(self, validation, csv_file):
        limit = self.settings.train_set_size
        if validation: limit = -1
        variable_input_shapes = self.settings.validation_whole_input if validation else self.settings.variable_input_shapes
        return TissueDataset(img_size=self.settings.image_size,
                             img_dir=self.settings.data_dir,
                             cache_dir=self.settings.copy_dir,
                             filetype=self.settings.filetype,
                             csv_fname=csv_file,
                             augmentations=(not validation),
                             limit_size=limit,
                             variable_input_shapes=variable_input_shapes,
                             tile_size=self.settings.tile_size,
                             multiply_len=self.settings.epoch_multiply if not validation else 1,
                             num_classes=self.settings.num_classes,
                             regression=self.settings.regression,
                             convert_to_vips=self.settings.convert_to_vips)

    def _train_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            progress_bar(batches_evaluated, math.ceil(len(self.train_dataset) / float(self.world_size)),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Train", avg_loss, avg_acc, loss))

    def _eval_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            progress_bar(batches_evaluated, math.ceil(len(self.validation_dataset) / float(self.world_size)),
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
        options.gather_batch_on_one_gpu = self.settings.gather_batch_on_one_gpu
        self.trainer = StreamingCheckpointedTrainer(options)

        self.validator = StreamingCheckpointedTrainer(options, sCNN=self.trainer.sCNN)
        self.validator.dataloader = self.validation_loader
        self.validator.accumulate_over_n_batches = 1

        # StreamingCheckpointedTrainer changes modules, reset optimizer!
        self._reset_optimizer()

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

        if self.settings.resuming and self.settings.resume_epoch == -1:
            resumed, state = self._try_resuming_last_checkpoint(resumed)

        if not resumed and self.settings.resume_epoch > -1:
            name = self.settings.resume_name if self.settings.resume_name else self.settings.name

            if self.settings.weight_averaging:
                window = 5
                resumed, state = self.resume_with_averaging(name,
                                                            self.settings.resume_epoch - math.floor(window / 2.),
                                                            self.settings.resume_epoch + math.floor(window / 2.))
            else:
                resumed, state = self._resume_with_epoch(name)

            print('Did not reset optimizer, maybe using lr from checkpoint')
            assert self.trainer.net == self.validator.net  # type:ignore
            assert resumed

        self.resumed = resumed
        self._calculate_starting_epoch(resumed, state)
        del state

    def _resume_with_epoch(self, name):
        resumed, state = self.trainer.load_checkpoint_if_available(name, self.settings.resume_epoch)
        resumed, state = self.validator.load_checkpoint_if_available(name, self.settings.resume_epoch)
        return resumed, state

    def resume_with_averaging(self, resume_name, begin_epoch, after_epoch, window=5):
        param_dict = {}

        # sum all parameters
        # checkpoint_range = np.arange(epoch - math.floor(window / 2.), epoch + math.ceil(window / 2.))
        checkpoint_range = np.arange(begin_epoch, after_epoch)
        for i in checkpoint_range:
            try:
                current_param_dict = dict(self.trainer.load_checkpoint(resume_name, i))
            except Exception as e:
                print(f'Did not find {i}', e)
                return False, None

            if not param_dict:
                param_dict = current_param_dict
            else:
                for key in ['state_dict_net', 'state_dict_checkpointed']:
                    for name in current_param_dict[key]:
                        param_dict[key][name].data.add_(current_param_dict[key][name].data)

        for key in ['state_dict_net', 'state_dict_checkpointed']:
            for name in param_dict[key]:
                param_dict[key][name].data.div_(float(len(checkpoint_range)))

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
        self._configure_optimizer()
        self.trainer.optimizer = self.optimizer
        self.validator.optimizer = self.optimizer

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
            pprint(dataclasses.asdict(self.settings))
            print()

if __name__ == "__main__":
    distributed = (torch.cuda.device_count() > 1)
    torch.manual_seed(0)
    np.random.seed(0)  # type:ignore

    exp_options = ExperimentOptions()
    parser = exp_options.configure_parser_with_options()
    args = parser.parse_args()
    exp_options.parser_to_options(vars(args))

    print('Starting', args.local_rank, 'of n gpus:', torch.cuda.device_count())

    if distributed: torch.distributed.init_process_group(backend='nccl', init_method='env://')  # type:ignore
    exp = Experiment(exp_options, distributed, torch.cuda.device_count())
    exp.run_experiment()
