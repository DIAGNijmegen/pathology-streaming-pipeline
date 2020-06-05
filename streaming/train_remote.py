import dataclasses
import glob
import os
import math
import subprocess
from io import StringIO
from subprocess import PIPE, Popen

import numpy as np
import PIL
import pyvips
import torch
import torch.hub
import torch.utils
import torch.utils.data
import wandb
from tqdm import tqdm

from streaming.torch_utils.utils import progress_bar
from streaming.train import Experiment, ExperimentOptions


def initialize_wandb(exp_name, nets, project, key=''):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    wandb.init(project=project, name=exp_name, resume=exp_name)
    for net in nets: wandb.watch(net, log='all')

@dataclasses.dataclass
class RemoteExperimentOptions(ExperimentOptions):
    home_dir: str = '/home/user/'
    wandb_key: str = ''
    wandb_project: str = 'streaming'
    copy: bool = True

def fix_seed():
    torch.manual_seed(0)
    np.random.seed(0)  # type:ignore

class RemoteExperiment(Experiment):
    settings: RemoteExperimentOptions

    def __init__(self, settings: RemoteExperimentOptions, distributed, world_size):
        super().__init__(settings, distributed, world_size)

    def copy_data(self):
        print("Copying data")

        rsync_param = "-av"
        ddir = self.settings.data_dir
        cdir = self.settings.copy_dir
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        command = f"rsync {rsync_param} --info=progress2 --info=name0 {ddir}/ {cdir}"

        retry = True 
        while retry:
            # kick gluster
            subprocess.check_output(["ls", self.settings.data_dir])
            retry = False
            with Popen(command.split(' '), stdout=PIPE, bufsize=1, universal_newlines=True) as p,  StringIO() as buf:
                i = 0
                for line in p.stdout:
                    if i % 100 == 0: print(line, end='')
                    i += 1
                    if 'vanished' in line: retry = True
                    buf.write(line)  # type:ignore

        print("Copying done")
        self.settings.data_dir = self.settings.copy_dir

    def configure_experiment(self):
        if self.settings.wandb_key and self.verbose:
            initialize_wandb(self.settings.name, [], self.settings.wandb_project, self.settings.wandb_key)
            # self._log_test_image()
        if self.verbose and self.settings.copy:
            self.copy_data()
        super().configure_experiment()

    def _log_test_image(self):
        print('Sending test images to wandb...')
        try:
            images = []
            for _ in range(3):
                for image, label in self.train_loader:
                    augmented_image = image[0].numpy()
                    augmented_image = augmented_image.transpose(1, 2, 0)
                    augmented_image = PIL.Image.fromarray(augmented_image)
                    augmented_image = augmented_image.resize((512, 512))
                    images.append(wandb.Image(augmented_image, caption=str(label)))
                    del augmented_image
                    break
            wandb.log({'data/example_train_image': images})

            images = []
            for _ in range(3):
                for image, label in self.validation_loader:
                    augmented_image = image[0].numpy()
                    augmented_image = augmented_image.transpose(1, 2, 0)
                    augmented_image = PIL.Image.fromarray(augmented_image)
                    augmented_image = augmented_image.resize((512, 512))
                    images.append(wandb.Image(augmented_image, caption=str(label)))
                    del augmented_image
                    break
            wandb.log({'data/example_val_image': images})
        except Exception as e:
            print(e)

    def _get_dataset(self, validation, csv_file):
        dataset = super()._get_dataset(validation, csv_file)
        return dataset

    def _train_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            if self.settings.wandb_key:
                if self.settings.mixedprecision:
                    try: wandb.log({'grad-scale': self.trainer.grad_scaler.get_scale()})
                    except Exception as e: print(e)
                else:
                    wandb.log({'train/batch_loss': loss})
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

    def log_train_metrics(self, preds, gt, e):
        super().log_train_metrics(preds, gt, e)
        if self.settings.wandb_key and self.verbose:
            try:
                wandb.log({'epoch': self.epoch, 'train/accuracy_epoch': self.trainer.average_epoch_accuracy()})
                wandb.log({'epoch': self.epoch, 'train/loss_epoch': self.trainer.average_epoch_loss()})
            except Exception as e: print(e)

    def log_eval_metrics(self, preds, gt, e):
        super().log_eval_metrics(preds, gt, e)
        if self.settings.wandb_key and self.verbose:
            try:
                wandb.log({'epoch': self.epoch, 'val/accuracy_epoch': self.validator.average_epoch_accuracy()})
                wandb.log({'epoch': self.epoch, 'val/loss_epoch': self.validator.average_epoch_loss()})
            except Exception as e: print(e)

if __name__ == "__main__":
    distributed = (torch.cuda.device_count() > 1)
    fix_seed()

    options = RemoteExperimentOptions()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))

    print('Starting', args.local_rank, 'of n gpus:', torch.cuda.device_count())

    if distributed: torch.distributed.init_process_group(backend='nccl', init_method='env://')  # type:ignore
    exp = RemoteExperiment(options, distributed, torch.cuda.device_count())
    exp.run_experiment()
