import dataclasses
import os
import subprocess
import glob
import pyvips
from tqdm import tqdm

import numpy as np
import torch
import torch.hub
import torch.utils
import torch.utils.data
import wandb
import PIL

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
    copy_dir: str = '/home/user/data'
    wandb_key: str = ''
    wandb_project: str = 'streaming'
    convert_to_v: bool = True

def fix_seed():
    torch.manual_seed(0)
    np.random.seed(0)  # type:ignore

class RemoteExperiment(Experiment):
    settings: RemoteExperimentOptions

    def __init__(self, settings: RemoteExperimentOptions, distributed, world_size):
        super().__init__(settings, distributed, world_size)

    def copy_data(self):
        print("Copying data")

        # kick gluster
        subprocess.check_output(["ls", self.settings.data_dir])

        rsync_param = "-av"
        ddir = self.settings.data_dir
        cdir = self.settings.copy_dir
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        command = f"rsync {rsync_param} --info=progress2 --info=name0 {ddir}/ {cdir}"
        subprocess.call(command, shell=True)
        print("Copying done")

    def convert_to_pyvips(self):
        cdir = self.settings.copy_dir
        images = glob.glob(f'{cdir}/*{self.settings.filetype}')
        for image_fname in tqdm(images):
            image = pyvips.Image.new_from_file(image_fname, access='sequential')
            image.write_to_file(image_fname.replace(self.settings.filetype, '.v'))
        self.settings.filetype = '.v'

    def configure_experiment(self):
        if self.verbose:
            self.copy_data()
        if self.settings.convert_to_v:
            self.convert_to_pyvips()
        super().configure_experiment()
        if self.settings.wandb_key and self.verbose:
            initialize_wandb(self.settings.name, [self.stream_net],
                             self.settings.wandb_project, self.settings.wandb_key)
            # self._log_test_image()

    def _log_test_image(self):
        augmented_image, label = self.train_dataset[0]
        augmented_image = augmented_image.numpy()
        augmented_image = augmented_image.transpose(1, 2, 0)
        augmented_image = PIL.Image.fromarray(augmented_image)
        augmented_image = augmented_image.resize((512, 512))
        wandb.log({'data/example_train_image': [wandb.Image(augmented_image, caption=str(label))]})

        augmented_image, label = self.train_dataset[-1]
        augmented_image = augmented_image.numpy()
        augmented_image = augmented_image.transpose(1, 2, 0)
        augmented_image = PIL.Image.fromarray(augmented_image)
        augmented_image = augmented_image.resize((512, 512))
        wandb.log({'data/example_train_image': [wandb.Image(augmented_image, caption=str(label))]})

        augmented_image, label = self.validation_dataset[0]
        augmented_image = augmented_image.numpy()
        augmented_image = augmented_image.transpose(1, 2, 0)
        augmented_image = PIL.Image.fromarray(augmented_image)
        augmented_image = augmented_image.resize((512, 512))
        wandb.log({'data/example_val_image': [wandb.Image(augmented_image, caption=str(label))]})

        augmented_image, label = self.validation_dataset[-1]
        augmented_image = augmented_image.numpy()
        augmented_image = augmented_image.transpose(1, 2, 0)
        augmented_image = PIL.Image.fromarray(augmented_image)
        augmented_image = augmented_image.resize((512, 512))
        wandb.log({'data/example_val_image': [wandb.Image(augmented_image, caption=str(label))]})

    def _get_dataset(self, validation, csv_file):
        self.settings.data_dir = self.settings.copy_dir
        dataset = super()._get_dataset(validation, csv_file)
        return dataset

    def _train_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            if self.settings.wandb_key:
                wandb.log({'epoch': self.epoch, 'train/accuracy_batch': accuracy})
                wandb.log({'epoch': self.epoch, 'train/loss_batch': loss})
            progress_bar(batches_evaluated, len(self.train_dataset),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Train", avg_loss, avg_acc, loss))

    def _eval_batch_callback(self, evaluator, batches_evaluated, loss, accuracy):
        if self.verbose and self.settings.progressbar:
            avg_acc = evaluator.average_epoch_accuracy()
            avg_loss = evaluator.average_epoch_loss()
            if self.settings.wandb_key:
                wandb.log({'epoch': self.epoch, 'val/accuracy_batch': accuracy})
                wandb.log({'epoch': self.epoch, 'val/loss_batch': loss})
            progress_bar(batches_evaluated, len(self.validation_dataset),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Val", avg_loss, avg_acc, loss))

    def log_train_metrics(self, preds, gt, e):
        super().log_train_metrics(preds, gt, e)
        if self.settings.wandb_key and self.verbose:
            wandb.log({'epoch': self.epoch, 'train/accuracy_epoch': self.trainer.average_epoch_accuracy()})
            wandb.log({'epoch': self.epoch, 'train/loss_epoch': self.trainer.average_epoch_loss()})

    def log_eval_metrics(self, preds, gt, e):
        super().log_eval_metrics(preds, gt, e)
        if self.settings.wandb_key and self.verbose:
            wandb.log({'epoch': self.epoch, 'val/accuracy_epoch': self.validator.average_epoch_accuracy()})
            wandb.log({'epoch': self.epoch, 'val/loss_epoch': self.validator.average_epoch_loss()})

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
