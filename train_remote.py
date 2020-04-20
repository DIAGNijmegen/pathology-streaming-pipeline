import numpy as np
import torch
import torch.hub
import torch.utils
import torch.utils.data

import os
import subprocess
import dataclasses

from train import ExperimentOptions, Experiment

@dataclasses.dataclass
class RemoteExperimentOptions(ExperimentOptions):
    home_dir: str = '/home/user/'
    copy_dir: str = '/home/user/data'

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
        subprocess.call(command, shell=True)
        print("Copying done")

    def configure_experiment(self):
        if self.verbose:
            self.copy_data()
        super().configure_experiment()

    def _get_dataset(self, validation, csv_file):
        dataset = super()._get_dataset(validation, csv_file)
        dataset.img_dir = self.settings.copy_dir
        return dataset

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
