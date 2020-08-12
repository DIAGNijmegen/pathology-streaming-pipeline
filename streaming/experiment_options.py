import dataclasses
import argparse

@dataclasses.dataclass
class ExperimentOptions:
    """ REQUIRED """
    name: str = ''  # The name of the current experiment, used for saving checkpoints
    num_classes: int = 1  # The number of classes in the task

    train_csv: str = ''  # The filenames (without extension) and labels of train set
    val_csv: str = ''  # The filenames (without extension) and labels of validation or test set
    data_dir: str = ''  # The directory where the images reside
    copy_dir: str = ''  # If convert_to_vips is on, this is the directory where the .v files are saved
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
    gather_batch_on_one_gpu: bool = False
    weighted_sampler: bool = False # Oversample minority class, only works in binary tasks

    validation: bool = True  # Whether to run on validation set
    validation_interval: int = 1  # How many times to run on validation set, after n train epochs
    validation_whole_input: bool = False
    epoch_multiply: int = 1  # This will increase the size of one train epoch by reusing train images

    # speed
    variable_input_shapes: bool = False  # When the images vary a lot with size, this helps with speed
    mixedprecision: bool = True  # Paper is trained with full precision, but this is way faster
    normalize_on_gpu: bool = True  # Helps with RAM usage of dataloaders
    num_workers: int = 2  # Number of dataloader workers
    convert_to_vips: bool = False

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

