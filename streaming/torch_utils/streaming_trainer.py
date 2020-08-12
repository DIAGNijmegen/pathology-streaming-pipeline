import numpy as np
import torch
import dataclasses
import os

from streaming.torch_utils.scnn import StreamingCNN
from streaming.torch_utils.checkpointed_trainer import CheckpointedTrainer, CheckpointedTrainerOptions

try:
    from torch.cuda.amp import autocast  # pylint: disable=import-error,no-name-in-module
except ModuleNotFoundError:
    pass

@dataclasses.dataclass
class StreamingTrainerOptions(CheckpointedTrainerOptions):
    tile_shape = (1, 3, 600, 600)
    train_streaming_layers: bool = True
    average_over_five_crops_of_size: int = -1
    normalize_on_gpu: bool = True

class StreamingCheckpointedTrainer(CheckpointedTrainer):
    def __init__(self, options, sCNN=None):
        self.tile_shape = options.tile_shape
        self.train_streaming_layers = options.train_streaming_layers
        super().__init__(options)
        for l in self.freeze: l.eval()
        if sCNN is None:
            # if os.path.isfile('cache'+str(options.tile_shape)):
            #     state_dict = torch.load('cache'+str(options.tile_shape))
            # else: 
            state_dict = None
            self.sCNN = StreamingCNN(self.checkpointed_net,
                                     self.tile_shape,
                                     verbose=(self.gpu_rank == 0),
                                     copy_to_gpu=True,
                                     statistics_on_cpu=(options.tile_shape[2] > 2800),
                                     normalize_on_gpu=options.normalize_on_gpu,
                                     state_dict=state_dict)
            # if self.gpu_rank != 1:
            self.sCNN.verbose = False
            # if state_dict is None:
            #     torch.save(self.sCNN.state_dict(), 'cache'+str(options.tile_shape))
        else:
            self.sCNN = sCNN

    def checkpoint_image_forward(self, x):
        if self.mixedprecision:
            with autocast(): 
                output = self.stream_image_forward(x)
        else:
            output = self.stream_image_forward(x)

        return output
        
    def stream_image_forward(self, x):
        if isinstance(self.checkpointed_net, torch.nn.Sequential) and len(self.checkpointed_net) == 0:
            output = x
        else:
            output = self.sCNN.forward(x)
        return output

    def backward_image(self, x, gradient):
        if self.train_streaming_layers:
            self.sCNN.backward(x, gradient[None])

class CheckpointedStreamingMultiClassTrainer(StreamingCheckpointedTrainer):
    def accuracy_with_predictions(self, predictions, labels):
        equal = np.equal(np.round(torch.sigmoid(predictions)), labels.numpy() == 1)
        equal_c = np.sum(equal, axis=1)
        correct = (equal_c == labels.shape[1]).sum()
        return correct / len(predictions)
