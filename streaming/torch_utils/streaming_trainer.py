import numpy as np
import torch
import dataclasses

from .scnn import StreamingCNN
from .checkpointed_trainer import CheckpointedTrainer, CheckpointedTrainerOptions

if '1.6' in torch.__version__:  # type:ignore
    from torch.cuda.amp import autocast

@dataclasses.dataclass
class StreamingTrainerOptions(CheckpointedTrainerOptions):
    tile_shape = (1, 3, 600, 600)
    train_streaming_layers: bool = True
    normalize_on_gpu: bool = False

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
                                     verbose=True,
                                     copy_to_gpu=True,
                                     statistics_on_cpu=True,
                                     normalize_on_gpu=options.normalize_on_gpu,
                                     state_dict=state_dict)
            self.sCNN.verbose = False
            self.sCNN.disable()
            # if state_dict is None:
            #     torch.save(self.sCNN.state_dict(), 'cache'+str(options.tile_shape))
        else:
            self.sCNN = sCNN

    def train_epoch(self, batch_callback):
        self.sCNN.enable()
        all_predictions, all_labels = super().train_epoch(batch_callback)
        self.sCNN.disable()
        return all_predictions, all_labels

    def validation_epoch(self, batch_callback):
        self.sCNN.disable()
        self.sCNN.enable()
        all_predictions, all_labels = super().validation_epoch(batch_callback)
        self.sCNN.disable()
        return all_predictions, all_labels

    def checkpoint_image_forward(self, x):
        if self.mixedprecision:
            with autocast(): 
                output = self.stream_image_forward(x)
        else:
            output = self.stream_image_forward(x)

        return output
        
    def stream_image_forward(self, x):
        if isinstance(self.checkpointed_net, torch.nn.Sequential):
            if len(self.checkpointed_net) > 0:
                output = self.sCNN.forward(x)
                return output
            else:
                return x
        else:
            return self.sCNN.forward(x)

    def backward_batch_checkpointed(self, fmap_grad):
        if self.train_streaming_layers:
            for i, x in enumerate(self.batch_images):
                self.sCNN.backward(x, fmap_grad[i][None])

class CheckpointedStreamingMultiClassTrainer(StreamingCheckpointedTrainer):
    def accuracy_with_predictions(self, predictions, labels):
        equal = np.equal(np.round(torch.sigmoid(predictions)), labels.numpy() == 1)
        equal_c = np.sum(equal, axis=1)
        correct = (equal_c == labels.shape[1]).sum()
        return correct / len(predictions)
