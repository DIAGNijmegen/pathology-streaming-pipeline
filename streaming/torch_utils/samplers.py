import torch
import numpy as np
import math
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
import torch.distributed as dist
from torch._six import int_classes as _int_classes

class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=1):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(
            len(self.dataset) * 1.0 / (self.num_replicas * batch_size)) * batch_size)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = np.array(indices).reshape(-1, self.batch_size)
        # test_indices = np.array(indices).reshape(-1, self.num_replicas)
        # indices = test_indices[:, self.rank] 

        test_indices = np.array(indices).reshape(self.num_replicas, -1)
        indices = test_indices[self.rank] 

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedWeightedRandomSampler(WeightedRandomSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = int(math.ceil(num_samples * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)  # type:ignore

        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples
