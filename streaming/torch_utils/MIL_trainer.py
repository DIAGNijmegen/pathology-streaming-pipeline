from collections import namedtuple

import torch
import wandb
import numpy as np

from .trainer import Trainer, TrainerOptions

InstancePred = namedtuple('InstancePred', 'index, pred, label, loss, img')

class MILTrainerOptions(TrainerOptions):
    batch_size: int = 16
    dataloader_indices_to_instance = None
    instances_dataloader = None
    instance_indices = []

class MILTrainer(Trainer):
    def __init__(self, options: MILTrainerOptions):
        self.batch_size = options.batch_size
        self.dataloader_indices_to_instance = options.dataloader_indices_to_instance
        self.instances_dataloader = options.instances_dataloader
        self.instance_indices = options.instance_indices
        super().__init__(options)

    def reset_epoch_stats(self):
        super().reset_epoch_stats()
        self.predictions_per_instance = {}

    def forward_batch(self, x, y):
        output = self.net.forward(x.cuda())
        return output

    def stack_epoch_predictions(self):
        if len(self.predictions_per_instance) > 0:
            values = sorted(self.predictions_per_instance.keys())
            self.all_predictions = [self.predictions_per_instance[ins].pred for ins in values]
            self.all_labels = [self.predictions_per_instance[ins].label for ins in values]
            self.all_max_preds = [self.predictions_per_instance[ins].pred.item() for ins in values]
            self.all_max_instances = [self.predictions_per_instance[ins].index for ins in values]
            self.all_max_labels = [self.predictions_per_instance[ins].label for ins in values]
            self.all_max_images = [self.predictions_per_instance[ins].img for ins in values]
        else:
            super().stack_epoch_predictions()

    def MIL_balanced_metrics(self):
        accuracies, losses, lengths = np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        for pred in self.predictions_per_instance.values():
            accuracies[int(pred.label)] += 1 if np.round(pred.pred.numpy()) == pred.label else 0
            losses[int(pred.label)] += pred.loss.item()
            lengths[int(pred.label)] += 1
        accuracy = (accuracies / lengths).sum() / 2
        loss = (losses / lengths).sum() / 2
        return accuracy, loss
    
    def MIL_metrics(self):
        accuracies, losses = np.array([0.]), np.array([0.])
        n = len(self.predictions_per_instance)
        if n == 0: return 0, 0
        for pred in self.predictions_per_instance.values():
            accuracies += 1 if np.round(pred.pred.numpy()) == pred.label else 0
            losses += pred.loss.item()
        accuracy = accuracies / n
        loss = losses / n
        return accuracy, loss

    def validation_epoch(self, batch_callback):
        super().validation_epoch(batch_callback)
        return self.all_predictions, self.all_labels, self.all_max_instances, self.all_max_preds, self.all_max_labels, self.all_max_images

    def evaluate_full_dataloader(self, batch_callback):
        bs = self.instances_dataloader.batch_size
        self.current_instance = -1
        for i, (x, y) in enumerate(self.instances_dataloader):
            if i % 100 == 0:
                images = []
                for t in x:
                    img = (np.array(t)).transpose(1, 2, 0)
                    img[:, :] *= np.array([0.229, 0.224, 0.225])
                    img[:, :] += np.array([0.485, 0.456, 0.406])
                    images.append(wandb.Image(img))
                wandb.log({"patch": images[0:20]})

            begin_i, end_i = i * bs, (i+1) * bs
            indices = self.instance_indices[begin_i:end_i]
            instances = self.dataloader_indices_to_instance[indices]
            loss, accuracy, predictions = self.evaluate_batch(x, y)
            self.accumulate_max_predictions(indices, instances, predictions, y, x)
            batch_callback(self, self.batches_evaluated, loss, accuracy)

        # should be in a function
        torch.save(self.current_max, f'/home/user/img_{self.current_instance}')

    def accumulate_max_predictions(self, indices, instances, predictions, labels, x):
        self.images_evaluated += len(labels)
        self.batches_evaluated += 1

        predictions = torch.tensor(predictions)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
        for patch_i, instance, pred, label, loss, img in zip(indices, instances, predictions, labels, losses, x):
            if instance in self.predictions_per_instance:
                max_pred = self.predictions_per_instance[instance]  # type: InstancePred
            else:
                max_pred = None  # type: ignore

            # max_label = label
            # if max_pred and label.item() != max_pred.label:
            # print('instance', instance, 'has multiple labels', label.item(), max_label)

            pred = torch.sigmoid(pred)
            label = label.numpy().copy()

            if not max_pred or pred > max_pred.pred:
                img = (np.array(img)).transpose(1, 2, 0)
                img[:, :] *= np.array([0.229, 0.224, 0.225])
                img[:, :] += np.array([0.485, 0.456, 0.406])
                img *= 255
                img = img.astype(np.uint8)

                if instance != self.current_instance:
                    if self.current_instance != -1:
                        torch.save(self.current_max, f'/home/user/img_{self.current_instance}')
                    self.current_instance = instance

                self.current_max = img
                new_prediction = InstancePred(patch_i, pred, label, loss, f'/home/user/img_{instance}')
                self.predictions_per_instance[instance] = new_prediction

