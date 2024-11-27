import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import List, Dict, Optional
from torchmetrics import Metric
from torchmetrics.text import BLEUScore

import pdb


class EdgeRelationAccuracy(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: Tensor, y_true: Tensor, mask: Tensor) -> None:
        y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
        b, seq_len, seq_len = y_pred.shape

        for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
            num_units = sample_mask.sum()
            pred = pred[:num_units, :num_units]
            true = true[:num_units, :num_units]
            true_indexes = true.argmax(dim=1)
            pred_indexes = pred.argmax(dim=1)
            self.correct += (pred_indexes == true_indexes).sum()
            self.total += pred_indexes.shape[0]
        

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
    


class TotalOrderAccuracy(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, y_pred: Tensor, y_true: Tensor, mask: Tensor) -> None:
        y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
        b, seq_len, seq_len = y_pred.shape

        for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
            num_units = sample_mask.sum()
            pred = pred[:num_units, :num_units]
            true = true[:num_units, :num_units]
            true_indexes = true.argmax(dim=1)
            pred_indexes = pred.argmax(dim=1)
            self.correct += int(torch.allclose(pred_indexes, true_indexes))
        self.total += b
        

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
    


class ROBleuScore(BLEUScore):
    def update(self, y_pred: Tensor, y_true: Tensor, mask: Tensor) -> None:
        y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
        b, seq_len, seq_len = y_pred.shape
        
        pred_seq, target_seq = [], [] 
        for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
            num_units = sample_mask.sum()
            pred = pred[:num_units, :num_units]
            true = true[:num_units, :num_units]
            true_indexes = true.argmax(dim=1)
            true_indexes = list(map(str, true_indexes.cpu().numpy().tolist()))
            pred_indexes = pred.argmax(dim=1)
            pred_indexes = list(map(str, pred_indexes.cpu().numpy().tolist()))
            pred_seq.append(' '.join(pred_indexes))
            target_seq.append([' '.join(true_indexes)])
        
        super().update(pred_seq, target_seq)
