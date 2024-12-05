import numpy as np
import torch
import logging
import pdb
import torch.nn.functional as F


def multilabel_categorical_crossentropy(y_pred, y_true, mask):
    """
    https://kexue.fm/archives/7359
    Multilabel classification cross entropy
    Description: y_true and y_pred have the same shape, y_true's elements are either 0 or 1,
                 1 indicates that the corresponding class is a target class, 0 indicates that the corresponding class is not a target class.
                 mask and y_true have the same shape, also either 0 or 1, 0 indicates that the corresponding position does not calculate loss.
    Warning: Please ensure that the range of y_pred is all real numbers, which means that y_pred generally does not need to add an activation function,
             especially it cannot use sigmoid or softmax! The predictive stage outputs the classes with y_pred greater than 0.
             If you have any questions, please carefully read and understand the article.
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - (y_true + 1-mask) * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1-y_true + 1-mask) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def globalpointer_loss(y_pred, y_true, mask=None):
    """
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    mask:(batch_size, seq_len) or None
    """
    batch_size, ent_type_size, seq_len, seq_len_ = y_pred.shape
    assert seq_len == seq_len_
    if mask is None:
        mask = torch.ones((batch_size, ent_type_size, seq_len, seq_len)).long()
    else:
        mask = torch.einsum('ij,ik->ijk', mask, mask)
        mask = mask.repeat(1, ent_type_size, 1, 1)
    mask = mask.to(y_pred.device)
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    mask = mask.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true, mask)
    return loss


def sparse_multilabel_categorical_crossentropy(
    y_true, y_pred, mask_zero=False, epsilon=1e-7, Inf=1e12
):
    """稀疏版多标签分类的交叉熵
    https://github.com/JunnYu/GPLinker_pytorch/blob/dev/models.py
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + Inf
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def sparse_globalpointer_loss(y_true, y_pred):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(
        y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()



def cycle_loss(pred: torch.Tensor):
    """
    Compute cycle consistency loss for a probability matrix.

    Args:
        pred (torch.Tensor): Prob matrix (after softmax or sigmoid), shape (nxn).

    Returns:
        torch.Tensor: Computed cycle loss.
    """
    n = pred.shape[0]  # actually n+2
    device = pred.device

    # Initialize the loss
    loss = torch.tensor(0.0, device=device)

    # Accumulate loss for cycle powers from 1 to n
    pred_power = torch.eye(n, device=device)  # Start with the identity matrix
    for i in range(1, n + 1):  # Includes the final cycle (i=n)
        pred_power = torch.matmul(pred_power, pred)  # Efficient power calculation
        pred_diag = torch.diagonal(pred_power)  # Get diagonal elements
        if i < n:
            loss += torch.norm(pred_diag, p=2)  # L2 norm of the diagonal
        else:
            loss += torch.norm(pred_diag - 1, p=2)  # Final cycle: diag elements close to 1

    return loss / n



def degree_loss(pred: torch.Tensor):
    """
    Compute degree consistency loss for a probability matrix.

    Args:
        pred (torch.Tensor): Prob matrix (after softmax or sigmoid), shape (nxn).

    Returns:
        torch.Tensor: Computed degree loss.
    """
    # Row and column sums
    row_sums = torch.sum(pred, dim=1)  # Sum along columns (row-wise)
    column_sums = torch.sum(pred, dim=0)  # Sum along rows (column-wise)

    # Combine row and column discrepancies
    discrepancies = (1 - row_sums) ** 2 + (1 - column_sums) ** 2

    # Use stable log operation for loss
    loss = torch.log1p(torch.sum(discrepancies))  # log(1 + x) for stability
    return loss



def softmax_ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):
    """
        y_true: shape (batch_size, ent_type_size, seq_len, seq_len)
        y_pred: shape (batch_size, ent_type_size, seq_len, seq_len)
        mask: shape (batch_size, seq_len)
    """
    y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
    b, seq_len, seq_len = y_pred.shape
    batch_losses = torch.zeros(b)

    for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
        num_units = sample_mask.sum()
        pred = pred[:num_units, :num_units]
        true = true[:num_units, :num_units]
        true_indexes = true.argmax(dim=1)
        # pdb.set_trace()
        loss = F.cross_entropy(pred, true_indexes, reduce='mean')
        batch_losses[i] = loss

    return batch_losses.mean()


def softmax_ce_cycle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):
    """
        y_true: shape (batch_size, ent_type_size, seq_len, seq_len)
        y_pred: shape (batch_size, ent_type_size, seq_len, seq_len)
        mask: shape (batch_size, seq_len)
    """
    y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
    b, seq_len, seq_len = y_pred.shape
    batch_losses = torch.zeros(b)

    for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
        num_units = sample_mask.sum()
        pred = pred[:num_units, :num_units]
        true = true[:num_units, :num_units]
        true_indexes = true.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, true_indexes, reduce='mean')

        # cycle loss
        c_loss = cycle_loss(torch.softmax(pred, dim=1))

        batch_losses[i] = ce_loss + c_loss

    return batch_losses.mean()


def pairwise_bce_loss(y_pred, y_true, mask):
    y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
    b, seq_len, seq_len = y_pred.shape
    batch_losses = torch.zeros(b)

    for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
        num_units = sample_mask.sum()
        pred = pred[:num_units, :num_units]
        true = true[:num_units, :num_units]
        loss = F.binary_cross_entropy_with_logits(pred, true.to(torch.float32), reduction='mean')
        batch_losses[i] = loss

    return batch_losses.mean()


def pairwise_bce_degree_loss(y_pred, y_true, mask):
    y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
    b, seq_len, seq_len = y_pred.shape
    batch_losses = torch.zeros(b)

    for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
        num_units = sample_mask.sum()
        pred = pred[:num_units, :num_units]
        true = true[:num_units, :num_units]
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.to(torch.float32), reduction='mean')

        # degree loss
        d_loss = degree_loss(torch.sigmoid(pred))

        # total loss
        batch_losses[i] = bce_loss + d_loss

    return batch_losses.mean()



def pairwise_bce_degree_cycle_loss(y_pred, y_true, mask):
    y_true, y_pred = y_true.squeeze(1), y_pred.squeeze(1)
    b, seq_len, seq_len = y_pred.shape
    batch_losses = torch.zeros(b)

    for i, (pred, true, sample_mask) in enumerate(zip(y_pred, y_true, mask)):
        num_units = sample_mask.sum()
        pred = pred[:num_units, :num_units]
        true = true[:num_units, :num_units]
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.to(torch.float32), reduction='mean')

        # degree loss
        d_loss = degree_loss(torch.sigmoid(pred))
        
        # cycle loss
        c_loss = cycle_loss(torch.sigmoid(pred))

        # total loss
        batch_losses[i] = bce_loss + d_loss + c_loss

    return batch_losses.mean()







all_losses = {
    'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
    'globalpointer_loss': globalpointer_loss,
    'softmax_ce_loss': softmax_ce_loss,
    'pairwise_bce_loss': pairwise_bce_loss,
    'pairwise_bce_and_degree_loss': pairwise_bce_degree_loss,
    'pairwise_bce_degree_cycle_loss': pairwise_bce_degree_cycle_loss,
    'softmax_ce_cycle_loss': softmax_ce_cycle_loss
}