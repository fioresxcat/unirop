import math
import pdb
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import LayoutLMv3PreTrainedModel, LayoutLMv3Model
from transformers.utils import ModelOutput

from ..global_pointers.globalpointer import GlobalPointer
from losses.main_losses import all_losses


class LayoutLMv3TwoHeadsForRORelation(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, lmv3_config, gp_config, dropout=0.1, criterion='globalpointer_loss'):
        super().__init__(lmv3_config)
        self.layoutlmv3 = LayoutLMv3Model(lmv3_config)
        self.dropout = nn.Dropout(dropout)
        self.global_pointer_1 = GlobalPointer(
            hidden_size=lmv3_config.hidden_size,
            **gp_config,
        )

        self.global_pointer_2 = GlobalPointer(
            hidden_size=lmv3_config.hidden_size,
            **gp_config,
        )

        self.criterion = all_losses[criterion]


    def forward(self,
                input_ids,
                attention_mask,
                bbox,
                unit_masks,
                global_pointer_masks,
                pixel_values=None,
                position_ids=None,
                grid_labels=None,
                prev_grid_labels=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs
                ):
        # encode tokens to get sequence_output through the encoder
        # normal layoutlmv3 formard, no use of unit_masks yet
        sequence_output = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        sequence_output = self.dropout(sequence_output)  # shape (bs, max_seq_len + IMAGE_LEN, 768)
        # pdb.set_trace()

        # aggegate features in each unit from whole sequence outputs
        masked_sequence_output = sequence_output.unsqueeze(1) * unit_masks.unsqueeze(-1) # (bs, num_units, max_seq_len + IMAGE_LEN, 768)
        summed_features = masked_sequence_output.sum(dim=2)  # [batch_size, num_units, feature_size]
        # 2. Calculate the mean of features in each unit (using clamp to avoid zero unit_mask)
        unit_counts = unit_masks.sum(dim=2, keepdim=True).clamp(min=1)  # [batch_size, num_units, 1]
        mean_features = summed_features / unit_counts  # [batch_size, num_units, feature_size]

        # global pointer
        logits1, _ = self.global_pointer_1(inputs=mean_features, attention_mask=global_pointer_masks)
        logits2, _ = self.global_pointer_2(inputs=mean_features, attention_mask=global_pointer_masks)

        if grid_labels is not None and prev_grid_labels is not None:
            grid_labels = grid_labels.unsqueeze(1)
            prev_grid_labels = prev_grid_labels.unsqueeze(1)
            loss1 = self.criterion(logits1, grid_labels, global_pointer_masks)
            loss2 = self.criterion(logits2, prev_grid_labels, global_pointer_masks)
        else:
            loss1, loss2 = None, None
        
        # pdb.set_trace()
        return ModelOutput(logits1=logits1, logits2=logits2, loss1=loss1, loss2=loss2)
    

from metrics.ro_relation_metrics import EdgeRelationAccuracy, TotalOrderAccuracy, REBleuScore

class LayoutLMv3TwoHeadsForRORelationModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        gp_config: dict,
        criterion: str,
        head_loss_weight: dict,
        optimizer_config: dict,
    ):
        super().__init__()
        self.optimizer_config = EasyDict(optimizer_config)
        self.head_loss_weight = head_loss_weight
        self.model = LayoutLMv3TwoHeadsForRORelation.from_pretrained(
            pretrained_path, gp_config=gp_config, criterion=criterion
        )
        
        self.train_edge_acc = EdgeRelationAccuracy()
        self.val_edge_acc = EdgeRelationAccuracy()

        self.train_sample_acc = TotalOrderAccuracy()
        self.val_sample_acc = TotalOrderAccuracy()

        self.train_bleu4 = REBleuScore(n_gram=4)
        self.val_bleu4 = REBleuScore(n_gram=4)

        self.train_prev_edge_acc = EdgeRelationAccuracy()
        self.val_prev_edge_acc = EdgeRelationAccuracy()

        self.train_prev_sample_acc = TotalOrderAccuracy()
        self.val_prev_sample_acc = TotalOrderAccuracy()

        self.train_prev_bleu4 = REBleuScore(n_gram=4)
        self.val_prev_bleu4 = REBleuScore(n_gram=4)


    def step(self, batch, batch_idx, split):
        outputs = self.model(**batch)
        loss1, loss2, logits1, logits2 = outputs.loss1, outputs.loss2, outputs.logits1, outputs.logits2

        edge_acc = getattr(self, f'{split}_edge_acc')
        edge_acc(logits1, batch['grid_labels'], batch['global_pointer_masks'])
        prev_edge_acc = getattr(self, f'{split}_prev_edge_acc')
        prev_edge_acc(logits2, batch['prev_grid_labels'], batch['global_pointer_masks'])

        sample_acc = getattr(self, f'{split}_sample_acc')
        sample_acc(logits1, batch['grid_labels'], batch['global_pointer_masks'])
        prev_sample_acc = getattr(self, f'{split}_prev_sample_acc')
        prev_sample_acc(logits2, batch['prev_grid_labels'], batch['global_pointer_masks'])

        bleu = getattr(self, f'{split}_bleu4')
        bleu(logits1, batch['grid_labels'], batch['global_pointer_masks'])
        prev_bleu = getattr(self, f'{split}_prev_bleu4')
        prev_bleu(logits2, batch['prev_grid_labels'], batch['global_pointer_masks'])

        self.log_dict({
            f'{split}_loss1': loss1,
            f'{split}_loss2': loss2
        }, on_step=True, on_epoch=True, prog_bar=True)

        self.log_dict({
            f'{split}_edge_acc': edge_acc,
            f'{split}_prev_edge_acc': prev_edge_acc,
            f'{split}_sample_acc': sample_acc,
            f'{split}_prev_sample_acc': prev_sample_acc,
            f'{split}_bleu4': bleu,
            f'{split}_prev_bleu4': prev_bleu
        }, on_step=False, on_epoch=True, prog_bar=True)

        return self.head_loss_weight['loss1'] * loss1 + self.head_loss_weight['loss2'] * loss2
    

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        scheduler_config = self.optimizer_config.lr_scheduler

        ckpt_callback = [cb for cb in self.trainer.callbacks if isinstance(cb, ModelCheckpoint)][0]
        
        sched_name = scheduler_config['name']
        scheduler = torch.optim.lr_scheduler.__dict__[sched_name](
            opt,
            **scheduler_config[sched_name]
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': ckpt_callback.monitor,
                'frequency': 1,
                'interval': 'epoch',
            }
        }
    

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch <= self.optimizer_config.n_warmup_epochs:
            lr_scale = 0.75 ** (self.optimizer_config.n_warmup_epochs - self.trainer.current_epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.optimizer_config.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def on_train_start(self) -> None:
        if self.optimizer_config.reset_optimizer:
            opt = type(self.trainer.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reset')


    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass