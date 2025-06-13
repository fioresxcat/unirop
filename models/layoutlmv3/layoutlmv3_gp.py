import os
os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'

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


class LayoutLMv3ForRORelation(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, lmv3_config, gp_config, dropout=0.1, criterion='globalpointer_loss'):
        super().__init__(lmv3_config)
        self.num_labels = lmv3_config.num_labels
        self.config = lmv3_config
        self.layoutlmv3 = LayoutLMv3Model(lmv3_config)
        self.dropout = nn.Dropout(dropout)
        self.global_pointer = GlobalPointer(
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
        logits, _ = self.global_pointer(inputs=mean_features, attention_mask=global_pointer_masks)
        if grid_labels is not None:
            grid_labels = grid_labels.unsqueeze(1)
            loss = self.criterion(logits, grid_labels, global_pointer_masks)
        else:
            loss = None
        
        # pdb.set_trace()
        return ModelOutput(logits=logits, loss=loss)
    

from metrics.ro_relation_metrics import EdgeRelationAccuracy, TotalOrderAccuracy, REBleuScore
from models.base_lightning_module import BaseLightningModule

class LayoutLMv3ForRORelationModule(BaseLightningModule):
    def __init__(
        self,
        pretrained_path: str,
        gp_config: dict,
        criterion: str,
        optimizer_config: dict,
    ):
        super().__init__(optimizer_config)
        self.model = LayoutLMv3ForRORelation.from_pretrained(
            pretrained_path, gp_config=gp_config, criterion=criterion
        )
        
        self.train_edge_acc = EdgeRelationAccuracy()
        self.val_edge_acc = EdgeRelationAccuracy()

        self.train_sample_acc = TotalOrderAccuracy()
        self.val_sample_acc = TotalOrderAccuracy()

        self.train_bleu4 = REBleuScore(n_gram=4)
        self.val_bleu4 = REBleuScore(n_gram=4)


    def step(self, batch, batch_idx, split):
        outputs = self.model(**batch)
        loss, logits = outputs.loss, outputs.logits

        edge_acc = getattr(self, f'{split}_edge_acc')
        edge_acc(logits, batch['grid_labels'], batch['global_pointer_masks'])

        sample_acc = getattr(self, f'{split}_sample_acc')
        sample_acc(logits, batch['grid_labels'], batch['global_pointer_masks'])

        bleu = getattr(self, f'{split}_bleu4')
        bleu(logits, batch['grid_labels'], batch['global_pointer_masks'])

        self.log_dict({
            f'{split}_loss': loss,
        }, on_step=True, on_epoch=True, prog_bar=True)

        self.log_dict({
            f'{split}_edge_acc': edge_acc,
            f'{split}_sample_acc': sample_acc,
            f'{split}_bleu4': bleu
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load('configs/vat-v2-re.yaml')
    config.model.gp_config.head_size=64
    model = LayoutLMv3ForRORelation.from_pretrained(
        # 'pretrained/layoutlmv3_base-layout_only-re',
        'hantian/layoutreader',
        gp_config=config.model.gp_config,
        criterion=config.model.criterion
    )
    # print(model)
    pdb.set_trace()