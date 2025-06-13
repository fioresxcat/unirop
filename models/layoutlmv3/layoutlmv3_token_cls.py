from models.base_lightning_module import BaseLightningModule
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3PreTrainedModel, LayoutLMv3Model
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.text import BLEUScore
from utils.token_cls_utils import parse_logits
import pdb


class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    



class LayoutLMv3ForGroupTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()


    def forward(self,
                input_ids,
                attention_mask,
                bbox,
                unit_masks,
                pixel_values=None,
                position_ids=None,
                labels=None,
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

        # head
        logits = self.classifier(mean_features)  # shape (b, num_units, num_labels)
        loss = None
        if labels is not None: # labels shape (b, max_num_units)
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        # pdb.set_trace()
        return ModelOutput(logits=logits, loss=loss)



class LayoutLMv3ForGroupTokenClassificationModule(BaseLightningModule):
    def __init__(
        self,
        pretrained_path: str,
        num_labels: int,
        use_image: bool,
        optimizer_config: dict,
    ):
        super().__init__(optimizer_config)
        self.model = LayoutLMv3ForGroupTokenClassification.from_pretrained(
            pretrained_path, visual_embed=use_image, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        self.train_f1 = MulticlassF1Score(num_classes=num_labels, average='micro')
        self.val_f1 = self.train_f1.clone()

        self.train_bleu = BLEUScore(n_gram=4)
        self.val_bleu = self.train_bleu.clone()


    def step(self, batch, batch_idx, split):
        outputs = self.model(**batch)
        loss, logits = outputs.loss, outputs.logits # logits shape (b, num_units, num_labels)
        num_units = batch['num_units']
        bs = logits.shape[0]
        labels = batch['labels']  # shape (b, max_num_units)

        pred_seqs, target_seqs = [], []
        pred_indexes, target_indexes = [], []
        labels = labels.cpu().numpy().tolist()
        for index in range(bs):
            sample_label = labels[index][:num_units[index]]
            sample_gt_str = ' '.join(list(map(str, sample_label)))
            target_indexes.extend(sample_label)

            sample_logits = logits[index]
            unpadded_logits = sample_logits[:num_units[index], :num_units[index]]
            preds = unpadded_logits.argmax(dim=-1)
            assert len(preds) == len(sample_label)
            pred_indexes.extend(preds.cpu().numpy().tolist())

            sample_orders = parse_logits(sample_logits, num_units[index])
            sample_pred_str = ' '.join(list(map(str, sample_orders)))
            pred_seqs.append(sample_pred_str)
            target_seqs.append([sample_gt_str])
        
        bleu = getattr(self, f'{split}_bleu')
        bleu(pred_seqs, target_seqs)
        f1_score = getattr(self, f'{split}_f1')
        f1_score(torch.tensor(pred_indexes), torch.tensor(target_indexes))
        

        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_f1': f1_score,
            f'{split}_bleu': bleu
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss