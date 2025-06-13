from .base import ItemTransform
import json
import pdb
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
from utils.utils import *
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin, LayoutLMv3Tokenizer, LayoutLMv3Processor



class TokenClassificationInputEncoder(ItemTransform):
    def __init__(self, keep_keys, processor_path, max_seq_len: int, max_num_units: int, 
                use_text: bool = True, use_image: bool = True, lower_text: bool = True, normalize_text: bool = True):
        super().__init__(keep_keys)
        self.processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
        self.tokenizer= self.processor.tokenizer
        self.max_seq_len = max_seq_len
        self.max_num_units = max_num_units
        self.SCALE_SIZE = 1000
        self.IMAGE_LEN = 14 ** 2 + 1 # num patches of layoutlmv3 + 1
        self.use_text = use_text
        self.use_image = use_image
        self.lower_text = lower_text
        self.normalize_text = normalize_text

    

    def _get_segment_spans(self, word_ids):
        segment_spans = []
        prev_segment = None
        start_token_index, end_token_index = 1, 0
        for token_index, segment_index in enumerate(word_ids):
            if token_index == 0:
                continue
            if prev_segment is not None and segment_index != prev_segment:
                segment_spans.append((start_token_index, end_token_index))
                start_token_index = token_index
                end_token_index = token_index
            else:
                end_token_index += 1
            if segment_index is None:
                break
            prev_segment = segment_index
        
        return segment_spans
    

    def process(self, item, mode='val'):
        list_segments = item['list_segments']
        if item['image'] is not None:
            im = item['image']
            im_h, im_w = im.shape[:2]
        else:
            im_h, im_w = 1000, 1000

        segment_texts, norm_boxes = [], []
        for segment_info in list_segments:
            if self.use_text:
                text = segment_info['text']
                if self.lower_text:
                    text = text.lower()
                if self.normalize_text:
                    text = unidecode.unidecode(text)
                segment_texts.append(text)
            else:
                segment_texts.append(self.tokenizer.unk_token)
            norm_boxes.append(normalize_bbox(segment_info['p4_bb'], im_w, im_h))

        if self.use_image:
            enc_inp = self.processor(
                im, segment_texts, boxes=norm_boxes,
                truncation=True, padding="max_length", max_length=self.max_seq_len,
                return_tensors="pt"
            )
        else:
            enc_inp = self.tokenizer(
                segment_texts, boxes=norm_boxes,
                truncation=True, padding="max_length", max_length=self.max_seq_len,
                return_tensors="pt"
            )

        # get labels
        orders = [segment_info['order'] for segment_info in list_segments]
        sorted_indexes = list(np.argsort(orders))
        labels = sorted_indexes + [-100] * (self.max_num_units-len(sorted_indexes)) # padding
        labels = torch.tensor(labels)  # not included <cls> and <sep> token
        labels[labels>self.max_num_units] = -100  # set label > MAX_LEN to -100, because original labels may be > MAX_LEN
        # pdb.set_trace()

        # construct segment spans
        segment_spans = self._get_segment_spans(enc_inp.word_ids(0))  # not included <cls> and <sep> token
        assert len(segment_spans) == len(list_segments)  # segment_spans and list_segments must be in the same order
        # print(segment_spans)
        # pdb.set_trace()

        # construct mask for each unit, not include mask for <cls> and <sep> token
        unit_masks = []
        for i, j in segment_spans:  # for start token index and end token index of each segment
            unit_mask = [0] * self.max_seq_len
            unit_mask[i:j+1] = [1] * (j+1 - i)
            unit_masks.append(unit_mask)
        unit_masks = unit_masks[:self.max_num_units] # unit_masks and list_segments must be in the same order

        num_units = torch.tensor(len(unit_masks), dtype=torch.int64)
        assert len(labels) == self.max_num_units

        # The number of unit masks needs to be padded to max_num_units
        while len(unit_masks) < self.max_num_units:
            unit_mask = [0] * self.max_seq_len
            unit_mask[0] = 1  # padding units will all be represented by the <cls> token.
            unit_masks.append(unit_mask)
        unit_masks = torch.tensor(unit_masks, dtype=torch.int64)

        # add mask for image token
        if self.use_image:
            unit_masks = torch.cat((unit_masks, torch.zeros((self.max_num_units, self.IMAGE_LEN), dtype=torch.int64)), dim=1)

        # squeeze
        for k, v in enc_inp.items():
            if isinstance(v, torch.Tensor):
                enc_inp[k] = v.squeeze(0)

        item.update(enc_inp)
        item.update({
            'labels': labels,
            'unit_masks': unit_masks,
            'num_units': num_units,
        })
        self.keep_keys(item)
        return item



if __name__ == '__main__':
    jp = 'data/CompHRDoc_reduce/val/1402.2741_17.json'
    ip = 'data/CompHRDoc_reduce/val/1402.2741_17.png'
    with open(jp) as f:
        json_data = json.load(f)
    im = cv2.imread(ip)
    item = {'image': im, 'json_data': json_data}

    input_encoder = TokenClassificationInputEncoder(
        keep_keys=['grid_labels', 'unit_masks', 'num_units', 'global_pointer_masks', 'input_ids', 'attention_mask', 'bbox', 'pixel_values'],
        processor_path='pretrained/layoutlmv3-base-1024',
        max_seq_len=1024, 
        max_num_units=256,
        lower_text=True,
        normalize_text=True
    )

    input_encoder.process(item)

