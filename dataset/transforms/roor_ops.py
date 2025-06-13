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


class ROORInputEncoder(ItemTransform):
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

    
    def _add_cls_and_sep_segment(self, orig_segments, im_size):
        im_w, im_h = im_size
        reading_orders = [segment['order'] for segment in orig_segments]

        cls_segment = {
            'id': -1,
            'p4_bb': [0,0,min(im_w, 10),min(im_h, 10)],
            'text': self.tokenizer.cls_token,
            'order': min(reading_orders) - 1
        }

        sep_segment = {
            'id': -2,
            'p4_bb': [max(0, im_w-10),max(0, im_h-10),im_w,im_h],
            'text': self.tokenizer.sep_token,
            'order': max(reading_orders) + 1
        }

        list_segments = [cls_segment] + orig_segments + [sep_segment]
        return list_segments
    

    def _get_directed_edges(self, list_segments):
        reading_orders = [segment['order'] for segment in list_segments]
        sorted_indexes = np.argsort(reading_orders)
        edges = []

        for index, segment_index in enumerate(sorted_indexes):
            if index == len(sorted_indexes) - 1:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[0]]['id']
                edges.append((id1, id2))
            else:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[index+1]]['id']
                edges.append((id1, id2))
        
        return edges


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
        im = item['image']
        if im is None:
            assert self.use_image == False
            im_h, im_w = 1000, 1000
        else:
            im_h, im_w = im.shape[:2]

        # add cls and sep segments
        list_segments = self._add_cls_and_sep_segment(list_segments, (im_w, im_h))

        # get directed edges
        edges = self._get_directed_edges(list_segments)
        id2index = {segment['id']: i for i, segment in enumerate(list_segments)}

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

        # construct segment spans
        segment_spans = self._get_segment_spans(enc_inp.word_ids(0))
        try:
            assert len(segment_spans) == len(list_segments)  # segment_spans and list_segments must be in the same order
        except:
            pdb.set_trace()
        # print(segment_spans)
        # pdb.set_trace()

        # get grid labels
        grid_labels = torch.zeros(size=(self.max_num_units, self.max_num_units), dtype=torch.int64)
        for id_i, id_j in edges:
            grid_labels[id2index[id_i]][id2index[id_j]] = 1

        # construct mask for each unit
        unit_masks = []
        for i, j in segment_spans:  # for start token index and end token index of each segment
            unit_mask = [0] * self.max_seq_len
            unit_mask[i:j+1] = [1] * (j+1 - i)
            unit_masks.append(unit_mask)
        unit_masks = unit_masks[:self.max_num_units] # unit_masks and list_segments must be in the same order

        global_pointer_masks = torch.tensor([1] * len(unit_masks) + [0] * (self.max_num_units - len(unit_masks)), dtype=torch.int64)
        num_units = torch.tensor(len(unit_masks), dtype=torch.int64)

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
            'grid_labels': grid_labels,
            'unit_masks': unit_masks,
            'num_units': num_units,
            'global_pointer_masks': global_pointer_masks
        })
        self.keep_keys(item)
        return item


class ROORInputEncoderTwoHeads(ROORInputEncoder):
    def _get_directed_edges(self, list_segments):
        reading_orders = [segment['order'] for segment in list_segments]
        sorted_indexes = np.argsort(reading_orders)
        next_edges, prev_edges = [], []

        for index, segment_index in enumerate(sorted_indexes):
            if index == len(sorted_indexes) - 1:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[0]]['id']
                next_edges.append((id1, id2))
            else:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[index+1]]['id']
                next_edges.append((id1, id2))

        
        for index, segment_index in enumerate(sorted_indexes):
            if index == 0:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[-1]]['id']
                prev_edges.append((id1, id2))
            else:
                id1 = list_segments[segment_index]['id']
                id2 = list_segments[sorted_indexes[index-1]]['id']
                prev_edges.append((id1, id2))
        
        return next_edges, prev_edges


    def process(self, item, mode='val'):
        list_segments = item['list_segments']
        im = item['image']
        im_h, im_w = im.shape[:2]

        # add cls and sep segments
        list_segments = self._add_cls_and_sep_segment(list_segments, (im_w, im_h))

        # get directed edges
        next_edges, prev_edges = self._get_directed_edges(list_segments)
        id2index = {segment['id']: i for i, segment in enumerate(list_segments)}

        segment_texts, norm_boxes = [], []
        for segment_info in list_segments:
            text = segment_info['text']
            if self.lower_text:
                text = text.lower()
            if self.normalize_text:
                text = unidecode.unidecode(text)
            segment_texts.append(text)
            norm_boxes.append(normalize_bbox(segment_info['p4_bb'], im_w, im_h))

        enc_inp = self.processor(
            im, segment_texts, boxes=norm_boxes,
            truncation=True, padding="max_length", max_length=self.max_seq_len,
            return_tensors="pt"
        )

        # construct segment spans
        segment_spans = self._get_segment_spans(enc_inp.word_ids(0))
        assert len(segment_spans) == len(list_segments)  # segment_spans and list_segments must be in the same order
        # print(segment_spans)
        # pdb.set_trace()

        # get grid labels
        grid_labels = torch.zeros(size=(self.max_num_units, self.max_num_units), dtype=torch.int64)
        for id_i, id_j in next_edges:
            grid_labels[id2index[id_i]][id2index[id_j]] = 1

        # get prev grid labels
        prev_grid_labels = torch.zeros(size=(self.max_num_units, self.max_num_units), dtype=torch.int64)
        for id_i, id_j in prev_edges:
            prev_grid_labels[id2index[id_i]][id2index[id_j]] = 1

        # construct mask for each unit
        unit_masks = []
        for i, j in segment_spans:  # for start token index and end token index of each segment
            unit_mask = [0] * self.max_seq_len
            unit_mask[i:j+1] = [1] * (j+1 - i)
            unit_masks.append(unit_mask)
        unit_masks = unit_masks[:self.max_num_units] # unit_masks and list_segments must be in the same order

        global_pointer_masks = torch.tensor([1] * len(unit_masks) + [0] * (self.max_num_units - len(unit_masks)), dtype=torch.int64)
        num_units = torch.tensor(len(unit_masks), dtype=torch.int64)

        # The number of unit masks needs to be padded to max_num_units
        while len(unit_masks) < self.max_num_units:
            unit_mask = [0] * self.max_seq_len
            unit_mask[0] = 1  # padding units will all be represented by the <cls> token.
            unit_masks.append(unit_mask)
        unit_masks = torch.tensor(unit_masks, dtype=torch.int64)

        # add mask for image token
        unit_masks = torch.cat(
            (unit_masks, torch.zeros((self.max_num_units, self.IMAGE_LEN), dtype=torch.int64)), dim=1
        )

        # squeeze
        for k, v in enc_inp.items():
            if isinstance(v, torch.Tensor):
                enc_inp[k] = v.squeeze(0)
        item.update(enc_inp)
        item.update({
            'grid_labels': grid_labels,
            'prev_grid_labels': prev_grid_labels,
            'unit_masks': unit_masks,
            'num_units': num_units,
            'global_pointer_masks': global_pointer_masks
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

    input_encoder = ROORInputEncoder(
        keep_keys=['grid_labels', 'unit_masks', 'num_units', 'global_pointer_masks', 'input_ids', 'attention_mask', 'bbox', 'pixel_values'],
        processor_path='pretrained/layoutlmv3-base-1024',
        max_seq_len=1024, 
        max_num_units=256,
        lower_text=True,
        normalize_text=True
    )

    input_encoder.process(item)

