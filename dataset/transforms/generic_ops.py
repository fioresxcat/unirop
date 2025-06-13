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



class LoadImageAndJson(ItemTransform):
    def __init__(self, keep_keys):
        super().__init__(keep_keys)

    
    def process(self, item, mode='val'):
        if os.path.exists(item['image_path']):
            item['image'] = cv2.imread(str(item['image_path']))
        else:
            item['image'] = None
        with open(item['json_path']) as f:
            list_segments = json.load(f)
        
        # remove empty segment
        new_list_segments = []
        for segment in list_segments:
            if unidecode.unidecode(segment['text']) != '':
                new_list_segments.append(segment)

        item['list_segments'] = new_list_segments
        self.keep_keys(item)
        return item
    


class ChunkInput(ItemTransform):
    def __init__(self, keep_keys, processor_path: str, use_text: bool, stride: int, max_seq_len: int, max_num_units: int,
                 lower_text: bool, normalize_text: bool, return_first_chunk_prob: float):
        super().__init__(keep_keys)
        self.processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
        self.tokenizer= self.processor.tokenizer
        self.max_seq_len = max_seq_len
        self.max_num_units = max_num_units
        self.stride = stride
        self.use_text = use_text
        self.lower_text = lower_text
        self.normalize_text = normalize_text
        self.return_first_chunk_prob = return_first_chunk_prob


    def chunk_segments(self, list_segments):
        max_seq_len = self.max_seq_len - 4

        texts = []
        for segment in list_segments:
            if self.use_text:
                text = segment['text']
                if self.lower_text:
                    text = text.lower()
                if self.normalize_text:
                    text = unidecode.unidecode(text)
            else:
                text = self.tokenizer.unk_token
            texts.append(text)
        bboxes = [[0,0,0,0]] * len(texts)
        enc_inp = self.tokenizer(texts, boxes=bboxes, truncation=False, padding=False, return_tensors='pt')
        word_ids = enc_inp.word_ids(0)[1:-1]  # exclude the <cls> and <sep> token.

        chunk_indexes = []
        start_index, end_index = 0, 0
        while end_index < len(word_ids):
            end_index = start_index + max_seq_len - 1
            chunk_indexes.append((start_index, min(end_index, len(word_ids)-1)))
            start_index += self.stride

        segment_chunks = []
        for start_index, end_index in chunk_indexes:
            start_segment_index = word_ids[start_index]
            if start_index == 0 or word_ids[start_index-1] != start_segment_index:
                pass
            else:
                start_segment_index += 1
            end_segment_index = word_ids[end_index]
            if end_index == len(word_ids) - 1 or word_ids[end_index+1] != end_segment_index:
                pass
            else:
                end_segment_index -= 1

            chunk = []
            for index in range(start_segment_index, end_segment_index+1):
                chunk.append(list_segments[index])
            if len(chunk) > 0:
                segment_chunks.append(chunk)

            # chunk_texts = [unidecode.unidecode(segment['text']).lower() for segment in chunk]
            # chunk_bboxes = [segment['p4_bb'] for segment in chunk]
            # chunk_enc_inp = self.tokenizer(chunk_texts, boxes=chunk_bboxes, truncation=False, padding=False, return_tensors='pt')
            # if chunk_enc_inp.input_ids.shape[1] > self.max_seq_len:
            #     pdb.set_trace()
            # # pdb.set_trace()
        
        return segment_chunks
    


    def process(self, item, mode='val'):
        list_segments = item['list_segments']

        # chunk
        segment_chunks = self.chunk_segments(list_segments)
        if mode != 'train' or np.random.rand() < self.return_first_chunk_prob:
            list_segments = segment_chunks[0]  # return the first chunk
        else:
            index = np.random.randint(0, len(segment_chunks))
            list_segments = segment_chunks[index]

        item['list_segments'] = list_segments[:self.max_num_units-2]  # -2 for cls and sep segment
        self.keep_keys(item)
        return item



class ShuffleInput(ItemTransform):
    def __init__(self, keep_keys, shuffle_prob: float):
        super().__init__(keep_keys)
        self.shuffle_prob = shuffle_prob


    def process(self, item, mode='val'):
        list_segments = item['list_segments']
        if mode == 'train' and np.random.rand() < self.shuffle_prob:
            np.random.shuffle(list_segments)
        else: # sort from top2bot, left2right
            bb2segment = {}
            for segment in list_segments:
                bb = segment['p4_bb']
                bb2segment[tuple(bb)] = segment
            bbs = list(bb2segment.keys())
            bbs, _ = sort_bbs(bbs)
            list_segments = [bb2segment[tuple(bb)] for bb in bbs]

        item['list_segments'] = list_segments
        self.keep_keys(item)
        return item