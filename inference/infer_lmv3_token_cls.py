import os
os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'

from pathlib import Path
import pdb
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from itertools import permutations
import json
from utils.utils import *
from utils.token_cls_utils import *
from .utils import *
from models.layoutlmv3.layoutlmv3_token_cls import LayoutLMv3ForGroupTokenClassification
from dataset.transforms.token_cls_ops import *
from dataset.transforms.generic_ops import *
from torchmetrics.text import BLEUScore
from torchmetrics.classification import Accuracy
from easydict import EasyDict


class Predictor:
    def __init__(self, ckpt_path, config: dict, device: str):
        super().__init__()
        self.device = device
        state_dict = self.parse_state_dict(ckpt_path)
        self.model = LayoutLMv3ForGroupTokenClassification.from_pretrained(
            'microsoft/layoutlmv3-large',
            visual_embed=config.common.use_image, num_labels=config.common.max_num_units
        )
        self.model.load_state_dict(state_dict)
        print('Load state dict OK!')
        self.model.eval().to(device)
        self.bleu4 = BLEUScore().to(device)
        self.sample_acc = Accuracy(task='binary').to(device)

    

    def _reset_metric(self):
        self.bleu4.reset()
        self.sample_acc.reset()


    def compute_metric(self):
        total_bleu4 = self.bleu4.compute().cpu().item()
        total_sample_acc = self.sample_acc.compute().cpu().item()
        self._reset_metric()
        return {
            'bleu_score': round(total_bleu4, 3),
            'sample_accuracy': round(total_sample_acc, 3)
        }


    def parse_state_dict(self, ckpt_path):
        state_dict = torch.load(ckpt_path, weights_only=True)['state_dict']
        keys = list(state_dict.keys())
        for k in keys:
            state_dict[k.replace('model.', '')] = state_dict.pop(k)
        return state_dict


    @torch.no_grad()
    def predict(self, inp):
        outputs = self.model(**inp)
        logits, loss = outputs.logits, outputs.loss
        num_units = inp['num_units'][0]
        logits = logits[0]
        orders = parse_logits(logits, num_units)

        # update metrics
        labels = inp['labels'][0][:num_units].cpu().numpy().tolist()
        gt_str = ' '.join(list(map(str, labels)))
        pred_str = ' '.join(list(map(str, orders)))
        bleu_score = self.bleu4([pred_str], [[gt_str]]).cpu().item()
        if bleu_score == 1:
            acc = self.sample_acc(torch.tensor([1]), torch.tensor([1])).cpu().item()
        else:
            acc = self.sample_acc(torch.tensor([0]), torch.tensor([1])).cpu().item()
        metric = {
            'bleu_score': round(bleu_score, 3),
            'sample_acc': round(acc, 3)
        }
        return orders, metric
    


def main(args):
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(args.ckpt_path).parent / 'config.yaml')
    predictor = Predictor(args.ckpt_path, config, args.device)

    # get transform config
    transform_config = {}
    for trans in config.data.transform_ops:
        class_name = trans['class_path'].split('.')[-1]
        transform_config[class_name] = trans['init_args']

    # override transform config
    transform_config['ShuffleInput']['shuffle_prob'] = args.shuffle_prob
    transform_config['TokenClassificationInputEncoder']['keep_keys'].append('list_segments')

    # load transforms
    load_image_and_json = LoadImageAndJson(**transform_config['LoadImageAndJson'])
    chunk_input = ChunkInput(**transform_config['ChunkInput'])
    shuffle_input = ShuffleInput(**transform_config['ShuffleInput'])
    input_encoder = TokenClassificationInputEncoder(**transform_config['TokenClassificationInputEncoder'])

    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        'files': {},
        'total_metrics': {}
    }
    jpaths = [fp for fp in Path(args.src_dir).glob('*.json')]
    jpaths.sort()
    for jp in jpaths:
        # if '00000823-1' not in ip.name:
        #     continue
        ip = get_img_fp_from_json_fp(jp)

        print(f'\n----------- Processing {jp} -----------')
        item = {'image_path': ip, 'json_path': jp}
        
        item = load_image_and_json.process(item)
        orig_segments = item['list_segments']
        im = item['image']
        if len(orig_segments) == 0:
            print(f'{jp} has no texts inside! Skipping ...')
            continue

        item = chunk_input.process(item)
        item = shuffle_input.process(item)
        list_segments = item['list_segments']
        
        item = input_encoder.process(item)
        # add batch dim
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                item[k] = v.unsqueeze(0).to(args.device)
        sorted_indexes, metrics = predictor.predict(item)
        print(f'METRICS: {metrics}')
        result['files'][str(jp)] = metrics
        save_path = os.path.join(args.out_dir, 'drawed', ip.name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if im is not None:
            visualize_inference_result(im, list_segments, sorted_indexes, save_path, scale=args.scale)

        # # debug
        # for i in sorted_indexes:
        #     print(list_segments[i]['text'], '-', list_segments[i]['p4_bb'])
    
    # final metrics
    total_metrics = predictor.compute_metric()
    print('TOTAL METRICS: ', total_metrics)
    result['total_metrics'] = total_metrics
    if args.save_metrics:
        with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
            json.dump(result, f)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--shuffle_prob', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--save_metrics', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    main(args)