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
from utils.roor_utils import *
from .utils import *
from models.layoutlmv3.layoutlmv3_gp import LayoutLMv3ForRORelation
from dataset.transforms.roor_ops import *
from dataset.transforms.generic_ops import *
from metrics.ro_relation_metrics import EdgeRelationAccuracy, TotalOrderAccuracy, REBleuScore



class LayoutLMv3RORelationPredictor:
    def __init__(self, ckpt_path, config, device, act='softmax'):
        super().__init__()
        self.device = device
        if ckpt_path.endswith('.ckpt'):
            state_dict = self.parse_state_dict(ckpt_path)
            self.model = LayoutLMv3ForRORelation.from_pretrained(config.model.pretrained_path, gp_config=config.model.gp_config)
            self.model.load_state_dict(state_dict)
        else:
            self.model = LayoutLMv3ForRORelation.from_pretrained(ckpt_path, gp_config=config.model.gp_config)
            
        print('Load state dict OK!')
        self.model.to(device)
        self.act = act  # activation function
        self.edge_acc = EdgeRelationAccuracy().to(device)
        self.sample_acc = TotalOrderAccuracy().to(device)
        self.bleu4 = REBleuScore(n_gram=4).to(device)
        
    
    def _reset_metric(self):
        self.edge_acc.reset()
        self.sample_acc.reset()
        self.bleu4.reset()


    def compute_metric(self):
        total_edge_acc = self.edge_acc.compute().cpu().item()
        total_sample_acc = self.sample_acc.compute().cpu().item()
        total_bleu4 = self.bleu4.compute().cpu().item()
        self._reset_metric()
        return {
            'edge_accuracy': round(total_edge_acc, 3),
            'sample_accuracy': round(total_sample_acc, 3),
            'bleu_score': round(total_bleu4, 3)
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
        raw_logits, loss = outputs.logits, outputs.loss
        logits = raw_logits.squeeze(1).squeeze(0)  # (max_num_units, max_num_units)
        num_units = inp['global_pointer_masks'][0].sum()
        logits = logits[:num_units, :num_units]
        if self.act == 'softmax':
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape (num_units + 2, num_units + 2)
        elif self.act == 'sigmoid':
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            raise ValueError(f'Unsupported activation function: {self.act}')
        best_path, best_probs = find_best_path_with_expansion(probs)
        
        # update metrics
        grid_labels = inp['grid_labels'].unsqueeze(1)
        edge_acc = self.edge_acc(raw_logits, grid_labels, inp['global_pointer_masks']).cpu().item()
        sample_acc = self.sample_acc(raw_logits, grid_labels, inp['global_pointer_masks']).cpu().item()
        sample_bleu = self.bleu4(raw_logits, grid_labels, inp['global_pointer_masks']).cpu().item()
        metric = {
            'edge_accuracy': round(edge_acc, 3),
            'sample_accuracy': round(sample_acc, 3),
            'bleu_score': round(sample_bleu, 3)
        }
        
        # # debug
        # for i in range(probs.shape[0]-1):
        #     print(probs[i][i+1])
        # pdb.set_trace()
        
        return best_path, metric
    


def main(args):
    from omegaconf import OmegaConf

    if args.ckpt_path.endswith('.ckpt'):
        config = OmegaConf.load(Path(args.ckpt_path).parent / 'config.yaml')
    else:
        config = OmegaConf.load(Path(args.ckpt_path) / 'config.yaml')
    predictor = LayoutLMv3RORelationPredictor(args.ckpt_path, config, args.device)

    # get transform config
    transform_config = {}
    for trans in config.data.transform_ops:
        class_name = trans['class_path'].split('.')[-1]
        transform_config[class_name] = trans['init_args']

    # override transform config
    transform_config['LoadImageAndJson'].pop('use_image', None)
    transform_config['ChunkInput']['max_num_units'] = config.common.max_num_units
    transform_config['ShuffleInput']['shuffle_prob'] = 0
    transform_config['ROORInputEncoder']['keep_keys'].append('list_segments')

    # load transforms
    load_image_and_json = LoadImageAndJson(**transform_config['LoadImageAndJson'])
    chunk_input = ChunkInput(**transform_config['ChunkInput'])
    shuffle_input = ShuffleInput(**transform_config['ShuffleInput'])
    input_encoder = ROORInputEncoder(**transform_config['ROORInputEncoder'])

    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        'files': {},
        'total_metrics': {}
    }
    ipaths = sorted([ip for ip in Path(args.src_dir).glob('*') if is_image(ip)])
    cnt = 0
    for ip in ipaths:
        print()
        # if ip.suffix == '.png':
        #     continue
        # if 'capture_34' not in ip.name:
        #     continue
        # jp = ip.with_suffix('.json')
        jp = ip.parent / (ip.stem + '-rop.json')
        if not jp.exists():
            continue

        print(f'\n----------- Processing {ip} -----------')
        item = {'image_path': ip, 'json_path': jp}
        
        item = load_image_and_json.process(item)
        orig_segments = item['list_segments']
        im = cv2.imread(str(ip))
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
        result['files'][str(ip)] = metrics
        if args.vis:
            save_path = os.path.join(args.out_dir, 'drawed', ip.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if args.vis:
            visualize_inference_result(im, list_segments, sorted_indexes, save_path, scale=args.scale)

        if args.save_paddle:
            annos = []
            for idx in sorted_indexes:
                segment = list_segments[idx]
                bb = list(map(int, segment['p4_bb']))
                xmin, ymin, xmax, ymax = bb
                text = segment['text']
                anno = {
                    "transcription": text,
                    "points": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                    "difficult": False
                }
                annos.append(anno)
            
            anno_str = json.dumps(annos, ensure_ascii=False)
            line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
            with open(os.path.join(args.out_dir, 'Label.txt'), 'a') as f:
                f.write(line)
            
            line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
            with open(os.path.join(args.out_dir, 'Cache.cach'), 'a') as f:
                f.write(line)
            
            line = f'{str(ip)}\t1\n'
            with open(os.path.join(args.out_dir, 'fileState.txt'), 'a') as f:
                f.write(line)
            print(f'saved annotation')
            
        print(f'done {ip}')
        print()
        cnt += 1
        # if cnt >= 10:
        #     break

        # # debug
        # for i in sorted_indexes:
        #     print(list_segments[i]['text'], '-', list_segments[i]['p4_bb'])
    
    # final metrics
    total_metrics = predictor.compute_metric()
    print('TOTAL METRICS: ', total_metrics)
    result['total_metrics'] = total_metrics
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f)
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--save_paddle', action='store_true', help='save inference result in paddle format')
    parser.add_argument('--scale', type=float, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    main(args)