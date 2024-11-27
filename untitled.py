import os
import pdb
from pathlib import Path
import cv2
import numpy as np
import json
from omegaconf import OmegaConf
from easydict import EasyDict


def add_id_to_segment():
    dir = 'data'
    for jp in Path(dir).rglob('*.json'):
        with open(jp) as f:
            list_segments = json.load(f)

        for i, segment in enumerate(list_segments):
            segment['id'] = i

        with open(jp, 'w') as f:
            json.dump(list_segments, f, ensure_ascii=False)

        print(f'done {jp}')


def nothing():
    config = OmegaConf.load('configs/comp_hrdoc.yaml')
    print(config)
    pdb.set_trace()


if __name__ == '__main__':
    pass
    nothing()
    # add_id_to_segment()