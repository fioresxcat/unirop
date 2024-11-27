import os
import pdb
from pathlib import Path
import cv2
import numpy as np
import json
from utils.utils import *
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
    im = cv2.imread('raw_data/VAT_data/images/00000001-0.jpg')
    with open('raw_data/VAT_data/images/00000001-0.json') as f:
        js_data = json.load(f)
    
    shapes = np.random.choice(js_data['shapes'], size=5)
    for shape in shapes:
        bb = poly2box(shape['points'])
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        text = shape['text']
        cv2.putText(im, text, (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite('test.jpg', im)


if __name__ == '__main__':
    pass
    nothing()
    # add_id_to_segment()