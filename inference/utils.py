import numpy as np
import cv2
import pdb
import heapq


def visualize_inference_result(im, list_segments, sorted_indexes, save_path, scale=1.0):
    sorted_segments = [list_segments[i] for i in sorted_indexes]
    im = im.copy()
    im_h, im_w = im.shape[:2]
    if scale:
        im = cv2.resize(im, (int(im_w * scale), int(im_h * scale)))
    for i, segment_info in enumerate(sorted_segments):
        order = i
        bb = segment_info['p4_bb']
        if scale:
            bb = list(map(lambda x: int(x * scale), bb))
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        cv2.putText(im, str(order), (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(save_path, im)


if __name__ == '__main__':
    import json
    import cv2

    ip = 'data/CompHRDoc_reduce/val/1507.02346_0.png'
    jp = 'data/CompHRDoc_reduce/val/1507.02346_0.json'
    im = cv2.imread(ip)
    with open(jp) as f:
        list_segments = json.load(f)
    
    visualize_inference_result(im, list_segments, list(range(len(list_segments))), 'test.jpg', 2.0)