from pathlib import Path
import xml.etree.ElementTree as ET
import unicodedata
import numpy as np
import json
import cv2
from copy import deepcopy
from PIL import Image
from shapely.geometry import Polygon
import unidecode
import os


def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


def row_bbs(bbs):
    polys = []
    poly2bb = {}
    for bb in bbs:
        poly = [bb[0], bb[1], bb[2], bb[1], bb[2], bb[3], bb[0], bb[3]]
        polys.append(poly)
        poly2bb[tuple(poly)] = bb
    poly_rows = row_polys(polys)
    bb_rows = []
    for row in poly_rows:
        bb_row = []
        for poly in row:
            bb_row.append(poly2bb[tuple(poly)])
        bb_rows.append(bb_row)
    return bb_rows


def sort_bbs(bbs):
    bb2idx_original = {tuple(bb): i for i, bb in enumerate(bbs)}
    bb_rows = row_bbs(bbs)
    sorted_bbs = [bb for row in bb_rows for bb in row]
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly_clusters = row_polys(polys)
    polys = []
    for row in poly_clusters:
        polys.extend(row)
    return polys, poly_clusters


def sort_json(json_data):
    polys = []
    poly2label = {}
    poly2text = {}
    poly2idx_original = {}
    poly2row = {}
    for i, shape in enumerate(json_data['shapes']):
        if shape['shape_type'] == 'rectangle':
            continue
        if len(shape['points']) != 4:
            raise ValueError('Json contains shape with more than 4 points')
        
        poly = shape['points']
        poly = [int(coord) for pt in poly for coord in pt]
        polys.append(poly)
        poly2label[tuple(poly)] = shape['label']
        poly2text[tuple(poly)] = shape['text']
        poly2idx_original[tuple(poly)] = i
    rows = row_polys(polys)
    for row_idx, row in enumerate(rows):
        for poly_idx, poly in enumerate(row):
            rows[row_idx][poly_idx] = tuple(poly)
    for row_idx, row in enumerate(rows):
        for poly in row:
            poly2row[tuple(poly)] = row_idx
    return poly2label, poly2text, rows, poly2idx_original, poly2row


def get_img_fp_from_json_fp(json_fp):
    if isinstance(json_fp, str):
        json_fp = Path(json_fp)
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None

def is_image(fp):
    if isinstance(fp, str):
        fp = Path(fp)
    return fp.suffix in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def denormalize_bbox(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]

def poly2box(poly):
    poly = np.array(poly).flatten().tolist()
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return xmin, ymin, xmax, ymax


def mask_image_poly(img: np.ndarray, poly):
    xmin, ymin, xmax, ymax = poly2box(poly)
    img[ymin:ymax, xmin:xmax] = np.random.randint(240, 255)
    return img


def str_similarity(str1, str2, normalize=False, remove_space=True):
    import Levenshtein
    import re

    str1 = str1.lower()
    str2 = str2.lower()

    if normalize:
        str1 = unicodedata.unidecode(str1)
        str2 = unicodedata.unidecode(str2)

    if remove_space:
        str1 = re.sub(r'\s+', '', str1)
        str2 = re.sub(r'\s+', '', str2)

    distance = Levenshtein.distance(str1, str2)
    score = 1 - (distance / max(len(str1), len(str2)))
    return score


def get_pdf_name_from_file_path(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return '-'.join(file_path.stem.split('-')[:-1])




def iou_poly(poly1, poly2):
    poly1 = np.array(poly1).flatten().tolist()
    poly2 = np.array(poly2).flatten().tolist()

    xmin1, xmax1 = min(poly1[::2]), max(poly1[::2])
    ymin1, ymax1 = min(poly1[1::2]), max(poly1[1::2])
    xmin2, xmax2 = min(poly2[::2]), max(poly2[::2])
    ymin2, ymax2 = min(poly2[1::2]), max(poly2[1::2])

    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return 0, 0, 0

    if len(poly1) == 4:  # if poly1 is a box
        x1, y1, x2, y2 = poly1
        poly1 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    if len(poly2) == 4:  # if poly2 is a box
        x1, y1, x2, y2 = poly2
        poly2 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if len(poly1) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly1
        poly1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    if len(poly2) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly2
        poly2 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    ratio1 = intersect / poly1.area
    ratio2 = intersect / poly2.area
    iou = intersect / union
    
    return ratio1, ratio2, iou
