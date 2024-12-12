import os
import pdb
from pathlib import Path
import cv2
import numpy as np
import json
from utils.utils import *
from omegaconf import OmegaConf
import fitz
import xml.etree.ElementTree as ET
from easydict import EasyDict


def write_to_xml(boxes, labels, size, xml_path):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = Path(xml_path).stem + '.jpg'
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = map(str, box)
    ET.ElementTree(root).write(xml_path)


def parse_txt(txt_fp, img_size, class2idx):
    idx2class = {v: k for k, v in class2idx.items()}
    img_w, img_h = img_size

    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    boxes, names = [], []
    for line in lines:
        class_id, x, y, w, h = line.split()
        class_id = int(class_id)
        class_name = idx2class[class_id]
        x, y, w, h = float(x), float(y), float(w), float(h)
        xmin = int((x - w/2) * img_w)
        ymin = int((y - h/2) * img_h)
        xmax = int((x + w/2) * img_w)
        ymax = int((y + h/2) * img_h)
        boxes.append([xmin, ymin, xmax, ymax])
        names.append(class_name)
    return np.array(boxes, dtype=np.float32), names


def txt2xml(txt_fp, out_xp, img_size, class2idx):
    boxes, labels = parse_txt(txt_fp, img_size, class2idx)
    write_to_xml(boxes, labels, img_size, out_xp)
    print(f'Done converting {txt_fp} to {out_xp}')



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



def remove_box_in_table():
    json_dir = 'raw_data/VAT_acb_captured/segment_jsons'
    table_dir = 'raw_data/VAT_acb_captured/table_xmls'

    for jp in Path(json_dir).glob('*.json'):
        with open(jp) as f:
            js_data = json.load(f)

        xp = os.path.join(table_dir, jp.stem+'.xml')
        if not os.path.exists(xp):
            continue
        boxes, names = parse_xml(xp)

        remove_indexes = []
        for i, shape in enumerate(js_data['shapes']):
            pts = shape['points']
            for bb in boxes:
                r1, r2, iou = iou_poly(pts, bb)
                if r1 > 0.7:
                    remove_indexes.append(i)
                    break

        js_data['shapes'] = [shape for i, shape in enumerate(js_data['shapes']) if i not in remove_indexes]

        with open(jp, 'w') as f:
            json.dump(js_data, f, ensure_ascii=False)
        
        print(f'done {jp}')



def remove_linhtinh_boxes():
    json_dir = 'raw_data/VAT_acb_captured/segment_jsons'

    for jp in Path(json_dir).glob('*.json'):
        with open(jp) as f:
            js_data = json.load(f)


        remove_indexes = []
        for i, shape in enumerate(js_data['shapes']):
            pts = shape['points']
            xmin, ymin, xmax, ymax = poly2box(pts)
            if (ymax-ymin) / (xmax-xmin) >= 1.5:
                remove_indexes.append(i)

        js_data['shapes'] = [shape for i, shape in enumerate(js_data['shapes']) if i not in remove_indexes]

        with open(jp, 'w') as f:
            json.dump(js_data, f, ensure_ascii=False)
        
        print(f'done {jp}')




def infer_table():
    from ultralytics import YOLO

    model = YOLO('utils/table_detect.onnx')
    model.predict(
        source='raw_data/VAT_acb_captured/images',
        imgsz=640, conf=0.3, iou=0.3, save_txt=True, save=False
    )


def txt2xml_dir():
    dir = 'runs/detect/predict/labels'
    im_dir = 'raw_data/VAT_acb_captured/images'
    out_dir = 'raw_data/VAT_acb_captured/table_xmls'
    os.makedirs(out_dir, exist_ok=True)
    for tp in Path(dir).glob('*.txt'):
        ip = os.path.join(im_dir, tp.stem+'.jpg')
        im = cv2.imread(ip)
        im_h, im_w = im.shape[:2]
        txt2xml(tp, os.path.join(out_dir, tp.stem+'.xml'), (im_w, im_h), {'table': 0})
        print(f'done {tp}')



def visualize_pdf_blocks():
    data_dir = '../raw_data/Doclaynet_extra/PDF'
    scale = 2
    for fp in Path(data_dir).glob('*.pdf'):
        if '32b2df5281feed2c1805aa0bb7e7c72faeb0ff94a4702146eea94885be152841' not in fp.name:
            continue
        doc = fitz.open(fp)
        for page_index, page in enumerate(doc):
            # page: Page = page
            # page.get_textbox((0, 0, page.rect.width, page.rect.height))
            # page.get_textpage()
            # page.get_displaylist()

            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            shape = (pix.height, pix.width, 3)
            image = np.ndarray(shape, dtype=np.uint8, buffer=pix.samples)  # this is rgb image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # after this it is actually bgr image

            page_blocks = page.get_text("dict", flags=11)["blocks"]
            line_index = -1
            for b_idx, block in enumerate(page_blocks):
                block_bb = list(map(lambda x: int(x*scale), block['bbox']))
                block_order = str(block['number'])
                
                for l_idx, line in enumerate(block['lines']):
                    line_bb = list(map(lambda x: int(x*scale), line['bbox']))
                    line_index += 1
                    cv2.rectangle(image, (line_bb[0], line_bb[1]), (line_bb[2], line_bb[3]), (0, 0, 255), 2)
                    cv2.putText(image, str(line_index), (line_bb[0], line_bb[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite('test.jpg', image)
            pdb.set_trace()



def nothing():
    dir = 'data/VAT_epdf_vng/images'
    max_len = 0
    for jp in Path(dir).rglob('*.json'):
        with open(jp) as f:
            data = json.load(f)
        print(len(data))
        max_len = max(max_len, len(data))
    print('MAX LEN: ', max_len)


if __name__ == '__main__':
    pass
    # nothing()
    # add_id_to_segment()
    # infer_table()
    # txt2xml_dir()
    # remove_box_in_table()
    # remove_linhtinh_boxes()
    visualize_pdf_blocks()