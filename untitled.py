from collections import defaultdict
import os
import pdb
import shutil
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



def get_max_num_units():
    dir = 'data/ReadingBank'
    max_len = 0
    for jp in Path(dir).rglob('*.json'):
        with open(jp) as f:
            list_segments = json.load(f)
        max_len = max(max_len, len(list_segments))
        print('MAX LEN: ', max_len)
    print(max_len)



def get_max_token_length():
    import os
    os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'
    from transformers import LayoutLMv3TokenizerFast

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    dir = 'data/VAT_scan_local_images/images'
    max_len = 0
    for jp in Path(dir).rglob('*.json'):
        with open(jp) as f:
            list_segments = json.load(f)
        texts = [unidecode.unidecode(seg['text']).lower() for seg in list_segments]
        boxes = [seg['p4_bb'] for seg in list_segments]
        input_ids = tokenizer.encode(texts, boxes=boxes)
        print(f'{jp}: {len(input_ids)}')
        max_len = max(max_len, len(input_ids))
        print('MAX LEN: ', max_len)


def get_ducbm3_data():
    dir = '/data/ducbm3/keypoints/BaoHiem'
    out_dir = 'data/InsuranceData/ducbm3_data/BaoHiem_rotate'
    np.random.seed(42)
    num_parts = 2
    for doc_type in os.listdir(dir):
        doc_dir = os.path.join(dir, doc_type)
        doc_paths = sorted([ip for ip in Path(doc_dir).rglob('*') if is_image(ip) and 'augment' not in str(ip)])
        for _ in range(10): np.random.shuffle(doc_paths)
        file_per_part = len(doc_paths) // num_parts
        for i, ip in enumerate(doc_paths):
            # if 'DB-25-177647_103' not in ip.stem: 
            #     continue
            split = ip.parent.name
            part_idx = i // file_per_part if file_per_part > 0 else 0
            part_idx += 1
            part_idx = min(part_idx, num_parts)
            save_dir = os.path.join(out_dir, split, f'part{part_idx}')
            os.makedirs(save_dir, exist_ok=True)

            # load image and crop
            im = cv2.imread(str(ip))
            jp = ip.with_suffix('.json')
            is_rotate = False
            if jp.exists():
                with open(jp) as f:
                    data = json.load(f)
                if len(data['shapes']) > 0:
                    pts = data['shapes'][0]['points']
                    label = data['shapes'][0]['label']
                    if '_' in label:
                        is_rotate = True
                        degree = int(label.split('_')[-1])
                        pts = np.array(pts, dtype=np.int32)
                        print(degree)
                        try:
                            im = four_point_transform(im, pts)
                            if degree == 90:
                                im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            elif degree == 180:
                                im = cv2.rotate(im, cv2.ROTATE_180)
                            elif degree == 270:
                                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                        except Exception as e:
                            print(f'Error: {e}')
            # save
            if is_rotate:
                new_name = f"{doc_type.replace('-', '_').replace(' ', '_')}-{ip.stem}.jpg"
                cv2.imwrite(os.path.join(save_dir, new_name), im)
                # cv2.imwrite('test.jpg', im)
                print(f'done {ip}')


def split_data():
    dir = '/data/tungtx2/reading_order/unirop/data/InsuranceData/ducbm3_data/BaoHiem/val/part2'
    num_parts = 2
    ipaths = sorted([ip for ip in Path(dir).glob('*') if is_image(ip)])
    doc2paths = defaultdict(list)
    for ip in ipaths:
        doc_type = ip.stem.split('-')[0]
        doc2paths[doc_type].append(ip)
    np.random.seed(42)
    for doc_type, doc_paths in doc2paths.items():
        for _ in range(10): np.random.shuffle(doc_paths)
        num_files = len(doc_paths)
        file_per_part = num_files // num_parts
        for idx, ip in enumerate(doc_paths):
            part_idx = idx // file_per_part if file_per_part > 0 else 0
            part_idx += 1
            part_idx = min(part_idx, num_parts)
            if part_idx == 1:
                out_dir = os.path.join(Path(dir).parent, f'part_2.1')
            elif part_idx == 2:
                out_dir = os.path.join(Path(dir).parent, f'part_2.2')
            os.makedirs(out_dir, exist_ok=True)
            shutil.move(str(ip), os.path.join(out_dir, ip.name))
            jp = ip.with_suffix('.json')
            if jp.exists():
                shutil.move(str(jp), os.path.join(out_dir, jp.name))
            jp = ip.parent / f'{ip.stem}-rop.json'
            if jp.exists():
                shutil.move(str(jp), os.path.join(out_dir, jp.name))
            xp = ip.parent / f'{ip.stem}-table.xml'
            if xp.exists():
                shutil.move(str(xp), os.path.join(out_dir, xp.name))
            print(f'done {ip}')

def nothing():
    dir = '/data/tungtx2/reading_order/unirop/data/InsuranceData/ducbm3_data/BaoHiem_rotate'
    for fp in Path(dir).rglob('*'):
        if fp.is_file():
            dst_fp = str(fp).replace('/BaoHiem_rotate/', '/BaoHiem/')
            shutil.copy(str(fp), dst_fp)
            print(f'done {fp}')

if __name__ == '__main__':
    pass
    # nothing()
    # get_max_token_length()
    # get_max_num_units()
    # add_id_to_segment()
    # infer_table()
    # txt2xml_dir()
    # remove_box_in_table()
    # remove_linhtinh_boxes()
    # visualize_pdf_blocks()
    # get_data()
    split_data()
    # get_ducbm3_data()