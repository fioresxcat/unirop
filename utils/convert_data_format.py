import json
import os
from pathlib import Path
import pdb
import shutil
import numpy as np
import cv2
from PIL import Image
from .utils import *
from PIL import Image, ImageDraw, ImageFont


def convert_CompHRDoc():
    np.random.seed(42)

    out_dir = 'data/CompHRDoc_reduce/test'
    os.makedirs(out_dir, exist_ok=True)

    with open('raw_data/Comp-HRDoc/HRDH_MSRA_POD_TEST/hdsa_test.json') as f:
        all_anno = json.load(f)
    image_infos = all_anno['images']
    anno_infos = all_anno['annotations']

    # random sample elements
    indexes = np.random.choice(len(anno_infos), 500, replace=False)
    image_infos = [image_infos[i] for i in indexes]
    anno_infos = [anno_infos[i] for i in indexes]
    assert len(image_infos) == len(anno_infos) == 500

    img_dir = 'raw_data/Comp-HRDoc/HRDH_MSRA_POD_TEST/Images'
    for file_index, (doc_anno, image_info) in enumerate(zip(anno_infos, image_infos)):
        in_doc_annos = doc_anno['in_doc_annotations']  # annotation for each pdf
        im_names = image_info['file_name']
        pdf_name = '_'.join(im_names[0].split('_')[:-1])
        pdf_im_dir = os.path.join(img_dir, pdf_name)
        if not os.path.exists(pdf_im_dir):
            continue
        
        for ip in Path(pdf_im_dir).glob('*.png'):
            page_index = ip.stem
            save_name = f'{pdf_name}_{page_index}'
            # copy img
            shutil.copy(ip, os.path.join(out_dir, f'{save_name}.png'))
            # get all annos in this page
            list_elements = [el for el in in_doc_annos if el['page_id'] == int(page_index)]
            # get json anno
            cur_index = 0
            im_anno = []
            for el_index, el in enumerate(list_elements):
                for line_index, (poly, content) in enumerate(zip(el['textline_polys'], el['textline_contents'])):
                    xmin, xmax = min(poly[::2]), max(poly[::2])
                    ymin, ymax = min(poly[1::2]), max(poly[1::2])

                    im_anno.append({
                        'p4_bb': [xmin, ymin, xmax, ymax],
                        'p8_bb': poly,
                        'text': content,
                        'order': cur_index,
                    })
                    cur_index += 1

            with open(os.path.join(out_dir, f'{save_name}.json'), 'w') as f:
                json.dump(im_anno, f, ensure_ascii=False)
            
            print(f'Done {save_name}')

    num_keep = 50
    files = [file for file in os.listdir(out_dir) if file.endswith('.png')]
    for _ in range(10): 
        np.random.shuffle(files)
    for file in files[num_keep:]:
        os.remove(os.path.join(out_dir, file))
        os.remove(os.path.join(out_dir, file.replace('.png', '.json')))
        print(f'Delete {file}')


def convert_ReadingBank():
    def group_box_and_text(text_anno, layout_anno):
        target_boxes = []
        target_texts = []
        last_box = [0, 0, 0, 0]
        last_text = ""
        for s, box in zip(text_anno["tgt"].split(), layout_anno["tgt"]):
            if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                # skip invalid box
                continue
            if (
                box[1] == last_box[1]
                and box[3] == last_box[3]
                and box[0] >= last_box[2]
                and (box[0] - last_box[2])
                < ((last_box[2] - last_box[0]) / max(len(last_text), 1))  # khoảng cách x-axis giữa 2 box nhỏ hơn chiều rộng của 1 character
            ):
                # merge box of the same line
                last_box[2] = box[2]
                last_text += " " + s
            else:
                if last_text != "":
                    target_boxes.append(last_box.copy())
                    target_texts.append(last_text)
                # renew buffer
                last_box = box.copy()
                last_text = s
        if last_text != "":
            target_boxes.append(last_box.copy())
            target_texts.append(last_text)

        for left, top, right, bottom in target_boxes:
            assert left <= right <= 1000 and top <= bottom <= 1000

        return target_boxes, target_texts
    

    dir = '../raw_data/ReadingBank'
    out_dir = 'data/ReadingBank/'
    for split in os.listdir(dir):
        split_dir = os.path.join(dir, split)
        out_split_dir = os.path.join(out_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)
        for layout_jp in Path(split_dir).glob('dataset-*-s2s-layout-m*.json'):
            with open(layout_jp) as f:
                layout_lines = f.readlines()
            text_jp = layout_jp.parent / (layout_jp.name.replace('layout', 'text'))
            with open(text_jp) as f:
                text_lines = f.readlines()
            print(f'------------------------ Processing {layout_jp} ----------------------')
            for file_index, (layout_line, text_line) in enumerate(zip(layout_lines, text_lines)):  # iterate each single file
                l_anno = json.loads(layout_line)
                t_anno = json.loads(text_line)
                boxes, texts = group_box_and_text(t_anno, l_anno)
                annos = []
                for box_index, (box, text) in enumerate(zip(boxes, texts)):
                    annos.append({
                        'p4_bb': box,
                        'p8_bb': [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]],
                        'text': text,
                        'order': box_index,
                        'id': box_index
                    })
                part_name = layout_jp.stem.split('-')[-1]
                save_path = os.path.join(out_split_dir, f'{part_name}-{file_index}.json')
                with open(save_path, 'w') as f:
                    json.dump(annos, f, ensure_ascii=False)
                print(f'done {save_path}')
        # break
        

def draw_ReadingBank_data():

    # Load JSON file
    jp = Path('data/ReadingBank/train/m1-100.json')
    with open(jp, 'r') as f:
        data = json.load(f)

    # Create a blank white image of size (1000, 1000)
    image = Image.new('RGB', (1000, 1000), 'white')
    draw = ImageDraw.Draw(image)

    # Load a font (use default PIL font if no custom font is available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default(size=14)

    # Iterate through the elements in the data and draw the text in the corresponding position
    for index, item in enumerate(data):
        p4_bb = item['p4_bb']  # Bounding box [x1, y1, x2, y2]
        text = item['text']
        
        # # Draw a rectangle around the bounding box (optional)
        # draw.rectangle(p4_bb, outline="black", width=1)
        
        # Draw the text within the bounding box
        draw.text((p4_bb[0], p4_bb[1]), text, fill="black", font=font)

        # # draw the index
        # draw.text((p4_bb[2], p4_bb[1]), str(index), fill="black", font=font)

    # Save the resulting image
    image.save('data/ReadingBank/sample/m1-100.png')
    shutil.copy(jp, 'data/ReadingBank/sample/m1-100.json')



def gen_ReadingBank_image():
    num_sample = 100
    dir = 'data/ReadingBank/test'
    out_dir = f'data/ReadingBank/test-{num_sample}_samples'
    os.makedirs(out_dir, exist_ok=True)
    jpaths = list(Path(dir).glob('*.json'))
    np.random.seed(42)
    np.random.shuffle(jpaths)
    for jp in jpaths[:100]:
        with open(jp, 'r') as f:
            data = json.load(f)

        # Create a blank white image of size (1000, 1000)
        image = Image.new('RGB', (1000, 1000), 'white')
        draw = ImageDraw.Draw(image)

        # Load a font (use default PIL font if no custom font is available)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default(size=14)

        # Iterate through the elements in the data and draw the text in the corresponding position
        for index, item in enumerate(data):
            p4_bb = item['p4_bb']  # Bounding box [x1, y1, x2, y2]
            text = item['text']
            
            # # Draw a rectangle around the bounding box (optional)
            # draw.rectangle(p4_bb, outline="black", width=1)
            
            # Draw the text within the bounding box
            draw.text((p4_bb[0], p4_bb[1]), text, fill="black", font=font)

        image.save(os.path.join(out_dir, jp.stem + '.png'))
        shutil.copy(jp, out_dir)
        print(f'done {jp}')


def json2paddle():
    im_dir = 'data/test_imgs_2'
    json_dir = 'data/test_imgs_2'

    for ip in Path(im_dir).glob('*'):
        if not is_image(ip): continue
        jp = os.path.join(json_dir, ip.stem+'.json')
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            js_data = json.load(f)
        
        annos = []
        for shape in js_data['shapes']:
            (xmin, ymin), (xmax, ymax) = shape['points']
            anno = {
                "transcription": shape['text'],
                "points": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                "difficult": False
            }
            annos.append(anno)
        
        anno_str = json.dumps(annos, ensure_ascii=False)
        line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
        with open(os.path.join(im_dir, 'Label.txt'), 'a') as f:
            f.write(line)
        
        line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
        with open(os.path.join(im_dir, 'Cache.cach'), 'a') as f:
            f.write(line)
        
        line = f'{str(ip)}\t1\n'
        with open(os.path.join(im_dir, 'fileState.txt'), 'a') as f:
            f.write(line)
        
        print(f'done {ip}')
    

def paddle2rop():
    dir = 'data/InsuranceData/part1_label/train/part_1.1'
    with open(os.path.join(dir, 'Label.txt')) as f:
        labels = [line.strip() for line in f.readlines()]
    
    for line in labels:
        fp, paddle_annos = line.split('\t')
        paddle_annos = json.loads(paddle_annos)
        fn = fp.split('/')[-1]
        rop_annos = []
        for i, paddle_anno in enumerate(paddle_annos):
            p4_bb = poly2box(paddle_anno['points'])
            p8_bb = np.array(paddle_anno['points']).flatten().tolist()
            rop_anno = {
                'p4_bb': p4_bb,
                'p8_bb': p8_bb,
                'text': paddle_anno['transcription'],
                'order': i,
                'id': i
            }
            rop_annos.append(rop_anno)
        ip = Path(os.path.join(dir, fn))
        with open(ip.parent / (ip.stem + '-rop.json'), 'w') as f:
            json.dump(rop_annos, f, ensure_ascii=False)
        print(f'done {fp}')



def view_rop_data():
    ip = Path('data/InsuranceData/part1_label/val/part_1.1/bang_ke-DB-25-177647_107.jpg')
    jp = Path(ip).parent / (ip.stem + '-rop.json')
    im = cv2.imread(str(ip))
    with open(jp) as f:
        anno = json.load(f)
    for el in anno:
        order = el['order']
        cv2.rectangle(im, (el['p4_bb'][0], el['p4_bb'][1]), (el['p4_bb'][2], el['p4_bb'][3]), (0, 0, 255), 2)
        cv2.putText(im, str(order), (el['p4_bb'][0], el['p4_bb'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite('test.jpg', im)


def convert_to_paddle_format():
    """
        from text detect json result, convert to paddle format for labeling
    """
    dir = '/data/tungtx2/InsurancePipeline/data/InsuranceData/classified_data-resplit/test'
    out_dir = 'data/InsuranceData/classified_data-resplit/test'
    os.makedirs(out_dir, exist_ok=True)

    for jp in Path(dir).glob('*.json'):
        with open(jp) as f:
            json_data = json.load(f)
        ip = get_img_fp_from_json_fp(jp)
        if ip is None:
            print(f'{jp} not found')
            continue
        
        # ------------- box2segment -------------
        bbs, texts = [], []
        for shape in json_data['shapes']:
            bb = poly2box(shape['points'])
            text = shape['text'] if 'text' in shape else ' '
            bbs.append(bb)
            texts.append(text)
        
        list_hs = []
        for x1, y1, x2, y2 in bbs:
            list_hs.append(y2 - y1)
        avg_h = np.average(list_hs)
        page_text_lines, page_bb_lines = sort_and_merge_box(bbs, texts, 2*avg_h)

        annos = []
        for i, (bb, text) in enumerate(zip(page_bb_lines, page_text_lines)):
            bb = list(map(int, bb))
            xmin, ymin, xmax, ymax = bb
            anno = {
                "transcription": text,
                "points": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                "difficult": False
            }
            annos.append(anno)
        
        anno_str = json.dumps(annos, ensure_ascii=False)
        line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
        with open(os.path.join(out_dir, 'Label.txt'), 'a') as f:
            f.write(line)
        
        line = f'{ip.parent.name}/{ip.name}\t{anno_str}\n'
        with open(os.path.join(out_dir, 'Cache.cach'), 'a') as f:
            f.write(line)
        
        line = f'{str(ip)}\t1\n'
        with open(os.path.join(out_dir, 'fileState.txt'), 'a') as f:
            f.write(line)
        
        print(f'done {ip}')
    

def convert_to_rop_format(src_dir):
    """
        from text detect json result, convert to rop format
    """
    for jp in Path(src_dir).rglob('*.json'):
        if '-rop' in jp.stem:
            continue
        with open(jp) as f:
            json_data = json.load(f)
        ip = get_img_fp_from_json_fp(jp)
        if ip is None:
            continue
        # ------------- box2segment -------------
        bbs, texts = [], []
        for shape in json_data['shapes']:
            bb = poly2box(shape['points'])
            text = shape['text'] if 'text' in shape else ' '
            bbs.append(bb)
            texts.append(text)
        
        list_hs = []
        for x1, y1, x2, y2 in bbs:
            list_hs.append(y2 - y1)
        avg_h = np.average(list_hs)
        page_text_lines, page_bb_lines = sort_and_merge_box(bbs, texts, 1.*avg_h)

        # ------------- remove boxes inside table -------------
        table_xml_fp = jp.parent / f'{jp.stem}-table.xml'
        if os.path.exists(table_xml_fp):
            boxes, names = parse_xml(table_xml_fp)
            if len(boxes) > 0:
                cnt = 0
                table_bb = boxes[0]
                new_text_lines, new_bb_lines = [], []
                for bb, text in zip(page_bb_lines, page_text_lines):
                    r1, r2, iou = iou_poly(bb, table_bb)
                    if r1 > 0.7: # box inside table
                        cnt += 1
                        continue
                    new_text_lines.append(text)
                    new_bb_lines.append(bb)
                page_text_lines, page_bb_lines = new_text_lines, new_bb_lines
                print(f'remove {cnt} boxes inside table')
        # ------------- to rop format -------------
        rop_annos = []
        for i, (bb, text) in enumerate(zip(page_bb_lines, page_text_lines)):
            bb = list(map(int, bb))
            p4_bb = bb
            xmin, ymin, xmax, ymax = bb
            p8_bb = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            rop_anno = {
                'p4_bb': p4_bb,
                'p8_bb': p8_bb,
                'text': text,
                'order': i,
                'id': i
            }
            rop_annos.append(rop_anno)
        with open(jp.parent / (jp.stem + '-rop.json'), 'w') as f:
            json.dump(rop_annos, f, ensure_ascii=False)
        print(f'done {jp}')



if __name__ == '__main__':
    pass
    # convert_CompHRDoc()
    # view_rop_data()
    # json2paddle()
    paddle2rop()
    # convert_ReadingBank()
    # draw_ReadingBank_data()
    # gen_ReadingBank_image()
    # convert_to_rop_format()
    # convert_to_paddle_format()