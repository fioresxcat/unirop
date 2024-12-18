import json
import os
from pathlib import Path
import pdb
import shutil
import numpy as np
import cv2
from PIL import Image
from .utils import poly2box


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


def json2paddle():
    im_dir = 'raw_data/VAT_scan_local_images/images'
    json_dir = 'raw_data/VAT_scan_local_images/segment_jsons'

    for ip in Path(im_dir).glob('*.jpg'):
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
    dir = '/home/fiores/Desktop/VNG/unirop/raw_data/VAT_data/images'
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
        with open(os.path.join(dir, fn.replace('.jpg', '.json')), 'w') as f:
            json.dump(rop_annos, f, ensure_ascii=False)
        print(f'done {fp}')

    


def view_data():
    ip = 'data/CompHRDoc_reduce/test/1808.07823_3.png'
    jp = 'data/CompHRDoc_reduce/test/1808.07823_3.json'
    im = cv2.imread(str(ip))
    with open(jp) as f:
        anno = json.load(f)
    for el in anno:
        order = el['order']
        cv2.rectangle(im, (el['p4_bb'][0], el['p4_bb'][1]), (el['p4_bb'][2], el['p4_bb'][3]), (0, 0, 255), 2)
        cv2.putText(im, str(order), (el['p4_bb'][0], el['p4_bb'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite('test.jpg', im)


if __name__ == '__main__':
    pass
    # convert_CompHRDoc()
    # view_data()
    json2paddle()
    # paddle2rop()