import json
import os
from pathlib import Path
import pdb
import shutil
import numpy as np
import cv2
from PIL import Image


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
    convert_CompHRDoc()
    # view_data()