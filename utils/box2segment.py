from .utils import *


def main():
    src_dir = '/data/tungtx2/reading_order/unirop/data/test_imgs_2'
    out_dir = '/data/tungtx2/reading_order/unirop/data/test_imgs_2'
    os.makedirs(out_dir, exist_ok=True)
    for ip in Path(src_dir).glob('*.jpg'):
        # if ip.stem not in ['2C23THG_00000007-0', '00001004-0']:
        #     continue
        jp = ip.with_suffix('.json')
        if not jp.exists():
            continue
        with open(jp) as f:
            json_data = json.load(f)
        
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
        
        # # visualize
        # im = cv2.imread(str(ip))
        # for bb_index, bb in enumerate(page_bb_lines):
        #     bb = list(map(int, bb))
        #     cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        #     cv2.putText(im, str(bb_index), (bb[0], bb[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.imwrite('test.jpg', im)

        # write to new json
        new_json_data = deepcopy(json_data)
        new_shapes = []
        for bb, text in zip(page_bb_lines, page_text_lines):
            bb = list(map(int, bb))
            shape = {
                "label": "text",
                "points": [
                    [bb[0], bb[1]],
                    [bb[2], bb[3]],
                 ],
                "shape_type": "rectangle",
                "text": text
            }
            new_shapes.append(shape)
        new_json_data['shapes'] = new_shapes
        new_json_data['imagePath'] = f'../images/{ip.name}'
        with open(os.path.join(out_dir, jp.name), 'w') as f:
            json.dump(new_json_data, f, ensure_ascii=False)
        
        print(f'done {jp}')
    

if __name__ == '__main__':
    main()