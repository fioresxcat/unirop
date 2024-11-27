from .utils import *


def sort_and_merge_box(bbs, texts, d_thres):
    """
        bbs: List of (xmin, ymin, xmax, ymax) boxes
        texts: List of words corresponding to bbs
        d_thres: height distance threshold between 2 consecutive lines.
    """
    bbs_clusters = [(b, t) for b, t in zip(bbs, texts)]
    bbs_clusters.sort(key=lambda x: x[0][0])

    # group cluster 1st time
    clusters, y_min, cluster_texts = [], [], []
    for tgt_node, text  in bbs_clusters:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
            continue
        matched = None
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            overlap_y = ((src_node[3] - src_node[1]) + (tgt_node[3] - tgt_node[1])) - (max(src_node[3], tgt_node[3]) - min(src_node[1], tgt_node[1]))
            overlap_x = ((src_node[2] - src_node[0]) + (tgt_node[2] - tgt_node[0])) - (max(src_node[2], tgt_node[2]) - min(src_node[0], tgt_node[0]))
            distance = tgt_node[0] - src_node[2]
            if overlap_y > 0.8*min(src_node[3] - src_node[1], tgt_node[3] - tgt_node[1]) and overlap_x < 0.6*min(src_node[2] - src_node[0], tgt_node[2] - tgt_node[0]):
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
            cluster_texts[idx].append(text)
    zip_clusters = list(zip(clusters, y_min, cluster_texts))
    zip_clusters.sort(key=lambda x: x[1])

    # break lines
    page_text_lines = []
    page_bb_lines = []
    for bb_cluster in zip_clusters:
        bbs, _, texts = bb_cluster
        text_lines = []
        bb_lines = []
        text_line = []
        bb_line = []
        for bb, text in zip(bbs, texts):
            if len(text_line) == 0:
                text_line.append(text)
                bb_line.append(bb)
            else:
                if bb[0] - bb_line[-1][2] > d_thres:
                    text_lines.append(text_line)
                    bb_lines.append(bb_line)
                    text_line = [text]
                    bb_line = [bb]
                else:
                    text_line.append(text)
                    bb_line.append(bb)
        if len(text_line) != 0:
            text_lines.append(text_line)
            bb_lines.append(bb_line)
        for text_line, bb_line in zip(text_lines, bb_lines):
            bb_line = np.array(bb_line)
            xmin = np.min(bb_line[:, 0])
            ymin = np.min(bb_line[:, 1])
            xmax = np.max(bb_line[:, 2])
            ymax = np.max(bb_line[:, 3])
            page_text_lines.append(' '.join(text_line))
            page_bb_lines.append([xmin, ymin, xmax, ymax])
    return page_text_lines, page_bb_lines


def main():
    src_dir = 'raw_data/VAT_data/images'
    out_dir = 'raw_data/VAT_data/segment_jsons'
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
        
        # visualize
        im = cv2.imread(str(ip))
        for bb_index, bb in enumerate(page_bb_lines):
            bb = list(map(int, bb))
            cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
            cv2.putText(im, str(bb_index), (bb[0], bb[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite('test.jpg', im)

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
            json.dump(new_json_data, f)
        
        print(f'done {jp}')
    

if __name__ == '__main__':
    main()