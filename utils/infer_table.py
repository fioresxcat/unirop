import cv2
from ultralytics import YOLO
from pathlib import Path
from .utils import is_image
from .utils import write_to_xml


def main(args):
    model = YOLO('/data/tungtx2/reading_order/unirop/utils/table_detect.onnx')
    idx2name = {0: 'table'}
    dir = args.src_dir
    for ip in Path(dir).rglob('*'):
        if not is_image(ip):
            continue
        
        res = model.predict(source=str(ip), conf=0.3, iou=0.3, save=False, save_txt=False, verbose=False)
        if res[0].boxes is None:
            continue

        data_boxes = res[0].boxes.data.detach().cpu().numpy()
        boxes, scores, classes = data_boxes[:, :4], data_boxes[:, 4], data_boxes[:, 5]

        # save detection result to xml
        out_xml_fp = ip.parent / f'{ip.stem}-table.xml'
        img = cv2.imread(str(ip))
        img_w, img_h = img.shape[1], img.shape[0]
        p_boxes = boxes.astype(int).tolist()
        p_names = [idx2name[cl] for cl in classes]
        write_to_xml(p_boxes, p_names, (img_w, img_h), out_xml_fp)
        print(f'done {ip}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)