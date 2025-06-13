import subprocess
import os
from .convert_data_format import convert_to_rop_format
import sys


def data_pipeline(args):
    """
        1. infer text detect
        2. infer table
        3. convert to rop
        4. infer to paddle
        5. do labeling
    """
    current_dir = os.getcwd()
    project_dir = '/data/tungtx2/reading_order/unirop/'

    src_dir = args.src_dir

    # # ------- infer text detect ---------
    # os.chdir('/data/tungtx2/common_code/text_detect_ocr')
    # cmd = ['python', 'text_detect_module.py', '--src_dir', src_dir]
    # result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    # print("stdout:\n", result.stdout)
    # print("stderr:\n", result.stderr)
    # result.check_returncode()
    # print("Text detection step completed")

    # # ------- infer ocr ---------
    # os.chdir('/data/tungtx2/common_code/text_detect_ocr')
    # cmd = ['python', 'ocr_module.py', '--src_dir', src_dir]
    # result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    # print("stdout:\n", result.stdout)
    # print("stderr:\n", result.stderr)
    # result.check_returncode()
    # print("OCR step completed")

    # # ------- infer table ---------
    # os.chdir(project_dir)
    # cmd = ['python', '-m', 'utils.infer_table', '--src_dir', src_dir]
    # result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    # print("stdout:\n", result.stdout)
    # print("stderr:\n", result.stderr)
    # result.check_returncode()
    # print("Table detection step completed")

    # # ------- convert to rop ---------
    # os.chdir(project_dir)
    # convert_to_rop_format(src_dir)

    # # ------- infer to paddle ---------
    # os.chdir(project_dir)
    # ckpt_path = 'pretrained/layoutlmv3_large-layout_only-re'
    # out_dir = src_dir
    # cmd = ['python', '-m', 'inference.infer_lmv3_re', '--ckpt_path', ckpt_path, 
    #        '--src_dir', src_dir, '--out_dir', out_dir,
    #        '--save_paddle', '--device', 'cuda:0']
    # # cmd = ['python', '-m', 'inference.infer_lmv3_re', '--ckpt_path', ckpt_path, 
    # #        '--src_dir', src_dir, '--out_dir', 'model_output/temp',
    # #        '--vis', '--device', 'cuda:0']
    # subprocess.run(cmd, check=True)
    # print("Inference step completed")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    args = parser.parse_args()
    
    data_pipeline(args)