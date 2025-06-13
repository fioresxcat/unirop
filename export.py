import os
os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'
import torch
import yaml
from easydict import EasyDict
from pathlib import Path


def to_fp16(fp32_model_path, out_fp16_model_path):
    import onnx
    from onnxconverter_common import float16
    
    model = onnx.load(fp32_model_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, out_fp16_model_path)
    print(f'saved model at {out_fp16_model_path}')
    os.remove(fp32_model_path)


def main(args):
    import pdb
    from models.layoutlmv3.layoutlmv3_gp import LayoutLMv3ForRORelationModule
    
    device = 'cuda:0'
    ckpt_path = args.ckpt_path
    exp_name = ckpt_path.split('/')[-2]
    epoch = ckpt_path.split('/')[-1].split('-')[0]
    ep = epoch.split('=')[1]
    
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    
    state = torch.load(ckpt_path)['state_dict']
    lightning_model = LayoutLMv3ForRORelationModule.load_from_checkpoint(ckpt_path, **config.model)
    lightning_model.to(device).eval()

    # export onnx
    if args.format == 'onnx':
        save_path = os.path.join(Path(ckpt_path).parent, f'epoch{ep}.onnx')
        print('Exporting ...')
        lightning_model.to_onnx(save_path, opset=14, fp16=False, device=device)
        print(f'Onnx model saved to {save_path}')
        print('Converting to float16 ...')
        to_fp16(
            os.path.join(Path(ckpt_path).parent, f'epoch{ep}.onnx'),
            os.path.join(Path(ckpt_path).parent, f'epoch{ep}_fp16.onnx')
        )
    elif args.format == 'hf':  
        # export hf format
        model = lightning_model.model
        model.save_pretrained(args.save_path)
        print(f'Huggingface model saved to {args.save_path}')
        if args.push_to_hub:
            print(f'Pushing model to hub ...')
            model.push_to_hub("fioresxcat/layoutlmv3_base-layout_only-re")
            print('Push to hub done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--format', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='default')
    parser.add_argument('--push_to_hub', action='store_true')

    args = parser.parse_args()

    main(args)
    # check_output_diff()