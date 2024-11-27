import torch
import yaml
from easydict import EasyDict
from pathlib import Path
import os

def to_fp16(fp32_model_path, out_fp16_model_path):
    import onnx
    from onnxconverter_common import float16
    
    model = onnx.load(fp32_model_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, out_fp16_model_path)
    print(f'saved model at {out_fp16_model_path}')
    os.remove(fp32_model_path)


def main():
    import pdb
    from models.lilt import LiltTokenClassificationModule
    from models.layoutlmv3 import LayoutLMv3TokenClassificationModule
    
    device = 'cuda:0'
    ckpt_path = 'ckpt/lct/exp8_retrain_exp7_modify_lr/epoch=71-train_loss=0.3207-balanced_train_f1=0.9990-micro_train_f1=0.9999-val_loss=0.3230-balanced_val_f1=0.9940-micro_val_f1=0.9991-non_text_val_acc=0.9945.ckpt'
    exp_name = ckpt_path.split('/')[-2]
    epoch = ckpt_path.split('/')[-1].split('-')[0]
    
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    
    state = torch.load(ckpt_path)['state_dict']
    model = LayoutLMv3TokenClassificationModule.load_from_checkpoint(ckpt_path, **config.model)
    model.to(device).eval()
    
    ep = epoch.split('=')[1]
    save_path = os.path.join(Path(ckpt_path).parent, f'epoch{ep}.onnx')
    print('Exporting ...')
    model.to_onnx(save_path, opset=14, fp16=False, device=device)
    print(f'Onnx model saved to {save_path}')
    
    print('Converting to float16 ...')
    to_fp16(
        os.path.join(Path(ckpt_path).parent, f'epoch{ep}.onnx'),
        os.path.join(Path(ckpt_path).parent, f'epoch{ep}_fp16.onnx')
    )

if __name__ == '__main__':
    main()
    # check_output_diff()