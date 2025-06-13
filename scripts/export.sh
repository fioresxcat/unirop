CUDA_VISIBLE_DEVICES='3' python export.py \
    --ckpt_path ckpt/ReadingBank/exp3-lmv3_base-layout_and_text/epoch=6-train_loss=0.002-val_loss=0.085.ckpt \
    --format hf \
    --save_path pretrained/layoutlmv3_base-layout_and_text-re \