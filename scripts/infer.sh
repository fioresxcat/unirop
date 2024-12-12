python -m inference.infer \
    --ckpt_path ckpt/VAT/exp7-retrain_exp4-shuffle_val/epoch=111-train_loss=0.001-train_edge_acc=1.000-train_sample_acc=0.994-val_loss=0.392-val_edge_acc=0.959-val_sample_acc=0.656.ckpt \
    --src_dir data/VAT_acb_captured/images/val \
    --out_dir model_output/VAT_acb_captured/exp7/val-seed42\
    --device cuda:3 \
    --shuffle_prob 1 \
    --seed 42 \
    --save_metrics \
    --scale 1.3 \


# python -m inference.infer_2_heads \
#     --ckpt_path ckpt/VAT/exp1_2_heads/epoch=104-train_loss=0.000-train_edge_acc=0.999-train_sample_acc=0.954-val_loss=0.000-val_edge_acc=0.939-val_sample_acc=0.600.ckpt \
#     --src_dir data/VAT_data/images/val \
#     --out_dir model_output/VAT_data/exp1_2_heads/val_seed42\
#     --shuffle \
#     --seed 42 \
#     --device cuda:0 \
#     --save_metrics \