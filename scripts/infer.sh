python -m inference.infer \
    --ckpt_path ckpt/VAT/exp0/epoch=88-train_loss=0.000-train_edge_acc=1.000-train_sample_acc=0.995-val_loss=0.363-val_edge_acc=0.950-val_sample_acc=0.600.ckpt \
    --src_dir data/VAT_data/images/val \
    --out_dir model_output/VAT_data/exp0/val_seed42\
    --shuffle \
    --seed 42 \
    --device cuda:3 \
    # --save_metrics \


# python -m inference.infer_2_heads \
#     --ckpt_path ckpt/VAT/exp1_2_heads/epoch=104-train_loss=0.000-train_edge_acc=0.999-train_sample_acc=0.954-val_loss=0.000-val_edge_acc=0.939-val_sample_acc=0.600.ckpt \
#     --src_dir data/VAT_data/images/val \
#     --out_dir model_output/VAT_data/exp1_2_heads/val_seed42\
#     --shuffle \
#     --seed 42 \
#     --device cuda:0 \
#     --save_metrics \