python -m inference.infer_lmv3_re \
    --ckpt_path ckpt/VAT/exp19-lmv3_large-layout_only-only_acb_data/epoch=6-train_loss=0.007-train_edge_acc=0.999-train_sample_acc=0.966-val_loss=0.104-val_edge_acc=0.978-val_sample_acc=0.703.ckpt \
    --src_dir data/VAT_acb_captured/images/val \
    --out_dir model_output/VAT_acb_captured/exp20-lmv3_large-epoch6/val\
    --device cuda:0 \
    --shuffle_prob 0. \
    --seed 42 \
    --scale 1.3 \
    --save_metrics \


# python -m inference.infer_2_heads \
#     --ckpt_path ckpt/VAT/exp1_2_heads/epoch=104-train_loss=0.000-train_edge_acc=0.999-train_sample_acc=0.954-val_loss=0.000-val_edge_acc=0.939-val_sample_acc=0.600.ckpt \
#     --src_dir data/VAT_data/images/val \
#     --out_dir model_output/VAT_data/exp1_2_heads/val_seed42\
#     --shuffle \
#     --seed 42 \
#     --device cuda:0 \
#     --save_metrics \


# python -m inference.infer_lmv3_token_cls \
#     --ckpt_path ckpt/VAT/exp10-token_cls-layout_only-non_pretrained/epoch=72-train_loss=1.307-train_f1=0.634-train_bleu=0.636-val_loss=0.671-val_f1=0.872-val_bleu=0.872.ckpt \
#     --src_dir data/VAT_acb_captured/images/val \
#     --out_dir model_output/VAT_acb_captured/exp10-token_cls-layout_only-non_pretrained/val \
#     --device cuda:3 \
#     --shuffle_prob 0 \
#     --seed 42 \
#     --save_metrics \
#     --scale 1.3 \