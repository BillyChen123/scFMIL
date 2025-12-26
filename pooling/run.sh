  # tranning
  python -m pooling2.train \
    --adata /project/Age_classcify/emb_pt/AIDA.h5ad \
    --resolution 0.5 \
    --epochs 10 \
    --out_dir /outputs/aida \
    --device cuda:4

# validing
python -m pooling2.extract \
    --adata /project/Age_classcify/emb_pt/AIDA.h5ad \
    --ckpt_path /outputs/aida/model.pt \
    --sample_key donor_id \
    --resolution 0.5 \
    --out_dir /outputs/aida \
    --device cuda:4