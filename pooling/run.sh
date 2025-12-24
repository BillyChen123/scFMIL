python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --device cuda:0 --epochs 100 --batch_size 8 \
    --out_dir outputs/pooling_pbmc_ch \
    --save_weights --diagnose

python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --emb_key X \
    --label_key age_value \
    --sample_key donor_id \
    --resolution 5 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --attn_hidden 64 \
    --dropout 0.2 \
    --seed 42 \
    --save_weights \
    --diagnose \
    --out_dir outputs/pooling_pbmc_ch_meanpool\
    --device cuda:6
    # --cluster_pool mean

python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --emb_key X \
    --label_key age_value \
    --sample_key donor_id \
    --resolution 2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --attn_hidden 64 \
    --dropout 0.2 \
    --seed 42 \
    --save_weights \
    --diagnose \
    --cluster_entropy_weight 0.05 \
    --out_dir outputs/pooling_pbmc_ch_entropy \
    --device cuda:6

python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --emb_key X \
    --label_key age_value \
    --sample_key donor_id \
    --resolution 2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --attn_hidden 64 \
    --dropout 0.2 \
    --seed 42 \
    --save_weights \
    --diagnose \
    --cluster_var_weight 0.2 \
    --cluster_var_target 0.2 \
    --out_dir outputs/pooling_pbmc_ch_var \
    --device cuda:6

python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --resolution 2 \
    --epochs 10 \
    --cluster_pool mean \
    --save_weights --diagnose \
    --out_dir outputs/pooling_pbmc_ch_mean_first \
    --device cuda:6

python -m pooling.train \
    --adata /data/chenyz/project/Age_classcify/emb_pt_v3/PBMC_CH.h5ad \
    --resolution 2 \
    --epochs 10 \
    --cluster_pool mean_attn \
    --cluster_alpha 0.1 \
    --save_weights --diagnose \
    --out_dir outputs/pooling_pbmc_ch_mean_attn \
    --device cuda:2