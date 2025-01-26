#!/bin/bash

#SBATCH --job-name=m3ae_train
#SBATCH --output=/mnt/nas/slurm_account/junyoon/m3ae_public/logs/%j.out
#SBATCH --error=/mnt/nas/slurm_account/junyoon/m3ae_public/logs/%j.err
#SBATCH --gres=gpu:4090:8
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=3-00:00:00  # 3 days runtime

# Load conda environment
source ~/miniconda3/bin/activate
conda activate m3ae_public

# Change to project directory
cd /mnt/nas/slurm_account/junyoon/m3ae_public

# Setup wandb
export WANDB_API_KEY=cbc758f8b3dd320229b848097505e9712d6f6895
wandb online

# Run training command
python3 -m m3ae.m3ae_main \
    --m3ae.model_type='small' \
    --m3ae.image_mask_ratio=0.75 \
    --m3ae.embedding_mask_ratio=0.75 \
    --seed=42 \
    --epochs=100 \
    --lr_warmup_epochs=5 \
    --batch_size=512 \
    --accumulate_grad_steps=8 \
    --discretized_image=False \
    --dataloader_n_workers=16 \
    --log_freq=500 \
    --plot_freq=2000 \
    --save_model_freq=10000 \
    --image_loss_weight=1.0 \
    --text_loss_weight=0.5 \
    --lr_peak_value=1.5e-4 \
    --weight_decay=0.05 \
    --load_checkpoint='' \
    --data.path="/mnt/nas/slurm_account/junyoon/output_wembedding.h5" \
    --data.transform_type='pretrain' \
    --data.image_normalization='imagenet'