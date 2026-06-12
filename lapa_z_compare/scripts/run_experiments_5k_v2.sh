#!/bin/bash
# run_experiments_5k_v2.sh
# Runs the main 5K tier experiment runs using the updated model parameters, batch size 64, decoupled betas, and maximum 90-minute limit.
set -e

# Load python environment
source ~/anaconda3/bin/activate llm312

TRAIN_CSV="data_cache/sthv2_splits/train_5k.csv"
VAL_CSV="data_cache/sthv2_splits/val_1k.csv"
TEST_CSV="data_cache/sthv2_splits/test_1k.csv"
STEPS=5000
IMAGE_SIZE=224
LATENT_DIM=32
BATCH_SIZE=64
MAX_MINUTES=90.0

echo "=== Starting Upgraded 5K Experiment Runs (v2) ==="

mkdir -p lapa_z_compare/outputs/5k_v2/lapa_nsvq
mkdir -p lapa_z_compare/outputs/5k_v2/adaworld_bvae
mkdir -p lapa_z_compare/outputs/5k_v2/univla_dino_vq
mkdir -p lapa_z_compare/outputs/5k_v2/villax_proxy
mkdir -p lapa_z_compare/outputs/5k_v2/vicreg

# We will run them in parallel, assigning specific GPUs to balance load.
# GPU 0: lapa_nsvq, univla_dino_vq, vicreg
# GPU 1: adaworld_bvae, villax_proxy

# 1. Launch Training
PIDS=()

# lapa_nsvq
echo "Launching training for lapa_nsvq on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python lapa_z_compare/scripts/train_latent_model.py \
    --method lapa_nsvq \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --out_dir "lapa_z_compare/outputs/5k_v2/lapa_nsvq" \
    --image_size "$IMAGE_SIZE" \
    --gap 12 \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --save_every 2500 \
    --eval_every 2500 \
    --precision bf16 \
    --vq_beta 0.25 \
    --max_minutes "$MAX_MINUTES" > lapa_z_compare/outputs/5k_v2/lapa_nsvq/train.log 2>&1 &
PIDS+=($!)

# adaworld_bvae
echo "Launching training for adaworld_bvae on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python lapa_z_compare/scripts/train_latent_model.py \
    --method adaworld_bvae \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --out_dir "lapa_z_compare/outputs/5k_v2/adaworld_bvae" \
    --image_size "$IMAGE_SIZE" \
    --gap 12 \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --save_every 2500 \
    --eval_every 2500 \
    --precision bf16 \
    --beta 2e-4 \
    --max_minutes "$MAX_MINUTES" > lapa_z_compare/outputs/5k_v2/adaworld_bvae/train.log 2>&1 &
PIDS+=($!)

# univla_dino_vq
echo "Launching training for univla_dino_vq on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python lapa_z_compare/scripts/train_latent_model.py \
    --method univla_dino_vq \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --out_dir "lapa_z_compare/outputs/5k_v2/univla_dino_vq" \
    --image_size "$IMAGE_SIZE" \
    --gap 12 \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --save_every 2500 \
    --eval_every 2500 \
    --precision bf16 \
    --vq_beta 0.25 \
    --max_minutes "$MAX_MINUTES" > lapa_z_compare/outputs/5k_v2/univla_dino_vq/train.log 2>&1 &
PIDS+=($!)

# villax_proxy
echo "Launching training for villax_proxy on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python lapa_z_compare/scripts/train_latent_model.py \
    --method villax_proxy \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --out_dir "lapa_z_compare/outputs/5k_v2/villax_proxy" \
    --image_size "$IMAGE_SIZE" \
    --gap 12 \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --save_every 2500 \
    --eval_every 2500 \
    --precision bf16 \
    --vq_beta 0.25 \
    --lambda_s 1.0 \
    --max_minutes "$MAX_MINUTES" > lapa_z_compare/outputs/5k_v2/villax_proxy/train.log 2>&1 &
PIDS+=($!)

# vicreg
echo "Launching training for vicreg on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python lapa_z_compare/scripts/train_latent_model.py \
    --method vicreg \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --out_dir "lapa_z_compare/outputs/5k_v2/vicreg" \
    --image_size "$IMAGE_SIZE" \
    --gap 12 \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --save_every 2500 \
    --eval_every 2500 \
    --precision bf16 \
    --lambda_var 1.0 \
    --lambda_cov 0.05 \
    --max_minutes "$MAX_MINUTES" > lapa_z_compare/outputs/5k_v2/vicreg/train.log 2>&1 &
PIDS+=($!)

echo "All training runs launched. Waiting for completion (PIDs: ${PIDS[*]})..."
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo "All training runs completed! Starting exports and evaluations..."

# 2. Run Exports and Evaluations (using GPU 1 sequentially to avoid conflict)
export CUDA_VISIBLE_DEVICES=1
METHODS=("lapa_nsvq" "adaworld_bvae" "univla_dino_vq" "villax_proxy" "vicreg")

for METHOD in "${METHODS[@]}"; do
    OUT_DIR="lapa_z_compare/outputs/5k_v2/${METHOD}"
    CKPT="${OUT_DIR}/checkpoints/checkpoint_final.pt"
    NPZ="${OUT_DIR}/latents_test.npz"
    CSV="${OUT_DIR}/metrics.csv"
    
    echo "Exporting latents for ${METHOD}..."
    python lapa_z_compare/scripts/export_latents.py \
        --ckpt "$CKPT" \
        --test_csv "$TEST_CSV" \
        --out_dir "$OUT_DIR"
        
    echo "Evaluating latents for ${METHOD}..."
    python lapa_z_compare/scripts/evaluate_latent_quality.py \
        --npz "$NPZ" \
        --out_csv "$CSV"
done

echo ""
echo "===================================================="
echo "               FINAL RESULTS SUMMARY                "
echo "===================================================="
python -c "
import pandas as pd
import glob
files = glob.glob('lapa_z_compare/outputs/5k_v2/*/metrics.csv')
dfs = []
for f in files:
    method = f.split('/')[-2]
    df = pd.read_csv(f)
    df.insert(0, 'method', method)
    dfs.append(df)
if dfs:
    summary = pd.concat(dfs, ignore_index=True)
    pd.set_option('display.max_columns', None)
    print(summary.to_string(index=False))
else:
    print('No metrics found.')
"
