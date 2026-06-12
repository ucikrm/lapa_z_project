#!/bin/bash
set -e

# Load python environment
source ~/anaconda3/bin/activate llm312

# Use GPU 1 to be polite and avoid conflict on GPU 0
export CUDA_VISIBLE_DEVICES=1

TRAIN_CSV="data_cache/sthv2_splits/train_5k.csv"
VAL_CSV="data_cache/sthv2_splits/val_1k.csv"
TEST_CSV="data_cache/sthv2_splits/test_1k.csv"
STEPS=5000
IMAGE_SIZE=224
LATENT_DIM=32
BATCH_SIZE=64

echo "=== Starting 5K Experiment Runs on GPU 1 ==="

METHODS=("lapa_nsvq" "adaworld_bvae" "univla_dino_vq" "villax_proxy" "vicreg")
PIDS=()

for METHOD in "${METHODS[@]}"; do
    OUT_DIR="lapa_z_compare/outputs/5k/${METHOD}"
    mkdir -p "$OUT_DIR"
    
    echo "Launching training for ${METHOD} in background..."
    python lapa_z_compare/scripts/train_latent_model.py \
        --method "$METHOD" \
        --train_csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --out_dir "$OUT_DIR" \
        --image_size "$IMAGE_SIZE" \
        --gap 12 \
        --latent_dim "$LATENT_DIM" \
        --batch_size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --save_every 2500 \
        --eval_every 2500 \
        --precision bf16 > "${OUT_DIR}/train.log" 2>&1 &
    
    PIDS+=($!)
done

echo "All 5 training runs launched. Waiting for completion (PIDs: ${PIDS[*]})..."
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo "All training runs completed! Starting exports..."

# Run exports and evaluations sequentially
for METHOD in "${METHODS[@]}"; do
    OUT_DIR="lapa_z_compare/outputs/5k/${METHOD}"
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
files = glob.glob('lapa_z_compare/outputs/5k/*/metrics.csv')
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
