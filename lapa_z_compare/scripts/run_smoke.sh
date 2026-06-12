#!/bin/bash
# run_smoke.sh
# Runs a quick 20-step smoke test for all models to verify forward, backward, export, evaluation, and correctness.
set -e

# Load python environment
source ~/anaconda3/bin/activate llm312

# Use GPU 1 to be polite and avoid conflict on GPU 0
export CUDA_VISIBLE_DEVICES=1

TRAIN_CSV="data_cache/sthv2_splits/debug/train.csv"
VAL_CSV="data_cache/sthv2_splits/debug/val.csv"
TEST_CSV="data_cache/sthv2_splits/debug/test.csv"
STEPS=20
IMAGE_SIZE=96
LATENT_DIM=32
BATCH_SIZE=8

echo "=== Starting Smoke Test Runs on GPU 1 ==="

METHODS=("lapa_nsvq" "adaworld_bvae" "univla_dino_vq" "villax_proxy" "vicreg")

for METHOD in "${METHODS[@]}"; do
    OUT_DIR="lapa_z_compare/outputs/smoke/${METHOD}"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    
    echo "----------------------------------------------------"
    echo "Running training smoke test for ${METHOD}..."
    echo "----------------------------------------------------"
    
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
        --save_every 20 \
        --eval_every 20 \
        --precision bf16
        
    CKPT="${OUT_DIR}/checkpoints/checkpoint_final.pt"
    NPZ="${OUT_DIR}/latents_test.npz"
    CSV="${OUT_DIR}/metrics.csv"
    
    echo "Running export smoke test for ${METHOD}..."
    python lapa_z_compare/scripts/export_latents.py \
        --ckpt "$CKPT" \
        --test_csv "$TEST_CSV" \
        --out_dir "$OUT_DIR"
        
    echo "Running evaluation smoke test for ${METHOD}..."
    python lapa_z_compare/scripts/evaluate_latent_quality.py \
        --npz "$NPZ" \
        --out_csv "$CSV"
        
    echo "Running programmatic smoke assertions..."
    python lapa_z_compare/scripts/verify_smoke.py \
        --method "$METHOD" \
        --ckpt "$CKPT" \
        --npz "$NPZ" \
        --csv "$CSV" \
        --train_csv "$TRAIN_CSV"
done

echo "=============================================="
echo "          ALL SMOKE TESTS PASSED!             "
echo "=============================================="
