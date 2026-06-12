#!/bin/bash
set -e

# Load python environment
source ~/anaconda3/bin/activate llm312

METHODS=("lapa_nsvq" "adaworld_bvae" "univla_dino_vq" "villax_proxy" "vicreg")
TEST_CSV="data_cache/sthv2_splits/debug/test.csv"

echo "=== Running Export & Evaluation for All 5 Debug Models ==="

for METHOD in "${METHODS[@]}"; do
    CKPT="lapa_z_compare/outputs/debug/${METHOD}/checkpoints/checkpoint_final.pt"
    OUT_DIR="lapa_z_compare/outputs/debug/${METHOD}"
    NPZ="${OUT_DIR}/latents_test.npz"
    CSV="${OUT_DIR}/metrics.csv"

    if [ -f "$CKPT" ]; then
        echo ""
        echo "----------------------------------------"
        echo "Processing Method: ${METHOD}"
        echo "----------------------------------------"
        
        # 1. Export latents
        python lapa_z_compare/scripts/export_latents.py \
            --ckpt "$CKPT" \
            --test_csv "$TEST_CSV" \
            --out_dir "$OUT_DIR"
            
        # 2. Evaluate latents
        python lapa_z_compare/scripts/evaluate_latent_quality.py \
            --npz "$NPZ" \
            --out_csv "$CSV"
    else
        echo "Warning: Checkpoint for ${METHOD} not found at ${CKPT}"
    fi
done

# Concatenate all CSVs to show a summary table
echo ""
echo "===================================================="
echo "                   SUMMARY TABLE                    "
echo "===================================================="
python -c "
import pandas as pd
import glob
files = glob.glob('lapa_z_compare/outputs/debug/*/metrics.csv')
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
