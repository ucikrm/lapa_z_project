# lapa_z_project

Comparative study of **latent action** representations on Something-Something V2 (STHV2): five methods (LAPA-NSVQ, AdaWorld β-VAE, UniVLA DINO-VQ, villa-X proxy, VICReg) trained under a shared harness, with geometry metrics and decoder-free retrieval + VLM progress evaluation.

The main implementation lives in **`lapa_z_compare/`**. The top-level **`scripts/`** folder is an older prototype lineage (VQ vs continuous phase-1).

## Setup

- **Training / export / geometry eval:** Python 3.12+ with PyTorch, torchvision, transformers (for UniVLA backbones), pandas, opencv.
- **VLM scoring (TOPReward / GVL):** use a separate env (e.g. `.venv_vlm`) with `transformers`, `qwen-vl-utils`, and a GPU large enough for Qwen3-VL-8B.

Point `data` at your STHV2 video root (or set paths in split CSVs). Build splits with:

```bash
python lapa_z_compare/scripts/build_sthv2_index.py
```

## Main workflow (`lapa_z_compare`)

```bash
# 1. Smoke / tiny overfit (optional)
bash lapa_z_compare/scripts/run_smoke.sh

# 2. Train all five methods (5k tier, ~90 min each)
bash lapa_z_compare/scripts/run_experiments_5k_v2.sh

# 3. Export latents + geometry metrics (also run by the script above)
python lapa_z_compare/scripts/export_latents.py \
  --ckpt lapa_z_compare/outputs/5k_v2/vicreg/checkpoints/checkpoint_final.pt \
  --test_csv data_cache/sthv2_splits/test_1k.csv \
  --out_dir lapa_z_compare/outputs/5k_v2/vicreg

python lapa_z_compare/scripts/evaluate_latent_quality.py \
  --npz lapa_z_compare/outputs/5k_v2/vicreg/latents_test.npz \
  --out_csv lapa_z_compare/outputs/5k_v2/vicreg/metrics.csv

# 4. Compile HTML/PDF results table
python lapa_z_compare/scripts/compile_results.py

# 5. Decoder-free retrieval + VLM progress (requires .venv_vlm)
python lapa_z_compare/scripts/retrieval_eval.py ...
python lapa_z_compare/scripts/score_retrieval_vlm.py ...
python lapa_z_compare/scripts/build_retrieval_report.py ...
```

Checkpoints, exported `.npz`, metrics CSVs, and reports under `lapa_z_compare/outputs/` are **local artifacts** (not committed).

## Layout

```text
lapa_z_compare/
  src/models/          # five latent-action models + shared encoder/decoder
  src/data/            # STHV2 frame-pair dataset
  src/losses/          # VICReg loss
  scripts/             # train, export, evaluate, retrieval, VLM scoring
  vlm_eval/            # TOPReward + GVL scorers
scripts/               # legacy prototype scripts
```

## Methods (proxies)

| Method | Latent type | Notes |
|--------|-------------|--------|
| LAPA-NSVQ | Discrete (NSVQ) | Pixel recon + commitment |
| AdaWorld β-VAE | Continuous (VAE) | Conditional frame-pair VAE, small β |
| UniVLA DINO-VQ | Discrete (VQ) | DINOv2 + T5 features, feature-space recon |
| villa-X proxy | Discrete (VQ) | Pixel recon + Sobel structural head |
| VICReg (ours) | Continuous | Pixel recon + variance/covariance on z |

Implementations are **paper-inspired proxies** on a shared backbone for fair comparison, not full reproductions of each paper's system.
