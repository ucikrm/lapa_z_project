# lapa_z_project

Comparative study of **latent action** representations on Something-Something V2 (STHV2): five methods
(LAPA-NSVQ, AdaWorld beta-VAE, UniVLA DINO-VQ, villa-X visual-FDM proxy, and VICReg) trained under a shared
harness, with geometry metrics and decoder-free retrieval + VLM progress evaluation.

The main implementation lives in **`lapa_z_compare/`**. The top-level **`scripts/`** folder is an older
prototype lineage (VQ vs continuous phase-1).

## What is reproducible here

This repository can reproduce the 5k-clip training-tier comparison if you have:

- Something-Something V2 videos as `{video_id}.webm`.
- STHV2 labels containing `train.json` and `validation.json`.
- A CUDA environment for training.
- A separate, larger GPU environment for Qwen3-VL-8B if you want to reproduce the VLM progress scores.

The published baselines are implemented as **comparable controlled proxies**, not full reproductions of each
paper's full system.

## Environment

The convenience shell scripts currently assume a local conda env named `llm312`:

```bash
source ~/anaconda3/bin/activate llm312
```

If your environment has a different name, edit the `source ... llm312` line in:

- `lapa_z_compare/scripts/run_smoke.sh`
- `lapa_z_compare/scripts/run_experiments_5k_v2.sh`

Training/export/evaluation dependencies:

```bash
pip install numpy pandas tqdm pillow opencv-python scikit-learn scipy matplotlib
pip install transformers sentencepiece accelerate
# Install torch/torchvision using the command appropriate for your CUDA version:
# https://pytorch.org/get-started/locally/
```

VLM scoring dependencies should live in a separate env, for example `.venv_vlm`:

```bash
python -m venv .venv_vlm
source .venv_vlm/bin/activate
pip install numpy pandas pillow opencv-python scipy transformers accelerate qwen-vl-utils
# Install a CUDA-enabled torch build appropriate for the VLM GPU.
```

The VLM scripts load `Qwen/Qwen3-VL-8B-Instruct`, so the VLM machine needs enough GPU memory for that model.

## Data and splits

Expected raw data layout:

```text
STHV2_VIDEOS/
  12345.webm
  67890.webm
  ...

STHV2_LABELS/
  train.json
  validation.json
```

Build the full filtered split CSVs and debug split:

```bash
python lapa_z_compare/scripts/build_sthv2_index.py \
  --videos_root /path/to/20bn-something-something-v2 \
  --labels_root /path/to/labels \
  --out_dir data_cache/sthv2_splits \
  --num_train 50000 \
  --num_val 5000 \
  --num_test 5000 \
  --seed 42
```

The main 5k runner expects `train_5k.csv`, `val_1k.csv`, and `test_1k.csv`. Create those deterministic subsets
from the generated full splits:

```bash
python -c "import pandas as pd, pathlib; r=pathlib.Path('data_cache/sthv2_splits'); pd.read_csv(r/'train.csv').head(5000).to_csv(r/'train_5k.csv', index=False); pd.read_csv(r/'val.csv').head(1000).to_csv(r/'val_1k.csv', index=False); pd.read_csv(r/'test.csv').head(1000).to_csv(r/'test_1k.csv', index=False)"
```

## Main workflow

### 1. Smoke test

This checks forward/backward, export, geometry evaluation, and basic assertions on the small debug split.

```bash
bash lapa_z_compare/scripts/run_smoke.sh
```

### 2. Train all five 5k-tier models

The default script launches five jobs in parallel across two GPUs:

- GPU 0: LAPA-NSVQ, UniVLA DINO-VQ, VICReg
- GPU 1: AdaWorld beta-VAE, villa-X visual-FDM proxy

If you have a different GPU setup, edit the `CUDA_VISIBLE_DEVICES=...` assignments in the script or run the
`train_latent_model.py` commands manually.

```bash
bash lapa_z_compare/scripts/run_experiments_5k_v2.sh
```

This writes checkpoints, logs, test latents, and geometry metrics under:

```text
lapa_z_compare/outputs/5k_v2/{method}/
```

The method keys are:

```text
lapa_nsvq
adaworld_bvae
univla_dino_vq
villax_proxy
vicreg
```

### 3. Export train-bank latents for decoder-free retrieval

The 5k runner exports `latents_test.npz`. Retrieval also needs `latents_train.npz` for the train bank:

```bash
for METHOD in lapa_nsvq adaworld_bvae univla_dino_vq villax_proxy vicreg; do
  OUT_DIR="lapa_z_compare/outputs/5k_v2/${METHOD}"
  python lapa_z_compare/scripts/export_latents.py \
    --ckpt "${OUT_DIR}/checkpoints/checkpoint_final.pt" \
    --test_csv data_cache/sthv2_splits/train_5k.csv \
    --out_dir "${OUT_DIR}" \
    --out_name latents_train.npz \
    --split train
done
```

### 4. Compile geometry results

```bash
python lapa_z_compare/scripts/compile_results.py
```

Outputs:

```text
lapa_z_results_table.html
lapa_z_results_table.pdf   # only if matplotlib is installed
```

### 5. Run decoder-free retrieval

This uses each method's `z` to retrieve real future frames from the train bank. It does not use decoded frames.

```bash
RETR="lapa_z_compare/outputs/5k_v2/retrieval"

for METHOD in lapa_nsvq adaworld_bvae univla_dino_vq villax_proxy vicreg; do
  python lapa_z_compare/scripts/retrieval_eval.py \
    --method "${METHOD}" \
    --bank "lapa_z_compare/outputs/5k_v2/${METHOD}/latents_train.npz" \
    --query "lapa_z_compare/outputs/5k_v2/${METHOD}/latents_test.npz" \
    --train_csv data_cache/sthv2_splits/train_5k.csv \
    --test_csv data_cache/sthv2_splits/test_1k.csv \
    --out_dir "${RETR}/${METHOD}" \
    --frame_cache "${RETR}/frame_cache" \
    --topk 5 \
    --buffer 50 \
    --num_score 120 \
    --gap 12 \
    --seed 0
done
```

This writes same-template retrieval metrics and VLM frame manifests under:

```text
lapa_z_compare/outputs/5k_v2/retrieval/{method}/
```

### 6. Validate and run VLM progress scoring

Activate the VLM environment:

```bash
source .venv_vlm/bin/activate
```

Optional but recommended: validate that the scorer orders real STHV2 frames by progress.

```bash
python lapa_z_compare/scripts/validate_scorer_voc.py \
  --test_csv data_cache/sthv2_splits/test_1k.csv \
  --num_videos 40 \
  --k 8 \
  --out_csv lapa_z_compare/outputs/5k_v2/_vlm_eval/scorer_validation.csv
```

Score the retrieved real futures with TOPReward/Qwen3-VL:

```bash
python lapa_z_compare/scripts/score_retrieval_vlm.py \
  --retrieval_dir lapa_z_compare/outputs/5k_v2/retrieval \
  --methods lapa_nsvq adaworld_bvae univla_dino_vq villax_proxy vicreg \
  --topk 5
```

Build the self-contained retrieval report:

```bash
python lapa_z_compare/scripts/build_retrieval_report.py \
  --retrieval_dir lapa_z_compare/outputs/5k_v2/retrieval \
  --out lapa_z_compare/outputs/5k_v2/retrieval/retrieval_report.html \
  --n_grid 6
```

## Outputs

Local experiment artifacts are intentionally not committed. After a full run, the main files are:

```text
lapa_z_compare/outputs/5k_v2/{method}/checkpoints/checkpoint_final.pt
lapa_z_compare/outputs/5k_v2/{method}/latents_test.npz
lapa_z_compare/outputs/5k_v2/{method}/latents_train.npz
lapa_z_compare/outputs/5k_v2/{method}/metrics.csv
lapa_z_compare/outputs/5k_v2/retrieval/{method}/retrieval_metrics.csv
lapa_z_compare/outputs/5k_v2/retrieval/{method}/vlm_manifest.csv
lapa_z_compare/outputs/5k_v2/retrieval/vlm_retrieval_summary.csv
lapa_z_compare/outputs/5k_v2/retrieval/retrieval_report.html
lapa_z_results_table.html
lapa_z_results_table.pdf
```

## Layout

```text
lapa_z_compare/
  src/models/          # five latent-action models + shared encoder/decoder templates
  src/data/            # STHV2 frame-pair dataset
  src/losses/          # VICReg loss
  scripts/             # train, export, evaluate, retrieval, VLM scoring
  vlm_eval/            # TOPReward + GVL scorers
scripts/               # legacy prototype scripts
blog/                  # writeup and figures
```

## Methods

| Method | Latent type | Notes |
|--------|-------------|-------|
| LAPA-NSVQ | Discrete (NSVQ) | Pixel reconstruction + NSVQ commitment |
| AdaWorld beta-VAE | Continuous (VAE) | Conditional frame-pair VAE with small beta |
| UniVLA DINO-VQ | Discrete (VQ) | Frozen DINOv2/T5 features, feature-space reconstruction |
| villa-X visual-FDM proxy | Discrete (VQ) | Pixel reconstruction + Sobel-edge visual-FDM proxy |
| VICReg (ours) | Continuous | Pixel reconstruction + variance/covariance regularization on `z` |

Again, these are controlled proxy implementations in one shared harness, not full paper reproductions.
