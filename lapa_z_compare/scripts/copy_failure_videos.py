"""Copy the source STHV2 .webm clips for each blog failure case so the blog
plays them offline. For each case we copy three clips:
  query.webm    - the test query clip
  vicreg.webm   - VICReg's top-1 retrieved train clip
  baseline.webm - the baseline's top-1 retrieved train clip
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("lapa_z_compare/outputs/5k_v2/retrieval")
OUT = Path("blog/assets/failures")
BASELINES = ["lapa_nsvq", "adaworld_bvae", "univla_dino_vq", "villax_proxy"]

tr = pd.read_csv("data_cache/sthv2_splits/train_5k.csv")
te = pd.read_csv("data_cache/sthv2_splits/test_1k.csv")
tr_path = {str(r.video_id): r.video_path for r in tr.itertuples()}
te_path = {str(r.video_id): r.video_path for r in te.itertuples()}

# query video ids (any method's test npz has the same order)
q_vid = np.load("lapa_z_compare/outputs/5k_v2/vicreg/latents_test.npz", allow_pickle=True)["video_id"].astype(str)
vic_bank_vid = np.load("lapa_z_compare/outputs/5k_v2/vicreg/latents_train.npz", allow_pickle=True)["video_id"].astype(str)
vic_nbr = np.load(ROOT / "vicreg" / "retrieval_neighbors.npz", allow_pickle=True)["neighbor_idx"]

failures = json.loads((OUT / "failures.json").read_text())

def copy_clip(video_id, dst):
    src = tr_path.get(video_id) or te_path.get(video_id)
    if src and Path(src).exists():
        shutil.copy(src, dst)
        return True
    print(f"[warn] missing video {video_id} -> {src}")
    return False

for b, info in failures.items():
    q = info["q_idx"]
    d = OUT / b
    b_bank_vid = np.load(f"lapa_z_compare/outputs/5k_v2/{b}/latents_train.npz", allow_pickle=True)["video_id"].astype(str)
    b_nbr = np.load(ROOT / b / "retrieval_neighbors.npz", allow_pickle=True)["neighbor_idx"]
    copy_clip(q_vid[q], d / "query.webm")
    copy_clip(vic_bank_vid[vic_nbr[q, 0]], d / "vicreg.webm")
    copy_clip(b_bank_vid[b_nbr[q, 0]], d / "baseline.webm")
    print(f"[{b}] q={q} copied query+vicreg+baseline clips")

print("done")
