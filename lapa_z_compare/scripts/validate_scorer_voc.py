"""
Phase 1 scorer validation (runs in `.venv_vlm`).

Certifies that the VLM scorers actually track progress on REAL STHV2 clips before we
trust them on decoded frames. For M videos we sample K chronological frames and compute
Value-Order Correlation (VOC) = Spearman rank-corr between predicted progress and true
frame order.

TOPReward value for frame k = normalized p(True) for the anchored pair [frame_0, frame_k]
(the same 2-image query we use in Phase 3). GVL value = parsed per-frame percentage over
shuffled frames. A working scorer should have median VOC clearly > 0.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.vlm_eval.topreward import TOPRewardScorer


def sample_frames(video_path, k, tmp, prefix):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < k:
        cap.release()
        return None
    idxs = np.linspace(0, n - 1, k).astype(int)
    paths = []
    for j, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        p = tmp / f"{prefix}_{j}.png"
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(p)
        paths.append(str(p))
    cap.release()
    return paths


def voc(values):
    vals = np.asarray(values, dtype=float)
    if np.isnan(vals).any() or np.allclose(vals, vals[0]):
        return np.nan
    rho, _ = spearmanr(vals, np.arange(len(vals)))
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data_cache/sthv2_splits/test_1k.csv")
    ap.add_argument("--num_videos", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out_csv", default="lapa_z_compare/outputs/5k_v2/_vlm_eval/scorer_validation.csv")
    ap.add_argument("--with_gvl", action="store_true")
    ap.add_argument("--gvl_only", action="store_true", help="only run GVL (avoids loading two 8B models)")
    args = ap.parse_args()

    df = pd.read_csv(args.test_csv).head(args.num_videos)
    top = None if args.gvl_only else TOPRewardScorer()
    gvl = None
    if args.with_gvl or args.gvl_only:
        from lapa_z_compare.vlm_eval.gvl import GVLScorer
        gvl = GVLScorer()

    tmp = Path(tempfile.mkdtemp(prefix="vlm_voc_"))
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        paths = sample_frames(row["video_path"], args.k, tmp, f"v{i}")
        if paths is None:
            continue
        instr = row["template"]
        rec = {"video_id": row["video_id"]}
        if top is not None:
            top_vals = [top.reward([paths[0], paths[k]], instr)[1] for k in range(len(paths))]
            rec["voc_topreward"] = voc(top_vals)
        if gvl is not None:
            rec["voc_gvl"] = voc(gvl.predict_values(paths, instr))
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(df)} videos")

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("\n=== Scorer Validation (real STHV2) ===")
    print(f"videos scored: {len(out)}")
    if "voc_topreward" in out.columns:
        print(f"TOPReward VOC: median={out['voc_topreward'].median():.3f}  "
              f"mean={out['voc_topreward'].mean():.3f}  %positive={ (out['voc_topreward'] > 0).mean()*100:.1f}%")
    if gvl is not None:
        print(f"GVL VOC:       median={out['voc_gvl'].median():.3f}  "
              f"mean={out['voc_gvl'].mean():.3f}  %positive={ (out['voc_gvl'] > 0).mean()*100:.1f}%")
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
