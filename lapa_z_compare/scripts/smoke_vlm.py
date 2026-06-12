"""
Phase 0 smoke test (runs in `.venv_vlm`).

Loads the TOPReward scorer and checks that token-logit access works and that a real
later frame scores higher progress than a no-change pair, on one STHV2 clip.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.vlm_eval.topreward import TOPRewardScorer


def grab_frames(video_path, fracs=(0.05, 0.95)):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for fr in fracs:
        idx = max(0, min(n - 1, int(fr * (n - 1))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {idx} of {video_path}")
        out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data_cache/sthv2_splits/test_1k.csv")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.test_csv)
    scorer = TOPRewardScorer()
    print("Scorer loaded. true_ids:", scorer.true_ids, "false_ids:", scorer.false_ids)

    tmp = Path(tempfile.mkdtemp(prefix="vlm_smoke_"))
    for i in range(min(args.n, len(df))):
        row = df.iloc[i]
        early, late = grab_frames(row["video_path"])
        ep, lp, sp = tmp / f"{i}_e.png", tmp / f"{i}_l.png", tmp / f"{i}_same.png"
        early.save(ep); late.save(lp); early.save(sp)
        log_prog, prog = scorer.reward([str(ep), str(lp)], row["template"])
        log_same, same = scorer.reward([str(ep), str(sp)], row["template"])
        print(f"[{i}] '{row['template']}'  progress(early->late)={prog:.4f}  "
              f"progress(no-change)={same:.4f}  delta={prog - same:+.4f}")


if __name__ == "__main__":
    main()
