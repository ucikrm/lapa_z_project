"""
Phase 3 (scoring step) — runs in `.venv_vlm`.

Reads each method's decode manifest and scores task progress of the decoded future with
TOPReward, plus controls that hold the decoder fixed:

  prog_true   = progress([x_t, x_hat(true z)]   | template)
  prog_zero   = progress([x_t, x_hat(z=0)]       | template)
  prog_random = progress([x_t, x_hat(random z)]  | template)
  prog_real   = progress([x_t, real x_{t+H}]     | template)   (method-independent, cached)
  prog_wrong  = progress([x_t, x_hat(true z)]    | WRONG template)

Aggregated semantic metrics per method:
  progress_true           mean prog_true
  zero_z_progress_gap     mean(prog_true - prog_zero)      (>0: z adds task-relevant info)
  random_z_progress_gap   mean(prog_true - prog_random)    (>0: z is sample-specific)
  wrong_template_contrast mean(prog_true - prog_wrong)     (>0: future matches THIS action)
  real_vs_decoded_gap     mean(prog_real - prog_true)      (smaller: decoded ~ real future)
  ordering_acc            frac(prog_zero <= prog_true <= prog_real)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.vlm_eval.topreward import TOPRewardScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="lapa_z_compare/outputs/5k_v2/_vlm_eval")
    ap.add_argument("--methods", nargs="+",
                    default=["lapa_nsvq", "adaworld_bvae", "univla_dino_vq", "villax_proxy", "vicreg"])
    ap.add_argument("--limit", type=int, default=-1, help="cap samples per method (debug)")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    scorer = TOPRewardScorer()
    real_cache = {}  # idx -> prog_real (method-independent)
    summary = []

    for method in args.methods:
        manifest = eval_root / method / "manifest.csv"
        if not manifest.exists():
            print(f"skip {method}: no manifest")
            continue
        df = pd.read_csv(manifest)
        if args.limit > 0:
            df = df.head(args.limit)

        recs = []
        for _, r in df.iterrows():
            tmpl = str(r["template"])
            prog_true = scorer.reward([r["xt"], r["true"]], tmpl)[1]
            prog_zero = scorer.reward([r["xt"], r["zero"]], tmpl)[1]
            prog_rand = scorer.reward([r["xt"], r["random"]], tmpl)[1]
            prog_wrong = scorer.reward([r["xt"], r["true"]], str(r["wrong_template"]))[1]
            idx = int(r["idx"])
            if idx not in real_cache:
                real_cache[idx] = scorer.reward([r["xt"], r["real"]], tmpl)[1]
            prog_real = real_cache[idx]
            recs.append({
                "idx": idx, "video_id": r["video_id"], "template": tmpl,
                "prog_true": prog_true, "prog_zero": prog_zero, "prog_random": prog_rand,
                "prog_real": prog_real, "prog_wrong": prog_wrong,
            })
        pm = pd.DataFrame(recs)
        pm.to_csv(eval_root / method / "progress_samples.csv", index=False)

        order_ok = ((pm["prog_zero"] <= pm["prog_true"]) & (pm["prog_true"] <= pm["prog_real"])).mean()
        row = {
            "method": method,
            "progress_true": pm["prog_true"].mean(),
            "zero_z_progress_gap": (pm["prog_true"] - pm["prog_zero"]).mean(),
            "random_z_progress_gap": (pm["prog_true"] - pm["prog_random"]).mean(),
            "wrong_template_contrast": (pm["prog_true"] - pm["prog_wrong"]).mean(),
            "real_vs_decoded_gap": (pm["prog_real"] - pm["prog_true"]).mean(),
            "ordering_acc": order_ok,
            "n": len(pm),
        }
        summary.append(row)
        print(f"[{method}] true={row['progress_true']:.3f} zeroGap={row['zero_z_progress_gap']:+.3f} "
              f"randGap={row['random_z_progress_gap']:+.3f} wrongContrast={row['wrong_template_contrast']:+.3f} "
              f"realGap={row['real_vs_decoded_gap']:+.3f} ordAcc={row['ordering_acc']:.3f}")

    out = pd.DataFrame(summary)
    out_path = eval_root / "progress_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
