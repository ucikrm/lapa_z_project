"""
Part B — runs in `.venv_vlm`.

Scores REAL retrieved futures (never decoder output) with TOPReward, isolating z
quality from the weak decoder. For each query in every method's vlm_manifest.csv:

  prog_nochange = progress([x_t, x_t]            | instruction)   (lower ref)
  prog_real     = progress([x_t, real x_{t+H}]   | instruction)   (upper ref)
  prog_nn1      = progress([x_t, NN1_future]      | instruction)   (z top-1 retrieval)
  prog_nn_topk  = mean over available NN futures
  prog_random   = progress([x_t, random_future]  | instruction)   (uniform baseline)
  prog_wrong    = progress([x_t, wrong_tmpl_fut] | instruction)   (wrong-template control)

Aggregate (per method):
  topreward_lift_over_random = mean(prog_nn1) - mean(prog_random)
  wrong_template_contrast    = mean(prog_nn1) - mean(prog_wrong)
  aggregate_retention        = (mean(prog_nn1)-mean(prog_nochange)) /
                               (mean(prog_real)-mean(prog_nochange))   [gated denom>0.05]
  ordering_acc               = frac(prog_real > prog_nochange)         [scorer sanity]
A global pair cache keyed on (xt,cand) dedupes method-independent frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lapa_z_compare.vlm_eval.topreward import TOPRewardScorer


def boot_ci(diffs, n=2000, seed=0):
    diffs = np.asarray([d for d in diffs if d is not None], dtype=float)
    if len(diffs) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_dir", required=True, help="dir with <method>/vlm_manifest.csv")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--retention_min_gap", type=float, default=0.05)
    args = ap.parse_args()

    scorer = TOPRewardScorer(model_id=args.model_id)
    cache = {}

    def score(xt, cand, instr):
        key = f"{xt}||{cand}||{instr}"
        if key in cache:
            return cache[key]
        _, p = scorer.reward([xt, cand], instr)
        cache[key] = p
        return p

    out_root = Path(args.retrieval_dir)
    all_summ = []
    for method in args.methods:
        man_p = out_root / method / "vlm_manifest.csv"
        if not man_p.exists():
            print(f"[skip] no manifest for {method}")
            continue
        man = pd.read_csv(man_p).fillna("")
        rows = []
        for r in man.itertuples():
            instr = r.instruction
            xt = r.xt
            p_nochange = score(xt, xt, instr)
            p_real = score(xt, r.real_future, instr)
            p_random = score(xt, r.random_future, instr)
            p_wrong = score(xt, r.wrong_future, instr)
            nn_scores = []
            for k in range(args.topk):
                p = getattr(r, f"nn{k}", "")
                if isinstance(p, str) and p:
                    nn_scores.append(score(xt, p, instr))
            p_nn1 = nn_scores[0] if nn_scores else float("nan")
            p_nntopk = float(np.mean(nn_scores)) if nn_scores else float("nan")
            rows.append({
                "method": method, "q_idx": int(r.q_idx),
                "prog_nochange": p_nochange, "prog_real": p_real,
                "prog_random": p_random, "prog_wrong": p_wrong,
                "prog_nn1": p_nn1, "prog_nn_topk": p_nntopk,
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_root / method / "vlm_progress_persample.csv", index=False)

        m_nochange = df["prog_nochange"].mean()
        m_real = df["prog_real"].mean()
        denom = m_real - m_nochange
        m_nn1 = df["prog_nn1"].mean()
        retention = (m_nn1 - m_nochange) / denom if denom > args.retention_min_gap else float("nan")
        lift = m_nn1 - df["prog_random"].mean()
        lo, hi = boot_ci((df["prog_nn1"] - df["prog_random"]).tolist())
        summ = {
            "method": method,
            "prog_nochange": m_nochange,
            "prog_real": m_real,
            "prog_random": df["prog_random"].mean(),
            "prog_wrong": df["prog_wrong"].mean(),
            "prog_nn1": m_nn1,
            "prog_nn_top5_mean": df["prog_nn_topk"].mean(),
            "topreward_lift_over_random": lift,
            "lift_ci95_lo": lo,
            "lift_ci95_hi": hi,
            "wrong_template_contrast": m_nn1 - df["prog_wrong"].mean(),
            "aggregate_retention": retention,
            "retention_denom": denom,
            "ordering_acc": float((df["prog_real"] > df["prog_nochange"]).mean()),
            "n": len(df),
        }
        all_summ.append(summ)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})

    out = pd.DataFrame(all_summ)
    out.to_csv(out_root / "vlm_retrieval_summary.csv", index=False)
    with open(out_root / "pair_cache.json", "w") as f:
        json.dump({"n_pairs": len(cache)}, f)
    print(f"\nSaved -> {out_root/'vlm_retrieval_summary.csv'} (cache pairs: {len(cache)})")


if __name__ == "__main__":
    main()
