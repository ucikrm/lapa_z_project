"""Mine one strong failure case per baseline for the blog.

A "failure case" = a query where VICReg's top-1 retrieved future is the SAME
template as the query (correct), but the baseline's top-1 is a DIFFERENT template
(wrong). Among candidates we prefer the largest VLM progress gap
(prog_nn1[vicreg] - prog_nn1[baseline]). Copies the 4 real frames
(x_t, real future, VICReg top-1, baseline top-1) into blog/assets/failures/.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("lapa_z_compare/outputs/5k_v2/retrieval")
OUT = Path("blog/assets/failures")
BASELINES = ["lapa_nsvq", "adaworld_bvae", "univla_dino_vq", "villax_proxy"]
LABEL = {
    "lapa_nsvq": "LAPA-NSVQ",
    "adaworld_bvae": "AdaWorld \u03b2-VAE",
    "univla_dino_vq": "UniVLA DINO-VQ",
    "villax_proxy": "villa-X structural",
    "vicreg": "VICReg (ours)",
}


def load(method):
    bank = np.load(f"lapa_z_compare/outputs/5k_v2/{method}/latents_train.npz", allow_pickle=True)
    nbr = np.load(ROOT / method / "retrieval_neighbors.npz", allow_pickle=True)["neighbor_idx"]
    man = pd.read_csv(ROOT / method / "vlm_manifest.csv").set_index("q_idx")
    ps = pd.read_csv(ROOT / method / "vlm_progress_persample.csv").set_index("q_idx")
    return bank["template"], nbr, man, ps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in BASELINES + ["vicreg"]}
    vic_tmpl, vic_nbr, vic_man, vic_ps = data["vicreg"]

    chosen = {}
    used = set()
    for b in BASELINES:
        b_tmpl, b_nbr, b_man, b_ps = data[b]
        cands = []
        for q in vic_man.index:
            qt = str(vic_man.loc[q, "template"])
            vic_nn0_t = str(vic_tmpl[vic_nbr[q, 0]])
            base_nn0_t = str(b_tmpl[b_nbr[q, 0]])
            if vic_nn0_t == qt and base_nn0_t != qt:
                vic_p = float(vic_ps.loc[q, "prog_nn1"]) if q in vic_ps.index else np.nan
                base_p = float(b_ps.loc[q, "prog_nn1"]) if q in b_ps.index else np.nan
                gap = (vic_p - base_p) if np.isfinite(vic_p) and np.isfinite(base_p) else -np.inf
                cands.append((gap, q, vic_p, base_p, qt, base_nn0_t))
        if not cands:
            print(f"[{b}] no clean failure case found")
            continue
        cands.sort(reverse=True)
        # prefer a query not already used by another baseline (distinct examples)
        pick = next((c for c in cands if c[1] not in used), cands[0])
        gap, q, vic_p, base_p, qt, base_nn0_t = pick
        used.add(q)
        d = OUT / b
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(vic_man.loc[q, "xt"], d / "xt.png")
        shutil.copy(vic_man.loc[q, "real_future"], d / "real_future.png")
        shutil.copy(vic_man.loc[q, "nn0"], d / "vicreg_nn0.png")
        shutil.copy(b_man.loc[q, "nn0"], d / "baseline_nn0.png")
        chosen[b] = {
            "baseline": b, "baseline_label": LABEL[b], "q_idx": int(q),
            "instruction": str(vic_man.loc[q, "instruction"]),
            "query_template": qt,
            "baseline_retrieved_template": base_nn0_t,
            "vicreg_prog_nn1": round(vic_p, 3), "baseline_prog_nn1": round(base_p, 3),
            "vlm_gap": round(gap, 3),
            "n_candidates": len(cands),
        }
        print(json.dumps(chosen[b], indent=2))

    (OUT / "failures.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nWrote {OUT/'failures.json'} with {len(chosen)} cases")


if __name__ == "__main__":
    main()
