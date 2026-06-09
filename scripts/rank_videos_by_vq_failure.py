from pathlib import Path
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--vq_dir", type=str, required=True)
parser.add_argument("--cont_dir", type=str, required=True)
parser.add_argument("--subset_meta", type=str, required=True)
parser.add_argument("--out_csv", type=str, required=True)
args = parser.parse_args()

vq_dir = Path(args.vq_dir)
cont_dir = Path(args.cont_dir)

Z_vq = np.load(vq_dir / "z_quantized.npy")
vq_indices = np.load(vq_dir / "indices.npy")
meta_vq = pd.read_csv(vq_dir / "metadata.csv")

Z_cont = np.load(cont_dir / "z_continuous.npy")

subset_meta = pd.read_csv(args.subset_meta)

label_map = {}
for _, row in subset_meta.iterrows():
    traj = f"traj_{int(row['traj_idx']):06d}"
    label_map[traj] = row.get("label", "")

def effective_rank(Z):
    if len(Z) < 2:
        return 0.0
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

def path_length(Z):
    if len(Z) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(Z, axis=0), axis=1).sum())

meta_vq["traj"] = meta_vq["x1_path"].apply(lambda p: Path(str(p)).parent.name)

rows = []
for traj, group in meta_vq.groupby("traj"):
    idxs = group.index.to_numpy()
    codes = vq_indices[idxs]
    z_vq = Z_vq[idxs]
    z_cont = Z_cont[idxs]

    unique, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    top_frac = float(probs.max())

    vq_rank = effective_rank(z_vq)
    cont_rank = effective_rank(z_cont)

    vq_path = path_length(z_vq)
    cont_path = path_length(z_cont)

    # High score = good failure case for VQ:
    # VQ uses few codes, one code dominates, continuous space is richer.
    score = (
        top_frac
        + (1.0 / (1.0 + entropy))
        + max(0.0, cont_rank - vq_rank) / 32.0
        + max(0.0, cont_path - vq_path) / (cont_path + 1e-8)
    )

    rows.append({
        "traj": traj,
        "label": label_map.get(traj, ""),
        "num_pairs": len(group),
        "vq_used_codes": len(unique),
        "vq_entropy": entropy,
        "vq_top_code_fraction": top_frac,
        "vq_effective_rank": vq_rank,
        "continuous_effective_rank": cont_rank,
        "vq_path_length": vq_path,
        "continuous_path_length": cont_path,
        "failure_score": score,
        "vq_codes": " ".join(map(str, codes.tolist())),
    })

df = pd.DataFrame(rows)
df = df.sort_values("failure_score", ascending=False)
Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out_csv, index=False)

print("Saved:", args.out_csv)
print(df.head(15).to_string(index=False))
