from pathlib import Path
import argparse
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--rank_csv", type=str, required=True)
parser.add_argument("--subset_meta", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--top_k", type=int, default=5)
args = parser.parse_args()

rank_df = pd.read_csv(args.rank_csv).head(args.top_k)
meta = pd.read_csv(args.subset_meta)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for i, row in rank_df.iterrows():
    traj = row["traj"]
    traj_idx = int(traj.split("_")[-1])

    m = meta[meta["traj_idx"] == traj_idx]
    if len(m) == 0:
        print("Missing metadata for", traj)
        continue

    source_video = Path(m.iloc[0]["source_video"])
    label = str(m.iloc[0]["label"]).replace("/", "_").replace(" ", "_")[:80]

    dst = out_dir / f"rank_{i+1:02d}_{traj}_{label}{source_video.suffix}"
    shutil.copy2(source_video, dst)

    print("Copied:", dst)
