from pathlib import Path
import argparse
import pandas as pd
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--rank_csv", type=str, required=True)
parser.add_argument("--subset_meta", type=str, required=True)
parser.add_argument("--frames_root", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--fps", type=int, default=5)
args = parser.parse_args()

rank_df = pd.read_csv(args.rank_csv).head(args.top_k)
meta = pd.read_csv(args.subset_meta)

frames_root = Path(args.frames_root)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for rank_idx, row in enumerate(rank_df.itertuples(), start=1):
    traj = row.traj
    traj_idx = int(traj.split("_")[-1])

    m = meta[meta["traj_idx"] == traj_idx]
    label = str(m.iloc[0]["label"]) if len(m) else str(getattr(row, "traj_label", ""))

    traj_dir = frames_root / traj
    if not traj_dir.exists():
        print("Missing traj dir:", traj_dir)
        continue

    label_safe = label.replace(":", "").replace("'", "").replace(",", "").replace("/", "_")
    out_path = out_dir / f"rank_{rank_idx:02d}_{traj}.mp4"

    title = (
        f"Rank {rank_idx}: {label_safe} | "
        f"VQ codes={int(row.traj_vq_used_codes)}, "
        f"VQ rank={row.traj_vq_effective_rank:.2f}, "
        f"Cont rank={row.traj_cont_effective_rank:.2f}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(traj_dir / "%06d.jpg"),
        "-vf",
        f"scale=640:-2,"
        f"drawtext=text='{title}':x=10:y=10:fontsize=20:"
        f"fontcolor=white:box=1:boxcolor=black@0.6",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    subprocess.run(cmd, check=True)
    print("Saved:", out_path)
