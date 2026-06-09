from pathlib import Path
import argparse
import json
import random
import csv
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--videos_dir", type=str, required=True)
parser.add_argument("--labels_json", type=str, required=True)
parser.add_argument("--out_root", type=str, default="data/sthv2_subset_100")
parser.add_argument("--num_videos", type=int, default=100)
parser.add_argument("--fps", type=int, default=5)
parser.add_argument("--max_frames", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--include_keywords", type=str, default="")
args = parser.parse_args()

random.seed(args.seed)

videos_dir = Path(args.videos_dir)
labels_json = Path(args.labels_json)
out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

with open(labels_json, "r") as f:
    items = json.load(f)

print("Loaded label items:", len(items))

keywords = [k.strip().lower() for k in args.include_keywords.split(",") if k.strip()]

if keywords:
    filtered = []
    for item in items:
        label_text = str(item.get("label", item.get("template", ""))).lower()
        if any(k in label_text for k in keywords):
            filtered.append(item)
    items = filtered
    print("After keyword filter:", len(items))

random.shuffle(items)

selected = []
for item in items:
    vid = str(item["id"])
    label = item.get("label", item.get("template", ""))

    candidates = []
    for ext in [".webm", ".mp4", ".mkv", ".avi"]:
        candidates.extend(videos_dir.rglob(vid + ext))

    if candidates:
        selected.append((vid, label, candidates[0]))

    if len(selected) >= args.num_videos:
        break

print("Selected videos:", len(selected))

metadata_rows = []
usable = 0

for i, (vid, label, video_path) in enumerate(selected):
    traj_dir = out_root / f"traj_{i:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={args.fps},scale=224:-1",
        "-frames:v", str(args.max_frames),
        str(traj_dir / "%06d.jpg"),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    num_frames = len(list(traj_dir.glob("*.jpg")))

    if num_frames < 2:
        for p in traj_dir.glob("*"):
            p.unlink()
        traj_dir.rmdir()
        continue

    metadata_rows.append([usable, vid, label, str(video_path), num_frames])
    usable += 1

    if usable % 25 == 0:
        print(f"Usable trajectories: {usable}/{args.num_videos}")

with open(out_root / "metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["traj_idx", "video_id", "label", "source_video", "num_frames"])
    writer.writerows(metadata_rows)

print("Done.")
print("Output:", out_root)
print("Usable trajectories:", usable)
print("Total frames:", sum(row[-1] for row in metadata_rows))
