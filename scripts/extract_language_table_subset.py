from pathlib import Path
import argparse
import csv
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="gs://gresearch/robotics/language_table_sim/0.0.1/")
parser.add_argument("--out_root", type=str, default="data/language_table_subset_20")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--max_frames_per_episode", type=int, default=40)
args = parser.parse_args()

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

print("Loading builder from:", args.dataset_path)
builder = tfds.builder_from_directory(args.dataset_path)

print("Opening split:", args.split)
ds = builder.as_dataset(split=args.split)

all_actions = []
metadata_rows = []

num_frames = 0
num_pairs = 0

for ep_idx, episode in enumerate(tfds.as_numpy(ds.take(args.max_episodes))):
    traj_dir = out_root / f"traj_{ep_idx:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    actions = []

    # episode["steps"] is iterable, not a dict.
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx >= args.max_frames_per_episode:
            break

        rgb = step["observation"]["rgb"]
        action = step["action"]

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        Image.fromarray(rgb).save(traj_dir / f"{step_idx:06d}.jpg", quality=95)

        frames.append(str(traj_dir / f"{step_idx:06d}.jpg"))
        actions.append(action)
        num_frames += 1

    # Action at t corresponds to transition frame t -> t+1.
    for t in range(len(frames) - 1):
        all_actions.append(actions[t])
        metadata_rows.append([
            num_pairs,
            ep_idx,
            t,
            frames[t],
            frames[t + 1],
        ])
        num_pairs += 1

    print(f"episode {ep_idx + 1}/{args.max_episodes}: frames={len(frames)}, total_pairs={num_pairs}")

all_actions = np.asarray(all_actions, dtype=np.float32)
np.save(out_root / "actions.npy", all_actions)

with open(out_root / "action_metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pair_idx", "episode_idx", "t", "x1_path", "x2_path"])
    writer.writerows(metadata_rows)

print("Done.")
print("Output:", out_root)
print("Total frames:", num_frames)
print("Total pairs:", num_pairs)
print("Actions shape:", all_actions.shape)
