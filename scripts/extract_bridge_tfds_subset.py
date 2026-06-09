from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


parser = argparse.ArgumentParser()
parser.add_argument("--out_root", type=str, default="data/bridge_subset_20")
parser.add_argument("--tfds_data_dir", type=str, default="/mnt/d/tfds_data")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--max_frames_per_episode", type=int, default=40)
parser.add_argument("--image_key", type=str, default="")
args = parser.parse_args()

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

print("Loading TFDS builder...")
builder = tfds.builder("bridge", data_dir=args.tfds_data_dir)

print("Downloading/preparing if needed. This may take a while the first time...")
builder.download_and_prepare()

print("Opening split:", args.split)
ds = builder.as_dataset(split=args.split)

num_saved = 0
used_key = None

for ep_idx, episode in enumerate(tfds.as_numpy(ds.take(args.max_episodes))):
    traj_dir = out_root / f"traj_{ep_idx:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    steps = episode["steps"]
    saved_in_ep = 0

    for step in steps:
        if saved_in_ep >= args.max_frames_per_episode:
            break

        obs = step["observation"]

        if args.image_key:
            arr = obs[args.image_key]
            used_key = args.image_key
        else:
            arr = None
            for key, candidate in obs.items():
                if isinstance(candidate, np.ndarray) and candidate.ndim == 3 and candidate.shape[-1] in (1, 3, 4):
                    arr = candidate
                    used_key = key
                    break

            if arr is None:
                raise KeyError(f"No image-like observation found. Observation keys: {list(obs.keys())}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        img = Image.fromarray(arr)
        img.save(traj_dir / f"{saved_in_ep:06d}.jpg", quality=95)

        saved_in_ep += 1
        num_saved += 1

    if ep_idx == 0:
        print("Used image key:", used_key)

    print(f"Extracted episode {ep_idx + 1}/{args.max_episodes}; total frames={num_saved}")

print("Done.")
print("Output:", out_root)
print("Total frames:", num_saved)
