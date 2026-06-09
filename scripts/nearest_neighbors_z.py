from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, choices=["vq", "continuous"], default="continuous")
parser.add_argument("--max_items", type=int, default=5000)
parser.add_argument("--num_queries", type=int, default=10)
args = parser.parse_args()

root = Path("..")
fig_dir = root / "figures" / "nn"
fig_dir.mkdir(parents=True, exist_ok=True)

if args.method == "continuous":
    Z = np.load(root / "z_exports/continuous_z/z_continuous.npy")
    meta = pd.read_csv(root / "z_exports/continuous_z/metadata.csv")
else:
    Z = np.load(root / "z_exports/vq_lapa/z_quantized.npy")
    meta = pd.read_csv(root / "z_exports/vq_lapa/metadata.csv")

n = min(len(Z), args.max_items)
Z = Z[:n]
meta = meta.iloc[:n].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
nn.fit(Z)

query_ids = np.linspace(0, n - 1, num=min(args.num_queries, n), dtype=int)

def load_pair(row, size=128):
    x1 = Image.open(row["x1_path"]).convert("RGB").resize((size, size))
    x2 = Image.open(row["x2_path"]).convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size * 2, size), "white")
    canvas.paste(x1, (0, 0))
    canvas.paste(x2, (size, 0))
    return canvas

for qid in query_ids:
    dists, inds = nn.kneighbors(Z[qid:qid+1])
    inds = inds[0]

    tiles = []
    for rank, idx in enumerate(inds):
        tile = load_pair(meta.iloc[idx])
        draw = ImageDraw.Draw(tile)
        draw.text((5, 5), f"rank {rank} idx {idx}", fill=(255, 0, 0))
        tiles.append(tile)

    w, h = tiles[0].size
    grid = Image.new("RGB", (w, h * len(tiles)), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, (0, i * h))

    grid.save(fig_dir / f"{args.method}_query_{qid:05d}.jpg")

print("Saved nearest-neighbor grids to", fig_dir)
