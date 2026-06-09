from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("--z_path", type=str, required=True)
parser.add_argument("--meta_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--max_items", type=int, default=5000)
parser.add_argument("--num_queries", type=int, default=10)
parser.add_argument("--size", type=int, default=128)
args = parser.parse_args()

z_path = Path(args.z_path)
meta_path = Path(args.meta_path)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

Z = np.load(z_path)
meta = pd.read_csv(meta_path)

n = min(len(Z), len(meta), args.max_items)
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
    dists = dists[0]

    tiles = []
    for rank, (idx, dist) in enumerate(zip(inds, dists)):
        tile = load_pair(meta.iloc[idx], size=args.size)
        draw = ImageDraw.Draw(tile)
        draw.text((5, 5), f"rank {rank} idx {idx} d={dist:.3f}", fill=(255, 0, 0))
        tiles.append(tile)

    w, h = tiles[0].size
    grid = Image.new("RGB", (w, h * len(tiles)), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, (0, i * h))

    grid.save(out_dir / f"query_{qid:05d}.jpg")

print("Saved nearest-neighbor grids to", out_dir)
