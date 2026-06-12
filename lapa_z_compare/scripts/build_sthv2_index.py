import argparse
import json
import random
from pathlib import Path
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_root", type=str, default="/data/mohantyk/lapa_z_data/sthv2_videos/20bn-something-something-v2")
    parser.add_argument("--labels_root", type=str, default="/data/mohantyk/lapa_z_data/labels/labels")
    parser.add_argument("--out_dir", type=str, default="data_cache/sthv2_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_keywords", type=str, default="pushing,pulling,moving,lifting,putting,picking,dropping,throwing,turning,opening,closing,tearing,folding,unfolding,separating,covering,uncovering,pretending,squeezing")
    parser.add_argument("--num_train", type=int, default=50000)
    parser.add_argument("--num_val", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=5000)
    args = parser.parse_args()

    random.seed(args.seed)

    videos_root = Path(args.videos_root)
    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labels...")
    all_items = []
    
    # We load both train.json and validation.json to have a larger pool of labeled videos.
    for filename in ["train.json", "validation.json"]:
        file_path = labels_root / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                items = json.load(f)
                all_items.extend(items)
                print(f"Loaded {len(items)} items from {filename}")
        else:
            print(f"Warning: {file_path} not found.")

    # Remove duplicates if any
    unique_items = {}
    for item in all_items:
        unique_items[item["id"]] = item
    items = list(unique_items.values())
    print(f"Total unique items: {len(items)}")

    # Filter by keywords if provided
    keywords = [k.strip().lower() for k in args.include_keywords.split(",") if k.strip()]
    if keywords:
        filtered = []
        for item in items:
            label_text = str(item.get("label", item.get("template", ""))).lower()
            if any(k in label_text for k in keywords):
                filtered.append(item)
        items = filtered
        print(f"After keyword filter: {len(items)}")

    # Check if files exist
    valid_items = []
    print("Verifying video file existence...")
    for idx, item in enumerate(items):
        video_id = item["id"]
        video_path = videos_root / f"{video_id}.webm"
        if video_path.exists():
            valid_items.append(item)
        if (idx + 1) % 50000 == 0:
            print(f"Verified {idx + 1}/{len(items)} items")

    print(f"Total valid items with files: {len(valid_items)}")
    
    # Shuffle for split
    random.shuffle(valid_items)

    total_needed = args.num_train + args.num_val + args.num_test
    if len(valid_items) < total_needed:
        print(f"Warning: Not enough valid items ({len(valid_items)}) to satisfy the requested split sizes (Total needed: {total_needed}). Scaling down splits proportionally.")
        ratio_train = args.num_train / total_needed
        ratio_val = args.num_val / total_needed
        args.num_train = int(len(valid_items) * ratio_train)
        args.num_val = int(len(valid_items) * ratio_val)
        args.num_test = len(valid_items) - args.num_train - args.num_val
        print(f"Scaled splits to: Train={args.num_train}, Val={args.num_val}, Test={args.num_test}")

    train_split = valid_items[:args.num_train]
    val_split = valid_items[args.num_train:args.num_train+args.num_val]
    test_split = valid_items[args.num_train+args.num_val:args.num_train+args.num_val+args.num_test]

    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    for name, split_items in splits.items():
        csv_path = out_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "video_path", "label", "template"])
            for item in split_items:
                video_id = item["id"]
                video_path = str(videos_root / f"{video_id}.webm")
                label = item.get("label", "")
                template = item.get("template", "")
                writer.writerow([video_id, video_path, label, template])
        print(f"Saved {len(split_items)} rows to {csv_path}")

    # Generate a debug/Stage A split as well (500 train, 100 val, 100 test)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    debug_splits = {
        "train": train_split[:500],
        "val": val_split[:100],
        "test": test_split[:100]
    }
    for name, split_items in debug_splits.items():
        csv_path = debug_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "video_path", "label", "template"])
            for item in split_items:
                video_id = item["id"]
                video_path = str(videos_root / f"{video_id}.webm")
                label = item.get("label", "")
                template = item.get("template", "")
                writer.writerow([video_id, video_path, label, template])
        print(f"Saved debug {len(split_items)} rows to {csv_path}")

    print("Index build finished successfully!")

if __name__ == "__main__":
    main()
