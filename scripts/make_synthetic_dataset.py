from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

out_root = Path("data/tiny_debug")
out_root.mkdir(parents=True, exist_ok=True)

num_traj = 200
frames_per_traj = 30
W = H = 128

random.seed(0)

for tid in range(num_traj):
    traj_dir = out_root / f"traj_{tid:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    x = random.randint(20, 108)
    y = random.randint(20, 108)

    angle = random.random() * 2 * math.pi
    speed = random.uniform(1.0, 3.5)
    dx = speed * math.cos(angle)
    dy = speed * math.sin(angle)

    radius = random.randint(7, 13)
    color = tuple(random.randint(60, 240) for _ in range(3))

    for t in range(frames_per_traj):
        img = Image.new("RGB", (W, H), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        # add static distractors
        for k in range(5):
            rx = (17 * k + 13 * tid) % W
            ry = (31 * k + 7 * tid) % H
            draw.rectangle([rx, ry, rx + 4, ry + 4], fill=(60, 60, 60))

        cx = int(x + dx * t)
        cy = int(y + dy * t)

        # bounce effect
        cx = max(radius, min(W - radius, cx))
        cy = max(radius, min(H - radius, cy))

        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color
        )

        img.save(traj_dir / f"{t:06d}.jpg", quality=95)

print(f"Created {num_traj} trajectories at {out_root}")
print(f"Total frames: {num_traj * frames_per_traj}")
print(f"Approx frame pairs: {num_traj * (frames_per_traj - 1)}")
