from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FramePairDataset(Dataset):
    def __init__(self, root, image_size=128, gap=1):
        self.root = Path(root)
        self.image_size = image_size
        self.gap = gap

        self.pairs = []
        traj_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])

        for traj in traj_dirs:
            frames = sorted(
                list(traj.glob("*.jpg")) +
                list(traj.glob("*.png")) +
                list(traj.glob("*.jpeg"))
            )
            for i in range(len(frames) - gap):
                self.pairs.append((frames[i], frames[i + gap]))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No frame pairs found in {root}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1_path, x2_path = self.pairs[idx]

        x1 = Image.open(x1_path).convert("RGB")
        x2 = Image.open(x2_path).convert("RGB")

        x1 = self.transform(x1)
        x2 = self.transform(x2)

        return {
            "x1": x1,
            "x2": x2,
            "x1_path": str(x1_path),
            "x2_path": str(x2_path),
        }
