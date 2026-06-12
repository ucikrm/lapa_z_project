import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.io as io
import pandas as pd
import numpy as np
import cv2
import random
from pathlib import Path

class SthV2PairDataset(Dataset):
    def __init__(self, csv_path, image_size=224, gap=12, split="train", compute_flow=False):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.gap = gap
        self.split = split
        self.compute_flow = compute_flow

        # Set up torchvision transforms
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            # Normalization will be handled after conversion to float tensor
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        video_id = str(row["video_id"])
        label = str(row["label"])
        template = str(row["template"])

        # Try loading with OpenCV first for a 10x speedup
        opencv_success = False
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                F_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if F_len > 0:
                    # Sampling logic
                    if F_len <= self.gap:
                        t = 0
                        t_next = F_len - 1
                    else:
                        if self.split == "train":
                            t = random.randint(0, F_len - self.gap - 1)
                            t_next = t + self.gap
                        else:
                            t = (F_len - self.gap) // 2
                            t_next = t + self.gap
                    
                    frames = {}
                    target_indices = {t, t_next}
                    idx = 0
                    while len(frames) < len(target_indices):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx in target_indices:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames[idx] = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                        idx += 1
                    cap.release()
                    
                    if t in frames and t_next in frames:
                        x1 = frames[t]
                        x2 = frames[t_next]
                        opencv_success = True
        except Exception:
            pass

        if not opencv_success:
            try:
                # Fallback to PyAV backend if OpenCV fails
                video, _, _ = io.read_video(video_path, pts_unit="sec", output_format="THWC")
                F_len = video.shape[0]
                
                if F_len <= self.gap:
                    t = 0
                    t_next = F_len - 1
                else:
                    if self.split == "train":
                        t = random.randint(0, F_len - self.gap - 1)
                        t_next = t + self.gap
                    else:
                        t = (F_len - self.gap) // 2
                        t_next = t + self.gap

                frame1 = video[t]
                frame2 = video[t_next]
                x1 = frame1.permute(2, 0, 1).float() / 255.0
                x2 = frame2.permute(2, 0, 1).float() / 255.0
            except Exception as e:
                # Fallback if video loading completely fails: return a dummy batch
                print(f"Error loading {video_path}: {e}")
                dummy_frame = torch.zeros((3, self.image_size, self.image_size))
                dummy_flow = torch.zeros((2, 14, 14)) if self.compute_flow else torch.zeros(1)
                return {
                    "x1": dummy_frame,
                    "x2": dummy_frame,
                    "video_id": video_id,
                    "label": label,
                    "template": template,
                    "t": 0,
                    "num_frames": 0,
                    "flow": dummy_flow
                }

        # Apply spatial resizing
        x1 = self.transform(x1)
        x2 = self.transform(x2)

        # Optional optical flow computation using Farneback algorithm
        flow_tensor = torch.zeros(1)
        if self.compute_flow:
            try:
                # Convert back to numpy for opencv
                img1_np = (x1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                img2_np = (x2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                
                gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
                
                # Compute Farneback flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                # Downsample flow to [14, 14, 2] to save space and match model resolution
                flow_resized = cv2.resize(flow, (14, 14), interpolation=cv2.INTER_AREA)
                # Convert to channel first [2, 14, 14]
                flow_tensor = torch.from_numpy(flow_resized).permute(2, 0, 1).float()
            except Exception as e:
                # Fallback on flow error
                flow_tensor = torch.zeros((2, 14, 14))

        return {
            "x1": x1,
            "x2": x2,
            "video_id": video_id,
            "label": label,
            "template": template,
            "t": t,
            "num_frames": F_len,
            "flow": flow_tensor
        }
