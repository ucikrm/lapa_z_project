"""
GVL-style scorer (runs in the `.venv_vlm` env), used as a secondary cross-check on
multi-frame real sequences. The VLM is shown SHUFFLED frames and asked to emit a
0-100 completion percentage per frame (Ma et al., GVL). We parse the percentages,
map them back to chronological order, and the caller computes Value-Order Correlation.
"""
import re
import random

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .prompts import GVL_SYSTEM, GVL_FRAME_REQUEST


class GVLScorer:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda", dtype=torch.bfloat16,
                 max_new_tokens=512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()

    @torch.no_grad()
    def predict_values(self, image_paths, instruction, seed=0):
        """Returns a list of predicted completion fractions in the ORIGINAL frame order
        (NaN where parsing failed)."""
        n = len(image_paths)
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        images = [Image.open(image_paths[i]).convert("RGB") for i in order]

        content = [{"type": "text", "text": GVL_SYSTEM.format(instruction=instruction)}]
        for k, im in enumerate(images):
            content.append({"type": "text", "text": f"Frame {k}:"})
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": GVL_FRAME_REQUEST})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out = gen[0, inputs["input_ids"].shape[1]:]
        decoded = self.processor.tokenizer.decode(out, skip_special_tokens=True)

        shuffled_vals = [float("nan")] * n
        for m in re.finditer(r"Frame\s+(\d+)\s*:\s*(\d+(?:\.\d+)?)\s*%", decoded):
            k = int(m.group(1))
            v = float(m.group(2))
            if 0 <= k < n:
                shuffled_vals[k] = v / 100.0

        chrono = [float("nan")] * n
        for k, orig_i in enumerate(order):
            chrono[orig_i] = shuffled_vals[k]
        return chrono
