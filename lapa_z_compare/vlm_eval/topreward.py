"""
TOPReward-style scorer (runs in the `.venv_vlm` env).

Instead of asking the VLM to emit a numeric progress value (GVL), we read the
probability mass the model assigns to the affirmative completion token ("True")
for a binary "did this make progress on the task?" query (Chen et al., TOPReward).

We use Qwen3-VL-8B-Instruct. The paper notes a raw (no chat template) formulation
is best for white-box logit reading; with an instruct model we use the chat template
with a generation prompt and read the FIRST assistant-token distribution, which is the
practical open-weights adaptation. We report both the raw log p(True) and the
normalized p(True) / (p(True) + p(False)).
"""
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .prompts import TOPREWARD_QUERY


class TOPRewardScorer:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()
        tok = self.processor.tokenizer
        # Candidate token ids whose surface form is True/False (with/without leading space).
        self.true_ids = self._variant_ids(tok, ["True", " True"])
        self.false_ids = self._variant_ids(tok, ["False", " False"])

    @staticmethod
    def _variant_ids(tok, words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) >= 1:
                ids.add(enc[0])
        return sorted(ids)

    @torch.no_grad()
    def _first_token_probs(self, image_paths, query_text):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": query_text})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_true = probs[self.true_ids].sum().item()
        p_false = probs[self.false_ids].sum().item()
        return p_true, p_false

    def reward(self, image_paths, instruction):
        """Returns (log p(True), normalized p(True)) for the [frames | instruction] query."""
        q = TOPREWARD_QUERY.format(instruction=instruction)
        p_true, p_false = self._first_token_probs(image_paths, q)
        eps = 1e-12
        log_p_true = float(torch.log(torch.tensor(p_true + eps)))
        p_true_norm = p_true / (p_true + p_false + eps)
        return log_p_true, p_true_norm
