import json
import os

import torch
import uform
from uform import models


class VisualEncoder_FP16(models.VisualEncoder):
    pass


class TextEncoder_FP16(models.TextEncoder):
    def get_attention_mask(
        self, attn_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        attn_mask = 1.0 - attn_mask.to(dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 1.0, torch.finfo(dtype).min)
        return attn_mask.unsqueeze(1).expand(-1, attn_mask.shape[1], -1).unsqueeze(1)


class VLM_FP16(models.VLM):
    def __init__(self, config: dict, tokenizer_path: os.PathLike):
        super().__init__(config=config, tokenizer_path=tokenizer_path)

        setattr(self, "text_encoder", TextEncoder_FP16(**config["text_encoder"]))
        setattr(self, "image_encoder", VisualEncoder_FP16(**config["image_encoder"]))


def get_fp16_model(model_name: str, token: str | None = None) -> VLM_FP16:
    config_path, state, tokenizer_path = uform.get_checkpoint(model_name, token)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = VLM_FP16(config, tokenizer_path)
    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model
