# Adapted from https://github.com/apple/ml-ane-transformers

import json
import os

import torch
import uform
from torch import nn
from uform import models

from .constants import EPS


class LayerNorm_ANE(nn.Module):
    def __init__(
        self,
        num_channels: int,
        clip_mag=None,
        eps: float = EPS,
        elementwise_affine=True,
    ):
        super().__init__()

        self.expected_rank = len("BC1S")

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

        self._register_load_state_dict_pre_hook(
            self.correct_for_bias_scale_order_inversion
        )

    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        input_rank = len(inputs.size())

        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)) * self.weight.view(
                1, self.num_channels, 1, 1
            )

        return out

    @staticmethod
    def correct_for_bias_scale_order_inversion(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict[prefix + "bias"] = (
            state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
        )
        return state_dict


class MLP_ANE(models.MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(
            self,
            "hidden_layer",
            nn.Conv2d(self.dim, self.dim * self.dim_expand_factor, 1),
        )

        setattr(
            self,
            "output_layer",
            nn.Conv2d(self.dim * self.dim_expand_factor, self.dim, 1),
        )

        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    @staticmethod
    def linear_to_conv2d_map(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for k in state_dict:
            is_proj = any(
                [
                    "hidden_layer.weight" in k,
                    "output_layer.weight" in k,
                ]
            )
            if is_proj and len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


class Attention_ANE(models.Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(self, "query", nn.Conv2d(self.dim, self.dim, 1))
        setattr(self, "key", nn.Conv2d(self.dim, self.dim, 1))
        setattr(self, "value", nn.Conv2d(self.dim, self.dim, 1))
        setattr(self, "out", nn.Conv2d(self.dim, self.dim, 1))

        self.dropout = (
            nn.Dropout(self.dropout_prob) if self.dropout_prob > 0.0 else nn.Identity()
        )

        self.apply(self._reset_parameters)
        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def _attention_fn(self, q, k, v, qk_mask, k_mask, return_weights):
        mh_q = q.split(
            self.dim // self.num_heads, dim=1
        )  # n_head * (batch_size, d_qk/n_head, 1, tgt_seq_len)

        mh_k = k.transpose(1, 3).split(
            self.dim // self.num_heads, dim=3
        )  # n_head * (batch_size, src_seq_len, 1, d_qk/n_head)
        mh_v = v.split(
            self.dim // self.num_heads, dim=1
        )  # n_head * (batch_size, d_v/n_head, 1, src_seq_len)

        attn_weights = [
            torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * self.scale
            for qi, ki in zip(mh_q, mh_k)
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        if qk_mask is not None:
            for head_idx in range(self.num_heads):
                attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
        if k_mask is not None:
            for head_idx in range(self.num_heads):
                attn_weights[head_idx] = attn_weights[head_idx] + k_mask

        attn_weights = [
            aw.softmax(dim=1) for aw in attn_weights
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)
        mh_w = [
            self.dropout(aw) for aw in attn_weights
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        attn = [
            torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(mh_w, mh_v)
        ]  # n_head * (batch_size, d_v/n_head, 1, tgt_seq_len)
        attn = torch.cat(attn, dim=1)  # (batch_size, d_v, 1, tgt_seq_len)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def _forward_impl(
        self,
        q,
        k,
        v,
        qpos=None,
        kpos=None,
        vpos=None,
        qk_mask=None,
        k_mask=None,
        return_weights=True,
    ):
        assert len(q.size()) == 4 and len(k.size()) == 4 and len(v.size()) == 4
        b, ct, ht, wt = q.size()
        b, cs, hs, ws = k.size()

        tgt_seq_len = ht * wt
        src_seq_len = hs * ws

        if qpos is not None:
            q = q + qpos
        if kpos is not None:
            k = k + kpos
        if vpos is not None:
            v = v + kpos

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        expected_qk_mask_shape = [b, src_seq_len, 1, tgt_seq_len]
        if qk_mask is not None:
            if qk_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`qk_mask` must be of type torch.float32, received {qk_mask.dtype}"
                )
            if list(qk_mask.size()) != expected_qk_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `qk_mask` (Expected {expected_qk_mask_shape}, got {list(qk_mask.size())}"
                )

        expected_k_mask_shape = [b, src_seq_len, 1, 1]
        if k_mask is not None:
            if k_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`k_mask` must be of type torch.float32, received {k_mask.dtype}"
                )
            if list(k_mask.size()) != expected_k_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `k_mask` (Expected {expected_k_mask_shape}, got {list(k_mask.size())}"
                )

        attn, attn_weights = self._attention_fn(
            q, k, v, qk_mask, k_mask, return_weights
        )

        attn = attn.contiguous().view(b, self.dim, ht, wt)
        attn = self.out(attn)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def forward(
        self,
        x: torch.Tensor,  # (B, S, n*E)
        attn_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        assert not is_causal, "`is_causal` not yet supported"
        # TODO: Add support for the causal mode.
        # if is_causal:
        #     assert attn_mask is None
        #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        #     attn_bias.to(query.dtype)

        context = context if context is not None else x
        attn, _ = self._forward_impl(x, context, context, k_mask=attn_mask)
        return attn

    @staticmethod
    def linear_to_conv2d_map(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for k in state_dict:
            is_proj = any(
                [
                    "query.weight" in k,
                    "key.weight" in k,
                    "value.weight" in k,
                    "out.weight" in k,
                ]
            )
            if is_proj and len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


class TextEncoderBlock_ANE(models.TextEncoderBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(self, "norm_attn", LayerNorm_ANE(self.dim, eps=EPS))
        setattr(
            self,
            "attention",
            Attention_ANE(self.dim, self.num_heads, self.dropout_prob),
        )

        if self.cross_attention:
            setattr(self, "norm_crossattn", LayerNorm_ANE(self.dim, eps=EPS))
            setattr(
                self,
                "crossattn",
                Attention_ANE(self.dim, self.num_heads, self.dropout_prob),
            )

        setattr(self, "norm_mlp", LayerNorm_ANE(self.dim, eps=EPS))
        setattr(self, "mlp", MLP_ANE(self.dim))


class TextEncoder_ANE(models.TextEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(self, "layer_norm", LayerNorm_ANE(self.dim, eps=EPS))
        setattr(
            self,
            "blocks",
            nn.ModuleList(
                [
                    TextEncoderBlock_ANE(
                        self.dim,
                        self.num_heads,
                        self.dropout_prob,
                        layer_id in self.multimodal_layers_ids,
                    )
                    for layer_id in range(self.num_layers)
                ]
            ),
        )

        setattr(
            self,
            "embedding_projection",
            nn.Conv2d(self.dim, self.embedding_dim, 1, bias=False),
        )
        setattr(
            self,
            "matching_head",
            nn.Conv2d(self.dim, 1 if self.head_one_neuron else 2, 1),
        )

        if self.context_dim != self.dim:
            setattr(
                self,
                "context_projection",
                nn.Conv2d(self.context_dim, self.dim, 1, bias=False),
            )

        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    def embed_text(self, x: torch.Tensor) -> torch.Tensor:
        positional_embedding = self.position_embeddings(self.get_position_ids(x))
        x = self.word_embeddings(x) + positional_embedding
        x = x.transpose(-1, -2).unsqueeze(2)
        return self.dropout(self.layer_norm(x))

    def get_attention_mask(
        self, attn_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        attn_mask = 1.0 - attn_mask.to(dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 1.0, torch.finfo(dtype).min)
        return attn_mask.unsqueeze(-1).unsqueeze(-1)

    def pool_features(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        return (x * attn_mask).sum(dim=-1, keepdim=True) / attn_mask.sum(
            dim=-1, keepdim=True
        )

    @staticmethod
    def linear_to_conv2d_map(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for k in state_dict:
            is_proj = any(
                [
                    "embedding_projection.weight" in k,
                    "matching_head.weight" in k,
                    "context_projection.weight" in k,
                ]
            )
            if is_proj and len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


class LayerScale_ANE(models.LayerScale):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(
            self, "gamma", nn.Parameter(self.init_values * torch.ones(self.dim, 1, 1))
        )

        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    @staticmethod
    def linear_to_conv2d_map(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for k in state_dict:
            if "gamma" in k:
                state_dict[k] = state_dict[k][:, None, None]


class VisualEncoderBlock_ANE(models.VisualEncoderBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setattr(
            self,
            "norm1",
            LayerNorm_ANE(self.dim, eps=EPS),
        )

        setattr(
            self,
            "attn",
            Attention_ANE(dim=self.dim, num_heads=self.num_heads),
        )

        setattr(
            self,
            "ls1",
            LayerScale_ANE(self.dim),
        )

        setattr(
            self,
            "norm2",
            LayerNorm_ANE(self.dim, eps=EPS),
        )

        setattr(
            self,
            "mlp",
            MLP_ANE(dim=self.dim),
        )

        setattr(
            self,
            "ls2",
            LayerScale_ANE(self.dim),
        )


class VisualEncoder_ANE(models.VisualEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        seq_len = (self.image_size // self.patch_size) ** 2

        setattr(
            self,
            "pos_embed",
            nn.Parameter(torch.randn(1, self.dim, 1, seq_len) * 0.02),
        )

        setattr(
            self,
            "cls_token",
            nn.Parameter(torch.zeros(1, self.dim, 1, 1)),
        )

        setattr(
            self,
            "blocks",
            nn.Sequential(
                *[
                    VisualEncoderBlock_ANE(self.dim, self.num_heads)
                    for _ in range(self.num_layers)
                ]
            ),
        )

        setattr(
            self,
            "norm",
            LayerNorm_ANE(self.dim, eps=EPS),
        )

        setattr(
            self,
            "embedding_projection",
            nn.Conv2d(self.dim, self.embedding_dim, 1, bias=False),
        )

        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(start_dim=2).unsqueeze(2)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token, x), dim=-1)
        x = self.blocks(x)
        return self.norm(x)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            x = x[:, :, :, 0:1]
        else:
            x = x.mean(dim=-1, keepdim=True)

        return self.embedding_projection(x)

    @staticmethod
    def linear_to_conv2d_map(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for k in state_dict:
            is_proj = "embedding_projection.weight" in k
            if is_proj and len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]
            elif "pos_embed" in k:
                state_dict[k] = state_dict[k].transpose(-1, -2).unsqueeze(2)
            elif "cls_token" in k:
                state_dict[k] = state_dict[k].transpose(-1, -2).unsqueeze(-1)


class VLM_ANE(models.VLM):
    def __init__(self, config: dict, tokenizer_path: os.PathLike):
        super().__init__(config=config, tokenizer_path=tokenizer_path)

        setattr(self, "text_encoder", TextEncoder_ANE(**config["text_encoder"]))
        setattr(self, "image_encoder", VisualEncoder_ANE(**config["image_encoder"]))


def get_ane_model(model_name: str, token: str | None = None) -> VLM_ANE:
    config_path, state, tokenizer_path = uform.get_checkpoint(model_name, token)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = VLM_ANE(config, tokenizer_path)
    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model
