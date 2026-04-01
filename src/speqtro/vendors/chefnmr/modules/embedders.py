# Vendored from chefnmr/src/model/modules/embedders.py — inference-only.
# NMR spectra tokenization + transformer embedding.

import torch
from torch import nn
from einops import rearrange
from typing import Optional
import numpy as np
import math

from speqtro.vendors.chefnmr.modules.utils import get_1d_sincos_pos_embed_from_grid


# -----------------------------------------------------------------------------
# Tokenizers
# -----------------------------------------------------------------------------
class SpectraTokenizerPatch1D(nn.Module):
    """1-D Patch Tokenizer: splits a 1D sequence into patches and projects them."""

    def __init__(self, patch_size: int, stride: int, hidden_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(patch_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        x = self.proj(patches)
        return x

    @staticmethod
    def num_tokens(input_size: int, patch_size: int, stride: int) -> int:
        return (input_size - patch_size) // stride + 1


class SpectraTokenizerConv1D(nn.Module):
    """1-D Convolutional Tokenizer: Conv1d -> ReLU -> MaxPool1d blocks."""

    def __init__(self, input_size, hidden_dim, pool_sizes, kernel_sizes, out_channels):
        super().__init__()

        num_layers = len(pool_sizes)
        assert num_layers > 0
        assert len(kernel_sizes) == num_layers
        assert len(out_channels) == num_layers

        self.conv_blocks = nn.ModuleList()
        in_channel = 1
        for i in range(num_layers):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.MaxPool1d(pool_sizes[i]),
            )
            self.conv_blocks.append(block)
            in_channel = out_channels[i]

        self.linear_after_conv = nn.Linear(out_channels[-1], hidden_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        for block in self.conv_blocks:
            x = block(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear_after_conv(x)
        return x

    @staticmethod
    def _calculate_dim_after_conv(L_in, kernel, padding, dilation, stride):
        numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        return math.floor((numerator / stride) + 1)

    @staticmethod
    def _calculate_dim_after_pool(pool_variation, L_in, kernel, padding, dilation, stride):
        if pool_variation == "max":
            numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
            return math.floor((numerator / stride) + 1)
        elif pool_variation == "avg":
            numerator = L_in + (2 * padding) - kernel
            return math.floor((numerator / stride) + 1)

    @staticmethod
    def num_tokens(input_size: int, kernel_sizes: list, pool_sizes: list) -> int:
        L_current = input_size
        for conv_kernel, pool_kernel in zip(kernel_sizes, pool_sizes):
            L_current = SpectraTokenizerConv1D._calculate_dim_after_conv(
                L_in=L_current, kernel=conv_kernel, padding=0, dilation=1, stride=1
            )
            L_current = SpectraTokenizerConv1D._calculate_dim_after_pool(
                pool_variation="max",
                L_in=L_current,
                kernel=pool_kernel,
                padding=0,
                dilation=1,
                stride=pool_kernel,
            )
        return L_current


# -----------------------------------------------------------------------------
# Transformer primitives
# -----------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class EmbedderAttention(nn.Module):
    """Multi-head Self Attention (used inside the NMR spectra embedder)."""

    def __init__(self, dim: int, heads: int, dim_head: Optional[int] = None):
        super().__init__()
        if dim_head is None:
            assert dim % heads == 0
            dim_head = dim // heads
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attn(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout):
        super().__init__()
        mlp_dim = dim * mlp_ratio
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, EmbedderAttention(dim, heads=heads, dim_head=dim_head)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.final_norm(x)


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------
class AttnPoolToken(nn.Module):
    """Attention-based pooling using a learnable CLS token."""

    def __init__(self, dim, out_dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.attn = EmbedderAttention(dim, heads, dim_head)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        cls = self.cls.expand(x.size(0), -1, -1)
        y = torch.cat([cls, x], dim=1)
        y = self.attn(y)[:, 0]
        return self.proj(y)


# -----------------------------------------------------------------------------
# Main embedding module
# -----------------------------------------------------------------------------
class NMRSpectraEmbedder(nn.Module):
    """Embedder for NMR Spectra (1H and/or 13C)."""

    def __init__(
        self,
        *,
        use_hnmr: bool = True,
        use_cnmr: bool = False,
        hnmr_dim: int = 10000,
        cnmr_dim: int = 10000,
        hidden_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1,
        pooling: str = "flatten",
        tokenizer_args: dict = None,
        transformer_args: dict = None,
    ):
        super().__init__()
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.hnmr_dim = hnmr_dim
        self.cnmr_dim = cnmr_dim
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        default_tokenizer_args = {
            "h_tokenizer": "conv",
            "c_tokenizer": "conv",
            "h_pool_sizes": [8, 12],
            "h_kernel_sizes": [5, 9],
            "h_out_channels": [64, 128],
            "c_pool_sizes": [8, 12],
            "c_kernel_sizes": [5, 9],
            "c_out_channels": [64, 128],
            "h_patch_size": 256,
            "h_patch_stride": 128,
            "c_patch_size": 256,
            "c_patch_stride": 128,
            "h_mask_token": False,
            "c_mask_token": False,
        }
        tokenizer_args = {**default_tokenizer_args, **(tokenizer_args or {})}

        default_transformer_args = {
            "pos_enc": "learnable",
            "type_enc": True,
            "depth": 4,
            "heads": 8,
            "dim_head": None,
            "mlp_ratio": 4,
        }
        transformer_args = {**default_transformer_args, **(transformer_args or {})}

        self.h_tokenizer = tokenizer_args["h_tokenizer"]
        self.c_tokenizer = tokenizer_args["c_tokenizer"]
        self.use_h_mask_token = tokenizer_args["h_mask_token"]
        self.use_c_mask_token = tokenizer_args["c_mask_token"]

        pos_enc = transformer_args["pos_enc"]
        type_enc = transformer_args["type_enc"]
        depth = transformer_args["depth"]
        heads = transformer_args["heads"]
        dim_head = transformer_args["dim_head"]
        mlp_ratio = transformer_args["mlp_ratio"]

        self.h_embed, self.h_token_num = None, 0
        self.c_embed, self.c_token_num = None, 0

        if use_hnmr:
            self.h_embed, self.h_token_num = self._initialize_tokenizer(
                tokenizer_type=self.h_tokenizer,
                input_dim=hnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args["h_pool_sizes"],
                kernel_sizes=tokenizer_args["h_kernel_sizes"],
                out_channels=tokenizer_args["h_out_channels"],
                patch_size=tokenizer_args["h_patch_size"],
                stride=tokenizer_args["h_patch_stride"],
            )
            if self.use_h_mask_token:
                self.h_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.h_mask_token.weight, std=0.02)
                self.h_token_num += 1

        if use_cnmr:
            self.c_embed, self.c_token_num = self._initialize_tokenizer(
                tokenizer_type=self.c_tokenizer,
                input_dim=cnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args["c_pool_sizes"],
                kernel_sizes=tokenizer_args["c_kernel_sizes"],
                out_channels=tokenizer_args["c_out_channels"],
                patch_size=tokenizer_args["c_patch_size"],
                stride=tokenizer_args["c_patch_stride"],
            )
            if self.use_c_mask_token:
                self.c_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.c_mask_token.weight, std=0.02)
                self.c_token_num += 1

        total_tokens = self.h_token_num + self.c_token_num
        if pos_enc == "sincos":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, total_tokens, hidden_dim), requires_grad=False
            )
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1], np.arange(total_tokens)
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
            self.pos_encode = lambda x: x + self.pos_embed
        elif pos_enc == "learnable":
            self.learnable_pos_embed = nn.Parameter(
                torch.zeros(1, total_tokens, hidden_dim)
            )
            nn.init.normal_(self.learnable_pos_embed, std=0.02)
            self.pos_encode = lambda x: x + self.learnable_pos_embed
        elif pos_enc is None:
            self.pos_encode = nn.Identity()
        else:
            raise ValueError(f"Unknown positional encoding: {pos_enc}")

        if use_hnmr and use_cnmr and type_enc:
            self.type_embedding = nn.Embedding(2, hidden_dim)
            nn.init.normal_(self.type_embedding.weight, std=0.02)
        else:
            self.type_embedding = None

        if depth > 0:
            self.transformer = TransformerEncoder(
                dim=hidden_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            self.transformer = nn.Identity()

        if pooling == "flatten":
            flatten_dim = total_tokens * hidden_dim
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(flatten_dim),
                nn.Dropout(dropout),
                nn.Linear(flatten_dim, output_dim),
            )
        elif pooling == "attn":
            self.head = AttnPoolToken(
                hidden_dim, output_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def _initialize_tokenizer(self, tokenizer_type, input_dim, hidden_dim, **kwargs):
        if tokenizer_type == "conv":
            embed_layer = SpectraTokenizerConv1D(
                input_size=input_dim,
                hidden_dim=hidden_dim,
                pool_sizes=kwargs["pool_sizes"],
                kernel_sizes=kwargs["kernel_sizes"],
                out_channels=kwargs["out_channels"],
            )
            num_tokens = SpectraTokenizerConv1D.num_tokens(
                input_size=input_dim,
                kernel_sizes=kwargs["kernel_sizes"],
                pool_sizes=kwargs["pool_sizes"],
            )
        elif tokenizer_type == "patch":
            embed_layer = SpectraTokenizerPatch1D(
                patch_size=kwargs["patch_size"],
                stride=kwargs["stride"],
                hidden_dim=hidden_dim,
            )
            num_tokens = SpectraTokenizerPatch1D.num_tokens(
                input_size=input_dim,
                patch_size=kwargs["patch_size"],
                stride=kwargs["stride"],
            )
        elif tokenizer_type == "embed":
            embed_layer = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=0)
            num_tokens = input_dim
            nn.init.normal_(embed_layer.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        return embed_layer, num_tokens

    def _embed_spectrum(self, x, tokenizer_type, embed_layer, input_dim):
        if tokenizer_type in ("conv", "patch"):
            return embed_layer(x)
        elif tokenizer_type == "embed":
            indices = torch.arange(
                1, input_dim + 1, device=x.device, dtype=torch.long
            )
            x = x.long() * indices
            return embed_layer(x)
        else:
            raise ValueError(f"Unknown tokenizer type '{tokenizer_type}'.")

    def _embed_hnmr(self, x):
        h_embed = self._embed_spectrum(x, self.h_tokenizer, self.h_embed, self.hnmr_dim)
        if not self.use_h_mask_token:
            return h_embed
        h_missing_mask = (x == 0).all(dim=1)
        h_missing_tokens = self.h_mask_token(h_missing_mask.long()).unsqueeze(1)
        return torch.cat([h_missing_tokens, h_embed], dim=1)

    def _embed_cnmr(self, x):
        c_embed = self._embed_spectrum(x, self.c_tokenizer, self.c_embed, self.cnmr_dim)
        if not self.use_c_mask_token:
            return c_embed
        c_missing_mask = (x == 0).all(dim=1)
        c_missing_tokens = self.c_mask_token(c_missing_mask.long()).unsqueeze(1)
        return torch.cat([c_missing_tokens, c_embed], dim=1)

    def _separate_spectra_components(self, x):
        hnmr_x = x[:, : self.hnmr_dim]
        cnmr_x = x[:, self.hnmr_dim : self.hnmr_dim + self.cnmr_dim]
        return hnmr_x, cnmr_x

    def forward(self, x) -> torch.Tensor:
        hnmr_spectra, cnmr_spectra = self._separate_spectra_components(x)

        tokens = []
        type_ids = []
        if self.use_hnmr:
            t = self._embed_hnmr(hnmr_spectra)
            tokens.append(t)
            type_ids.append(
                torch.zeros(t.size(1), device=t.device, dtype=torch.long)
            )
        if self.use_cnmr:
            t = self._embed_cnmr(cnmr_spectra)
            tokens.append(t)
            type_ids.append(
                torch.ones(t.size(1), device=t.device, dtype=torch.long)
            )

        x = torch.cat(tokens, dim=1)
        x = self.pos_encode(x)

        if self.type_embedding is not None:
            type_ids_tensor = torch.cat(type_ids)
            type_emb = self.type_embedding(type_ids_tensor)
            type_emb = type_emb.unsqueeze(0).expand(x.size(0), -1, -1)
            x = x + type_emb

        x = self.dropout(x)
        x = self.transformer(x)

        return self.head(x)
