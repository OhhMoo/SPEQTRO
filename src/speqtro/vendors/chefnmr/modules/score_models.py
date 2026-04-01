# Vendored from chefnmr/src/model/modules/score_models.py — inference-only.
# DiffusionModuleTransformer: DiT-based score model for atom diffusion.

import torch
from torch import nn
from typing import Dict, Optional, Union, List
from omegaconf import ListConfig

from speqtro.vendors.chefnmr.modules.layers import (
    TimestepEmbedder,
    DiTBlock,
    FinalLayer,
)
from speqtro.vendors.chefnmr.modules.embedders import NMRSpectraEmbedder


class DiffusionModuleTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) for molecular structure generation
    conditioned on NMR spectra and chemical formula.
    """

    def __init__(
        self,
        in_atom_feature_size: int = 10,
        out_atom_coords_size: int = 3,
        condition: str = "H1C13NMRSpectrum",
        in_condition_size: Union[int, List[int], ListConfig] = [10000, 80],
        max_n_atoms: int = 300,
        drop_transform: str = "zero",
        n_blocks: int = 10,
        n_heads: int = 8,
        hidden_size: int = 512,
        mlp_ratio: float = 4.0,
        embedder_args: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()

        self.x_embedder = nn.Sequential(
            nn.Linear(in_atom_feature_size + 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.condition = condition
        if condition in ["H1NMRSpectrum", "C13NMRSpectrum", "H1C13NMRSpectrum"]:
            use_hnmr = "H1" in condition
            use_cnmr = "C13" in condition
            hnmr_dim = 0
            cnmr_dim = 0

            if use_hnmr and use_cnmr:
                assert isinstance(in_condition_size, (list, tuple, ListConfig))
                hnmr_dim = in_condition_size[0]
                cnmr_dim = in_condition_size[1]
            elif use_hnmr:
                assert isinstance(in_condition_size, int)
                hnmr_dim = in_condition_size
            elif use_cnmr:
                assert isinstance(in_condition_size, int)
                cnmr_dim = in_condition_size

            self.y_embedder = NMRSpectraEmbedder(
                use_hnmr=use_hnmr,
                use_cnmr=use_cnmr,
                hnmr_dim=hnmr_dim,
                cnmr_dim=cnmr_dim,
                hidden_dim=embedder_args["hidden_dim"],
                output_dim=hidden_size,
                dropout=embedder_args["dropout"],
                pooling=embedder_args["pooling"],
                tokenizer_args=embedder_args["tokenizer_args"],
                transformer_args=embedder_args["transformer_args"],
            )
        else:
            raise NotImplementedError(
                f"Condition embedding {condition} not implemented."
            )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_n_atoms, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio) for _ in range(n_blocks)]
        )

        self.final_layer = FinalLayer(hidden_size, out_atom_coords_size)
        self.initialize_weights()

        self.drop_transform = drop_transform

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        model_inputs: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        guidance_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        atom_mask = (
            model_inputs["atom_mask"].repeat_interleave(multiplicity, 0).bool()
        )
        padded_atom_mask = atom_mask[..., None]
        atom_one_hot = model_inputs["atom_one_hot"].repeat_interleave(multiplicity, 0)
        condition = model_inputs["condition"].repeat_interleave(multiplicity, 0)

        if guidance_scale != 0.0:
            r_noisy = torch.cat([r_noisy, r_noisy], dim=0)
            times = torch.cat([times, times], dim=0)
            atom_mask = torch.cat([atom_mask, atom_mask], dim=0)
            padded_atom_mask = torch.cat([padded_atom_mask, padded_atom_mask], dim=0)
            atom_one_hot = torch.cat([atom_one_hot, atom_one_hot], dim=0)

            if self.drop_transform == "zero":
                uncondition = torch.zeros_like(condition)
            else:
                raise ValueError(f"Unsupported drop_transform: {self.drop_transform}")

            condition = torch.cat([condition, uncondition], dim=0)

        x = torch.cat([r_noisy, atom_one_hot], dim=-1)
        x = self.x_embedder(x) * padded_atom_mask

        t = self.t_embedder(times)
        y = self.y_embedder(condition)
        c = t + y

        for block in self.blocks:
            x = block(x, c, atom_mask)

        x = self.final_layer(x, c)
        x = x * padded_atom_mask

        if guidance_scale != 0.0:
            x_guided = x[: x.shape[0] // 2]
            x_unconditioned = x[x.shape[0] // 2 :]
            x = (1 + guidance_scale) * x_guided - guidance_scale * x_unconditioned

        return dict(r_update=x)
