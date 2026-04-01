# Vendored from chefnmr/src/model/model.py — inference-only.
# NMRTo3DStructureElucidation LightningModule, trimmed to checkpoint loading + sampling.
# All training steps, validation loops, metric computation, visualization, and optimizer
# configuration have been removed.

from typing import Dict, Any, List
from omegaconf import DictConfig
import numpy as np
import torch
from lightning.pytorch import LightningModule

from speqtro.vendors.chefnmr.modules.diffusion import AtomDiffusion
from speqtro.vendors.chefnmr.modules.utils import ExponentialMovingAverage


class NMRTo3DStructureElucidation(LightningModule):
    """
    Inference-only LightningModule for NMR-to-3D structure elucidation.

    Use ``load_from_checkpoint(path)`` to instantiate, then call
    ``sample_from_spectra()`` or use the lower-level ``self.model.sample()``.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset_args = self.cfg.dataset_args
        self.neural_network_args = self.cfg.neural_network_args
        self.diffusion_process_args = self.cfg.diffusion_process_args
        self.score_model_args = self.cfg.score_model_args

        # Configure the backbone score model
        self._configure_score_model()

        self.model = AtomDiffusion(
            score_model_args=self.score_model_args,
            **self.diffusion_process_args,
        )

        # EMA setup
        self.ema = None
        self.use_ema = self.neural_network_args.use_ema
        self.ema_decay = self.neural_network_args.ema_decay

        self.save_hyperparameters()

        self.loaded_ckpt_epoch = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure_score_model(self):
        score_model_name = self.score_model_args.model_name
        if score_model_name != "DiffusionModuleTransformer":
            raise NotImplementedError(
                f"Score model {score_model_name} is not supported."
            )

        self.score_model_args[score_model_name].in_atom_feature_size = len(
            self.dataset_args.atom_decoder
        )
        self.score_model_args[score_model_name].max_n_atoms = (
            self.dataset_args.max_n_atoms
        )

        self.score_model_args[score_model_name].condition = (
            self.dataset_args.input_generator
        )
        self.score_model_args[score_model_name].drop_transform = (
            self.dataset_args.multitask_args.drop_transform
        )

        condition = self.dataset_args.input_generator
        if condition == "H1NMRSpectrum":
            self.score_model_args[score_model_name].in_condition_size = (
                self.dataset_args.input_generator_addn_args["h1nmr"]["input_dim"]
            )
        elif condition == "C13NMRSpectrum":
            self.score_model_args[score_model_name].in_condition_size = (
                self.dataset_args.input_generator_addn_args["c13nmr"]["input_dim"]
            )
        elif condition == "H1C13NMRSpectrum":
            self.score_model_args[score_model_name].in_condition_size = [
                self.dataset_args.input_generator_addn_args["h1nmr"]["input_dim"],
                self.dataset_args.input_generator_addn_args["c13nmr"]["input_dim"],
            ]
        else:
            raise NotImplementedError(
                f"Condition {condition} is not supported."
            )

    # ------------------------------------------------------------------
    # EMA checkpoint hooks (needed for load_from_checkpoint)
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.use_ema and "ema" in checkpoint:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
            if self.ema.compatible(checkpoint["ema"]["shadow_params"]):
                self.ema.load_state_dict(
                    checkpoint["ema"], device=torch.device("cpu")
                )
            else:
                self.ema = None
        self.loaded_ckpt_epoch = checkpoint.get("epoch")

    def prepare_eval(self) -> None:
        """Apply EMA weights for inference (if available)."""
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        if self.use_ema and self.ema is not None:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def restore_after_eval(self) -> None:
        """Restore original weights after EMA inference."""
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.parameters())

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def sample_from_inputs(
        self,
        model_inputs: Dict[str, torch.Tensor],
        n_samples: int = 10,
        num_sampling_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Run diffusion sampling given pre-built model_inputs tensors.

        Args:
            model_inputs: Dict with 'atom_mask', 'atom_one_hot', 'condition'.
            n_samples: Number of candidate structures per input.
            num_sampling_steps: Diffusion denoising steps.

        Returns:
            Dict with 'atom_coords' (numpy), 'atom_one_hot' (numpy),
            'atom_mask' (numpy).
        """
        self.prepare_eval()
        self.eval()

        try:
            with torch.no_grad():
                predicted_atom_coords, predicted_atom_coords_chains = self.model.sample(
                    model_inputs=model_inputs,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=n_samples,
                    n_chain_frames=1,
                )
        finally:
            self.restore_after_eval()

        return {
            "atom_coords": predicted_atom_coords.cpu().detach().numpy(),
            "atom_mask": model_inputs["atom_mask"].cpu().detach().numpy(),
            "atom_one_hot": model_inputs["atom_one_hot"].cpu().detach().numpy(),
        }

    # ------------------------------------------------------------------
    # Stubs required by LightningModule (unused at inference time)
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use sample_from_inputs() for inference.")

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("This vendored module is inference-only.")

    def configure_optimizers(self):
        raise NotImplementedError("This vendored module is inference-only.")
