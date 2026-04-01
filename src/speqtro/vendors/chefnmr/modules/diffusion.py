# Vendored from chefnmr/src/model/modules/diffusion.py — inference-only.
# Training-only code (forward, compute_loss, train_sigma_distribution) removed.
# Original license: MIT (boltz).

import torch
from torch.nn import Module
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from omegaconf import DictConfig

from einops import rearrange
from math import sqrt

from speqtro.vendors.chefnmr.modules import score_models
from speqtro.vendors.chefnmr.modules.utils import log, default


class AtomDiffusion(Module):
    """
    Atom diffusion module implementing EDM-style reverse diffusion for sampling.
    Training code has been removed; this class is inference-only.
    """

    def __init__(
        self,
        score_model_args: DictConfig,
        sample_sigma_schedule_type: str = "edm",
        sample_gamma_schedule_type: str = "edm",
        num_sampling_steps: int = 50,
        sigma_min: float = 0.0004,
        sigma_max: float = 80.0,
        gamma_min: float = 1.0,
        noise_scale: float = 1.0,
        step_scale: float = 1.0,
        guidance_scale: float = 0.0,
        edm_args: Optional[Dict] = None,
        # Accept and ignore training-only kwargs so load_from_checkpoint works
        **kwargs,
    ):
        super().__init__()
        score_model_name = score_model_args.model_name
        self.score_model = getattr(score_models, score_model_args.model_name)(
            **score_model_args[score_model_name],
        )

        self.sample_sigma_schedule_type = sample_sigma_schedule_type
        self.sample_gamma_schedule_type = sample_gamma_schedule_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.guidance_scale = guidance_scale

        default_edm_args = {
            "sigma_data": 3.0,
            "rho": 7,
            "use_heun_solver": True,
            "gamma_0": 0.8,
        }
        self.edm_args = default(edm_args, default_edm_args)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def float_to_tensor(
        self, value: Union[float, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(value, float):
            value = torch.full((batch_size,), value, device=device)
        return value

    def pad_sigma(
        self, sigma: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        sigma = self.float_to_tensor(sigma, batch_size, device)
        return rearrange(sigma, "b -> b 1 1")

    # --- Preconditioning (Karras et al. 2022) ---
    def c_in(self, sigma):
        return 1 / (torch.sqrt(sigma**2 + self.edm_args.sigma_data**2))

    def d_sigma(self, sigma):
        return -self.edm_args.sigma_data / torch.sqrt(
            sigma**2 + self.edm_args.sigma_data**2
        )

    def b_sigma(self, sigma):
        return sigma

    def noised_coords_in_network(self, atom_coords, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.c_in(padded_sigma) * atom_coords

    def sigma_in_network(self, sigma):
        return log(sigma) * 0.25

    # --- Sampling ---
    def sample_sigma_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        if self.sample_sigma_schedule_type in ["edm", "af3"]:
            inv_rho = 1 / self.edm_args.rho
            steps = torch.arange(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
            sigmas = (
                self.sigma_max**inv_rho
                + steps
                / (num_sampling_steps - 1)
                * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
            ) ** self.edm_args.rho
            if self.sample_sigma_schedule_type == "af3":
                sigmas = sigmas * self.edm_args.sigma_data
            sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    def sample_gamma_schedule(self, sigmas):
        return torch.where(sigmas > self.gamma_min, self.edm_args.gamma_0, 0.0)

    def sample(
        self,
        model_inputs: Dict[str, torch.Tensor],
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1,
        n_chain_frames: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples using the reverse diffusion process.

        Args:
            model_inputs: Dict with 'atom_mask', 'atom_one_hot', 'condition'.
            num_sampling_steps: Number of diffusion steps.
            multiplicity: Number of samples per input.
            n_chain_frames: Number of intermediate frames to save.

        Returns:
            (final_atom_coords, atom_coords_chains)
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)

        if num_sampling_steps < n_chain_frames:
            save_chain_indices = torch.arange(
                0, num_sampling_steps, device=self.device
            )
        else:
            save_chain_indices = (
                torch.arange(0, num_sampling_steps * n_chain_frames, num_sampling_steps)
                // n_chain_frames
            )
        save_chain_indices = save_chain_indices.tolist()
        atom_coords_chains = torch.tensor([], device=self.device)

        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_sigma_schedule(num_sampling_steps)
        gammas = self.sample_gamma_schedule(sigmas)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)

        for i, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            if i in save_chain_indices:
                atom_coords_chains = torch.cat(
                    [atom_coords_chains, atom_coords[:, None, ...]], dim=1
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            eps = torch.randn(shape, device=self.device)

            atom_coords_next = self._sample_one_step_edm(
                model_inputs=model_inputs,
                multiplicity=multiplicity,
                atom_coords=atom_coords,
                sigma_tm=sigma_tm,
                sigma_t=sigma_t,
                gamma=gamma,
                eps=eps,
            )

            atom_coords = atom_coords_next

        atom_coords_chains = torch.cat(
            [atom_coords_chains, atom_coords[:, None, ...]], dim=1
        )

        return atom_coords, atom_coords_chains

    def _sample_one_step_edm(
        self, model_inputs, multiplicity, atom_coords, sigma_tm, sigma_t, gamma, eps
    ):
        t_hat = sigma_tm * (1 + gamma)
        eps = eps * self.noise_scale * sqrt(t_hat**2 - sigma_tm**2)
        noisy_atom_coords = atom_coords + eps

        with torch.no_grad():
            net_out = self.neural_network_forward(
                noisy_atom_coords,
                t_hat,
                network_condition_kwargs=dict(
                    multiplicity=multiplicity,
                    model_inputs=model_inputs,
                    guidance_scale=self.guidance_scale,
                ),
            )

        denoised_atom_coords = self.predict_denoised_atom_coords(
            noisy_atom_coords, net_out, t_hat
        )

        velocity = self.predict_velocity(
            noisy_atom_coords=noisy_atom_coords,
            net_out=net_out,
            sigma=t_hat,
            denoised_atom_coords=denoised_atom_coords,
        )

        atom_coords_next = (
            noisy_atom_coords + self.step_scale * (sigma_t - t_hat) * velocity
        )

        if self.edm_args.use_heun_solver and sigma_t > 0:
            with torch.no_grad():
                net_out = self.neural_network_forward(
                    atom_coords_next,
                    sigma_t,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_inputs=model_inputs,
                        guidance_scale=self.guidance_scale,
                    ),
                )

            denoised_atom_coords = self.predict_denoised_atom_coords(
                atom_coords_next, net_out, sigma_t
            )

            velocity_next = self.predict_velocity(
                noisy_atom_coords=atom_coords_next,
                net_out=net_out,
                sigma=sigma_t,
                denoised_atom_coords=denoised_atom_coords,
            )

            atom_coords_next = (
                noisy_atom_coords
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity_next
            )

        return atom_coords_next

    def neural_network_forward(
        self,
        noisy_atom_coords: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        network_condition_kwargs: dict,
    ) -> Dict[str, torch.Tensor]:
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        sigma = self.float_to_tensor(sigma, batch_size, device)
        net_out = self.score_model(
            r_noisy=self.noised_coords_in_network(noisy_atom_coords, sigma),
            times=self.sigma_in_network(sigma),
            **network_condition_kwargs,
        )
        return net_out

    def predict_denoised_atom_coords(self, noisy_atom_coords, net_out, sigma):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)

        denoised_atom_coords = (
            self.d_sigma(padded_sigma) * noisy_atom_coords
            - self.b_sigma(padded_sigma) * net_out["r_update"]
        )
        denoised_atom_coords = (
            denoised_atom_coords
            * -self.edm_args.sigma_data
            * self.c_in(padded_sigma)
        )

        return denoised_atom_coords

    def predict_velocity(
        self, noisy_atom_coords, net_out, sigma, denoised_atom_coords=None
    ):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)

        if denoised_atom_coords is None:
            denoised_atom_coords = self.predict_denoised_atom_coords(
                noisy_atom_coords, net_out, sigma
            )
        velocity = (noisy_atom_coords - denoised_atom_coords) / padded_sigma

        return velocity
