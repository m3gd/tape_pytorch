from typing import Optional, Tuple, Union, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from rin_pytorch.rin_pytorch import RIN

class RINModelOutput(BaseOutput):
    """
    Output of the RIN model.

    Args:
        sample (`torch.FloatTensor`): The output sample.
    """
    sample: torch.FloatTensor


class RINModel(ModelMixin, ConfigMixin):
    """
    RIN (Recurrent Interface Network) model for diffusion.

    Args:
        sample_size (int): Sample size (image size).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimension of the model.
        image_size (int): Size of the input image.
        patch_size (int): Size of patches.
        depth (int): Number of RIN blocks.
        latent_self_attn_depth (int): Depth of latent self-attention.
        dim_latent (int): Dimension of latent vectors.
        num_latents (int): Number of latent vectors.
        learned_sinusoidal_dim (int): Dimension of learned sinusoidal embeddings.
        latent_token_time_cond (bool): Whether to use latent token time conditioning.
        dual_patchnorm (bool): Whether to use dual patch normalization.
        patches_self_attn (bool): Whether to use patches self-attention.
        attention (dict): Attention parameters (heads, dim_head, flash, qk_norm).
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 512,
        image_size: int = 64,
        patch_size: int = 16,
        depth: int = 6,
        latent_self_attn_depth: int = 2,
        dim_latent: Optional[int] = None,
        num_latents: int = 256,
        learned_sinusoidal_dim: int = 16,
        latent_token_time_cond: bool = False,
        dual_patchnorm: bool = True,
        patches_self_attn: bool = True,
        attention: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Default attention parameters if not provided
        if attention is None:
            attention = {
                "heads": 8,
                "dim_head": 64,
                #"flash": False,
                "qk_norm": False
            }

        # Initialize the RIN model
        self.model = RIN(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            channels=in_channels,  # RIN expects channels, not in_channels
            depth=depth,
            latent_self_attn_depth=latent_self_attn_depth,
            dim_latent=dim_latent,
            num_latents=num_latents,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            latent_token_time_cond=latent_token_time_cond,
            dual_patchnorm=dual_patchnorm,
            patches_self_attn=patches_self_attn,
            **attention
        )

        self.latent_self_cond = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[RINModelOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            sample (torch.FloatTensor): Input tensor of shape [batch_size, channels, height, width].
            timestep (torch.Tensor or float or int): Time embedding value for conditioning.
            return_dict (bool): Whether to return a dictionary or a tuple.

        Returns:
            RINModelOutput or tuple: Output of the model.
        """
        # Process timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        # Ensure timestep has the right shape and type
        timestep = timestep.to(dtype=torch.float32)

        # Call the RIN model
        # RIN expects x_self_cond and latent_self_cond, but we'll handle these differently in training
        output = self.model(
            sample,
            timestep,
            x_self_cond=None,
            latent_self_cond=self.latent_self_cond,
            return_latents=False
        )

        if not return_dict:
            return (output,)

        return RINModelOutput(sample=output)
