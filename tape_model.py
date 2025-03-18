import math
from functools import partial
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange, repeat

class TAPEModelOutput(BaseOutput):
    """
    Output of the TAPE model.

    Args:
        sample (`torch.FloatTensor`): The output sample.
    """
    sample: torch.FloatTensor


class ScalarEmbedding(nn.Module):
    """Scalar embedding layers."""

    def __init__(self, dim, scaling, expansion=4):
        super().__init__()
        self.scalar_encoding = lambda x: positional_encoding(x * scaling, dim)
        self.dense_0 = nn.Linear(dim, dim * expansion)
        self.dense_1 = nn.Linear(dim * expansion, dim * expansion)

    def forward(self, x, last_swish=True, normalize=False):
        y = None
        if x.dim() > 1:
            assert x.dim() == 2
            x, y = x[..., 0], x[..., 1:]
        x = self.scalar_encoding(x)[0]
        if normalize:
            x_mean = torch.mean(x, dim=-1, keepdim=True)
            x_std = torch.std(x, dim=-1, keepdim=True)
            x = (x - x_mean) / (x_std + 1e-6)
        x = F.silu(self.dense_0(x))
        x = x if y is None else torch.cat([x, y], dim=-1)
        x = self.dense_1(x)
        return F.silu(x) if last_swish else x


def positional_encoding(coords, dim):
    """Positional encoding for coordinates."""
    batch_size = coords.shape[0]
    angle_rads = get_angles(coords.unsqueeze(-1),
                          torch.arange(dim)[None, None, :].to(coords.device),
                          dim)

    # Apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[:, :, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[:, :, 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], dim=-1)

    return pos_encoding


def get_angles(pos, i, dim):
    angle_rates = 1 / (10000 ** (2 * (i//2).float() / dim))
    return pos.float() * angle_rates


class LayerNorm(nn.Module):
    """Layer normalization without bias."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class PEG(nn.Module):
    """Positional Encoding Generator."""

    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) d -> b d h w', h=hw)
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class FeedForward(nn.Module):
    """Feed forward module with time conditioning."""

    def __init__(self, dim, mult=4, time_cond_dim=None):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.time_cond = None

        if time_cond_dim is not None:
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                nn.Unflatten(1, (1, -1))
            )
            # Initialize to zeros
            nn.init.zeros_(self.time_cond[1].weight)
            nn.init.zeros_(self.time_cond[1].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time=None):
        x = self.norm(x)

        if self.time_cond is not None and time is not None:
            time_cond = self.time_cond(time)
            scale, shift = time_cond.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim,
        dim_context=None,
        num_heads=8,
        head_dim=64,
        dropout=0.0,
        time_cond_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)

        context_dim = dim_context if dim_context is not None else dim
        self.to_kv = nn.Linear(context_dim, 2 * num_heads * head_dim, bias=False)

        self.to_out = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.time_cond = None
        if time_cond_dim is not None:
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                nn.Unflatten(1, (1, -1))
            )
            # Initialize to zeros
            nn.init.zeros_(self.time_cond[1].weight)
            nn.init.zeros_(self.time_cond[1].bias)

    def forward(self, x, context=None, time=None, mask=None):
        batch_size = x.shape[0]

        if self.time_cond is not None and time is not None:
            time_cond = self.time_cond(time)
            scale, shift = time_cond.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        q = self.to_q(x)

        context = x if context is None else context
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        out = self.to_out(out)

        return out


class TAPEBlock(nn.Module):
    """TAPE block with reading and writing operations."""

    def __init__(
        self,
        dim,
        latent_dim,
        num_heads=8,
        time_cond_dim=None,
        dropout=0.0,
        num_layers=1,
    ):
        super().__init__()
        self.num_layers=num_layers
        self.peg = PEG(dim)

        # Latents attend to tape
        self.latents_attend_to_tape = MultiHeadAttention(
            latent_dim, dim_context=dim, num_heads=num_heads, dropout=dropout,
            time_cond_dim=time_cond_dim
        )
        self.latent_ff = FeedForward(latent_dim, time_cond_dim=time_cond_dim)

        # Latent self attention
        self.latent_self_attn = MultiHeadAttention(
            latent_dim, num_heads=num_heads, dropout=dropout,
            time_cond_dim=time_cond_dim
        )
        self.latent_self_ff = FeedForward(latent_dim, time_cond_dim=time_cond_dim)

        # Tape attends to latents
        self.tape_attends_to_latents = MultiHeadAttention(
            dim, dim_context=latent_dim, num_heads=num_heads, dropout=dropout,
            time_cond_dim=time_cond_dim
        )
        self.tape_ff = FeedForward(dim, time_cond_dim=time_cond_dim)

        # Tape self attention
        self.tape_self_attn = MultiHeadAttention(
            dim, num_heads=num_heads, dropout=dropout,
            time_cond_dim=time_cond_dim
        )
        self.tape_self_ff = FeedForward(dim, time_cond_dim=time_cond_dim)

        # Normalization layers
        self.norm_latent_1 = LayerNorm(latent_dim)
        self.norm_latent_2 = LayerNorm(latent_dim)
        self.norm_latent_3 = LayerNorm(latent_dim)
        self.norm_tape_1 = LayerNorm(dim)
        self.norm_tape_2 = LayerNorm(dim)
        self.norm_tape_3 = LayerNorm(dim)

    def forward(self, tape, latents, time=None):
        # Apply positional encoding
        tape = self.peg(tape) + tape

        # Latents attend to tape
        latents_residual = latents
        latents = self.norm_latent_1(latents)
        latents = latents_residual + self.latents_attend_to_tape(latents, tape, time)
        latents = latents + self.latent_ff(latents, time)

        # Latent self attention
        latents_residual = latents
        latents = self.norm_latent_2(latents)
        latents = latents_residual + self.latent_self_attn(latents, time=time)
        latents = latents + self.latent_self_ff(latents, time)

        # Tape self attention
        for _ in range(self.num_layers):
            tape_residual = tape
            tape = self.norm_tape_1(tape)
            tape = tape_residual + self.tape_self_attn(tape, time=time)
            tape = tape + self.tape_self_ff(tape, time)

        # Tape attends to latents
        tape_residual = tape
        tape = self.norm_tape_2(tape)
        latents = self.norm_latent_3(latents)
        tape = tape_residual + self.tape_attends_to_latents(tape, latents, time)
        tape = tape + self.tape_ff(tape, time)

        return tape, latents


class TAPEModel(ModelMixin, ConfigMixin):
    """
    TAPE model for diffusion.

    Args:
        sample_size (int): Sample size (image size).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (Tuple[int]): Number of TAPE layers per block.
        latent_dim (int): Dimension of latent vectors.
        tape_dim (int): Dimension of tape vectors.
        num_latents (int): Number of latent vectors.
        patch_size (int): Size of patches.
        num_heads (int): Number of attention heads.
        time_embedding_dim (int): Dimension of time embeddings.
        dropout (float): Dropout rate.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        num_layers: Tuple = (4, 4),
        latent_dim: int = 512,
        tape_dim: int = 512,
        num_latents: int = 16,
        patch_size: int = 8,
        num_heads: int = 8,
        time_embedding_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.time_embedding_dim = time_embedding_dim

        # Calculate number of patches
        self.num_patches = (sample_size // patch_size) ** 2

        # Time embedding
        self.time_embedding = ScalarEmbedding(
            dim=time_embedding_dim // 4,
            scaling=10000,
            expansion=4
        )

        # Patch embedding (stem)
        self.patch_embedding = nn.Conv2d(
            in_channels, tape_dim,
            kernel_size=patch_size, stride=patch_size, padding=0
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, tape_dim) * 0.02)

        # Latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        # Normalization layers
        self.tape_norm = LayerNorm(tape_dim)
        self.output_norm = LayerNorm(tape_dim)

        # TAPE blocks
        self.tape_blocks = nn.ModuleList([
            TAPEBlock(
                dim=tape_dim,
                latent_dim=latent_dim,
                num_heads=num_heads,
                time_cond_dim=time_embedding_dim,
                dropout=dropout,
                num_layers=num_layers[i],
            )
            for i in range(len(num_layers))
        ])

        # Output projection
        self.to_output = nn.Linear(tape_dim, patch_size * patch_size * out_channels)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True
    ) -> Union[TAPEModelOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            sample (torch.FloatTensor): Input tensor of shape [batch_size, channels, height, width].
            timestep (torch.Tensor or float or int): Time embedding value for conditioning.
            return_dict (bool): Whether to return a dictionary or a tuple.

        Returns:
            TAPEModelOutput or tuple: Output of the model.
        """
        # Process timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        # Time embedding
        time_emb = self.time_embedding(timestep)

        # Extract patches
        batch_size = sample.shape[0]
        x = self.patch_embedding(sample)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add positional embedding
        x = x + self.pos_embedding

        # Initialize latents
        latents = repeat(self.latents, '1 n d -> b n d', b=batch_size)

        # Process through TAPE blocks
        for block in self.tape_blocks:
            x, latents = block(x, latents, time_emb)

        # Output projection
        x = self.output_norm(x)
        x = self.to_output(x)

        # Reshape to image
        patch_size = self.patch_size
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                    h=h, w=w, p1=patch_size, p2=patch_size, c=self.out_channels)

        if not return_dict:
            return (x,)

        return TAPEModelOutput(sample=x)
