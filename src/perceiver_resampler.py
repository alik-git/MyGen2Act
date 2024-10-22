"""
Taken from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
which itself is based on
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

# Cross Attention without media variables
class CrossAttention(nn.Module):
    """ 
    In the open-flamingo implementation, they are wanting the text tokens to attend to the media tokens.
    In out context, we want the latent tokens to attend to the media tokens. This is the key difference.
    """
    def __init__(
        self,
        *,
        dim,
        dim_context, # their dim_visual is our dim_context
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False) # their dim_visual is our dim_context
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context): # their media is our context
        """
        Args:
            x (torch.Tensor): input features
                shape (b, n, d)
            context (torch.Tensor): context features
                shape (b, m, d)
        """
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(context).chunk(2, dim=-1) # their media is our context
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

# Gated Cross Attention Block without media variables
class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_context, # their dim_visual is our dim_context
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = CrossAttention(
            dim=dim,
            dim_context=dim_context, # their dim_visual is our dim_context
            dim_head=dim_head,
            heads=heads,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        context, # their media is our context
    ):
        x = (
            self.attn(
                x,
                context, # their media is our context
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


# Perceiver Resampler adjusted to your use case
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2, # from the Gen2Act paper
        dim_head=64,
        heads=8,
        num_latents=64,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, dim_context=dim, ff_mult=ff_mult),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input features
                shape (b, n, d)
        Returns:
            shape (b, num_latents, d)
        """
        b, n, d = x.shape
        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for cross_attn, ff in self.layers:
            latents = cross_attn(latents, x) + latents # because they are using perceiver attention here and we are using gated cross attention
            latents = ff(latents) + latents
        return self.norm(latents)