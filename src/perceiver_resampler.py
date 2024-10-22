"""
Based on https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
which itself is based on
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum

def exists(val):
    return val is not None


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert (
                media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"

        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

# This below is Ali's code (maybe wrong)
class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, num_layers=2, num_heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    GatedCrossAttentionBlock(dim=dim, num_heads=num_heads, dim_head=dim_head),
                    FeedForward(dim=dim, mult=ff_mult)
                ])
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: shape (batch_size, seq_len, dim)
        batch_size = x.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_latents, dim)
        for cross_attn, ff in self.layers:
            latents = cross_attn(latents, x) + latents
            latents = ff(latents) + latents
        return self.norm(latents)  # (batch_size, num_latents, dim)