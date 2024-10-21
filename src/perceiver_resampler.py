

import torch
import torch.nn as nn




# Define GatedCrossAttention, FeedForward, and PerceiverResampler modules
class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, context):
        # x: (batch_size, seq_len1, dim)
        # context: (batch_size, seq_len2, dim)
        x = self.norm(x)
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(x.size(0), x.size(1), self.num_heads, -1)
        k = k.view(context.size(0), context.size(1), self.num_heads, -1)
        v = v.view(context.size(0), context.size(1), self.num_heads, -1)

        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len1, dim_head)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len2, dim_head)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len2, dim_head)

        q = q * self.scale

        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len1, seq_len2)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len1, dim_head)
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len1, num_heads, dim_head)
        out = out.view(x.size(0), x.size(1), -1)  # (batch_size, seq_len1, inner_dim)
        out = self.to_out(out)
        out = out * self.gate.tanh()  # gating
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.net(x) * self.gate.tanh()

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, num_layers=2, num_heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    GatedCrossAttention(dim=dim, num_heads=num_heads, dim_head=dim_head),
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