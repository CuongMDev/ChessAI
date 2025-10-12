from torch import nn
import torch.nn.functional as F

from Agent.Network.Attention.Smolgen import Smolgen


class ResidualAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dff_dim, dropout, smolgen_info, initializer, board_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # Linear projections for QKV
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.smolgen = Smolgen(embed_dim, num_heads,
                               *smolgen_info,
                               board_size=board_size)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # FeedForward
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dff_dim),
            nn.Mish(inplace=True),
            nn.Linear(dff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # initializer
        initializer(self.qkv.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape

        # --- Attention ---
        residual = x

        # QKV: (B, N, 3*embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # smolgen bias (dynamic)
        smol_bias = self.smolgen(x) # (B, num_heads, N, N)

        # compute attention scores
        attn_out = F.scaled_dot_product_attention(Q, K, V, smol_bias, dropout_p=self.dropout)

        # merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.norm1(residual + self.out_proj(attn_out))  # post-norm attention

        # --- FeedForward ---
        residual = x
        x = self.mlp(x)
        x = self.norm2(residual + x)

        return x