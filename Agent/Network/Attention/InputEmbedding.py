import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, ffn_dim):
        """
        in_dim: input size per square (12 for one-hot piece type)
        embed_dim: embedding dimension C
        ffn_dim: hidden size của feedforward layer
        """
        super().__init__()
        # Linear projection
        self.linear = nn.Linear(in_dim, embed_dim, bias=False)

        # LayerNorm mới
        self.norm1 = nn.LayerNorm(embed_dim)

        # Gating
        self.gate_linear = nn.Linear(embed_dim, embed_dim)

        # LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)

        # FeedForward mới
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        """
        x: (B, N, C)
        return: (B, N, embed_dim)
        """

        # Step 1: linear projection
        x = self.linear(x)

        # Step 2: LayerNorm
        x_norm = self.norm1(x)

        # Step 3: Gating (add & multiply)
        g = torch.sigmoid(self.gate_linear(x_norm))
        x_gated = x_norm * g + x_norm

        # Step 4: FeedForward
        x_ffn = self.ffn(x_gated)

        # Step 5: Residual
        x_out = self.norm2(x_gated + x_ffn)

        return x_out