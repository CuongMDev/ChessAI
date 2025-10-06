import math

import torch
from torch import nn

class AttentionPolicyHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, board_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.board_size = board_size
        self.num_squares = board_size ** 2  # 64

        # 1. Policy embedding
        self.policy_embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.SELU(inplace=True)
        )

        # 2. Linear projections for Q, K
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)

        # 3. Promotion offsets (Q,R,B,N)
        self.promo_offset = nn.Linear(hidden_dim, 4, bias=False)

        # register constant indices for ONNX
        from_rank7 = torch.arange(board_size, 2 * board_size)  # 8..15
        to_rank8   = torch.arange(0, board_size)  # 0..7
        self.register_buffer("from_rank7", from_rank7, persistent=False)
        self.register_buffer("to_rank8", to_rank8, persistent=False)

    def forward(self, x):
        """
        x: (B, 64, in_dim) one-hot piece representation
        returns: (B, 5120) policy logits
        """
        # 1. Embedding
        x_emb = self.policy_embedding(x)  # (B, 64, hidden_dim)

        # 2. Q, K
        Q = self.q_linear(x_emb)  # (B, 64, hidden_dim)
        K = self.k_linear(x_emb)  # (B, 64, hidden_dim)

        dk = math.sqrt(self.hidden_dim)

        # 3. Attention scores
        matmul_qk = torch.bmm(Q, K.transpose(-2, -1)) # (B, 64, 64)
        attn_scores = matmul_qk / dk

        # -----------------------
        # Promotion part
        # -----------------------
        # knight baseline logits
        n_promo_logits = matmul_qk[:, self.from_rank7][:, :, self.to_rank8]  # (B, 8, 8)

        # get promotion keys (to-rank squares)
        promotion_keys = K[:, self.to_rank8, :]  # (B, 8, hidden_dim)

        # offsets: (B, 8, 4) -> (B, 4, 8)
        promotion_offsets = self.promo_offset(promotion_keys)  # (B, 8, 4)
        promotion_offsets = promotion_offsets.transpose(1, 2) * dk

        promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]

        # q,r,b use knight baseline + offset
        q_logits = (n_promo_logits + promotion_offsets[:, 0:1, :])
        r_logits = (n_promo_logits + promotion_offsets[:, 1:2, :])
        b_logits = (n_promo_logits + promotion_offsets[:, 2:3, :])

        # concat (Q,R,B) = (B, 8, 8 * 4)
        promotion_logits = torch.cat([q_logits, r_logits, b_logits, n_promo_logits], dim=-1)
        promotion_logits = promotion_logits / dk

        # scale the logits

        # -----------------------
        # Final flatten
        # -----------------------
        flat_policy = torch.cat([
            attn_scores.flatten(start_dim=1),        # (B, 64*64)
            promotion_logits.flatten(start_dim=1)    # (B, 8*32)
        ], dim=1)  # (B, 5120)

        return flat_policy
