from torch import nn

class Smolgen(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 hidden_channels, hidden_sz, gen_sz, shared_logit_proj, board_size):
        super().__init__()
        self.num_heads = num_heads

        self.smolgen_activation = nn.SiLU()

        # Step 1: reduce each token
        self.token_proj = nn.Linear(embed_dim, hidden_channels, bias=False)  # (C → 32)

        # Step 2: compress all tokens into hidden_sz
        self.compress = nn.Linear(board_size ** 2 * hidden_channels, hidden_sz, bias=False)  # 64*32 → 256
        self.norm1 = nn.LayerNorm(hidden_sz)

        # Step 3: head-specific projection
        self.head_proj = nn.Linear(hidden_sz, gen_sz * num_heads)  # (256 → 256*h)
        self.norm2 = nn.LayerNorm(gen_sz)

        # Step 4: shared generator to logits (256 → 64*64)
        self.logit_proj = shared_logit_proj

    def forward(self, x):
        # x: (B, N=64, C)
        B, N, C = x.shape
        h = self.num_heads

        # Step 1: token compression
        tokens = self.token_proj(x)  # (B, N, hidden_channels)

        # Step 2: flatten board
        flat = tokens.reshape(B, -1)  # (B, N*hidden_channels)

        # Step 3: global hidden
        hidden = self.compress(flat)  # (B, hidden_sz)
        hidden = self.smolgen_activation(hidden)
        hidden = self.norm1(hidden)

        # Step 4: head-specific hidden
        head_hidden = self.head_proj(hidden)  # (B, h*gen_sz)
        head_hidden = self.smolgen_activation(head_hidden)
        head_hidden = head_hidden.view(B, h, -1)  # (B, h, gen_sz)
        head_hidden = self.norm2(head_hidden)

        # Step 5: project to logits
        smol_logits = self.logit_proj(head_hidden)  # (B, h, N*N)
        smol_logits = smol_logits.view(B, h, N, N)

        return smol_logits
