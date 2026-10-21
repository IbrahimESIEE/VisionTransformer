import torch
import torch.nn as nn
import math


# ============================================================
#  Patch Embedding : remplace le tokenizer/vocab du texte.
#  Une Conv2d avec kernel_size = stride = patch_size découpe
#  l'image en patches non chevauchants ET les projette dans
#  l'espace embed_dim en une seule opération.
# ============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x : (B, C, H, W)
        x = self.proj(x)                 # (B, embed_dim, H/patch, W/patch)
        B, E, Hp, Wp = x.shape
        x = x.flatten(2)                 # (B, embed_dim, Hp*Wp)
        x = x.transpose(1, 2)            # (B, Hp*Wp, embed_dim)  -> séquence de patches
        return x, Hp, Wp


# ============================================================
#  Position embedding 2D appris : une table par ligne + une
#  table par colonne, additionnées. Plus économe qu'une table
#  1D géante et ça encode explicitement la structure spatiale.
# ============================================================
class PositionalEmbedding2D(nn.Module):
    def __init__(self, embed_dim, max_grid=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_grid, embed_dim // 2)
        self.col_embed = nn.Embedding(max_grid, embed_dim // 2)

    def forward(self, Hp, Wp, device):
        rows = torch.arange(Hp, device=device)
        cols = torch.arange(Wp, device=device)
        row_emb = self.row_embed(rows).unsqueeze(1).repeat(1, Wp, 1)   # (Hp, Wp, E/2)
        col_emb = self.col_embed(cols).unsqueeze(0).repeat(Hp, 1, 1)   # (Hp, Wp, E/2)
        pos = torch.cat([row_emb, col_emb], dim=-1)                    # (Hp, Wp, E)
        return pos.view(Hp * Wp, -1)                                   # (Hp*Wp, E)


def attention(Q, K, V, head_dim, causal=False):
    seq_len = Q.shape[-2]
    KTranspose = K.transpose(-2, -1)
    value = torch.matmul(Q, KTranspose) / math.sqrt(head_dim)

    if causal:
        masque = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        value = value.masked_fill(~masque, float('-inf'))
    # sinon : pas de masque -> chaque patch voit tous les autres (encodeur)

    soft = torch.softmax(value, dim=-1)
    return torch.matmul(soft, V)


class attentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, causal=False):
        super().__init__()
        self.causal = causal
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return attention(Q, K, V, self.W_q.out_features, causal=self.causal)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            attentionHead(embed_dim, self.head_dim, causal=causal) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sorties = [head(x) for head in self.heads]
        out = torch.cat(sorties, dim=-1)
        return self.proj(out)


class feedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        return self.feature(x)


class transformerBlock(nn.Module):
    """Bloc encodeur : pas de masque causal, attention bidirectionnelle entre patches."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, causal=False)
        self.ff = feedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ============================================================
#  Tête de profondeur : pour chaque patch, prédit une valeur
#  de profondeur, puis on reconstruit une carte basse résolution
#  (Hp x Wp) et on l'upsample à la résolution de l'image.
# ============================================================
class DepthHead(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # prédit patch_size*patch_size valeurs par patch -> on regagne la pleine résolution
        self.proj = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x, Hp, Wp):
        # x : (B, Hp*Wp, embed_dim)
        B = x.shape[0]
        x = self.proj(x)                                    # (B, Hp*Wp, patch*patch)
        x = x.view(B, Hp, Wp, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous()            # (B, Hp, patch, Wp, patch)
        x = x.view(B, Hp * self.patch_size, Wp * self.patch_size)  # (B, H, W)
        return x


class DepthTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_blocks=6,
                 patch_size=16, in_channels=3, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = PositionalEmbedding2D(embed_dim)
        self.blocks = nn.ModuleList([
            transformerBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.depth_head = DepthHead(embed_dim, patch_size)

    def forward(self, images):
        # images : (B, C, H, W), valeurs normalisées
        x, Hp, Wp = self.patch_embed(images)         # (B, N, E)
        pos = self.pos_embed(Hp, Wp, images.device)  # (N, E)
        x = x + pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        depth = self.depth_head(x, Hp, Wp)            # (B, H, W)
        return depth