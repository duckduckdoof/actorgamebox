
"""
smol_transformer.py

Author: Caleb Scott

---

Code for a smol transformer. For fun!

Works with characters as tokens.

"""

# IMPORTS
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# CONSTANTS
TXT_SM = \
    """
    From fairest creatures we desire increase, that thereby beauty's rose might never die.
    """

# FUNCTIONS
def get_chars_data(text=TXT_SM) -> tuple:
    # Get unique tokens + length of token vocab
    chars = sorted(set(text))
    c_len = len(chars)

    # Convert text tokens into integers, and back via dicts
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}

    # Return encode/decode funcs for any string from this text.
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    decode = lambda t: ''.join([itos[int(i)] for i in t])
    return chars, c_len, encode, decode

def split_data(data, split_ratio=0.9):
    n = int(split_ratio, * len(data))
    return data[:n], data[n:]

def get_batch(block_size, batch_size, data, device):
    b = min(block_size, max(2, len(data) - 2))
    hi = len(data) - b - 1
    if hi <= 0:
        x = data[:b].unsqueeze(0)
        y = data[1:b+1].unsqueeze(0)
        return x.to(device), y.to(device)
    idx = torch.randint(hi, (batch_size,))
    x = torch.stack([data[i:i+b] for i in idx])
    y = torch.stack([data[i+1:i+b+1] for i in idx])
    return x.to(device), y.to(device)

# CLASSES
@dataclass
class TransformerParams:
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size  = 128   # number of tokens captured in an attention 'window'
    batch_size  = 16    # standard batch size for training
    n_layers    = 2     # How many attention block layers
    n_heads     = 2     # Heads for multi-headed attention on the projection from token embeddings
    embed_dim   = 128   # How big to make each token
    dropout     = 0.1
    train_steps = 1200
    lr          = 3e-3

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, n_heads, attn_block_size, dropout):
        super().__init__()
        self.n_heads = n_heads

        # Q K V projection
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)

        # Causal masking from tril
        self.register_buffer(
            'mask', 
            torch.tril(
                torch.ones(
                    attn_block_size, 
                    attn_block_size
                )
            ).view(1,1, attn_block_size, attn_block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Split up the output of our projected q, k, v based on the number of attn heads.
        query = self.q(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        key = self.k(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        value = self.v(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = (query @ key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ value
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.res_drop(self.proj(y))

class MLP(nn.Module):

    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class AttnBlock(nn.Module):

    def __init__(self, embed_dim, n_heads, attn_block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim, 
            n_heads=n_heads, 
            attn_block_size=attn_block_size, 
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SmolTransformer(nn.Module):

    def __init__(self, vocab_size, n_layers, embed_dim, n_heads, block_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            AttnBlock(embed_dim, n_heads, block_size, dropout) for _ in n_layers
        ])
        self.block_size = block_size
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: from token embedding
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idxs = idx[:, -self.block_size:]
            logits, _ = self(idxs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
