"""Decoder-only transformer (GPT styl) psaný od nuly.

Tohle je srdce V2b — přesně ta architektura, na které stojí moderní jazykové
modely, jen malá a přehledná. Model čte sekvenci tokenů a pro každou pozici
předpovídá další token; při generování se tím pádem dá „rozmluvit" a psát dál.
Klíčové kousky: **kauzální self-attention** (každý token se dívá jen dozadu, ne
do budoucnosti), **transformer blok** (attention + MLP s residuály a pre-LayerNorm)
a **weight tying** (vstupní embedding a výstupní hlava sdílejí váhy). Žádná magie,
jen lineární algebra, softmax a pár reziduálních spojů.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention s kauzální maskou (token nevidí do budoucna)."""

    def __init__(self, config):
        """Vytvoří vrstvu podle konfigurace (n_embd musí být dělitelné n_head)."""
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)   # query, key, value naráz
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Dolní trojúhelníková maska: pozice t smí koukat jen na pozice ≤ t.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        """Spočítá self-attention nad vstupem x tvaru (B, T, C)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # rozdělíme na hlavy: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # skóre pozornosti, škálované, zamaskované do budoucnosti, softmax
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = att @ v                                    # vážený součet hodnot
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # zpět slepit hlavy
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Poziční dopředná síť bloku (rozšíří 4×, GELU, zpět)."""

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Aplikuje fc → GELU → proj → dropout."""
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Jeden transformer blok: pre-LN attention + pre-LN MLP, oba s residuálem."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Reziduálně přičte attention a pak MLP (obojí po LayerNormu)."""
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Malý decoder-only transformer pro podmíněnou generaci odpovědí."""

    def __init__(self, config):
        """Sestaví model: embeddingy → N bloků → LayerNorm → hlava (weight tying)."""
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # sdílené váhy vstupu a výstupu
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Inicializuje váhy (normální 0.02), biasy na nulu."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Předpoví logity dalšího tokenu; s `targets` spočítá i loss.

        Args:
            idx (LongTensor): Vstupní tokeny tvaru (B, T), T ≤ block_size.
            targets (LongTensor | None): Cílové tokeny (B, T); pozice s hodnotou
                -100 se do loss nezapočítají (maska promptu).

        Returns:
            tuple: (logits (B, T, vocab), loss nebo None).
        """
        B, T = idx.shape
        assert T <= self.block_size, f"sekvence {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, eos_id=None):
        """Autoregresivně dogeneruje tokeny za vstupní `idx` (sampling).

        Args:
            idx (LongTensor): Počáteční tokeny (1, T).
            max_new_tokens (int): Kolik nejvíc tokenů přidat.
            temperature (float): Teplota samplingu (nižší = jistější).
            top_k (int | None): Ponechat jen k nejlepších tokenů.
            top_p (float | None): Nucleus sampling — nejmenší množina do součtu top_p.
            eos_id (int | None): Když se vygeneruje, generování skončí.

        Returns:
            LongTensor: Vstup + vygenerované tokeny.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = self._filter(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx

    @staticmethod
    def _filter(logits, top_k, top_p):
        """Ořeže logity podle top-k a/nebo top-p (nucleus) před samplingem."""
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k)[0][..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()  # posun: první vždy ponech
            remove[..., 0] = False
            remove_original = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(remove_original, float("-inf"))
        return logits
