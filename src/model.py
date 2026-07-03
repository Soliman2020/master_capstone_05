"""Char-level Transformer decoder for the P5 incident-handling corpus.

Architecture (decoder-only, GPT-style, pre-LN, weight-tied):

    token embedding (V x d_model)
        + positional embedding (block_size x d_model)
        -> N x [LayerNorm -> CausalSelfAttention -> residual
                -> LayerNorm -> MLP -> residual]
        -> LayerNorm
        -> linear LM head (V x d_model)  # weights tied to token embedding

Where V = vocab_size (101 for our corpus: 97 unique chars + 4 special
tokens), 
block_size = 128, 
N = 6, d_model = 192, n_heads = 6,
d_mlp = 4 * d_model = 768. 
Total params ~ 2.7M.

The "one change" for the report is NOT an architectural variant — it's the
deliberate choice of character-level tokenization (vs. BPE/subword) in
dataset.py. The model file itself stays a single class; experiments live in train.py.

This module owns no data logic — dataset.py owns tokenization, train.py
owns the loop. The model is pure: takes a LongTensor of token ids, returns
logits over the vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

# Local conventions
SEED = 42

# Default model shape. Kept as module constants so train.py and the notebook
# can read the same numbers without re-declaring them.
BLOCK_SIZE = 128        # context length in tokens (chars in our case)
D_MODEL = 192
N_LAYERS = 6
N_HEADS = 6
D_MLP = 4 * D_MODEL     # 768 — GPT-2 ratio
DROPOUT = 0.1           # applied to attn + MLP; small corpus benefits from some reg


@dataclass
class GPTConfig:
    """All hyperparameters in one place so train.py can dump/load them
    alongside the state_dict for a clean reproducibility story."""
    block_size: int = BLOCK_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    d_mlp: int = D_MLP
    dropout: float = DROPOUT


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with pre-LN.

    Single QKV projection (3 * d_model) is faster on small models than
    three separate linears; we split into heads after projection. The
    causal mask is registered as a buffer so .to(device) follows the
    module automatically.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Non-learnable buffer: upper-triangular causal mask.
        # shape (1, 1, block_size, block_size) so it broadcasts over (B, H, T, T).
        mask = torch.triu(torch.ones(cfg.block_size, cfg.block_size), diagonal=1).bool()
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Scaled dot-product attention. is_causal=True lets SDPA build the
        # causal mask itself; the explicit `mask` buffer is kept for
        # introspection / tests, not passed here (its boolean-vs-additive
        # convention is easy to get wrong, and SDPA's built-in path is
        # both correct and faster on the chosen backend).
        y = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Two-layer feed-forward with GELU. 4x expansion + projection, GPT-2 style."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(nn.functional.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: LN -> attn -> residual -> LN -> MLP -> residual."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class CharTransformer(nn.Module):
    """Char-level GPT. Embedding tied to LM head.

    Forward returns logits of shape (B, T, V) where V is the configured
    vocab_size. Cross-entropy loss is computed by the caller (train.py)
    using a flat (B*T, V) view; see train.py for the standard
    `F.cross_entropy(logits.view(-1, V), targets.view(-1))` pattern.
    """

    def __init__(self, vocab_size: int, cfg: Optional[GPTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or GPTConfig()
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, self.cfg.d_model)            # what character is this?
        self.pos_emb = nn.Embedding(self.cfg.block_size, self.cfg.d_model)   # where in the sequence is it?
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        # LM head is the transposed embedding; we don't allocate a separate
        # Linear — we reuse tok_emb.weight via F.linear at forward time.
        # This is the standard GPT-2 weight-tying trick.

        # Deterministic init: same scheme as nanoGPT (small init for residuals).
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):  # residual projections: scale by 1/sqrt(2N)
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

        n_params = sum(p.numel() for p in self.parameters())
        # tok_emb is counted twice (once for the embed, once for the tied head)
        # so we subtract one copy to report the true unique-param count.
        print(f"[model] CharTransformer: vocab_size={vocab_size}, "
              f"block_size={self.cfg.block_size}, d_model={self.cfg.d_model}, "
              f"n_layers={self.cfg.n_layers}, n_heads={self.cfg.n_heads}, "
              f"unique_params={n_params - self.tok_emb.weight.numel():,}")

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            idx: (B, T) LongTensor of token ids.
            targets: optional (B, T) LongTensor of next-token targets. If
                provided, returns (logits, loss). If None, returns logits only.

        Returns:
            logits: (B, T, V) — predictions for every position.
            loss: scalar tensor, only if `targets` is given.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        pos = torch.arange(T, device=idx.device)
        tok = self.tok_emb(idx)            # (B, T, C)
        pos_e = self.pos_emb(pos)[None, :, :]  # (1, T, C) — broadcast over batch
        x = self.drop(tok + pos_e)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Weight-tied LM head: logits = x @ tok_emb.T
        logits = nn.functional.linear(x, self.tok_emb.weight)  # (B, T, V)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # reshape (not view): targets may be non-contiguous after
            # random_split's subset indexing; view would raise on stride.
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 forbidden_token_ids: tuple = (0, 1, 2, 3)) -> torch.Tensor:
        """Autoregressive sampling. Takes a (B, T) context, returns (B, T+max_new_tokens).

        temperature: 1.0 = neutral, <1.0 = sharper, >1.0 = more random.
        top_k: if set, restricts sampling to the top-k most likely next chars
            (avoids the long tail of nonsense).
        forbidden_token_ids: token ids that must never be sampled. Defaults
            to the four special tokens (PAD/BOS/EOS/UNK) so generated text
            never contains a literal "<BOS>" / "<EOS>" / "<PAD>" / "<UNK>".
            Pass `forbidden_token_ids=()` to disable masking.

        Sets eval mode internally so dropout is off during sampling, then
        restores the caller's mode — so this is safe to call mid-training.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Crop context to the last block_size tokens; the model can't
                # see further than that anyway, and SDPA gets O(T^2) memory.
                idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]  # crop context
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)       # last pos only
                # Never sample special tokens — they are control tokens, not
                # content. Mask them to -inf before softmax. This is the fix
                # for the "<BOS>/<EOS> in the output" bug seen in the notebook.
                if forbidden_token_ids:
                    logits[:, list(forbidden_token_ids)] = -float("inf")
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)       # sample
                idx = torch.cat([idx, idx_next], dim=1)                  # append
            return idx
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Self-check (no test framework)
# ---------------------------------------------------------------------------
# Three plain checks, each one line to read at 3am:
#   1. Shape:    a forward pass returns logits of shape (B, T, V).
#   2. Init loss: a fresh model with random weights outputs near-uniform, so
#                cross-entropy loss ≈ ln(vocab_size). Big deviation => init bug.
#   3. Causality: appending tokens to the END of the input must NOT change the
#                outputs at the positions BEFORE the append. (Exact equality,
#                not a fuzzy tolerance — if it's causal, the first-half outputs
#                are bit-identical regardless of the second half.)

if __name__ == "__main__":
    torch.manual_seed(SEED)

    vocab_size = 101   # matches the real corpus (97 chars + 4 special tokens)
    cfg = GPTConfig()
    model = CharTransformer(vocab_size=vocab_size, cfg=cfg)

    # ---- 1. Shape ----------------------------------------------------------
    B, T = 4, cfg.block_size
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, vocab_size), \
        f"[shape] expected {(B, T, vocab_size)}, got {tuple(logits.shape)}"
    assert loss.dim() == 0, f"[shape] loss must be scalar, got shape {loss.shape}"
    print(f"[1/3 shape]     logits {tuple(logits.shape)}, loss scalar  OK")

    # ---- 2. Init loss ≈ ln(V) ---------------------------------------------
    expected = math.log(vocab_size)
    assert abs(loss.item() - expected) < 0.10 * expected, \
        f"[init loss] {loss.item():.3f} far from ln(V)={expected:.3f} (random init should be near-uniform)"
    print(f"[2/3 init loss] {loss.item():.3f} vs ln(V)={expected:.3f}  OK")

    # ---- 3. Causality (exact) ---------------------------------------------
    # Run the model on the first half, then on the full sequence. The outputs
    # at the first-half positions must be bit-identical, because a causal model
    # cannot see the appended second-half tokens. We use torch.equal (exact),
    # not allclose: if causality holds, there is nothing to be "close" about.
    #
    # MUST be in eval mode: with dropout on, two forward passes on the same
    # input produce different outputs, and the equality check would fail for
    # a reason that has nothing to do with causality.
    model.eval()
    half = T // 2
    out_half, _ = model(idx[:, :half])         # (B, half, V)
    out_full, _ = model(idx)                    # (B, T,    V)
    assert torch.equal(out_half, out_full[:, :half, :]), \
        "[causality] first-half outputs changed when tokens were appended to the end"
    print(f"[3/3 causality] first-half outputs identical after append  OK")

    # ---- Generate sanity --------------------------------------------------
    out = model.generate(idx[:1, :4], max_new_tokens=20, temperature=0.8, top_k=20)
    assert out.shape == (1, 4 + 20), f"[generate] expected (1, 24), got {tuple(out.shape)}"
    print(f"[generate]      prompt (1,4) -> {tuple(out.shape)}  OK")

    print(f"\nCharTransformer self-check PASSED — all three checks OK.")
