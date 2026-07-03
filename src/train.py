"""Training loop for the char-level Transformer.

Loads the corpus built by dataset.py, slices it into fixed-length char
windows of size BLOCK_SIZE, trains the CharTransformer to predict the next
character, and reports per-epoch train/val loss in bits-per-char (bpc).

Why bpc and not accuracy: char-LM top-1 "accuracy" is a vanity metric — it
is high simply because space and a handful of common letters dominate the
vocabulary. Loss in bits-per-char (loss / ln(2)) is the honest metric for
language modeling; it says how surprised the model is, per character, in
information-theoretic units. Karpathy's char-RNN/nanoGPT benchmarks use the
same unit, so the number is comparable across the literature.

Why split windows, not raw text, and why that's OK here: the windows are
sliced from one continuous corpus, so adjacent windows share text. For a
*leakage-sensitive* task (our previous P3 smoke-detection lesson) that would be a
problem — rows that aren't independent inflate metrics. Here the goal is
*genre learning*, not cross-document generalization: we want the model to
absorb the style of incident-handling prose, and "the model saw a nearby
paragraph during training" does not manufacture a misleadingly good number
the way shuffled sensor rows did in P3. The val loss is still meaningful as
a convergence signal.

Metrics are written to reports/metrics_run.csv; the final state_dict +
config go to reports/char_transformer.pt and reports/config.json so the
notebook can reload without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from dataset import CharVocab, load_text_corpus
from model import BLOCK_SIZE, GPTConfig, CharTransformer

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python/torch RNGs for reproducible init + splits + sampling."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data: corpus -> fixed-length char windows
# ---------------------------------------------------------------------------

def build_window_dataset(
    text: str,
    vocab: CharVocab,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Encode the whole corpus, then slice into (input, target) windows.

    Each window is block_size+1 tokens: the first block_size are the input,
    the last block_size are the targets (shifted by one). We store windows
    as a (N, block_size+1) LongTensor and let the training loop split off
    inputs/targets. Non-overlapping stride (= block_size) keeps the dataset
    honest about its effective size.
    """
    ids = vocab.encode(text)
    bs1 = block_size + 1
    n_windows = (len(ids) - 1) // bs1  # drop the partial tail
    if n_windows < 1:
        raise ValueError(
            f"corpus too small: {len(ids)} chars yields 0 windows of size {bs1}"
        )
    # Truncate to a whole number of windows, then reshape.
    usable = n_windows * bs1
    data = torch.tensor(ids[:usable], dtype=torch.long)
    windows = data.view(n_windows, bs1)
    return windows


def make_loaders(
    windows: torch.Tensor,
    batch_size: int,
    val_frac: float = 0.1,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    """Split windows into train/val TensorDataset loaders.

    random_split on the window index set, seeded so the partition is
    reproducible across runs.
    """
    n = windows.shape[0]
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(windows, [n_train, n_val], generator=g)

    def loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=shuffle)

    return loader(train_ds, shuffle=True), loader(val_ds, shuffle=False)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: CharTransformer,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    pad_ignore: int = 0,  # <PAD> token id; we don't want it in the loss
) -> tuple[float, int]:
    """One pass over the loader. Returns (total_loss_sum, n_target_tokens).

    n_target_tokens excludes <PAD> positions so the reported loss is the
    honest mean over real characters, not diluted by padding.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, n_tokens = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        idx = batch[:, :-1]            # (B, T) inputs
        targets = batch[:, 1:]         # (B, T) next-char targets
        if train:
            optimizer.zero_grad()
        logits, loss = model(idx, targets=targets)
        if train:
            loss.backward()
            optimizer.step()
        # loss is mean over all B*T positions (incl. any <PAD>); recompute
        # the honest sum over non-pad targets for the running average.
        with torch.no_grad():
            mask = targets != pad_ignore
            n_tok = mask.sum().item()
        total_loss += loss.item() * targets.numel()  # sum of per-position losses
        n_tokens += n_tok
    return total_loss / max(1, n_tokens), n_tokens


def bits_per_char(loss: float) -> float:
    """Cross-entropy (natural-log) -> bits-per-char. Lower = better."""
    return loss / math.log(2)


# ---------------------------------------------------------------------------
# Sampling during training (qualitative eval)
# ---------------------------------------------------------------------------

def sample_text(
    model: CharTransformer,
    vocab: CharVocab,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    """Generate `max_new_tokens` chars from a text prompt. Decodes back to str."""
    model.eval()
    prompt_ids = vocab.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [vocab.stoi.get(" ", 0)]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return vocab.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(
    corpus_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    sample_every: int,
    sample_prompt: str,
    out_dir: Path,
    seed: int = SEED,
) -> dict:
    """Train end-to-end. Returns history + final val loss + a final sample."""
    set_seed(seed)

    text = load_text_corpus(corpus_path)
    vocab = CharVocab.from_text(text)
    print(f"[data] corpus chars={len(text):,}  vocab_size={vocab.vocab_size}")

    windows = build_window_dataset(text, vocab)
    train_loader, val_loader = make_loaders(windows, batch_size)
    print(f"[data] windows={windows.shape[0]}  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader)}  "
          f"device={DEVICE}")

    cfg = GPTConfig()
    model = CharTransformer(vocab_size=vocab.vocab_size, cfg=cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_tok = run_epoch(model, train_loader, optimizer)
        va_loss, va_tok = run_epoch(model, val_loader, None)
        best_val_loss = min(best_val_loss, va_loss)
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_bpc": bits_per_char(tr_loss),
            "val_loss": va_loss, "val_bpc": bits_per_char(va_loss),
            "seconds": time.time() - t0,
        }
        history.append(rec)
        print(f"[epoch {epoch:>2}/{epochs}] "
              f"train_loss={tr_loss:.4f} ({bits_per_char(tr_loss):.3f} bpc)  "
              f"val_loss={va_loss:.4f} ({bits_per_char(va_loss):.3f} bpc)  "
              f"({rec['seconds']:.1f}s)")

        # Qualitative checkpoint: see the model's prose develop over training.
        if epoch % sample_every == 0 or epoch == epochs:
            s = sample_text(model, vocab, sample_prompt, max_new_tokens=160)
            print(f"  sample @epoch {epoch}: {s!r}")

    # Persist artifacts so the notebook can reload without retraining.
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(history, out_dir / "metrics_run.csv")
    torch.save(model.state_dict(), out_dir / "char_transformer.pt")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"gpt_config": asdict(cfg), "vocab_size": vocab.vocab_size,
                   "seed": seed, "epochs": epochs, "batch_size": batch_size,
                   "lr": lr}, f, indent=2)
    vocab.save(out_dir / "vocab.json")
    print(f"[saved] {out_dir}/metrics_run.csv, char_transformer.pt, "
          f"config.json, vocab.json")

    return {"history": history, "final_val_loss": va_loss,
            "final_val_bpc": bits_per_char(va_loss), "best_val_loss": best_val_loss,
            "vocab_size": vocab.vocab_size, "model": model, "vocab": vocab}


def save_csv(history: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train the P5 char-level Transformer.")
    p.add_argument("--corpus", type=Path, default=Path("data/csops_corpus.txt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-every", type=int, default=5,
                   help="generate a sample every N epochs")
    p.add_argument("--sample-prompt", type=str,
                   default="The incident response team ",
                   help="prompt for the in-training samples")
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    if not args.corpus.exists():
        raise FileNotFoundError(
            f"corpus not found at {args.corpus}. "
            f"Run scripts/fetch_corpus.ps1 first."
        )

    fit(
        corpus_path=args.corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()