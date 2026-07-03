# Project 5 — Generative AI: Character-Level Transformer on Incident-Handling Prose

This project trains a small **character-level Transformer decoder** on a public
corpus of security incident-handling documents (NIST SP 800-61 + ENISA +
JPCERT/CC) and generates plausible, operator-genre prose. It is the Generative
AI project of the Udacity AI Mastery Capstone, and the smallest valid model on
the rubric's allow-list (GAN / VAE / Transformer): a char-level Transformer with
no subword tokenization, chosen deliberately so the model's failure modes stay
visible in every sample.

## The problem this trains for

The end user is the **SOC analyst in the control room** — the operator who
writes the escalation summary at 3am. This project does **not** write that
summary. Its job is upstream: demonstrate that we can train a generator on the
*response* genre of security prose (dated, status-driven, paragraph-form), so a
later project's retrieval-grounded summarizer can be trusted to sound like the
right genre. The genre choice matters: we considered NVD/CVE descriptions first
and rejected them — NVD is *disclosure* prose ("a vulnerability in X allows Y
via Z"), not the *response* prose the operator writes.

## What the model is

A decoder-only GPT, pre-LN, weight-tied LM head, ~2.7M unique parameters.

| Hyperparameter | Value |
|---|---|
| Context length (`block_size`) | 128 chars |
| Model width (`d_model`) | 192 |
| Layers | 6 |
| Attention heads | 6 |
| Feed-forward width (`d_mlp`) | 768 |
| Dropout | 0.1 |
| Vocabulary | 100 (96 chars + 4 special tokens) |

The metric is **bits-per-char** (`loss / ln(2)`) — the honest char-LM unit. Top-1
next-character "accuracy" is a vanity number (space and a handful of common
letters dominate the vocabulary), so it is not reported. bpc matches the
Karpathy char-RNN / nanoGPT literature, so the number is comparable across
references.

## Results

A 30-epoch run on CUDA (~2 minutes) reaches:

- **Validation loss: 1.53 bpc** (train 1.39 bpc)
- Clean, genre-correct samples, e.g. at temperature 0.7:
  > *"The incident response team should be communicated based on the
  > organization's that may be team shared ability to be detected an a bout
  > of events such as the incident data…"*

The model visibly absorbs vocabulary, document structure, and the formal
technical register of incident-handling prose. Its failure modes are visible by
design — factual nonsense (nonexistent products / CVE-IDs / procedures),
drift and repetition at longer sample lengths (the 128-char context window is
small), and occasional mis-placement of rare characters (curly quotes, em-dash,
§, ®). A subword model would hide these behind fluency; char-level keeps the
boundary between pattern-matching and real knowledge legible — which is what
the handoff to the later synthesis project requires.

## Repository layout

```
project_05_generative_ai/
├── data/
│   ├── csops_corpus.txt            # generated corpus (~342 KB, tracked)
│   └── csops_raw/                  # raw downloads (gitignored; reproducible)
├── notebooks/
│   └── generative_model.ipynb       # rubric deliverable; Restart & Run-All clean
├── reports/
│   ├── Generative_AI_Analysis_Report.md   # report source (markdown)
│   ├── Generative_AI_Analysis_Report.pdf  # rubric deliverable (PDF)
│   ├── metrics_run.csv             # per-epoch train/val loss + bpc (tracked)
│   ├── config.json                 # model config + seed + epochs (tracked)
│   ├── vocab.json                  # 96-char vocab + specials (tracked)
│   └── char_transformer.pt         # trained weights (gitignored; reproducible)
├── scripts/
│   ├── fetch_corpus.ps1            # live download + parse + write corpus
│   └── run_selfcheck.ps1           # offline self-check (no network, no PDFs)
├── src/
│   ├── dataset.py                  # catalog-driven download + parse + char vocab
│   ├── model.py                    # char-level GPT decoder + 3-part self-check
│   └── train.py                    # training loop + sampling + artifacts
├── requirements.txt                # pip freeze of the project venv
└── README.md                       # This file
```

## The corpus

Public, real, not synthetic, and not reused from any prior capstone project.
The catalog is driven by `src/dataset.py` → `SOURCE_CATALOG`:

| Source | Kind | License | Attribution |
|---|---|---|---|
| NIST SP 800-61 Rev. 2 — *Computer Security Incident Handling Guide* | PDF | Public domain (U.S. Govt) | NIST |
| NIST SP 800-61 Rev. 3 — *Incident Response Recommendations for CSF 2.0* | PDF | Public domain (U.S. Govt) | NIST |
| ENISA publications index | HTML | CC BY 4.0 | ENISA |
| JPCERT/CC English home | HTML | CC BY-NC 4.0 (non-commercial) | JPCERT/CC |

A live fetch returns 4/4 sources, 574 records, ~339K characters. The corpus is
~96% NIST SP 800-61 by character count — a genre bias worth naming in any
report's limitations section.

**Rejected sources** (documented in the code for transparency):
- SANS ISC diary archive and CISA advisories listing — JavaScript-rendered
  single-page apps with no static links to follow; scraping would need a
  headless browser, out of scope.


## How to run

All commands assume the working directory is `project_05_generative_ai/`.

### 1. Offline self-check (no network, no PDFs, ~1 second)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_selfcheck.ps1
```

Expected: `CharVocab self-check OK`, `HTML→text self-check OK`.

### 2. Build the corpus (downloads + parses + writes `data/csops_corpus.txt`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_corpus.ps1
```

Expected: 4/4 catalog entries fetched, 574 records, ~339K characters.

### 3. Model self-check (offline, needs torch → use `venv`)

```powershell
../venv/Scripts/python.exe -X utf8 src/model.py
```

Expected: `CharTransformer self-check PASSED — all three checks OK.`

### 4. Train (writes `reports/metrics_run.csv` + weights + config + vocab)

```powershell
../venv/Scripts/python.exe -X utf8 src/train.py `
    --corpus data/csops_corpus.txt --epochs 30 --batch-size 64
```

### 5. Run the notebook

The notebook's kernel is `venv`. Register it once as a Jupyter kernelspec:

```powershell
../venv/Scripts/python.exe `
    -m ipykernel install --user --name=venv --display-name="venv"
```

Then open `notebooks/generative_model.ipynb` in JupyterLab (or VS Code) and
**Run All**. Alternatively, execute headless:

```powershell
../venv/Scripts/python.exe `
    -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.kernel_name=venv notebooks/generative_model.ipynb
```

## Reproducibility

- `SEED = 42` is fixed in every module (data split, model init, sampling).
- The corpus is deterministic given the catalog: re-running
  `scripts/fetch_corpus.ps1` reproduces `data/csops_corpus.txt` (modulo any
  upstream site re-publishing).
- `reports/metrics_run.csv`, `reports/config.json`, and `reports/vocab.json`
  are tracked so a reviewer can verify the reported bpc without re-running.
- `reports/char_transformer.pt` (the 10 MB trained weights) is intentionally
  **not** tracked — it is reproducible from `src/train.py` + the corpus in ~2
  minutes on CUDA. It is excluded by `.gitignore`.

## Self-checks (in place of a pytest suite)

Each source module carries a `__main__` self-check runnable directly, in the
project's convention of "leave one runnable check behind":

- **`src/dataset.py`** — `CharVocab` encode/decode round-trip + HTML-to-text
  stripping (verifies the parser drops scripts/styles and decodes entities).
- **`src/model.py`** — three checks: (1) forward-pass shape `(B, T, V)` and
  scalar loss; (2) initial loss ≈ `ln(vocab_size)` since random weights are
  near-uniform; (3) exact causality — appending tokens to the end of the
  input must not change earlier-position outputs (`torch.equal`, in `eval()`
  mode so dropout does not interfere). These checks caught two real bugs
  during development (an inverted attention-mask convention, and a
  non-contiguous `view` after `random_split`) before any training ran.

## Ethics and responsible use

Generated text is **synthetic incident-handling prose, not real vulnerability
information or real incident reports.** Treat every sample as a modeling
artifact, never as authoritative security guidance. The model has no access
to real systems, real advisories beyond its training corpus, or real policy;
any factual claim in a sample is coincidence.

The corpus contains no personal data, no real organization names, and no
operational logs — it is published guidance text only. Per-source licenses
and attributions are recorded in `src/dataset.py` → `SOURCE_CATALOG` and in
the corpus file's header, and must be cited in any derived report.

## Software versions

- Python 3.12.3
- PyTorch 2.12.1+cu126 (CUDA build)
- `pypdf` 6.14.2, `certifi` 2026.6.17
- JupyterLab 4.6.0, matplotlib 3.11.0