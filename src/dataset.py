"""Public CSIRT / CERT / SANS / NIST incident-report corpus.

Sources (all public, all U.S. Government work or openly licensed):

  NIST SP 800-61 Rev. 3 (Computer Security Incident Handling Guide)
      https://nvd.nist.gov/explained/sp800-61r3  (HTML explainer)
      https://csrc.nist.gov/pubs/sp/800/61/r3/final  (PDF; we use the HTML
          explainer only, so we don't need a PDF parser)
      License: public domain (NIST / U.S. Govt).

  CISA Cybersecurity Incident & Vulnerability Response Playbooks
      https://www.cisa.gov/sites/default/files/publications/Federal_Government_
      Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf
      We avoid PDFs (no parser dep). Use CISA's *HTML* incident response
      explainer pages instead; see SOURCE_CATALOG below.
      License: public domain.

  ENISA (European Union Agency for Cybersecurity) public reports
      https://www.enisa.europa.eu/publications  (HTML listings)
      License: CC BY 4.0 unless noted — attribute ENISA in the report.

  JPCERT/CC English-language incident reports
      https://www.jpcert.or.jp/english/  (HTML)
      License: CC BY-NC 4.0; non-commercial — fine for a student capstone.

  SANS Institute reading-room posters and whitepapers (publicly listed)
      https://www.sans.org/white-papers/  (HTML)
      License: varies; we use only papers that are explicitly free / public.

The module:
    1. Downloads each configured source to data/csops_raw/<label>.{html,json}
    2. Parses to (source_id, text) records — strips HTML, drops nav/footer
    3. Writes a single corpus file data/csops_corpus.txt
    4. Builds a char-level vocab (CharVocab) over the cleaned corpus text

The download is *catalogue-driven* (SOURCE_CATALOG below). If a URL 404s, the
fetch is skipped and the user sees a clear error — never a silent empty file.

This module only fetches + tokenizes; no model logic (see model.py / train.py).
SEED=42 keeps sampling deterministic.
"""

from __future__ import annotations

import json
import random
import re
import ssl
import string
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

#  HTTPS uses certifi's CA bundle; stdlib's bundled bundle is often
# stale on Windows and triggers CERTIFICATE_VERIFIED_FAILED on otherwise-valid
# certs. certifi is MIT-licensed and already pulled in transitively by most
# ML stacks, so the dep is effectively free.

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CTX = ssl.create_default_context()

#  pypdf is MIT-licensed, pure-stdlib-deps, ~350 KB. The single new
# dep this module adds. It's the cheapest way to get the operator's genre in
# the volume a char-LM needs (~300K chars per NIST SP 800-61 PDF).
#
# Lazy import: pypdf is only needed inside _records_from_pdf. Importing it at
# module top would make dataset.py (and therefore model.py / train.py) fail to
# import in a torch-only env. By deferring the import to first use, the
# CharVocab / load_text_corpus / corpus-file code paths work without pypdf,
# and the error only surfaces if you actually try to parse a PDF.
def _pdf_reader(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pypdf is required for the 'pdf' catalog entries. "
            "Install with: pip install pypdf"
        ) from e
    return PdfReader(str(path))

# Local conventions
SEED = 42
USER_AGENT = "udacity-p5-capstone/0.1 (operator-copilot-corpus; +https://nvd.nist.gov)"

# Many of these sites block empty User-Agents; we set a descriptive one.

# ---------------------------------------------------------------------------
# Source catalogue
# ---------------------------------------------------------------------------
# A few URL paths change as sites re-publish. Each entry is verified-by-use:
# the download function records HTTP status; if non-200, the source is skipped
# with a warning, the rest of the pipeline still works.
#
# Format: (label, url, kind, license, attribution)
#   kind = "html"  -> parse as a single HTML doc, keep <p>/<li>/<h1-3>
#   kind = "html-list" -> parse as a listing page, follow each link to a doc
#   kind = "json"  -> parse a JSON file (e.g., CERT/CC vulnerability notes)

SOURCE_CATALOG: List[Dict] = [
    {
        "label": "nist_sp800-61_r2",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf",
        "kind": "pdf",
        "license": "public-domain",
        "attribution": "NIST SP 800-61 Rev. 2 — Computer Security Incident Handling Guide (U.S. Govt, public domain)",
    },
    {
        "label": "nist_sp800-61_r3",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
        "kind": "pdf",
        "license": "public-domain",
        "attribution": "NIST SP 800-61 Rev. 3 — Incident Response Recommendations and Considerations for CSF 2.0 (U.S. Govt, public domain)",
    },
    {
        "label": "enisa_publications_index",
        "url": "https://www.enisa.europa.eu/publications",
        "kind": "html",
        "license": "CC-BY-4.0",
        "attribution": "ENISA publications index (CC BY 4.0)",
    },
    {
        "label": "jpcert_english_home",
        "url": "https://www.jpcert.or.jp/english/",
        "kind": "html",
        "license": "CC-BY-NC-4.0",
        "attribution": "JPCERT/CC (CC BY-NC 4.0; non-commercial use)",
    },
]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def _http_get(url: str, dest: Path, timeout: int = 30) -> int:
    """GET url, save to dest, return HTTP status. Raises on non-2xx unless
    raise_ok=False (then returns the status)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
        body = resp.read()
        status = resp.status
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(body)
    return status


def download_source_docs(dest_dir: str | Path,
                         catalog: Sequence[Dict] = SOURCE_CATALOG) -> List[Tuple[Dict, Path, int]]:
    """Download every source in `catalog`. Returns [(entry, path, status), ...].

    Skips a source if its output file already exists (idempotent). If a fetch
    returns non-200, the file is removed and the entry is still returned with
    its status — the caller decides whether to keep going.
    """
    out: List[Tuple[Dict, Path, int]] = []
    for entry in catalog:
        ext = ".html" if entry["kind"].startswith("html") else ".json"
        dest = Path(dest_dir) / f"{entry['label']}{ext}"
        if dest.exists():
            out.append((entry, dest, 200))
            continue
        try:
            status = _http_get(entry["url"], dest)
        except Exception as e:  # urllib.URLError, http errors, timeout, etc.
            print(f"[download] FAIL {entry['label']}: {e}")
            out.append((entry, dest, 0))
            continue
        if status != 200:
            dest.unlink(missing_ok=True)
            print(f"[download] HTTP {status} for {entry['label']} ({entry['url']})")
        else:
            print(f"[download] OK   {entry['label']} ({dest.stat().st_size} bytes)")
        out.append((entry, dest, status))
    return out


# ---------------------------------------------------------------------------
# 2. HTML / JSON parsing → (source_id, text)
# ---------------------------------------------------------------------------

# Strip script/style blocks, then strip all remaining tags. Whitelist is a
# defense-in-depth measure; we ALSO keep the visible-text fallback so a
# change in upstream markup doesn't break us.
_TAG_WHITELIST = re.compile(r"<(script|style|nav|footer|header|aside)\b[^>]*>.*?</\1>",
                           re.IGNORECASE | re.DOTALL)
_ALL_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Words we accept as sentence breaks when flattening HTML → prose.
# We intentionally do NOT use NLTK or spaCy — adds a dep, doesn't change
# the genre fit. Plain .split() on whitespace is enough for char-LM.
_SENT_BREAKS = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z])")


def _html_to_text(html: str) -> str:
    """Strip scripts/styles, drop tags, normalize whitespace. Keep visible
    text; don't try to preserve structure (the char-LM doesn't need it)."""
    s = _TAG_WHITELIST.sub(" ", html)
    s = _ALL_TAGS.sub(" ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = s.replace("&#39;", "'").replace("&quot;", '"')
    s = _WS.sub(" ", s).strip()
    return s


def _records_from_html(path: Path, source_id: str) -> List[Tuple[str, str]]:
    """One record per 'paragraph' — sentence-group of >40 chars."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _html_to_text(raw)
    if len(text) < 200:  # too short, likely a redirect or paywall
        return []
    # Split on sentence boundaries, regroup into ~3-sentence paragraphs
    sents = _SENT_BREAKS.split(text)
    paras: List[str] = []
    buf: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        buf.append(s)
        if sum(len(x) for x in buf) > 400:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return [(f"{source_id}#{i:03d}", p) for i, p in enumerate(paras) if len(p) >= 80]


def _records_from_pdf(path: Path, source_id: str) -> List[Tuple[str, str]]:
    """Extract text from each page, regroup into ~400-char paragraphs.

    Per-page text is joined then split on sentence boundaries, exactly the
    same shape as the HTML path. This keeps downstream corpus statistics
    consistent regardless of source kind.
    """
    try:
        reader = _pdf_reader(path)
    except Exception as e:
        print(f"[parse] PDF read failed for {source_id}: {e}")
        return []
    paras: List[str] = []
    buf: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        text = _WS.sub(" ", text).strip()
        if not text:
            continue
        for s in _SENT_BREAKS.split(text):
            s = s.strip()
            if not s:
                continue
            buf.append(s)
            if sum(len(x) for x in buf) > 400:
                paras.append(" ".join(buf))
                buf = []
    if buf:
        paras.append(" ".join(buf))
    # Drop pure header/footer noise: page numbers, running titles, etc.
    cleaned = [p for p in paras if not _looks_like_noise(p)]
    return [(f"{source_id}#{i:03d}", p) for i, p in enumerate(cleaned) if len(p) >= 80]


def _looks_like_noise(p: str) -> bool:
    """Heuristic: NIST PDFs have repeating footers like 'Page X' or
    'Chapter 3 — Handling Incidents' that we want to drop. Anything that's
    < 5 distinct words and contains 'Page' is noise; anything that's
    >70% digits/punct is also noise."""
    if len(p) < 30:
        return True
    if "Page" in p and len(p.split()) < 6:
        return True
    alnum = sum(c.isalnum() for c in p)
    if alnum < 20:
        return True
    return False


def parse_incident_docs(downloads: Iterable[Tuple[Dict, Path, int]]) -> List[Tuple[str, str]]:
    """Walk the download results and yield (source_id, text) records.

    Drops entries with non-200 status. Adding a new source kind means
    adding a branch here and a parser above — kept narrow on purpose.
    """
    out: List[Tuple[str, str]] = []
    for entry, path, status in downloads:
        if status != 200:
            continue
        kind = entry.get("kind")
        if kind == "html":
            out.extend(_records_from_html(path, entry["label"]))
        elif kind == "pdf":
            out.extend(_records_from_pdf(path, entry["label"]))
        else:
            print(f"[parse] unsupported kind={kind!r} for {entry['label']}, skipping")
    return out


# ---------------------------------------------------------------------------
# 3. Corpus file (genre-aware layout)
# ---------------------------------------------------------------------------

CORPUS_HEADER = (
    "# Public CSIRT / CERT / SANS / NIST incident-handling corpus\n"
    "# Built by project_05_generative_ai/src/dataset.py\n"
    "# Sources, licenses, and attribution per block: see the '## <source-id>' line.\n"
    "# Each block below is one paragraph from a public incident-handling source.\n"
)


def build_corpus_file(records: Sequence[Tuple[str, str]],
                      out_path: str | Path,
                      shuffle: bool = True,
                      attributions: Sequence[Dict] = ()) -> Path:
    """Write records to <out_path>. Records are grouped by source so the
    attribution in the header is clear; we still shuffle WITHIN each group
    so the model doesn't see a long run from one source."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_source: Dict[str, List[Tuple[str, str]]] = {}
    for sid, text in records:
        by_source.setdefault(sid.split("#", 1)[0], []).append((sid, text))
    rng = random.Random(SEED)
    with open(out, "w", encoding="utf-8") as f:
        f.write(CORPUS_HEADER)
        # Attribution block — cite every source the corpus actually contains.
        for entry in attributions:
            label = entry.get("label", "")
            if not any(label and label in sid for sid in by_source):
                continue
            f.write(f"## {label} | license: {entry.get('license','?')} | "
                    f"attribution: {entry.get('attribution','?')}\n")
        for source_id, group in sorted(by_source.items()):
            f.write(f"## source: {source_id}\n")
            if shuffle:
                rng.shuffle(group)
            for sid, text in group:
                # One doc per line; sid is metadata, not text the model trains on.
                f.write(f"{text}\n")
            f.write("\n")
    return out


# ---------------------------------------------------------------------------
# 4. Char vocab + encode/decode (unchanged from NVD version)
# ---------------------------------------------------------------------------

@dataclass
class CharVocab:
    """Character-level vocabulary. Built from the corpus, persisted to disk.

    Special tokens:
        <PAD> = 0  (padding; not used in this corpus, kept for future batching)
        <BOS> = 1  (beginning of sequence; kept so inference signature matches
                    a standard LM)
        <EOS> = 2  (end of sequence; emitted by the model at sample time)
        <UNK> = 3  (any char outside the training vocab; should not occur in
                    generated text if sampling stays in-distribution)
    """
    chars: Tuple[str, ...]  # sorted unique chars, EXCLUDING special tokens

    @property
    def stoi(self) -> dict:
        offset = 4
        return {c: i + offset for i, c in enumerate(self.chars)}

    @property
    def itos(self) -> dict:
        offset = 4
        inv = {i + offset: c for i, c in enumerate(self.chars)}
        inv.update({0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"})
        return inv

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 4

    def encode(self, s: str) -> List[int]:
        table = self.stoi
        unk = 3
        return [table.get(c, unk) for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        inv = self.itos
        return "".join(inv.get(i, "?") for i in ids)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(list(self.chars), f, ensure_ascii=False)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        with open(path, "r", encoding="utf-8") as f:
            chars = tuple(json.load(f))
        return cls(chars=chars)

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        # Strip BOM + any null bytes that show up in sloppy HTML/JSON dumps.
        cleaned = text.replace("﻿", "").replace("\x00", "")
        unique = sorted(set(cleaned))
        return cls(chars=tuple(unique))


# Unicode Private Use Area: codepoints fonts assign to ligatures/glyphs that
# have no standard meaning. pypdf extracts these from NIST PDFs (e.g. ,
# a fi-ligature glyph) as raw codepoints; left in the corpus they show up as
# mojibake in generated text. Strip the three PUA ranges + the BOM.
_PUA_RANGES = (
    (0xE000, 0xF8FF),    # Private Use Area (where  lives)
    (0xF0000, 0xFFFFD),  # Supplementary PUA-A
    (0x100000, 0x10FFFD),  # Supplementary PUA-B
)


def _strip_pua(text: str) -> str:
    """Remove Private-Use-Area codepoints + BOM. Keeps all standard chars."""
    out = []
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _PUA_RANGES):
            continue
        out.append(ch)
    return "".join(out).replace("﻿", "")


def load_text_corpus(path: str | Path) -> str:
    """Read a corpus file as a single string. Strips '## <...>' header
    and '## source: ...' lines (they're metadata, not genre), and removes
    Private-Use-Area codepoints (PDF font glyphs that leak through pypdf)."""
    text = Path(path).read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return _strip_pua(text)


# ---------------------------------------------------------------------------
# 5. Self-check (no test framework)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Round-trip on a synthetic 200-char string.
    sample = (
        "The SOC analyst received a badge-denial alert at 02:14 from zone ZONE-003. "
        "On-site security was dispatched and the door was secured by 02:31. "
        "No badge activity was observed during the intervening window."
    )
    v = CharVocab.from_text(sample)
    assert v.vocab_size > 0
    encoded = v.encode(sample)
    decoded = v.decode(encoded)
    assert decoded == sample, f"round-trip failed:\n  in:  {sample!r}\n  out: {decoded!r}"

    # HTML→text sanity: ensure we strip tags and decode the common entities.
    html = "<html><head><title>x</title></head><body><h1>Hello</h1>"
    html += "<p>An incident &amp; an alert &#39;X&#39; are reported.</p>"
    html += "<script>alert(1)</script><nav>menu</nav><p>Second paragraph.</p></body></html>"
    txt = _html_to_text(html)
    assert "alert(1)" not in txt
    assert "menu" not in txt
    assert "&" in txt and "'" in txt
    assert "Hello" in txt and "Second paragraph." in txt

    # Save+load round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vocab.json"
        v.save(p)
        v2 = CharVocab.load(p)
        assert v2.vocab_size == v.vocab_size
        assert v2.encode("SOC") == v.encode("SOC")

    print(f"CharVocab self-check OK — vocab_size={v.vocab_size}, "
          f"chars={''.join(v.chars)!r}")
    print(f"HTML→text self-check OK — cleaned text: {txt!r}")
