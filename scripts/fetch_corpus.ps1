# scripts/fetch_corpus.ps1
#
# Live corpus build for P5: downloads the public CSIRT/CERT/NIST catalog,
# parses to (source_id, text) records, writes data/csops_corpus.txt.
#
# Idempotent: the downloader skips files that already exist. The first
# `rm -rf` is just for a clean re-run with the latest catalog.
#
# Run from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts/fetch_corpus.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$venvPy = Join-Path $PSScriptRoot "..\generative_env\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "generative_env not found at $venvPy. Create the venv first."
}

# Clean re-run. Comment out the next two lines for incremental re-fetches.
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "data\csops_raw"
Remove-Item -Force -ErrorAction SilentlyContinue "data\csops_corpus.txt"

& $venvPy -X utf8 -c @"
import sys
sys.path.insert(0, 'src')
import dataset as d
dl = d.download_source_docs('data/csops_raw', catalog=d.SOURCE_CATALOG)
print('---')
recs = d.parse_incident_docs(dl)
print(f'records: {len(recs)}, chars: {sum(len(t) for _,t in recs):,}')
d.build_corpus_file(
    recs, 'data/csops_corpus.txt',
    shuffle=True, attributions=d.SOURCE_CATALOG,
)
print('corpus written: data/csops_corpus.txt')
"@
