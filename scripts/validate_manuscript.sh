#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

required_files=(
  paper.org
  references.bib
  figures/screening/01_normalized_time.pdf
  figures/screening/02_voltage_rate_vs_voltage.pdf
  figures/scaling_v2/05_scaled_rate_residual.pdf
  figures/scaling_v4/12_monotone_ratio_bootstrap.pdf
  figures/scaling_v3/11_parameter_sensitivity.pdf
  figures/scaling_v5/18_drift_vs_scaling_residual.pdf
  figures/scaling_v5/19_leave_one_out_alpha.pdf
)

missing=0
for f in "${required_files[@]}"; do
  if [[ ! -s "$f" ]]; then
    printf 'MISSING: %s\n' "$f" >&2
    missing=1
  fi
done
if (( missing )); then
  cat >&2 <<'EOF'

Generate the manuscript figures first:
  uv run python scripts/01_screen_temporal_scaling.py
  uv run python scripts/02_temporal_scaling_v2.py
  uv run python scripts/03_temporal_scaling_v3.py
  uv run python scripts/04_temporal_scaling_v4.py
  uv run python scripts/05_order_confounding_v5.py
EOF
  exit 1
fi

if grep -nE '/Stage [0-9]|To be confirmed|To be completed|FINAL wording' paper.org; then
  echo "Draft placeholder(s) remain in paper.org" >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path
import re

text = Path("paper.org").read_text(encoding="utf-8")
bib = Path("references.bib").read_text(encoding="utf-8")

keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
cited = set()
for block in re.findall(r"\[cite:([^\]]+)\]", text):
    cited.update(re.findall(r"@([A-Za-z0-9_:-]+)", block))

missing = sorted(cited - keys)
unused = sorted(keys - cited)

if missing:
    raise SystemExit("Missing bibliography keys: " + ", ".join(missing))
print(f"citation keys: {len(cited)}; bibliography entries: {len(keys)}")
if unused:
    print("unused bibliography entries:", ", ".join(unused))

names = re.findall(r"^#\+name:\s*(\S+)\s*$", text, flags=re.M | re.I)
dupes = sorted({x for x in names if names.count(x) > 1})
if dupes:
    raise SystemExit("Duplicate Org names: " + ", ".join(dupes))
print(f"Org named objects: {len(names)}; duplicate names: 0")
PY

if command -v emacs >/dev/null 2>&1; then
  emacs --batch --quick paper.org     --eval '(require (quote ox-latex))'     --eval '(org-latex-export-to-latex)'
  echo "Org -> LaTeX export: OK"
else
  echo "SKIP: emacs not found; Org -> LaTeX export not tested" >&2
fi

if [[ -f paper.tex ]] && command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
  echo "LaTeX -> PDF build: OK"
elif [[ -f paper.tex ]]; then
  echo "SKIP: latexmk not found; PDF build not tested" >&2
fi

echo "Stage-9 static validation: OK"
