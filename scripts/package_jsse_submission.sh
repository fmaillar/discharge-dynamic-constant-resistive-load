#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained, flat submission ZIP for
# Journal of Solid State Electrochemistry.
#
# The workflow mirrors the first-article submission package:
# - export/validate the manuscript first;
# - include paper.tex, paper.bbl, references.bib;
# - include biblatex.cfg when present;
# - include all manuscript figures as separate PDF files;
# - include optional submission material (cover letter, graphical abstract,
#   supplementary information) when present;
# - rewrite figure paths in paper.tex so the flat ZIP compiles by itself;
# - test-compile the staged package before creating the ZIP.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUBMISSION_DIR="$REPO_ROOT/submission"
STAGE_DIR="$SUBMISSION_DIR/jsse-package"
ZIP_PATH="$SUBMISSION_DIR/discharge-dynamic-jsse-submission.zip"

FIGURES=(
  "figures/screening/01_normalized_time.pdf"
  "figures/screening/02_voltage_rate_vs_voltage.pdf"
  "figures/scaling_v2/05_scaled_rate_residual.pdf"
  "figures/scaling_v4/12_monotone_ratio_bootstrap.pdf"
  "figures/scaling_v3/11_parameter_sensitivity.pdf"
  "figures/scaling_v5/18_drift_vs_scaling_residual.pdf"
  "figures/scaling_v5/19_leave_one_out_alpha.pdf"
)

echo "==> Validating and exporting manuscript"
bash scripts/validate_manuscript.sh

required=(
  "paper.tex"
  "paper.bbl"
  "references.bib"
)

for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: required file missing: $file" >&2
    exit 1
  fi
done

for file in "${FIGURES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: manuscript figure missing: $file" >&2
    exit 1
  fi
done

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

echo "==> Staging manuscript sources"
cp paper.tex paper.bbl references.bib "$STAGE_DIR/"

if [[ -f biblatex.cfg ]]; then
  cp biblatex.cfg "$STAGE_DIR/"
fi

echo "==> Staging figures"
for file in "${FIGURES[@]}"; do
  cp "$file" "$STAGE_DIR/$(basename "$file")"
done

# Flatten the figure paths in the exported TeX source.
python3 - "$STAGE_DIR/paper.tex" <<'PY'
from pathlib import Path
import sys

tex_path = Path(sys.argv[1])
text = tex_path.read_text(encoding="utf-8")

figure_paths = [
    "figures/screening/01_normalized_time.pdf",
    "figures/screening/02_voltage_rate_vs_voltage.pdf",
    "figures/scaling_v2/05_scaled_rate_residual.pdf",
    "figures/scaling_v4/12_monotone_ratio_bootstrap.pdf",
    "figures/scaling_v3/11_parameter_sensitivity.pdf",
    "figures/scaling_v5/18_drift_vs_scaling_residual.pdf",
    "figures/scaling_v5/19_leave_one_out_alpha.pdf",
]

for path in figure_paths:
    text = text.replace(path, Path(path).name)

if "figures/" in text:
    raise SystemExit("ERROR: unresolved figures/ path remains in staged paper.tex")

tex_path.write_text(text, encoding="utf-8")
PY

echo "==> Adding optional submission files when present"
optional_names=(
  "cover_letter.pdf"
  "cover_letter.docx"
  "graphical_abstract.pdf"
  "graphical_abstract.png"
  "supplementary_information.pdf"
  "supplementary_information.tex"
)

for name in "${optional_names[@]}"; do
  if [[ -f "$REPO_ROOT/$name" ]]; then
    cp "$REPO_ROOT/$name" "$STAGE_DIR/"
    echo "    + $name"
  elif [[ -f "$SUBMISSION_DIR/$name" ]]; then
    cp "$SUBMISSION_DIR/$name" "$STAGE_DIR/"
    echo "    + $name"
  fi
done

echo "==> Test-compiling the staged flat package"
(
  cd "$STAGE_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
)

# Keep only submission inputs, not temporary TeX build products.
rm -f   "$STAGE_DIR/paper.pdf"   "$STAGE_DIR/paper.aux"   "$STAGE_DIR/paper.log"   "$STAGE_DIR/paper.out"   "$STAGE_DIR/paper.bcf"   "$STAGE_DIR/paper.run.xml"   "$STAGE_DIR/paper.blg"   "$STAGE_DIR/paper.fdb_latexmk"   "$STAGE_DIR/paper.fls"

mkdir -p "$SUBMISSION_DIR"
rm -f "$ZIP_PATH"

echo "==> Creating submission ZIP"
(
  cd "$STAGE_DIR"
  zip -9 -q "$ZIP_PATH" ./*
)

echo
echo "Created:"
echo "  $ZIP_PATH"
echo
echo "Contents:"
(
  cd "$STAGE_DIR"
  printf '  %s\n' ./*
)
echo
echo "ZIP listing:"
unzip -l "$ZIP_PATH"
