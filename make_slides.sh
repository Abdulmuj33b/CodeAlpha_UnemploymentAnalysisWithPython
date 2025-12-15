#!/usr/bin/env bash
# Optional script to convert SLIDES.md to PDF and PPTX using pandoc
# Requires: pandoc and a PDF engine (wkhtmltopdf or LaTeX) for PDF output

set -euo pipefail

MD_FILE="SLIDES.md"
OUT_PDF="slides.pdf"
OUT_PPTX="slides.pptx"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Pandoc not found. Install pandoc to enable automatic slide conversion." >&2
  exit 1
fi

echo "Converting $MD_FILE to PowerPoint ($OUT_PPTX)"
pandoc --standalone -t pptx -o "$OUT_PPTX" "$MD_FILE"

if command -v wkhtmltopdf >/dev/null 2>&1 || command -v pdflatex >/dev/null 2>&1; then
  echo "Converting $MD_FILE to PDF ($OUT_PDF)"
  pandoc -t beamer -o "$OUT_PDF" "$MD_FILE"
else
  echo "No PDF engine found (wkhtmltopdf or pdflatex). Skipping PDF generation." >&2
fi

echo "Done."
