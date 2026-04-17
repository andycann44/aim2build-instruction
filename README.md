# LEGO Instruction Manual Reader v1

Standalone Python CLI for downloading official LEGO instruction PDFs and estimating bag-section starts.

## What it does

- accepts a LEGO set number
- downloads one or more official instruction PDFs from LEGO when available
- scans every page with PyMuPDF
- estimates bag overview pages and bag-start pages
- writes JSON output to `output/<set_num>.json`
- exports reviewable candidate snippets when `--debug` is enabled

## Setup

```bash
cd /Users/olly/aim2build-instruction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m lego_reader.cli 21330
```

With debug snippet exports:

```bash
python -m lego_reader.cli 21330 --debug
```

Custom directories:

```bash
python -m lego_reader.cli 21330 --out output --instructions-dir instructions --debug-dir debug --debug
```

## Smoke test

```bash
python -m lego_reader.cli --help
python -m lego_reader.review_snippets --help
```

## Output

The CLI writes JSON like this to `output/21330.json`:

```json
{
  "set_num": "21330",
  "source_url": "https://www.lego.com/en-gb/service/building-instructions/21330",
  "page_title": "Building Instructions - Download",
  "downloaded_pdf_paths": [
    "/Users/olly/aim2build-instruction/instructions/21330/21330_01.pdf"
  ],
  "pdfs": [
    {
      "title": "Download",
      "source_url": "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6447727.pdf",
      "file": "/Users/olly/aim2build-instruction/instructions/21330/21330_01.pdf",
      "bag_count": 24,
      "bag_count_estimate": 24,
      "bag_start_pages": [13, 373, 433],
      "bags": [
        {"bag": 1, "start_page": 13, "confidence": 0.85},
        {"bag": 21, "start_page": 373, "confidence": 0.85},
        {"bag": 24, "start_page": 433, "confidence": 0.85}
      ],
      "overview_pages": [11, 12],
      "bag_start_like_pages": [13, 373, 433],
      "uncertain_pages": [58, 82]
    }
  ],
  "total_bags_detected": 24
}
```

## Snippet review workflow

When `--debug` is on, candidate pages are exported into the `candidates/` folder as a snippet pair:

- `candidate_page_013.png`
- `candidate_page_013.json`

The sidecar JSON includes:

- page metrics (`word_count`, `drawing_count`, `image_count`, `dark_pixel_ratio`)
- detected numbers
- current classification
- acceptance flag and reasons
- image path to the exported snippet image

Training folders are created here:

- `training/accepted/bag_start`
- `training/accepted/overview`
- `training/accepted/normal_step`
- `training/rejected`

Promote a reviewed snippet into the training set:

```bash
python -m lego_reader.review_snippets promote \
  debug/21330/21330_01/candidates/candidate_page_013.json \
  bag_start
```

Move instead of copy:

```bash
python -m lego_reader.review_snippets promote \
  debug/21330/21330_01/candidates/candidate_page_013.json \
  bag_start \
  --action move
```

Compare a future candidate against accepted snippets:

```bash
python -m lego_reader.review_snippets compare \
  debug/21330/21330_01/candidates/candidate_page_013.json
```

This comparison mode uses the saved page metrics only. It does not use OCR.

## Notes

- Candidate snippet export happens only when `--debug` is enabled.
- The snippet review system is focused on improving bag and overview detection confidence.
- No OCR is used.
- No part recognition or color recognition is implemented.
