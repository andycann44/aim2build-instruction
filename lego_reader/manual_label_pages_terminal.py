from __future__ import annotations

import argparse
import json
import re
import subprocess
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


PAGE_RE = re.compile(r"^page_(\d+)\.png$")
KEY_TO_LABEL = {
    "b": "true_bag_start",
    "s": "sticker_or_callout",
    "n": "normal_step",
}
VALID_LABELS = set(KEY_TO_LABEL.values())
UNSURE_KEY = "u"
QUIT_KEY = "q"
DEFAULT_DEBUG_ROOT = "debug"
DEFAULT_LABELS_ROOT = "manual_labels"


LABEL_ENTRY_RE = re.compile(r'"(\d+)"\s*:\s*"([a-z_]+)"')
SET_NUM_RE = re.compile(r'"set_num"\s*:\s*"([^"]+)"')
PDF_NAME_RE = re.compile(r'"pdf_name"\s*:\s*"([^"]+)"')
UNSURE_BLOCK_RE = re.compile(r'"unsure_pages"\s*:\s*\[(.*?)\]', re.DOTALL)
PAGE_TOKEN_RE = re.compile(r'"(\d+)"')


def _recover_label_state_from_text(raw_text: str, set_num: str, pdf_name: str) -> dict[str, Any]:
    labels = {
        page: label
        for page, label in LABEL_ENTRY_RE.findall(raw_text)
        if label in VALID_LABELS
    }

    unsure_pages: list[str] = []
    unsure_match = UNSURE_BLOCK_RE.search(raw_text)
    if unsure_match:
        unsure_pages = sorted(set(PAGE_TOKEN_RE.findall(unsure_match.group(1))), key=int)

    recovered_set_num = SET_NUM_RE.search(raw_text)
    recovered_pdf_name = PDF_NAME_RE.search(raw_text)
    return {
        "set_num": recovered_set_num.group(1) if recovered_set_num else set_num,
        "pdf_name": recovered_pdf_name.group(1) if recovered_pdf_name else pdf_name,
        "labels": labels,
        "unsure_pages": unsure_pages,
    }


def _page_number(path: Path) -> int:
    match = PAGE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected page filename: {path.name}")
    return int(match.group(1))


def _load_label_state(labels_path: Path, set_num: str, pdf_name: str) -> dict[str, Any]:
    if labels_path.exists():
        raw_text = labels_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except JSONDecodeError:
            payload = _recover_label_state_from_text(raw_text, set_num, pdf_name)
            print(f"Warning: recovered labels from malformed JSON in {labels_path}")
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid labels file format: {labels_path}")
    else:
        payload = {}

    payload["set_num"] = set_num
    payload["pdf_name"] = pdf_name

    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, dict):
        raw_labels = {}
    payload["labels"] = {
        str(page): label
        for page, label in raw_labels.items()
        if isinstance(label, str) and label in VALID_LABELS
    }

    raw_unsure = payload.get("unsure_pages", [])
    if not isinstance(raw_unsure, list):
        raw_unsure = []
    payload["unsure_pages"] = sorted(
        {
            str(page)
            for page in raw_unsure
            if isinstance(page, (str, int)) and str(page).isdigit()
        },
        key=int,
    )
    return payload


def _save_label_state(labels_path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(labels_path.parent)
    write_json(labels_path, payload)


def _page_paths(debug_pdf_dir: Path, start_page: int | None, end_page: int | None) -> list[Path]:
    pages_dir = debug_pdf_dir / "pages"
    if not pages_dir.is_dir():
        raise SystemExit(f"Page images directory not found: {pages_dir}")

    paths: list[Path] = []
    for path in sorted(pages_dir.glob("page_*.png"), key=_page_number):
        page_num = _page_number(path)
        if start_page is not None and page_num < start_page:
            continue
        if end_page is not None and page_num > end_page:
            continue
        paths.append(path)
    return paths


def _pending_pages(page_paths: list[Path], payload: dict[str, Any]) -> list[Path]:
    labels = payload.get("labels", {})
    unsure_pages = set(payload.get("unsure_pages", []))
    return [
        path
        for path in page_paths
        if str(_page_number(path)) not in labels and str(_page_number(path)) not in unsure_pages
    ]


def _dry_run_summary(labels_path: Path, payload: dict[str, Any], pending_pages: list[Path], page_paths: list[Path]) -> int:
    summary = {
        "labels_path": labels_path.as_posix(),
        "total_pages_in_range": len(page_paths),
        "labeled_pages": len(payload.get("labels", {})),
        "unsure_pages": len(payload.get("unsure_pages", [])),
        "pending_pages": len(pending_pages),
        "next_page": _page_number(pending_pages[0]) if pending_pages else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _open_in_preview(image_path: Path) -> None:
    try:
        subprocess.run(
            ["open", "-a", "Preview", image_path.as_posix()],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise SystemExit("Could not open the page image in Preview.") from exc


def _save_choice(payload: dict[str, Any], page_num: int, label: str | None) -> None:
    labels = payload.setdefault("labels", {})
    unsure_pages = set(payload.setdefault("unsure_pages", []))
    page_key = str(page_num)

    if label is None:
        labels.pop(page_key, None)
        unsure_pages.add(page_key)
    else:
        labels[page_key] = label
        unsure_pages.discard(page_key)

    payload["labels"] = labels
    payload["unsure_pages"] = sorted(unsure_pages, key=int)


def _prompt_choice(page_num: int, image_path: Path) -> str:
    print()
    print(f"Page {page_num}: {image_path}")
    print("[b] true_bag_start")
    print("[s] sticker_or_callout")
    print("[n] normal_step")
    print("[u] unsure")
    print("[q] quit")
    while True:
        choice = input("Choice: ").strip().lower()
        if choice in set(KEY_TO_LABEL) | {UNSURE_KEY, QUIT_KEY}:
            return choice
        print("Enter one of: b, s, n, u, q")


def manual_label_pages_terminal(args: argparse.Namespace) -> int:
    debug_root = Path(args.debug_root).expanduser().resolve()
    debug_pdf_dir = debug_root / args.set_num / args.pdf_name
    if not debug_pdf_dir.is_dir():
        raise SystemExit(f"Debug PDF directory not found: {debug_pdf_dir}")

    labels_root = Path(args.labels_root).expanduser().resolve()
    labels_path = labels_root / f"{args.set_num}_{args.pdf_name}.json"
    payload = _load_label_state(labels_path, args.set_num, args.pdf_name)
    page_paths = _page_paths(debug_pdf_dir, args.start_page, args.end_page)
    pending_pages = _pending_pages(page_paths, payload)

    if args.dry_run:
        return _dry_run_summary(labels_path, payload, pending_pages, page_paths)

    if not page_paths:
        raise SystemExit("No page images found for the requested range.")
    if not pending_pages:
        print(f"All pages in range are already labeled or marked unsure. Labels file: {labels_path}")
        return 0

    for index, image_path in enumerate(pending_pages, start=1):
        page_num = _page_number(image_path)
        labeled_count = len(payload.get("labels", {}))
        unsure_count = len(payload.get("unsure_pages", []))
        remaining = len(pending_pages) - index + 1

        print(
            f"\nSet {args.set_num} | PDF {args.pdf_name} | Page {page_num} | "
            f"Remaining {remaining} | Saved {labeled_count} | Unsure {unsure_count}"
        )
        _open_in_preview(image_path)
        choice = _prompt_choice(page_num, image_path)

        if choice == QUIT_KEY:
            _save_label_state(labels_path, payload, args.dry_run)
            print(f"Stopped. Labels saved to {labels_path}")
            return 0
        if choice == UNSURE_KEY:
            _save_choice(payload, page_num, None)
        else:
            _save_choice(payload, page_num, KEY_TO_LABEL[choice])

        _save_label_state(labels_path, payload, args.dry_run)
        print(f"Saved page {page_num} -> {payload['labels'].get(str(page_num), 'unsure')}")

    print(f"Done. Labels saved to {labels_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open rendered debug pages in Preview and save labels from terminal prompts.")
    parser.add_argument("--set-num", required=True, help="LEGO set number, for example 21330")
    parser.add_argument("--pdf-name", required=True, help="PDF folder name under debug/<set>, for example 21330_01")
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT, help="Root directory containing rendered debug page images.")
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT, help="Directory where manual label JSON files are saved.")
    parser.add_argument("--start-page", type=int, default=None, help="Optional first page number to include.")
    parser.add_argument("--end-page", type=int, default=None, help="Optional last page number to include.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary and labels path without opening Preview.")
    parser.set_defaults(func=manual_label_pages_terminal)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
