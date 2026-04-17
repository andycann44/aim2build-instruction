from __future__ import annotations

import argparse
import json
import math
import re
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


def _page_number(path: Path) -> int:
    match = PAGE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected page filename: {path.name}")
    return int(match.group(1))


def _load_label_state(labels_path: Path, set_num: str, pdf_name: str) -> dict[str, Any]:
    if labels_path.exists():
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
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


def _scaled_photo(tk_module: Any, image_path: Path, max_width: int, max_height: int) -> tuple[Any, Any]:
    original = tk_module.PhotoImage(file=image_path.as_posix())
    width = max(1, original.width())
    height = max(1, original.height())
    factor = max(1, math.ceil(width / max_width), math.ceil(height / max_height))
    if factor > 1:
        displayed = original.subsample(factor, factor)
    else:
        displayed = original
    return original, displayed


def manual_label_pages(args: argparse.Namespace) -> int:
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

    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("tkinter is unavailable on this Python install; cannot open the manual labeling viewer.") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit("Could not open the manual labeling viewer window.") from exc

    root.configure(background="white")
    info_var = tk.StringVar()
    hint_var = tk.StringVar(
        value="Keys: b=true_bag_start | s=sticker_or_callout | n=normal_step | u=skip/unsure | q=quit"
    )
    image_label = tk.Label(root, bg="white")
    info_label = tk.Label(root, textvariable=info_var, bg="white", font=("Helvetica", 14, "bold"))
    hint_label = tk.Label(root, textvariable=hint_var, bg="white", font=("Helvetica", 12))
    info_label.pack(padx=12, pady=(12, 6))
    image_label.pack(padx=12, pady=6, expand=True)
    hint_label.pack(padx=12, pady=(6, 12))

    state = {"index": 0}

    def save_label(page_num: int, label: str | None) -> None:
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
        _save_label_state(labels_path, payload, args.dry_run)

    def show_current() -> None:
        if state["index"] >= len(pending_pages):
            root.title("Manual labeling complete")
            info_var.set(f"Done. Labels saved to {labels_path}")
            image_label.configure(image="")
            image_label.image = None
            image_label.original_image = None
            return

        image_path = pending_pages[state["index"]]
        page_num = _page_number(image_path)
        screen_w = max(640, root.winfo_screenwidth() - 120)
        screen_h = max(480, root.winfo_screenheight() - 220)
        original, displayed = _scaled_photo(tk, image_path, screen_w, screen_h)
        image_label.configure(image=displayed)
        image_label.image = displayed
        image_label.original_image = original

        labeled_count = len(payload.get("labels", {}))
        unsure_count = len(payload.get("unsure_pages", []))
        info_var.set(
            f"Set {args.set_num} | PDF {args.pdf_name} | Page {page_num} | Remaining {len(pending_pages) - state['index']} | "
            f"Saved {labeled_count} | Unsure {unsure_count}"
        )
        root.title(f"Label page {page_num} ({state['index'] + 1}/{len(pending_pages)})")

    def on_key(event: Any) -> None:
        key = (event.keysym or "").lower()
        if key == QUIT_KEY:
            _save_label_state(labels_path, payload, args.dry_run)
            root.destroy()
            return
        if state["index"] >= len(pending_pages):
            if key in {"return", "space", QUIT_KEY}:
                root.destroy()
            return

        page_num = _page_number(pending_pages[state["index"]])
        if key in KEY_TO_LABEL:
            save_label(page_num, KEY_TO_LABEL[key])
        elif key == UNSURE_KEY:
            save_label(page_num, None)
        else:
            return

        state["index"] += 1
        show_current()

    root.bind("<Key>", on_key)
    show_current()
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label rendered debug pages one at a time with keyboard shortcuts.")
    parser.add_argument("--set-num", required=True, help="LEGO set number, for example 21330")
    parser.add_argument("--pdf-name", required=True, help="PDF folder name under debug/<set>, for example 21330_01")
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT, help="Root directory containing rendered debug page images.")
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT, help="Directory where manual label JSON files are saved.")
    parser.add_argument("--start-page", type=int, default=None, help="Optional first page number to include.")
    parser.add_argument("--end-page", type=int, default=None, help="Optional last page number to include.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary and labels path without opening the viewer.")
    parser.set_defaults(func=manual_label_pages)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
