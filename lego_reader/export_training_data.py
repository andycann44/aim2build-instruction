from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


VALID_LABELS = ("true_bag_start", "sticker_or_callout", "normal_step")
DEFAULT_LABELS: dict[tuple[str, str], dict[str, list[int]]] = {
    (
        "21330",
        "21330_01",
    ): {
        "true_bag_start": [13, 373, 433],
        "sticker_or_callout": [407, 110],
        "normal_step": [181, 192, 199],
    }
}
CROP_WIDTH_RATIO = 0.60
CROP_HEIGHT_RATIO = 0.40
CROP_NAMES = ("top_left", "top_right", "top_center")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pages(raw_pages: str) -> list[int]:
    pages: set[int] = set()
    for part in raw_pages.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise SystemExit(f"Invalid page number: {token!r}")
        page_number = int(token)
        if page_number < 1:
            raise SystemExit(f"Page number must be positive: {token!r}")
        pages.add(page_number)
    return sorted(pages)


def _parse_label_specs(raw_specs: list[str]) -> dict[str, list[int]]:
    labels: dict[str, set[int]] = {}
    for spec in raw_specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --label value: {spec!r}. Expected label=page,page")
        label, raw_pages = spec.split("=", 1)
        label = label.strip()
        if label not in VALID_LABELS:
            raise SystemExit(f"Unsupported label: {label!r}")
        labels.setdefault(label, set()).update(_parse_pages(raw_pages))
    return {label: sorted(page_numbers) for label, page_numbers in labels.items()}


def _resolve_labels(set_num: str, pdf_name: str, raw_specs: list[str]) -> dict[str, list[int]]:
    if raw_specs:
        return _parse_label_specs(raw_specs)

    default = DEFAULT_LABELS.get((set_num, pdf_name))
    if default:
        return {label: list(page_numbers) for label, page_numbers in default.items()}

    raise SystemExit(
        "No labels supplied. Use --label true_bag_start=13,373 or add a manual default mapping in export_training_data.py."
    )


def _candidate_json_path(debug_pdf_dir: Path, page_number: int) -> Path:
    return debug_pdf_dir / "candidates" / f"candidate_page_{page_number:03d}.json"


def _page_image_path(debug_pdf_dir: Path, page_number: int) -> Path:
    return debug_pdf_dir / "pages" / f"page_{page_number:03d}.png"


def _resolve_source_image(debug_pdf_dir: Path, page_number: int, payload: dict[str, Any]) -> Path | None:
    page_image = _page_image_path(debug_pdf_dir, page_number)
    if page_image.exists():
        return page_image

    image_path = payload.get("source_image_path") or payload.get("image_path")
    if image_path:
        candidate = Path(str(image_path))
        if candidate.exists():
            return candidate

    candidate_png = _candidate_json_path(debug_pdf_dir, page_number).with_suffix(".png")
    if candidate_png.exists():
        return candidate_png

    return None


def _crop_box(image_width: int, image_height: int, crop_name: str) -> tuple[int, int, int, int]:
    crop_width = max(1, int(image_width * CROP_WIDTH_RATIO))
    crop_height = max(1, int(image_height * CROP_HEIGHT_RATIO))
    y0 = 0

    if crop_name == "top_left":
        x0 = 0
    elif crop_name == "top_right":
        x0 = max(0, image_width - crop_width)
    elif crop_name == "top_center":
        x0 = max(0, (image_width - crop_width) // 2)
    else:
        raise ValueError(f"Unsupported crop name: {crop_name}")

    x1 = min(image_width, x0 + crop_width)
    y1 = min(image_height, y0 + crop_height)
    return (x0, y0, x1, y1)


def _export_crops(source_image: Path, destination_dir: Path, stem: str) -> tuple[dict[str, Path | None], str | None]:
    try:
        from PIL import Image
    except ImportError:
        return {crop_name: None for crop_name in CROP_NAMES}, "pillow unavailable"

    exported: dict[str, Path | None] = {crop_name: None for crop_name in CROP_NAMES}
    with Image.open(source_image) as image:
        suffix = source_image.suffix or ".png"
        for crop_name in CROP_NAMES:
            destination = destination_dir / f"{stem}_{crop_name}{suffix}"
            cropped = image.crop(_crop_box(image.width, image.height, crop_name))
            cropped.save(destination)
            exported[crop_name] = destination

    return exported, None


def export_training_data(args: argparse.Namespace) -> int:
    debug_root = Path(args.debug_root).expanduser().resolve()
    debug_pdf_dir = debug_root / args.set_num / args.pdf_name
    if not debug_pdf_dir.is_dir():
        raise SystemExit(f"Debug PDF directory not found: {debug_pdf_dir}")

    labels = _resolve_labels(args.set_num, args.pdf_name, args.label)
    output_root = Path(args.output_root).expanduser().resolve()
    exported_count = 0

    for label, page_numbers in labels.items():
        destination_dir = ensure_dir(output_root / label)
        for page_number in page_numbers:
            candidate_json = _candidate_json_path(debug_pdf_dir, page_number)
            payload = _load_json(candidate_json) if candidate_json.exists() else {}
            source_image = _resolve_source_image(debug_pdf_dir, page_number, payload)
            if source_image is None:
                raise SystemExit(f"No source image found for page {page_number} in {debug_pdf_dir}")

            stem = f"{args.set_num}_{args.pdf_name}_page_{page_number:03d}"
            exported_image = destination_dir / f"{stem}{source_image.suffix or '.png'}"
            exported_json = destination_dir / f"{stem}.json"
            shutil.copy2(source_image, exported_image)

            crop_paths: dict[str, Path | None] = {crop_name: None for crop_name in CROP_NAMES}
            crop_note: str | None = None
            if not args.no_crop:
                crop_paths, crop_note = _export_crops(source_image, destination_dir, stem)

            training_payload: dict[str, Any] = {
                "set_num": args.set_num,
                "pdf_name": args.pdf_name,
                "page_number": page_number,
                "label": label,
                "image_path": exported_image.as_posix(),
                "top_left_crop_path": crop_paths["top_left"].as_posix() if crop_paths["top_left"] else None,
                "top_right_crop_path": crop_paths["top_right"].as_posix() if crop_paths["top_right"] else None,
                "top_center_crop_path": crop_paths["top_center"].as_posix() if crop_paths["top_center"] else None,
                "source_image_path": source_image.as_posix(),
            }
            if candidate_json.exists():
                training_payload["source_candidate_json"] = candidate_json.as_posix()
            if crop_note:
                training_payload["crop_note"] = crop_note

            write_json(exported_json, training_payload)
            print(exported_json.as_posix())
            exported_count += 1

    print(f"Exported {exported_count} training example(s) to {output_root.as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export labeled training data from existing debug page renders.")
    parser.add_argument("--set-num", required=True, help="LEGO set number, for example 21330")
    parser.add_argument("--pdf-name", required=True, help="PDF folder name under debug/<set>, for example 21330_01")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Manual label specification like true_bag_start=13,373,433. Repeat for multiple labels.",
    )
    parser.add_argument("--debug-root", default="debug", help="Root directory containing rendered debug pages.")
    parser.add_argument("--output-root", default="training_data", help="Output directory for labeled training data.")
    parser.add_argument("--no-crop", action="store_true", help="Skip optional top-region crop export.")
    parser.set_defaults(func=export_training_data)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
