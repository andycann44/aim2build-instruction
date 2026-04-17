from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_dir


CROP_FIELDS = (
    "top_left_crop_path",
    "top_right_crop_path",
    "top_center_crop_path",
)
VALID_LABELS = ("true_bag_start", "sticker_or_callout", "normal_step")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_crop_dataset(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not training_root.is_dir():
        raise SystemExit(f"Training data root not found: {training_root}")

    exported = 0
    sidecar_paths = sorted(path for path in training_root.glob("*/*.json") if path.is_file())
    for sidecar_path in sidecar_paths:
        payload = _load_json(sidecar_path)
        label = str(payload.get("label", "")).strip()
        if label not in VALID_LABELS:
            continue

        destination_dir = ensure_dir(output_root / label)
        for field_name in CROP_FIELDS:
            crop_path_value = payload.get(field_name)
            if not crop_path_value:
                continue

            crop_path = Path(str(crop_path_value))
            if not crop_path.is_file():
                continue

            destination_path = destination_dir / crop_path.name
            shutil.copy2(crop_path, destination_path)
            print(destination_path.as_posix())
            exported += 1

    print(f"Exported {exported} crop image(s) to {output_root.as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy labeled crop images into a classifier-ready dataset tree.")
    parser.add_argument("--training-root", default="training_data", help="Root directory containing training sidecars.")
    parser.add_argument("--output-root", default="training_data_crops", help="Output directory for crop dataset.")
    parser.set_defaults(func=export_crop_dataset)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
