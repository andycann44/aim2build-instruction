from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ensure_dir


LABELS = ("bag_start", "overview", "normal_step", "rejected")
FEATURE_SCALES = {
    "word_count": 50.0,
    "drawing_count": 150.0,
    "image_count": 100.0,
    "dark_pixel_ratio": 0.25,
    "detected_numbers_count": 15.0,
}
FEATURE_WEIGHTS = {
    "word_count": 0.22,
    "drawing_count": 0.18,
    "image_count": 0.18,
    "dark_pixel_ratio": 0.24,
    "detected_numbers_count": 0.18,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _snippet_id(candidate_json: Path) -> str:
    parts = candidate_json.parts
    if "debug" in parts:
        idx = parts.index("debug")
        if len(parts) > idx + 3:
            set_num = parts[idx + 1]
            pdf_name = parts[idx + 2]
            return f"{set_num}_{pdf_name}_{candidate_json.stem}"
    return candidate_json.stem


def _find_candidate_image(candidate_json: Path, payload: dict[str, Any]) -> Path | None:
    image_path = payload.get("image_path") or payload.get("source_image_path")
    if image_path:
        candidate = Path(image_path)
        if candidate.exists():
            return candidate
    sibling_png = candidate_json.with_suffix(".png")
    if sibling_png.exists():
        return sibling_png
    return None


def _destination_dir(training_root: Path, label: str) -> Path:
    if label == "rejected":
        return ensure_dir(training_root / "rejected")
    return ensure_dir(training_root / "accepted" / label)


def _feature_vector(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "word_count": float(payload.get("word_count", 0.0)),
        "drawing_count": float(payload.get("drawing_count", 0.0)),
        "image_count": float(payload.get("image_count", 0.0)),
        "dark_pixel_ratio": float(payload.get("dark_pixel_ratio", 1.0)),
        "detected_numbers_count": float(len(payload.get("detected_numbers", []))),
    }


def _similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_vec = _feature_vector(a)
    b_vec = _feature_vector(b)
    total = 0.0
    for key, weight in FEATURE_WEIGHTS.items():
        scale = FEATURE_SCALES[key]
        diff = abs(a_vec[key] - b_vec[key])
        component = max(0.0, 1.0 - min(diff / scale, 1.0))
        total += component * weight
    return round(total, 4)


def _accepted_training_jsons(training_root: Path) -> list[Path]:
    accepted_root = training_root / "accepted"
    if not accepted_root.exists():
        return []
    return sorted(path for path in accepted_root.glob("*/*.json") if path.is_file())


def promote_snippet(args: argparse.Namespace) -> int:
    candidate_json = Path(args.candidate_json).expanduser().resolve()
    if not candidate_json.is_file():
        raise SystemExit(f"Candidate JSON not found: {candidate_json}")

    payload = _load_json(candidate_json)
    image_path = _find_candidate_image(candidate_json, payload)
    training_root = Path(args.training_root).expanduser().resolve()
    destination_dir = _destination_dir(training_root, args.label)
    snippet_id = _snippet_id(candidate_json)

    destination_json = destination_dir / f"{snippet_id}.json"
    destination_image = destination_dir / f"{snippet_id}{image_path.suffix if image_path else '.png'}"

    review_meta = {
        "label": args.label,
        "action": args.action,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_candidate_json": candidate_json.as_posix(),
    }
    if image_path:
        review_meta["source_candidate_image"] = image_path.as_posix()

    payload["review"] = review_meta
    payload["image_path"] = destination_image.as_posix() if image_path else None

    ensure_dir(destination_dir)
    _write_json(destination_json, payload)

    if image_path:
        shutil.copy2(image_path, destination_image)

    if args.action == "move":
        candidate_json.unlink(missing_ok=True)
        if image_path and image_path.exists() and image_path.parent == candidate_json.parent:
            image_path.unlink(missing_ok=True)

    print(destination_json.as_posix())
    return 0


def compare_snippet(args: argparse.Namespace) -> int:
    candidate_json = Path(args.candidate_json).expanduser().resolve()
    if not candidate_json.is_file():
        raise SystemExit(f"Candidate JSON not found: {candidate_json}")

    training_root = Path(args.training_root).expanduser().resolve()
    accepted_paths = _accepted_training_jsons(training_root)
    candidate_payload = _load_json(candidate_json)

    if not accepted_paths:
        result = {
            "candidate_json": candidate_json.as_posix(),
            "matches": [],
            "label_scores": {},
            "message": "No accepted training snippets found yet.",
        }
        print(json.dumps(result, indent=2))
        return 0

    scored_matches: list[dict[str, Any]] = []
    for accepted_json in accepted_paths:
        accepted_payload = _load_json(accepted_json)
        label = accepted_json.parent.name
        score = _similarity(candidate_payload, accepted_payload)
        scored_matches.append(
            {
                "label": label,
                "score": score,
                "snippet_json": accepted_json.as_posix(),
                "classification": accepted_payload.get("classification"),
                "page_number": accepted_payload.get("page_number"),
            }
        )

    scored_matches.sort(key=lambda item: item["score"], reverse=True)
    top_matches = scored_matches[: args.top_k]

    label_scores: dict[str, float] = {}
    for label in sorted({match["label"] for match in scored_matches}):
        label_specific = [match["score"] for match in scored_matches if match["label"] == label][: args.top_k]
        if label_specific:
            label_scores[label] = round(sum(label_specific) / len(label_specific), 4)

    result = {
        "candidate_json": candidate_json.as_posix(),
        "candidate_classification": candidate_payload.get("classification"),
        "label_scores": label_scores,
        "top_matches": top_matches,
    }
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review and compare exported candidate snippets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote_parser = subparsers.add_parser("promote", help="Copy or move a reviewed snippet into training.")
    promote_parser.add_argument("candidate_json", help="Path to an exported candidate JSON sidecar.")
    promote_parser.add_argument("label", choices=LABELS, help="Review label for the snippet.")
    promote_parser.add_argument("--action", choices=("copy", "move"), default="copy")
    promote_parser.add_argument("--training-root", default="training", help="Training data root directory.")
    promote_parser.set_defaults(func=promote_snippet)

    compare_parser = subparsers.add_parser("compare", help="Score a candidate against accepted snippets.")
    compare_parser.add_argument("candidate_json", help="Path to an exported candidate JSON sidecar.")
    compare_parser.add_argument("--training-root", default="training", help="Training data root directory.")
    compare_parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to show.")
    compare_parser.set_defaults(func=compare_snippet)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
