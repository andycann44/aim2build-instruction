#!/usr/bin/env python3
"""
Unified Instruction Reader pipeline runner.

Runs only the baseline stage scripts listed in docs/CLEAN_SETUP.md.
Does not modify stage logic — each stage is invoked as a subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent

DEFAULT_SET_NUM = "70618"
DEFAULT_PDF = ROOT_DIR / "pdfs" / "70618_01.pdf"
DEFAULT_BAG = 4
DEFAULT_PORT = 8000

STOPPOINTS = (
    "step-map",
    "crop-cache",
    "qty",
    "segmentation",
    "match",
)

BANNED_SCRIPTS = (
    "stage5_orchestrator.py",
    "stage5f_callout_quality.py",
)


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    stoppoint: str
    build_cmd: Callable[["RunConfig"], Sequence[str]]


@dataclass
class RunConfig:
    set_num: str
    pdf: Path
    bag: int
    run_id: Optional[str]
    stoppoint: str


def _resolve_pdf(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"PDF not found: {candidate}")
    return candidate


def _default_run_id(pdf: Path) -> str:
    return pdf.stem


def _run_stage(stage: Stage, config: RunConfig) -> None:
    script_path = ROOT_DIR / stage.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing baseline stage script: {script_path}")
    if stage.script in BANNED_SCRIPTS:
        raise RuntimeError(f"Refusing to run banned script: {stage.script}")

    cmd = [sys.executable, str(script_path), *stage.build_cmd(config)]
    print(f"\n==> {stage.key}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def _stages_for(config: RunConfig) -> List[Stage]:
    all_stages: List[Stage] = [
        Stage(
            key="stage0_set_context",
            script="stage0_set_context.py",
            stoppoint="step-map",
            build_cmd=lambda cfg: ["--set-num", cfg.set_num],
        ),
        Stage(
            key="phase1_pdf_pages",
            script="phase1_pdf_pages.py",
            stoppoint="step-map",
            build_cmd=lambda cfg: [
                "--pdf",
                str(cfg.pdf),
                "--run-id",
                cfg.run_id or _default_run_id(cfg.pdf),
            ],
        ),
        Stage(
            key="stage2_bag_candidates",
            script="stage2_bag_candidates.py",
            stoppoint="step-map",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage3b_bag_gap_review",
            script="stage3b_bag_gap_review.py",
            stoppoint="step-map",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage3_bag_map",
            script="stage3_bag_map.py",
            stoppoint="step-map",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage4_step_map",
            script="stage4_step_map.py",
            stoppoint="step-map",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage5_callout_crop_boxes",
            script="stage5_callout_crop_boxes.py",
            stoppoint="crop-cache",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage5d_sequence_completeness_diagnostics",
            script="stage5d_sequence_completeness_diagnostics.py",
            stoppoint="crop-cache",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage5e_export_crop_cache",
            script="stage5e_export_crop_cache.py",
            stoppoint="crop-cache",
            build_cmd=lambda cfg: ["--set-num", cfg.set_num, "--bag", str(cfg.bag)],
        ),
        Stage(
            key="stage6_qty_ocr",
            script="stage6_qty_ocr.py",
            stoppoint="qty",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage7_part_segmentation",
            script="stage7_part_segmentation.py",
            stoppoint="segmentation",
            build_cmd=lambda _cfg: [],
        ),
        Stage(
            key="stage8_match",
            script="stage8_match.py",
            stoppoint="match",
            build_cmd=lambda _cfg: [],
        ),
    ]

    stop_order = list(STOPPOINTS)
    limit = stop_order.index(config.stoppoint)
    allowed_stoppoints = set(stop_order[: limit + 1])
    return [stage for stage in all_stages if stage.stoppoint in allowed_stoppoints]


def _review_url(set_num: str, bag: int, host: str, port: int) -> str:
    return (
        f"http://{host}:{port}/debug/manual-match-review"
        f"?set_num={set_num}&bag={bag}"
    )


def _open_review(set_num: str, bag: int, host: str, port: int) -> None:
    crop_cache = REPO_ROOT / "debug" / "crop_cache" / f"{set_num}_bag{bag}.json"
    url = _review_url(set_num, bag, host, port)
    print(f"\nReview UI: {url}")
    print(f"Crop cache: {crop_cache}")
    if not crop_cache.exists():
        print(
            "Warning: crop cache missing. Run with --to crop-cache (or later) first.",
            file=sys.stderr,
        )
    print(
        f"\nStart V1 server (if not already running):\n"
        f"  uvicorn clean.main:app --reload --host {host} --port {port}"
    )
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}", file=sys.stderr)


def run_pipeline(config: RunConfig) -> None:
    stages = _stages_for(config)
    print(
        f"Instruction Reader run: set={config.set_num} bag={config.bag} "
        f"stop={config.stoppoint} stages={len(stages)}",
        flush=True,
    )
    for stage in stages:
        _run_stage(stage, config)
    print(f"\nDone. Reached stoppoint: {config.stoppoint}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the baseline Instruction Reader V2 pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stoppoints (in order):\n"
            "  step-map      stage0 → phase1 → stage2 → stage3b → stage3 → stage4\n"
            "  crop-cache    … → stage5 → stage5d → stage5e (exports debug/crop_cache)\n"
            "  qty           … → stage6\n"
            "  segmentation  … → stage7\n"
            "  match         … → stage8\n"
        ),
    )
    parser.add_argument(
        "--set-num",
        default=DEFAULT_SET_NUM,
        help=f"LEGO set number (default: {DEFAULT_SET_NUM})",
    )
    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF),
        help=f"Source instruction PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--bag",
        type=int,
        default=DEFAULT_BAG,
        help=f"Bag number for stage5e crop-cache export (default: {DEFAULT_BAG})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Phase1 page run id (default: PDF filename stem, e.g. 70618_01)",
    )
    parser.add_argument(
        "--to",
        choices=STOPPOINTS,
        default=None,
        help="Run pipeline through this stoppoint",
    )
    parser.add_argument(
        "--open-review",
        action="store_true",
        help="Open V1 manual-match-review UI in browser after run (or alone)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --open-review URL (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for --open-review URL (default: {DEFAULT_PORT})",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not args.to and not args.open_review:
        print("error: specify --to <stoppoint> and/or --open-review", file=sys.stderr)
        return 2

    set_num = str(args.set_num).strip()
    bag = int(args.bag)
    pdf = _resolve_pdf(Path(args.pdf))
    run_id = args.run_id or _default_run_id(pdf)

    if args.to:
        config = RunConfig(
            set_num=set_num,
            pdf=pdf,
            bag=bag,
            run_id=run_id,
            stoppoint=args.to,
        )
        run_pipeline(config)

    if args.open_review:
        _open_review(set_num, bag, args.host, args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
