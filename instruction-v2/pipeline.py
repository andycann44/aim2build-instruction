import argparse
import json

from phase1_pdf_pages import run_phase1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="instruction-v2 pipeline entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1", help="Run phase 1: PDF -> pages + page index")
    phase1.add_argument("--pdf", required=True, help="Path to source instruction PDF")
    phase1.add_argument("--run-id", default=None, help="Optional page run directory name")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "phase1":
        payload = run_phase1(pdf_path=args.pdf, run_id=args.run_id)
        print(json.dumps(payload, indent=2))
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
