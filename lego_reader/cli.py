from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bag_detector import analyze_pdf_for_bags
from .downloader import DownloadError, SetNotFoundError, download_set_pdfs
from .models import PdfBagAnalysis
from .pdf_reader import read_pdf_pages
from .utils import LOG, configure_logging, ensure_dir, normalize_set_num, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official LEGO instructions and detect bag sections.",
    )
    parser.add_argument("set_num", help="LEGO set number, for example 21330")
    parser.add_argument("--debug", action="store_true", help="Save debug artifacts.")
    parser.add_argument("--out", default="output", help="Output directory for result JSON.")
    parser.add_argument(
        "--instructions-dir",
        default="instructions",
        help="Directory where downloaded PDFs will be saved.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug",
        help="Directory for debug images and candidate snapshots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.debug)

    try:
        set_num = normalize_set_num(args.set_num)
    except ValueError as error:
        parser.error(str(error))
        return 2

    out_dir = ensure_dir(Path(args.out).expanduser().resolve())
    instructions_dir = ensure_dir(Path(args.instructions_dir).expanduser().resolve())
    debug_dir = ensure_dir(Path(args.debug_dir).expanduser().resolve())

    try:
        download_result = download_set_pdfs(
            set_num=set_num,
            instructions_dir=instructions_dir,
        )
    except SetNotFoundError as error:
        LOG.error("%s", error)
        payload = {
            "set_num": set_num,
            "source_url": error.source_url,
            "pdfs": [],
            "total_bags_detected": 0,
            "error": str(error),
        }
        output_path = out_dir / f"{set_num}.json"
        write_json(output_path, payload)
        print(output_path.as_posix())
        return 1
    except DownloadError as error:
        LOG.error("%s", error)
        return 1

    pdf_analyses: list[PdfBagAnalysis] = []
    total_bags = 0

    for downloaded_pdf in download_result.pdfs:
        LOG.info("Reading %s from %s", downloaded_pdf.local_path.name, downloaded_pdf.source_url)
        per_pdf_debug_dir = debug_dir / set_num / downloaded_pdf.local_path.stem
        page_data = read_pdf_pages(
            pdf_path=downloaded_pdf.local_path,
            debug=args.debug,
            debug_dir=per_pdf_debug_dir,
        )
        analysis = analyze_pdf_for_bags(
            pdf_path=downloaded_pdf.local_path,
            pages=page_data,
            debug=args.debug,
            debug_dir=per_pdf_debug_dir,
        )
        pdf_analyses.append(analysis)
        total_bags += analysis.bag_count_estimate

    payload = {
        "set_num": set_num,
        "source_url": download_result.source_url,
        "page_title": download_result.page_title,
        "downloaded_pdf_paths": [pdf.local_path.as_posix() for pdf in download_result.pdfs],
        "pdfs": [
            {
                "title": pdf.title,
                "source_url": pdf.source_url,
                "file": analysis.file,
                "bag_count": analysis.bag_count_estimate,
                "bag_count_estimate": analysis.bag_count_estimate,
                "bag_start_pages": [bag.start_page for bag in analysis.bags],
                "bags": [bag.to_output_dict() for bag in analysis.bags],
                "overview_pages": analysis.overview_pages,
                "bag_start_like_pages": analysis.bag_start_like_pages,
                "uncertain_pages": analysis.uncertain_pages,
                "expected_bags": analysis.expected_bags,
                "detected_bags": analysis.detected_bags,
                "missing_bags": analysis.missing_bags,
                "review_groups": analysis.review_groups,
            }
            for pdf, analysis in zip(download_result.pdfs, pdf_analyses)
        ],
        "total_bags_detected": total_bags,
    }
    output_path = out_dir / f"{set_num}.json"
    write_json(output_path, payload)
    print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
