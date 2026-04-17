from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DownloadedPdf:
    index: int
    title: str
    source_url: str
    local_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "source_url": self.source_url,
            "local_path": self.local_path.as_posix(),
        }


@dataclass
class DownloadResult:
    set_num: str
    source_url: str
    page_title: str
    pdfs: list[DownloadedPdf]


@dataclass
class PageData:
    page_index: int
    page_number: int
    text: str
    width: float
    height: float
    word_count: int
    drawing_count: int
    image_count: int
    image_path: Path | None = None


@dataclass
class BagCandidate:
    bag: int
    start_page: int
    confidence: float
    page_index: int
    reasons: list[str] = field(default_factory=list)

    def to_output_dict(self) -> dict[str, Any]:
        return {
            "bag": self.bag,
            "start_page": self.start_page,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PdfBagAnalysis:
    file: str
    bag_count_estimate: int
    bags: list[BagCandidate]
    overview_pages: list[int]
    bag_start_like_pages: list[int]
    uncertain_pages: list[int]
    expected_bags: list[int]
    detected_bags: list[int]
    missing_bags: list[int]
    review_groups: dict[str, Any] = field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "bag_count": self.bag_count_estimate,
            "bag_count_estimate": self.bag_count_estimate,
            "bag_start_pages": [bag.start_page for bag in self.bags],
            "bags": [bag.to_output_dict() for bag in self.bags],
            "overview_pages": self.overview_pages,
            "bag_start_like_pages": self.bag_start_like_pages,
            "uncertain_pages": self.uncertain_pages,
            "expected_bags": self.expected_bags,
            "detected_bags": self.detected_bags,
            "missing_bags": self.missing_bags,
            "review_groups": self.review_groups,
        }
