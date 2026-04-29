import json
from html import escape
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from clean.services import (
    auto_confirm_service,
    bag_truth_store,
    workflow_service,
    debug_service,
    step_detector_service,
    step_sequence_bag_service,
)
from clean.services.page_analyzer import analyze_page, configure_pages_dir
router = APIRouter()


def _list_rendered_pages(pages_dir):
    pages = []
    for path in sorted(pages_dir.glob("page_*.png")):
        stem = path.stem
        try:
            page_num = int(stem.replace("page_", ""))
        except ValueError:
            continue
        pages.append(page_num)
    return pages


def _build_bag_ranges(bag_starts, last_page):
    bag_ranges = []
    if not bag_starts or last_page is None:
        return bag_ranges

    sorted_bag_starts = sorted(bag_starts, key=lambda item: int(item["page"]))
    last_page = int(last_page)

    for index, item in enumerate(sorted_bag_starts):
        start_page = int(item["page"])
        next_bag_start_page = (
            int(sorted_bag_starts[index + 1]["page"])
            if index + 1 < len(sorted_bag_starts)
            else None
        )
        end_page = (
            int(next_bag_start_page) - 1
            if next_bag_start_page is not None
            else last_page
        )
        bag_ranges.append(
            {
                "bag_number": item["bag_number"],
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    return bag_ranges


def _boxes_overlap(box_a, box_b):
    if not box_a or not box_b:
        return False

    ax, ay, aw, ah = [int(v) for v in box_a]
    bx, by, bw, bh = [int(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False

    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh

    return not (
        a_right <= bx
        or b_right <= ax
        or a_bottom <= by
        or b_bottom <= ay
    )


def _load_page_main_steps(set_num: str, page: int):
    analyze_page(int(page), include_image=False)
    detected = step_detector_service.detect_steps(set_num, int(page))
    main_steps = detected.get("main_steps", []) or []
    return sorted(
        {
            int(item.get("value", 0) or 0)
            for item in main_steps
            if int(item.get("value", 0) or 0) > 0
        }
    )


def _fast_bag_prefilter(page_img) -> tuple[bool, dict]:
    if page_img is None or getattr(page_img, "size", 0) == 0:
        return True, {
            "white_ratio_top": None,
            "warm_ratio_top": None,
            "dark_ratio_top": None,
        }

    page_h, page_w = page_img.shape[:2]
    if page_h <= 0 or page_w <= 0:
        return True, {
            "white_ratio_top": None,
            "warm_ratio_top": None,
            "dark_ratio_top": None,
        }

    target_w = 300
    scale = float(target_w) / float(max(1, page_w))
    resized_h = max(1, int(round(page_h * scale)))
    small = cv2.resize(page_img, (target_w, resized_h), interpolation=cv2.INTER_AREA)

    top_h = max(1, int(round(small.shape[0] * 0.35)))
    top = small[:top_h, :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)

    white_ratio_top = float(np.mean(gray >= 225))
    dark_ratio_top = float(np.mean(gray <= 95))
    warm_mask = (
        (
            ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 140)
        )
        | (
            (hsv[:, :, 0] >= 8)
            & (hsv[:, :, 0] <= 28)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 140)
        )
    )
    warm_ratio_top = float(np.mean(warm_mask))

    metrics = {
        "white_ratio_top": round(white_ratio_top, 4),
        "warm_ratio_top": round(warm_ratio_top, 4),
        "dark_ratio_top": round(dark_ratio_top, 4),
    }
    passed = bool(white_ratio_top > 0.35 or warm_ratio_top > 0.002)
    return passed, metrics


def _filter_bag_starts_for_save(bag_starts, skipped_rows):
    filtered_bag_starts = []
    previous_saved_bag_number = None

    for item in bag_starts or []:
        bag_number = item.get("bag_number")
        if bag_number is None:
            continue

        candidate_bag_number = int(bag_number)
        card_found = bool(item.get("bag_start_card_found"))
        bag_start_layout = str(item.get("bag_start_layout", "") or "")
        transition_reset_layout = bag_start_layout in {
            "transition_reset",
            "hero_transition_reset",
            "white_box_transition",
        }
        sequence_valid = (
            candidate_bag_number == 1
            if previous_saved_bag_number is None
            else candidate_bag_number == previous_saved_bag_number + 1
        )

        if not card_found and not transition_reset_layout:
            skipped_rows.append(
                {
                    "page": int(item.get("page", 0) or 0),
                    "skipped": True,
                    "skip_reason": "no_bag_start_card",
                    "candidate_bag_number": candidate_bag_number,
                    "sequence_valid": bool(sequence_valid),
                    "bag_start_card_found": False,
                    "step_sequence_role": item.get("step_sequence_role"),
                    "step_drop": item.get("step_drop"),
                    "previous_max_step": item.get("previous_max_step"),
                    "bag_start_layout": bag_start_layout or None,
                    "slow_page": bool(item.get("slow_page")),
                }
            )
            continue

        filtered_bag_starts.append(item)
        previous_saved_bag_number = candidate_bag_number

    return filtered_bag_starts


def _build_existing_truth_by_page(existing_bag_truth):
    truth_by_page = {}
    for item in existing_bag_truth or []:
        start_page = int(item.get("start_page", 0) or 0)
        bag_number = int(item.get("bag_number", 0) or 0)
        if start_page <= 0 or bag_number <= 0:
            continue
        truth_by_page[start_page] = {
            "start_page": start_page,
            "bag_number": bag_number,
        }
    return truth_by_page


def _last_confirmed_bag_before_page(existing_bag_truth, page):
    target_page = int(page)
    last_confirmed_bag = 0
    for item in sorted(
        existing_bag_truth or [],
        key=lambda row: int(row.get("start_page", 0) or 0),
    ):
        start_page = int(item.get("start_page", 0) or 0)
        bag_number = int(item.get("bag_number", 0) or 0)
        if start_page <= 0 or bag_number <= 0:
            continue
        if start_page >= target_page:
            break
        last_confirmed_bag = bag_number
    return last_confirmed_bag


def _last_confirmed_bag_before_page_with_candidates(
    existing_bag_truth,
    provisional_bag_starts,
    page,
):
    last_confirmed_bag = _last_confirmed_bag_before_page(existing_bag_truth, page)
    for item in sorted(
        provisional_bag_starts or [],
        key=lambda row: int(row.get("page", 0) or 0),
    ):
        candidate_page = int(item.get("page", 0) or 0)
        candidate_bag_number = int(item.get("bag_number", 0) or 0)
        if candidate_page <= 0 or candidate_bag_number <= 0:
            continue
        if candidate_page >= int(page):
            break
        if candidate_bag_number > last_confirmed_bag:
            last_confirmed_bag = candidate_bag_number
    return last_confirmed_bag


def _split_bag_starts_for_existing_truth(bag_starts, existing_truth_by_page, skipped_rows):
    filtered_bag_starts = []
    bag_starts_to_save = []

    for item in bag_starts or []:
        page = int(item.get("page", 0) or 0)
        candidate_bag_number = int(item.get("bag_number", 0) or 0)
        existing_truth = (existing_truth_by_page or {}).get(page)
        if existing_truth is None:
            filtered_bag_starts.append(item)
            bag_starts_to_save.append(item)
            continue

        existing_bag_number = int(existing_truth.get("bag_number", 0) or 0)
        if existing_bag_number != candidate_bag_number:
            skipped_rows.append(
                {
                    "page": page,
                    "skipped": True,
                    "skip_reason": "same_page_different_bag_already_saved",
                    "existing_bag_number": existing_bag_number,
                    "candidate_bag_number": candidate_bag_number,
                    "step_sequence_role": item.get("step_sequence_role"),
                    "step_drop": item.get("step_drop"),
                    "previous_max_step": item.get("previous_max_step"),
                    "bag_start_layout": item.get("bag_start_layout"),
                    "inferred_bag_number_source": item.get("inferred_bag_number_source"),
                    "slow_page": bool(item.get("slow_page")),
                }
            )
            continue

        item["already_saved_same_page"] = True
        item["existing_truth_bag_number"] = existing_bag_number
        filtered_bag_starts.append(item)

    return filtered_bag_starts, bag_starts_to_save


def _has_strong_hero_transition_evidence(result):
    if bool(result.get("white_box_transition_found")):
        return True
    if bool(result.get("bag_start_card_found")):
        return True
    if float(result.get("bag_start_card_score", 0.0) or 0.0) >= 70.0:
        return True
    return bool(
        result.get("panel_found")
        and result.get("panel_source") == "strict_top_left"
        and float(result.get("panel_score", 0.0) or 0.0) >= 150.0
        and not bool(result.get("build_step_page"))
        and int(result.get("step_grid_number_count", 0) or 0) == 0
        and float(result.get("bag_region_quality", 0.0) or 0.0) >= 0.9
    )


def _resolve_transition_step_sequence_role(step_role_row, result):
    step_role_row = step_role_row or {}
    step_sequence_role = step_role_row.get("step_sequence_role")
    if step_sequence_role == "step_sequence_reset":
        return "step_sequence_reset"
    if bool(step_role_row.get("hero_transition_reset_candidate")) and _has_strong_hero_transition_evidence(result):
        return "hero_transition_reset"
    return step_sequence_role


def _infer_reset_bag_number(
    page,
    step_sequence_role,
    result,
    existing_bag_truth,
    existing_truth_by_page,
    provisional_bag_starts=None,
):
    if step_sequence_role not in {"step_sequence_reset", "hero_transition_reset"}:
        return None, None
    if not bool(result.get("panel_found")) or result.get("panel_source") != "strict_top_left":
        return None, None

    existing_truth_same_page = (existing_truth_by_page or {}).get(int(page))
    if existing_truth_same_page is not None:
        return (
            int(existing_truth_same_page.get("bag_number", 0) or 0),
            "existing_truth_same_page",
        )

    last_confirmed_bag = _last_confirmed_bag_before_page(
        existing_bag_truth=existing_bag_truth,
        page=int(page),
    )
    if provisional_bag_starts:
        last_confirmed_bag = _last_confirmed_bag_before_page_with_candidates(
            existing_bag_truth=existing_bag_truth,
            provisional_bag_starts=provisional_bag_starts,
            page=int(page),
        )
    if last_confirmed_bag <= 0:
        return 1, "last_confirmed_bag_sequence"
    return int(last_confirmed_bag) + 1, "last_confirmed_bag_sequence"


def _infer_white_box_transition_bag_number(
    page,
    result,
    existing_bag_truth,
    existing_truth_by_page,
    provisional_bag_starts=None,
):
    if not bool(result.get("white_box_transition_found")):
        return None, None

    existing_truth_same_page = (existing_truth_by_page or {}).get(int(page))
    if existing_truth_same_page is not None:
        return (
            int(existing_truth_same_page.get("bag_number", 0) or 0),
            "existing_truth_same_page",
        )

    last_confirmed_bag = _last_confirmed_bag_before_page(
        existing_bag_truth=existing_bag_truth,
        page=int(page),
    )
    if provisional_bag_starts:
        last_confirmed_bag = _last_confirmed_bag_before_page_with_candidates(
            existing_bag_truth=existing_bag_truth,
            provisional_bag_starts=provisional_bag_starts,
            page=int(page),
        )
    if last_confirmed_bag <= 0:
        return 1, "last_confirmed_bag_sequence"
    return int(last_confirmed_bag) + 1, "last_confirmed_bag_sequence"


def _build_bag_candidate_report_row(
    set_num,
    page,
    result,
    step_role_row,
    existing_bag_truth,
    existing_truth_by_page,
    expected_next_bag=None,
):
    page = int(page)
    step_role_row = step_role_row or {}
    step_sequence_role = _resolve_transition_step_sequence_role(
        step_role_row=step_role_row,
        result=result,
    )
    previous_max_step = step_role_row.get("previous_max_step")
    main_step_min = step_role_row.get("main_step_min")
    step_drop = step_role_row.get("step_drop")
    inferred_white_box_bag_number = None
    inferred_white_box_bag_number_source = None
    inferred_transition_bag_number = None
    inferred_bag_number_source = None

    if bool(result.get("white_box_transition_found")):
        (
            inferred_white_box_bag_number,
            inferred_white_box_bag_number_source,
        ) = _infer_white_box_transition_bag_number(
            page=page,
            result=result,
            existing_bag_truth=existing_bag_truth,
            existing_truth_by_page=existing_truth_by_page,
        )

    if result.get("bag_number") is None and inferred_white_box_bag_number is None:
        (
            inferred_transition_bag_number,
            inferred_bag_number_source,
        ) = _infer_reset_bag_number(
            page=page,
            step_sequence_role=step_sequence_role,
            result=result,
            existing_bag_truth=existing_bag_truth,
            existing_truth_by_page=existing_truth_by_page,
        )

    effective_bag_number = result.get("bag_number")
    if inferred_white_box_bag_number is not None:
        effective_bag_number = int(inferred_white_box_bag_number)
        inferred_bag_number_source = inferred_white_box_bag_number_source
    elif effective_bag_number is None and inferred_transition_bag_number is not None:
        effective_bag_number = int(inferred_transition_bag_number)

    accept_by_card = bool(result.get("bag_start_card_found"))
    accept_by_white_box_transition = bool(
        result.get("white_box_transition_found")
        and inferred_white_box_bag_number is not None
    )
    accept_by_transition_reset = (
        step_sequence_role in {"step_sequence_reset", "hero_transition_reset"}
        and effective_bag_number is not None
        and (
            bool(result.get("number_box_found"))
            or inferred_transition_bag_number is not None
        )
    )
    accept_by_strict_pattern = (
        result.get("panel_source") == "strict_top_left"
        and bool(result.get("number_box_found"))
    )

    bag_start_layout = None
    if accept_by_white_box_transition:
        bag_start_layout = "white_box_transition"
    elif accept_by_card:
        bag_start_layout = "card"
    elif accept_by_transition_reset:
        bag_start_layout = (
            "hero_transition_reset"
            if step_sequence_role == "hero_transition_reset"
            else "transition_reset"
        )
    elif accept_by_strict_pattern:
        bag_start_layout = "strict_top_left"

    rejection_reason = None
    if effective_bag_number is None:
        rejection_reason = "no_bag_number"
    elif bool(result.get("multi_step_green_boxes")):
        rejection_reason = "multi_step_green_boxes"
    elif (
        step_sequence_role == "build_step_sequence_page"
        or not bool(step_role_row.get("bag_start_allowed", True))
    ) and not accept_by_white_box_transition:
        rejection_reason = "build_step_sequence_page"
    elif (
        not bool(result.get("number_box_found"))
        and not accept_by_transition_reset
        and not accept_by_white_box_transition
    ):
        rejection_reason = "no_number_box"
    else:
        number_box_area = result.get("number_box_area")
        if number_box_area is not None:
            area = int(number_box_area)
            min_area = 20000 if expected_next_bag is not None else 40000
            if area <= min_area and not (
                accept_by_card
                or accept_by_transition_reset
                or accept_by_white_box_transition
            ):
                rejection_reason = "number_box_area_too_small"

        if rejection_reason is None:
            panel_box = result.get("panel_box")
            number_box = result.get("number_box")
            detected_step_boxes = list(step_role_row.get("detected_step_boxes", []) or [])
            main_step_boxes = [
                item.get("box")
                for item in detected_step_boxes
                if item.get("step_group") == "main_steps"
            ]
            for main_step_box in main_step_boxes:
                if _boxes_overlap(main_step_box, number_box):
                    rejection_reason = "step_box_contaminates_bag_number"
                    break

        if rejection_reason is None and not (
            accept_by_card
            or accept_by_transition_reset
            or accept_by_white_box_transition
            or accept_by_strict_pattern
        ):
            rejection_reason = "no_valid_bag_start_pattern"

    return {
        "page": page,
        "bag_number": effective_bag_number,
        "ocr_raw": result.get("ocr_raw", "") or "",
        "bag_start_card_found": bool(result.get("bag_start_card_found")),
        "bag_start_card_score": float(
            result.get("bag_start_card_score", 0.0) or 0.0
        ),
        "number_box_found": bool(result.get("number_box_found")),
        "number_box_area": result.get("number_box_area"),
        "step_sequence_role": step_sequence_role,
        "previous_max_step": previous_max_step,
        "main_step_min": main_step_min,
        "step_drop": step_drop,
        "would_accept_card": accept_by_card,
        "would_accept_reset": accept_by_transition_reset,
        "white_box_transition_found": bool(result.get("white_box_transition_found")),
        "final_decision": "accepted" if rejection_reason is None else "rejected",
        "rejection_reason": rejection_reason,
        "debug_page_image_url": (
            f"/debug/page-image?set_num={set_num}&page={page}"
        ),
        "bag_start_layout": bag_start_layout,
        "inferred_bag_number_source": inferred_bag_number_source,
    }


@router.get("/api/analyze-page-direct")
def analyze_page_direct(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "page": int(page),
            "error": "no rendered pages found for set",
        }

    configure_pages_dir(str(pages_dir))
    result = analyze_page(int(page), include_image=False)
    return result


@router.get("/api/bag-starts-scan")
def bag_starts_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    limit: int = Query(10, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    started_at = time.monotonic()
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "error": "no rendered pages found for set",
            "pages_dir": None,
            "page_count": 0,
            "scanned_pages_count": 0,
            "next_start_page": None,
            "remaining_pages": 0,
            "timed_out": False,
            "elapsed_seconds_total": 0.0,
            "page_timings": [],
            "slow_pages": [],
            "skipped_rows": [],
            "bag_starts": [],
            "bag_ranges": [],
        }

    pages = _list_rendered_pages(pages_dir)
    if start is not None:
        pages = [page for page in pages if int(page) >= int(start)]
    if end is not None:
        pages = [page for page in pages if int(page) <= int(end)]

    total_pages = len(pages)
    pages_to_scan = []
    for page in pages:
        if len(pages_to_scan) >= limit:
            break
        pages_to_scan.append(int(page))

    configure_pages_dir(str(pages_dir))
    step_role_map = step_sequence_bag_service.build_step_sequence_prepass_for_pages(
        set_num=set_num,
        pages=pages_to_scan,
    )
    step_role_by_page = {
        int(item.get("page", 0) or 0): item for item in (step_role_map or [])
    }
    existing_bag_truth = list(bag_truth_store.get_bag_truth(set_num) or [])
    existing_truth_by_page = _build_existing_truth_by_page(existing_bag_truth)

    bag_starts = []
    skipped_rows = []
    slow_pages = []
    page_timings = []
    scanned_pages_count = 0
    last_scanned_page = None
    for page in pages:
        if scanned_pages_count >= limit:
            break

        scanned_pages_count += 1
        last_scanned_page = int(page)
        prefilter = None
        page_image_path = debug_service.resolve_page_image_path(set_num, int(page))
        if page_image_path is not None:
            page_img = cv2.imread(str(page_image_path))
            if page_img is not None:
                prefilter_passed, prefilter = _fast_bag_prefilter(page_img)
                if not prefilter_passed:
                    skipped_rows.append(
                        {
                            "page": int(page),
                            "skipped": True,
                            "skip_reason": "fast_prefilter_reject",
                            "prefilter": prefilter,
                        }
                    )
                    continue
        page_started_at = time.monotonic()
        result = analyze_page(int(page), include_image=False)
        elapsed_seconds = round(time.monotonic() - page_started_at, 3)
        page_timings.append(
            {
                "page": int(page),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        slow_page = elapsed_seconds > 8.0
        if elapsed_seconds > 8.0:
            slow_row = {
                "page": int(page),
                "slow_page": True,
                "elapsed_seconds": elapsed_seconds,
            }
            slow_pages.append(slow_row)
        step_role_row = step_role_by_page.get(int(page))
        step_sequence_role = _resolve_transition_step_sequence_role(
            step_role_row=step_role_row,
            result=result,
        )
        previous_max_step = (step_role_row or {}).get("previous_max_step")
        step_drop = (step_role_row or {}).get("step_drop")
        inferred_white_box_bag_number = None
        inferred_white_box_bag_number_source = None
        inferred_transition_bag_number = None
        inferred_bag_number_source = None
        if bool(result.get("white_box_transition_found")):
            (
                inferred_white_box_bag_number,
                inferred_white_box_bag_number_source,
            ) = _infer_white_box_transition_bag_number(
                page=int(page),
                result=result,
                existing_bag_truth=existing_bag_truth,
                existing_truth_by_page=existing_truth_by_page,
                provisional_bag_starts=bag_starts,
            )
        if result.get("bag_number") is None and inferred_white_box_bag_number is None:
            (
                inferred_transition_bag_number,
                inferred_bag_number_source,
            ) = _infer_reset_bag_number(
                page=int(page),
                step_sequence_role=step_sequence_role,
                result=result,
                existing_bag_truth=existing_bag_truth,
                existing_truth_by_page=existing_truth_by_page,
                provisional_bag_starts=bag_starts,
            )

        effective_bag_number = result.get("bag_number")
        if inferred_white_box_bag_number is not None:
            effective_bag_number = int(inferred_white_box_bag_number)
            inferred_bag_number_source = inferred_white_box_bag_number_source
        elif effective_bag_number is None and inferred_transition_bag_number is not None:
            effective_bag_number = int(inferred_transition_bag_number)

        if effective_bag_number is None:
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "no_bag_number",
                    "step_sequence_role": step_sequence_role,
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": None,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "slow_page": slow_page,
                }
            )
            continue
        candidate_bag_number = int(effective_bag_number or 0)
        accept_by_card = bool(result.get("bag_start_card_found"))
        accept_by_white_box_transition = bool(
            result.get("white_box_transition_found")
            and inferred_white_box_bag_number is not None
        )
        if bool(result.get("multi_step_green_boxes")):
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "multi_step_green_boxes",
                    "candidate_bag_number": candidate_bag_number,
                    "bag_start_card_found": accept_by_card,
                    "step_sequence_role": step_sequence_role,
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": None,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "white_box_transition_found": bool(
                        result.get("white_box_transition_found")
                    ),
                    "slow_page": slow_page,
                }
            )
            continue
        accept_by_transition_reset = (
            step_sequence_role in {"step_sequence_reset", "hero_transition_reset"}
            and effective_bag_number is not None
            and (
                bool(result.get("number_box_found"))
                or inferred_transition_bag_number is not None
            )
        )
        bag_start_layout = None
        if accept_by_white_box_transition:
            bag_start_layout = "white_box_transition"
        elif accept_by_card:
            bag_start_layout = "card"
        elif accept_by_transition_reset:
            bag_start_layout = (
                "hero_transition_reset"
                if step_sequence_role == "hero_transition_reset"
                else "transition_reset"
            )
        elif (
            result.get("panel_source") == "strict_top_left"
            and bool(result.get("number_box_found"))
        ):
            bag_start_layout = "strict_top_left"
        if (
            step_sequence_role == "build_step_sequence_page"
            or not bool((step_role_row or {}).get("bag_start_allowed", True))
        ) and not accept_by_white_box_transition:
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "build_step_sequence_page",
                    "candidate_bag_number": candidate_bag_number,
                    "bag_start_card_found": accept_by_card,
                    "step_sequence_role": step_sequence_role,
                    "step_role_reason": (step_role_row or {}).get("step_role_reason"),
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": bag_start_layout,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "white_box_transition_found": bool(
                        result.get("white_box_transition_found")
                    ),
                    "main_steps": list((step_role_row or {}).get("main_steps", []) or []),
                    "slow_page": slow_page,
                }
            )
            continue
        if (
            not bool(result.get("number_box_found"))
            and not accept_by_transition_reset
            and not accept_by_white_box_transition
        ):
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "no_number_box",
                    "candidate_bag_number": candidate_bag_number,
                    "bag_start_card_found": accept_by_card,
                    "step_sequence_role": step_sequence_role,
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": bag_start_layout,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "white_box_transition_found": bool(
                        result.get("white_box_transition_found")
                    ),
                    "slow_page": slow_page,
                }
            )
            continue
        number_box_area = result.get("number_box_area")
        if number_box_area is not None:
            area = int(number_box_area)
            min_area = 20000 if expected_next_bag is not None else 40000
            if area <= min_area and not (
                accept_by_card
                or accept_by_transition_reset
                or accept_by_white_box_transition
            ):
                skipped_rows.append(
                    {
                        "page": int(page),
                        "skipped": True,
                        "skip_reason": "number_box_area_too_small",
                        "candidate_bag_number": candidate_bag_number,
                        "bag_start_card_found": accept_by_card,
                        "number_box_area": area,
                        "min_number_box_area": min_area,
                        "step_sequence_role": step_sequence_role,
                        "step_drop": step_drop,
                        "previous_max_step": previous_max_step,
                        "bag_start_layout": bag_start_layout,
                        "inferred_bag_number_source": inferred_bag_number_source,
                        "white_box_transition_found": bool(
                            result.get("white_box_transition_found")
                        ),
                        "slow_page": slow_page,
                    }
                )
                continue
        panel_box = result.get("panel_box")
        number_box = result.get("number_box")
        detected_step_boxes = list(
            (step_role_row or {}).get("detected_step_boxes", []) or []
        )
        main_step_boxes = [
            item.get("box")
            for item in detected_step_boxes
            if item.get("step_group") == "main_steps"
        ]
        contaminating_step_box = None
        for main_step_box in main_step_boxes:
            if _boxes_overlap(main_step_box, number_box):
                contaminating_step_box = main_step_box
                break
        if contaminating_step_box is not None:
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "step_box_contaminates_bag_number",
                    "candidate_bag_number": result.get("bag_number"),
                    "number_box": number_box,
                    "panel_box": panel_box,
                    "main_step_box": contaminating_step_box,
                    "step_sequence_role": step_sequence_role,
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": bag_start_layout,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "white_box_transition_found": bool(
                        result.get("white_box_transition_found")
                    ),
                    "slow_page": slow_page,
                }
            )
            continue
        accept_by_strict_pattern = (
            result.get("panel_source") == "strict_top_left"
            and bool(result.get("number_box_found"))
        )
        if not (
            accept_by_card
            or accept_by_transition_reset
            or accept_by_white_box_transition
            or accept_by_strict_pattern
        ):
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "no_valid_bag_start_pattern",
                    "candidate_bag_number": result.get("bag_number"),
                    "panel_source": result.get("panel_source"),
                    "bag_start_card_found": bool(result.get("bag_start_card_found")),
                    "bag_start_card_score": float(
                        result.get("bag_start_card_score", 0.0) or 0.0
                    ),
                    "bag_start_card_reasons": list(
                        result.get("bag_start_card_reasons", []) or []
                    ),
                    "step_sequence_role": step_sequence_role,
                    "step_drop": step_drop,
                    "previous_max_step": previous_max_step,
                    "bag_start_layout": bag_start_layout,
                    "inferred_bag_number_source": inferred_bag_number_source,
                    "white_box_transition_found": bool(
                        result.get("white_box_transition_found")
                    ),
                    "slow_page": slow_page,
                }
            )
            continue

        bag_starts.append(
            {
                "page": int(result.get("page", page) or page),
                "bag_number": effective_bag_number,
                "panel_found": bool(result.get("panel_found")),
                "panel_source": result.get("panel_source"),
                "number_box_found": bool(result.get("number_box_found")),
                "multi_step_green_boxes": bool(result.get("multi_step_green_boxes")),
                "page_kind": result.get("page_kind", "other"),
                "confidence": float(result.get("confidence", 0.0) or 0.0),
                "ocr_raw": result.get("ocr_raw", "") or "",
                "bag_start_card_found": bool(result.get("bag_start_card_found")),
                "bag_start_card_score": float(
                    result.get("bag_start_card_score", 0.0) or 0.0
                ),
                "bag_start_card_box": result.get("bag_start_card_box"),
                "bag_start_card_reasons": list(
                    result.get("bag_start_card_reasons", []) or []
                ),
                "step_sequence_role": step_sequence_role,
                "step_drop": step_drop,
                "previous_max_step": previous_max_step,
                "bag_start_layout": bag_start_layout,
                "inferred_bag_number_source": inferred_bag_number_source,
                "white_box_transition_found": bool(
                    result.get("white_box_transition_found")
                ),
                "slow_page": slow_page,
            }
        )

    bag_starts.sort(key=lambda item: int(item["page"]))
    bag_starts = _filter_bag_starts_for_save(
        bag_starts=bag_starts,
        skipped_rows=skipped_rows,
    )
    if expected_next_bag is not None:
        filtered_bag_starts = []
        current_expected_bag = int(expected_next_bag)
        for item in bag_starts:
            candidate_bag_number = item.get("bag_number")
            if (
                int(candidate_bag_number or 0) == 1
                and bool(item.get("bag_start_card_found"))
            ):
                filtered_bag_starts.append(item)
                current_expected_bag = 2
                continue
            if candidate_bag_number == current_expected_bag:
                filtered_bag_starts.append(item)
                current_expected_bag += 1
                continue

            skipped_rows.append(
                {
                    "page": int(item["page"]),
                    "skipped": True,
                    "skip_reason": "unexpected_bag_number",
                    "expected_bag_number": current_expected_bag,
                    "candidate_bag_number": candidate_bag_number,
                    "step_sequence_role": item.get("step_sequence_role"),
                    "step_drop": item.get("step_drop"),
                    "previous_max_step": item.get("previous_max_step"),
                    "bag_start_layout": item.get("bag_start_layout"),
                    "slow_page": bool(item.get("slow_page")),
                }
            )

        bag_starts = filtered_bag_starts

    kept_bag_starts = []
    previous_kept_bag_number = None
    for item in bag_starts:
        candidate_bag_number = item.get("bag_number")
        if previous_kept_bag_number is None:
            kept_bag_starts.append(item)
            previous_kept_bag_number = candidate_bag_number
            continue

        if candidate_bag_number == previous_kept_bag_number + 1:
            kept_bag_starts.append(item)
            previous_kept_bag_number = candidate_bag_number
            continue

        skipped_rows.append(
            {
                "page": int(item["page"]),
                "skipped": True,
                "skip_reason": "bag_number_jump",
                "previous_bag_number": previous_kept_bag_number,
                "candidate_bag_number": candidate_bag_number,
                "step_sequence_role": item.get("step_sequence_role"),
                "step_drop": item.get("step_drop"),
                "previous_max_step": item.get("previous_max_step"),
                "bag_start_layout": item.get("bag_start_layout"),
                "slow_page": bool(item.get("slow_page")),
            }
        )

    bag_starts = kept_bag_starts
    bag_starts, bag_starts_to_save = _split_bag_starts_for_existing_truth(
        bag_starts=bag_starts,
        existing_truth_by_page=existing_truth_by_page,
        skipped_rows=skipped_rows,
    )

    next_start_page = None
    remaining_pages = 0
    if last_scanned_page is not None:
        remaining_page_list = [page for page in pages if int(page) > int(last_scanned_page)]
        next_start_page = (
            int(remaining_page_list[0]) if remaining_page_list else None
        )
        remaining_pages = len(remaining_page_list)

    bag_ranges = []
    if scanned_pages_count > 0 and last_scanned_page is not None:
        bag_ranges = _build_bag_ranges(bag_starts, int(last_scanned_page))
    if bag_starts_to_save:
        save_summary = bag_truth_store.save_many_bag_starts(
            set_num=set_num,
            bag_starts=bag_starts_to_save,
            source="detector",
        )
    else:
        save_summary = {
            "saved_truth": bag_truth_store.get_bag_truth(set_num),
            "conflicts": bag_truth_store.get_conflicts(set_num),
        }

    return {
        "ok": True,
        "set_num": set_num,
        "pages_dir": str(pages_dir),
        "page_count": total_pages,
        "scanned_pages_count": scanned_pages_count,
        "next_start_page": next_start_page,
        "remaining_pages": remaining_pages,
        "timed_out": False,
        "elapsed_seconds_total": round(time.monotonic() - started_at, 3),
        "page_timings": page_timings,
        "slow_pages": slow_pages,
        "step_role_map": step_role_map,
        "skipped_rows": skipped_rows,
        "bag_starts": bag_starts,
        "bag_ranges": bag_ranges,
        "saved_truth": save_summary.get("saved_truth", []),
        "conflicts": save_summary.get("conflicts", []),
    }


@router.get("/api/full-bag-scan")
def full_bag_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    chunk_size: int = Query(5, ge=1),
    max_chunks: int = Query(3, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    started_at = time.monotonic()
    current_start = int(start) if start is not None else 1
    current_expected_bag = (
        int(expected_next_bag) if expected_next_bag is not None else None
    )
    chunks_processed = 0
    pages_scanned = 0
    last_scanned_page = None
    all_bag_starts = []
    all_skipped_rows = []
    all_step_role_map = []
    step_role_pages_seen = set()

    while True:
        if chunks_processed >= max_chunks:
            break

        chunk = bag_starts_scan(
            set_num=set_num,
            start=current_start,
            end=end,
            limit=chunk_size,
            expected_next_bag=current_expected_bag,
        )
        chunks_processed += 1

        if not chunk.get("ok"):
            return {
                "ok": False,
                "set_num": set_num,
                "error": chunk.get("error", "bag-starts-scan failed"),
                "bag_starts": [],
                "bag_ranges": [],
                "skipped_rows": [],
                "pages_scanned": pages_scanned,
                "chunks_processed": chunks_processed,
                "total_time_seconds": round(time.monotonic() - started_at, 3),
            }

        chunk_bag_starts = list(chunk.get("bag_starts", []) or [])
        chunk_skipped_rows = list(chunk.get("skipped_rows", []) or [])
        accepted_chunk_bag_starts = []
        if current_expected_bag is not None:
            next_expected_bag = int(current_expected_bag)
            for item in chunk_bag_starts:
                candidate_bag_number = item.get("bag_number")
                if (
                    int(candidate_bag_number or 0) == 1
                    and bool(item.get("bag_start_card_found"))
                ):
                    accepted_chunk_bag_starts.append(item)
                    next_expected_bag = 2
                    continue
                if candidate_bag_number == next_expected_bag:
                    accepted_chunk_bag_starts.append(item)
                    next_expected_bag += 1
                    continue

                chunk_skipped_rows.append(
                    {
                        "page": int(item.get("page", 0) or 0),
                        "skipped": True,
                        "skip_reason": "unexpected_bag_number",
                        "expected_bag_number": next_expected_bag,
                        "candidate_bag_number": candidate_bag_number,
                        "step_sequence_role": item.get("step_sequence_role"),
                        "step_drop": item.get("step_drop"),
                        "previous_max_step": item.get("previous_max_step"),
                        "bag_start_layout": item.get("bag_start_layout"),
                        "slow_page": bool(item.get("slow_page")),
                    }
                )
        else:
            accepted_chunk_bag_starts = chunk_bag_starts

        all_bag_starts.extend(accepted_chunk_bag_starts)
        if current_expected_bag is not None:
            if accepted_chunk_bag_starts:
                current_expected_bag = next_expected_bag
        elif accepted_chunk_bag_starts:
            current_expected_bag = (
                int(accepted_chunk_bag_starts[-1]["bag_number"]) + 1
            )
        all_skipped_rows.extend(chunk_skipped_rows)
        for row in chunk.get("step_role_map", []) or []:
            page = int(row.get("page", 0) or 0)
            if page in step_role_pages_seen:
                continue
            step_role_pages_seen.add(page)
            all_step_role_map.append(row)
        pages_scanned += int(chunk.get("scanned_pages_count", 0) or 0)

        chunk_next_start_page = chunk.get("next_start_page")
        if chunk.get("scanned_pages_count", 0):
            if chunk_next_start_page is not None:
                last_scanned_page = int(chunk_next_start_page) - 1
            else:
                page_count = chunk.get("page_count")
                if page_count is not None:
                    last_scanned_page = int(page_count)

        if chunk_next_start_page is None:
            break

        current_start = int(chunk_next_start_page)

    all_bag_starts.sort(key=lambda item: int(item["page"]))
    existing_bag_truth = list(bag_truth_store.get_bag_truth(set_num) or [])
    existing_truth_by_page = _build_existing_truth_by_page(existing_bag_truth)
    all_bag_starts, all_bag_starts_to_save = _split_bag_starts_for_existing_truth(
        bag_starts=all_bag_starts,
        existing_truth_by_page=existing_truth_by_page,
        skipped_rows=all_skipped_rows,
    )
    bag_ranges = _build_bag_ranges(all_bag_starts, last_scanned_page)
    if all_bag_starts_to_save:
        save_summary = bag_truth_store.save_many_bag_starts(
            set_num=set_num,
            bag_starts=all_bag_starts_to_save,
            source="detector",
        )
    else:
        save_summary = {
            "saved_truth": bag_truth_store.get_bag_truth(set_num),
            "conflicts": bag_truth_store.get_conflicts(set_num),
        }

    return {
        "ok": True,
        "set_num": set_num,
        "bag_starts": all_bag_starts,
        "bag_ranges": bag_ranges,
        "step_role_map": sorted(
            all_step_role_map,
            key=lambda item: int(item.get("page", 0) or 0),
        ),
        "skipped_rows": all_skipped_rows,
        "saved_truth": save_summary.get("saved_truth", []),
        "conflicts": save_summary.get("conflicts", []),
        "pages_scanned": pages_scanned,
        "chunks_processed": chunks_processed,
        "total_time_seconds": round(time.monotonic() - started_at, 3),
    }


@router.get("/api/bag-find-chunked")
def api_bag_find_chunked(
    set_num: str = Query(...),
    start: int = Query(1, ge=1),
    chunk_size: int = Query(10, ge=1),
):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="no rendered pages found for set")

    rendered_pages = _list_rendered_pages(pages_dir)
    total_pages = len(rendered_pages)
    last_rendered_page = int(rendered_pages[-1]) if rendered_pages else None
    scanned_start = int(start)
    requested_end = int(scanned_start) + int(chunk_size) - 1
    scanned_end = (
        min(int(requested_end), int(last_rendered_page))
        if last_rendered_page is not None
        else int(requested_end)
    )

    chunk = bag_starts_scan(
        set_num=set_num,
        start=scanned_start,
        end=requested_end,
        limit=chunk_size,
        expected_next_bag=None,
    )

    next_start_page = None
    for page in rendered_pages:
        if int(page) > int(requested_end):
            next_start_page = int(page)
            break

    return {
        "set_num": str(set_num),
        "total_pages": int(total_pages),
        "scanned_start": int(scanned_start),
        "scanned_end": int(scanned_end),
        "next_start_page": next_start_page,
        "done": next_start_page is None,
        "bag_starts": list(chunk.get("bag_starts", []) or []),
        "saved_truth": list(chunk.get("saved_truth", []) or []),
        "skipped_rows": list(chunk.get("skipped_rows", []) or []),
    }


@router.get("/api/bag-truth")
def api_bag_truth(
    set_num: str = Query(...),
):
    return {
        "set_num": str(set_num),
        "saved_truth": bag_truth_store.get_bag_truth(set_num),
        "conflicts": bag_truth_store.get_conflicts(set_num),
    }


@router.get("/api/bag-candidate-report")
def api_bag_candidate_report(
    set_num: str = Query(...),
    start: int = Query(..., ge=1),
    end: int = Query(..., ge=1),
):
    if int(end) < int(start):
        raise HTTPException(status_code=400, detail="end must be >= start")

    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="no rendered pages found for set")

    rendered_pages = [
        int(page)
        for page in _list_rendered_pages(pages_dir)
        if int(start) <= int(page) <= int(end)
    ]

    configure_pages_dir(str(pages_dir))
    step_role_map = step_sequence_bag_service.build_step_sequence_prepass_for_pages(
        set_num=set_num,
        pages=rendered_pages,
    )
    step_role_by_page = {
        int(item.get("page", 0) or 0): item for item in (step_role_map or [])
    }
    existing_bag_truth = list(bag_truth_store.get_bag_truth(set_num) or [])
    existing_truth_by_page = _build_existing_truth_by_page(existing_bag_truth)

    rows = []
    for page in rendered_pages:
        result = analyze_page(int(page), include_image=False)
        rows.append(
            _build_bag_candidate_report_row(
                set_num=set_num,
                page=int(page),
                result=result,
                step_role_row=step_role_by_page.get(int(page)),
                existing_bag_truth=existing_bag_truth,
                existing_truth_by_page=existing_truth_by_page,
                expected_next_bag=None,
            )
        )

    return {
        "set_num": str(set_num),
        "start": int(start),
        "end": int(end),
        "rows": rows,
    }


@router.get("/api/missing-bag-gap-report")
def api_missing_bag_gap_report(
    set_num: str = Query(...),
    from_bag: int = Query(..., ge=1),
    to_bag: int = Query(..., ge=1),
):
    if int(to_bag) <= int(from_bag):
        raise HTTPException(status_code=400, detail="to_bag must be > from_bag")

    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="no rendered pages found for set")

    bag_truth = sorted(
        list(bag_truth_store.get_bag_truth(set_num) or []),
        key=lambda item: int(item.get("start_page", 0) or 0),
    )
    from_bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) == int(from_bag)
        ),
        None,
    )
    to_bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) == int(to_bag)
        ),
        None,
    )
    if from_bag_row is None:
        raise HTTPException(status_code=404, detail=f"from_bag {from_bag} not found in bag_truth")
    if to_bag_row is None:
        raise HTTPException(status_code=404, detail=f"to_bag {to_bag} not found in bag_truth")

    from_bag_page = int(from_bag_row.get("start_page", 0) or 0)
    to_bag_page = int(to_bag_row.get("start_page", 0) or 0)
    if int(to_bag_page) <= int(from_bag_page):
        raise HTTPException(
            status_code=400,
            detail="to_bag start page must be after from_bag start page",
        )

    missing_bag_numbers = list(range(int(from_bag) + 1, int(to_bag)))
    rendered_pages = [
        int(page)
        for page in _list_rendered_pages(pages_dir)
        if int(from_bag_page) < int(page) < int(to_bag_page)
    ]

    configure_pages_dir(str(pages_dir))
    step_role_map = step_sequence_bag_service.build_step_sequence_prepass_for_pages(
        set_num=set_num,
        pages=rendered_pages,
    )
    step_role_by_page = {
        int(item.get("page", 0) or 0): item for item in (step_role_map or [])
    }

    working_truth = list(bag_truth)
    rows = []
    for page in rendered_pages:
        existing_truth_by_page = _build_existing_truth_by_page(working_truth)
        result = analyze_page(int(page), include_image=False)
        report_row = _build_bag_candidate_report_row(
            set_num=set_num,
            page=int(page),
            result=result,
            step_role_row=step_role_by_page.get(int(page)),
            existing_bag_truth=working_truth,
            existing_truth_by_page=existing_truth_by_page,
            expected_next_bag=None,
        )
        step_role = str(report_row.get("step_sequence_role") or "")
        enriched_row = {
            "page": int(page),
            "detected_bag_number": report_row.get("bag_number"),
            "bag_start_card_found": bool(report_row.get("bag_start_card_found")),
            "white_box_transition_found": bool(
                report_row.get("white_box_transition_found")
            ),
            "step_sequence_reset": step_role == "step_sequence_reset",
            "hero_transition_reset": step_role == "hero_transition_reset",
            "main_step_min": (step_role_by_page.get(int(page)) or {}).get("main_step_min"),
            "main_step_max": (step_role_by_page.get(int(page)) or {}).get("main_step_max"),
            "final_decision": report_row.get("final_decision"),
            "rejection_reason": report_row.get("rejection_reason"),
            "why": (
                report_row.get("rejection_reason")
                if report_row.get("final_decision") == "rejected"
                else str(report_row.get("bag_start_layout") or "accepted")
            ),
            "thumbnail_url": f"/debug/page-image?set_num={set_num}&page={int(page)}",
            "debug_page_image_url": report_row.get("debug_page_image_url"),
            "bag_start_layout": report_row.get("bag_start_layout"),
            "inferred_bag_number_source": report_row.get("inferred_bag_number_source"),
        }
        rows.append(enriched_row)

        if report_row.get("final_decision") == "accepted":
            accepted_bag_number = int(report_row.get("bag_number", 0) or 0)
            if accepted_bag_number > 0 and not any(
                int(item.get("start_page", 0) or 0) == int(page)
                for item in working_truth
            ):
                working_truth.append(
                    {
                        "bag_number": accepted_bag_number,
                        "start_page": int(page),
                    }
                )
                working_truth.sort(
                    key=lambda item: int(item.get("start_page", 0) or 0)
                )

    return {
        "set_num": str(set_num),
        "from_bag": int(from_bag),
        "to_bag": int(to_bag),
        "from_bag_page": int(from_bag_page),
        "to_bag_page": int(to_bag_page),
        "missing_bag_numbers": missing_bag_numbers,
        "scan_start_page": int(from_bag_page) + 1,
        "scan_end_page": int(to_bag_page) - 1,
        "rows": rows,
    }


@router.get("/api/bag-step-ranges")
def api_bag_step_ranges(
    set_num: str = Query(...),
    start_bag: Optional[int] = Query(None, ge=1),
    end_bag: Optional[int] = Query(None, ge=1),
    max_pages: int = Query(10, ge=1),
):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="no rendered pages found for set")

    pages = _list_rendered_pages(pages_dir)
    last_page = int(pages[-1]) if pages else None
    bag_truth = sorted(
        bag_truth_store.get_bag_truth(set_num),
        key=lambda item: (
            int(item.get("bag_number", 0) or 0),
            int(item.get("start_page", 0) or 0),
        ),
    )
    if start_bag is not None:
        bag_truth = [
            item for item in bag_truth if int(item.get("bag_number", 0) or 0) >= int(start_bag)
        ]
    if end_bag is not None:
        bag_truth = [
            item for item in bag_truth if int(item.get("bag_number", 0) or 0) <= int(end_bag)
        ]

    configure_pages_dir(str(pages_dir))
    bag_ranges = []
    total_bags_requested = len(bag_truth)
    pages_scanned = 0
    stopped_early = False
    next_bag_number = None
    progress = {
        "current_bag_number": None,
        "current_page": None,
        "total_bags_requested": total_bags_requested,
    }

    for index, item in enumerate(bag_truth):
        if pages_scanned >= int(max_pages):
            stopped_early = True
            next_bag_number = int(item["bag_number"])
            progress["current_bag_number"] = int(item["bag_number"])
            progress["current_page"] = int(item["start_page"])
            break

        bag_number = int(item["bag_number"])
        start_page = int(item["start_page"])
        next_start_page = (
            int(bag_truth[index + 1]["start_page"])
            if index + 1 < len(bag_truth)
            else None
        )
        end_page = int(next_start_page) - 1 if next_start_page is not None else None
        scan_end_page = end_page if end_page is not None else last_page

        main_step_values = []
        if scan_end_page is not None and int(scan_end_page) >= start_page:
            for page in pages:
                if int(page) < start_page:
                    continue
                if int(page) > int(scan_end_page):
                    break
                if pages_scanned >= int(max_pages):
                    stopped_early = True
                    next_bag_number = bag_number
                    progress["current_bag_number"] = bag_number
                    progress["current_page"] = int(page)
                    break
                progress["current_bag_number"] = bag_number
                progress["current_page"] = int(page)
                pages_scanned += 1
                main_step_values.extend(_load_page_main_steps(set_num, int(page)))

        result_end_page = end_page
        if stopped_early and progress.get("current_page") is not None:
            result_end_page = int(progress["current_page"]) - 1

        bag_ranges.append(
            {
                "bag_number": bag_number,
                "start_page": start_page,
                "end_page": result_end_page,
                "start_step": (
                    int(min(main_step_values)) if main_step_values else None
                ),
                "end_step": int(max(main_step_values)) if main_step_values else None,
            }
        )
        if stopped_early:
            break

    return {
        "set_num": str(set_num),
        "bag_ranges": bag_ranges,
        "pages_scanned": int(pages_scanned),
        "max_pages": int(max_pages),
        "stopped_early": bool(stopped_early),
        "next_bag_number": next_bag_number,
        "progress": progress,
    }


@router.get("/debug/full-bag-scan", response_class=HTMLResponse)
def debug_full_bag_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    chunk_size: int = Query(5, ge=1),
    max_chunks: int = Query(3, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    payload = full_bag_scan(
        set_num=set_num,
        start=start,
        end=end,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        expected_next_bag=expected_next_bag,
    )

    saved_path = None
    save_error = None
    try:
        out_path = debug_service.DEBUG_ROOT / str(set_num) / "full_bag_scan.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved_path = str(out_path)
    except Exception as exc:
        save_error = str(exc)

    raw_json_url = (
        f"/api/full-bag-scan?set_num={escape(str(set_num))}"
        f"{'' if start is None else f'&start={int(start)}'}"
        f"{'' if end is None else f'&end={int(end)}'}"
        f"&chunk_size={int(chunk_size)}"
        f"&max_chunks={int(max_chunks)}"
        f"{'' if expected_next_bag is None else f'&expected_next_bag={int(expected_next_bag)}'}"
    )

    def _rows(items):
        if not items:
            return "<tr><td colspan='6'>None</td></tr>"
        rows = []
        for item in items:
            rows.append(
                "<tr>"
                f"<td>{escape(str(item.get('page', '-')))}</td>"
                f"<td>{escape(str(item.get('bag_number', '-')))}</td>"
                f"<td>{escape(str(item.get('start_page', '-')))}</td>"
                f"<td>{escape(str(item.get('end_page', '-')))}</td>"
                f"<td>{escape(str(item.get('skip_reason', '-')))}</td>"
                f"<td><pre>{escape(json.dumps(item, indent=2))}</pre></td>"
                "</tr>"
            )
        return "\n".join(rows)

    bag_starts_rows = _rows(payload.get("bag_starts", []))
    bag_ranges_rows = _rows(payload.get("bag_ranges", []))
    skipped_rows = _rows(payload.get("skipped_rows", []))
    slow_pages_rows = _rows(payload.get("slow_pages", [])) if "slow_pages" in payload else "<tr><td colspan='6'>None</td></tr>"
    step_role_rows = (
        "\n".join(
            [
                "<tr>"
                f"<td>{escape(str(item.get('page', '-')))}</td>"
                f"<td>{escape(str(item.get('step_sequence_role', '-')))}</td>"
                f"<td>{escape(str(item.get('main_step_min', '-')))}</td>"
                f"<td>{escape(str(item.get('main_step_max', '-')))}</td>"
                f"<td>{escape(str(item.get('previous_max_step', '-')))}</td>"
                f"<td>{escape(str(item.get('step_drop', '-')))}</td>"
                f"<td>{escape(str(item.get('bag_start_allowed', '-')))}</td>"
                f"<td>{escape(str(item.get('step_role_reason', '-')))}</td>"
                "</tr>"
                for item in (payload.get('step_role_map', []) or [])
            ]
        )
        if (payload.get("step_role_map", []) or [])
        else "<tr><td colspan='8'>None</td></tr>"
    )
    page_timings_json = escape(json.dumps(payload.get("page_timings", []), indent=2))

    saved_block = (
        f"<p><strong>Saved debug JSON:</strong> <code>{escape(saved_path)}</code></p>"
        if saved_path
        else (
            f"<p><strong>Save error:</strong> {escape(save_error)}</p>"
            if save_error
            else ""
        )
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Full Bag Scan</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
            .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
            .btn {{ display:inline-block; padding:8px 12px; background:#222; color:#fff; text-decoration:none; border-radius:6px; margin-right:8px; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; margin-bottom: 16px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background: #f7f7f7; }}
            pre {{ white-space: pre-wrap; margin: 0; }}
            code {{ white-space: pre-wrap; }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Full Bag Scan for {escape(str(set_num))}</h2>
            <p><strong>Params:</strong> start={escape(str(start))}, end={escape(str(end))}, chunk_size={int(chunk_size)}, max_chunks={int(max_chunks)}, expected_next_bag={escape(str(expected_next_bag))}</p>
            <p><strong>Pages scanned:</strong> {int(payload.get("pages_scanned", 0) or 0)} | <strong>Chunks processed:</strong> {int(payload.get("chunks_processed", 0) or 0)} | <strong>Total seconds:</strong> {escape(str(payload.get("total_time_seconds", 0.0)))}</p>
            {saved_block}
            <p>
              <a class="btn" href="{raw_json_url}" target="_blank" download>Download JSON</a>
              <a class="btn" href="{raw_json_url}" target="_blank">Open Raw JSON</a>
            </p>
          </div>

          <div class="card"><h3>Accepted Bag Starts</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{bag_starts_rows}</tbody>
          </table>

          <div class="card"><h3>Bag Ranges</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{bag_ranges_rows}</tbody>
          </table>

          <div class="card"><h3>Skipped Rows</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{skipped_rows}</tbody>
          </table>

          <div class="card"><h3>Slow Pages</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{slow_pages_rows}</tbody>
          </table>

          <div class="card"><h3>Step Sequence Prepass</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Role</th><th>Main Min</th><th>Main Max</th><th>Prev Max</th><th>Step Drop</th><th>Allowed</th><th>Reason</th></tr></thead>
            <tbody>{step_role_rows}</tbody>
          </table>

          <div class="card">
            <h3>Page Timings</h3>
            <pre>{page_timings_json}</pre>
          </div>
        </body>
        </html>
        """
    )


@router.get("/api/set-workflow")
def set_workflow(set_num: str = Query(...)):
    return workflow_service.get_set_workflow(set_num)


@router.post("/api/auto-confirm-top-candidates")
def auto_confirm_top_candidates(
    set_num: Optional[str] = Query(None),
    set_num_form: Optional[str] = Form(None, alias="set_num"),
    redirect_to: Optional[str] = Form(None),
):
    selected_set_num = set_num_form or set_num
    if not selected_set_num:
        raise HTTPException(status_code=400, detail="set_num is required")

    summary = auto_confirm_service.auto_confirm_top_candidates(selected_set_num)
    if redirect_to:
        return RedirectResponse(url=redirect_to, status_code=303)
    return summary


@router.get("/debug/set-overview", response_class=HTMLResponse)
def debug_set_overview(set_num: str = Query(...)):
    workflow = workflow_service.get_set_workflow(set_num)
    mode = workflow.get("mode")

    if mode == "fresh_precheck":
        result = workflow.get("analyzer_scan", {})
        if not result.get("ok"):
            return HTMLResponse(
                f"""
                <!doctype html>
                <html>
                <head>
                  <meta charset="utf-8" />
                  <title>Set overview</title>
                  <style>
                    body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
                    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
                  </style>
                </head>
                <body>
                  <div class="card">
                    <h2>Set overview for {set_num}</h2>
                    <p><strong>Mode:</strong> fresh_precheck</p>
                    <p><strong>Source:</strong> legacy analyzer scan with lightweight precheck skips</p>
                    <p><strong>Error:</strong> {result.get("error", "Unknown error")}</p>
                  </div>
                </body>
                </html>
                """
            )

        rows_html = []
        for row in result.get("rows", []):
            page = int(row["page"])
            image_url = f"/api/debug/page-image?set_num={set_num}&page={page}"
            precheck_url = f"/api/page-precheck?set_num={set_num}&page={page}"
            analyze_url = f"/api/analyze-page-direct?set_num={set_num}&page={page}"
            summary = row.get("analyzer_summary", {})
            save_form = f"""
            <form method="post" action="/api/debug/save-bag-truth" style="display:inline-flex; gap:6px; align-items:center; margin:0;">
              <input type="hidden" name="set_num" value="{set_num}">
              <input type="hidden" name="start_page" value="{page}">
              <input type="hidden" name="redirect_to" value="/debug/set-overview?set_num={set_num}">
              <input type="number" name="bag_number" min="1" placeholder="Bag #" value="{'' if summary.get('bag_number') is None else summary.get('bag_number')}" required style="width:84px;">
              <button type="submit">Save as truth</button>
            </form>
            """

            rows_html.append(
                f"""
                <tr>
                  <td>{page}</td>
                  <td>{row.get("precheck_kind", "-")}</td>
                  <td>{'Y' if summary.get("panel_found") else ''}</td>
                  <td>{'Y' if summary.get("shell_found") else ''}</td>
                  <td>{'Y' if summary.get("grey_bag_found") else ''}</td>
                  <td>{'Y' if summary.get("number_found") else ''}</td>
                  <td>{'' if summary.get("bag_number") is None else summary.get("bag_number")}</td>
                  <td>{float(summary.get("confidence", 0.0) or 0.0):.2f}</td>
                  <td>
                    <a href="{image_url}" target="_blank">View image</a>
                    &nbsp;|&nbsp;
                    <a href="{precheck_url}" target="_blank">Precheck JSON</a>
                    &nbsp;|&nbsp;
                    <a href="{analyze_url}" target="_blank">Analyze direct</a>
                    &nbsp;|&nbsp;
                    {save_form}
                  </td>
                </tr>
                """
            )

        rows_block = (
            "\n".join(rows_html)
            if rows_html
            else "<tr><td colspan='9'>No analyzed rows</td></tr>"
        )

        return HTMLResponse(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Set overview</title>
              <style>
                body {{
                  font-family: Arial, sans-serif;
                  margin: 16px;
                  background: #f3f3f3;
                }}
                .card {{
                  background: #fff;
                  border: 1px solid #ddd;
                  border-radius: 8px;
                  padding: 12px;
                  margin-bottom: 16px;
                }}
                table {{
                  width: 100%;
                  border-collapse: collapse;
                  background: #fff;
                }}
                th, td {{
                  border: 1px solid #ddd;
                  padding: 8px;
                  text-align: left;
                  vertical-align: top;
                }}
                th {{
                  background: #f7f7f7;
                }}
                .btn {{
                  display:inline-block;
                  padding:8px 12px;
                  background:#222;
                  color:#fff;
                  text-decoration:none;
                  border-radius:6px;
                }}
                button {{
                  padding: 6px 10px;
                  cursor: pointer;
                }}
              </style>
            </head>
            <body>
              <div class="card">
                <h2>Set overview for {set_num}</h2>
                <p><strong>Mode:</strong> fresh_precheck</p>
                <p>This view skips only obvious junk pages, then ranks candidates using the legacy analyzer.</p>
                <p><strong>Pages dir:</strong> {result.get("pages_dir", "-")}</p>
                <p><strong>Total rendered pages:</strong> {result.get("page_count", 0)}</p>
                <p><strong>Rows shown:</strong> {result.get("row_count", 0)}</p>
                <form method="post" action="/api/auto-confirm-top-candidates" style="margin: 12px 0;">
                  <input type="hidden" name="set_num" value="{set_num}">
                  <input type="hidden" name="redirect_to" value="/debug/set-overview?set_num={set_num}">
                  <button type="submit">Auto-confirm strong candidates</button>
                </form>
                <p>
                  <a class="btn" href="/api/set-workflow?set_num={set_num}" target="_blank">Workflow JSON</a>
                  &nbsp;
                  <a class="btn" href="/api/scan-set-with-analyzer?set_num={set_num}&include_all=false" target="_blank">Raw analyzer JSON</a>
                  &nbsp;
                  <a class="btn" href="/debug/scan-set-with-analyzer?set_num={set_num}&include_all=false">Open analyzer table</a>
                </p>
              </div>

              <table>
                <thead>
                  <tr>
                    <th>Page</th>
                    <th>Precheck kind</th>
                    <th>Panel</th>
                    <th>Shell</th>
                    <th>Grey bag</th>
                    <th>Number</th>
                    <th>Bag number</th>
                    <th>Confidence</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {rows_block}
                </tbody>
              </table>
            </body>
            </html>
            """
        )

    sequence = workflow.get("sequence", {})
    confirmed_rows = sequence.get("confirmed", [])
    missing_numbers = sequence.get("missing_numbers", [])
    missing_windows = sequence.get("missing_bag_windows", [])

    confirmed_rows_html = []
    for row in confirmed_rows:
        page = int(row.get("start_page", row.get("page", 0)))
        bag_number = int(row["bag_number"])
        image_url = f"/api/debug/page-image?set_num={set_num}&page={page}"
        confirmed_rows_html.append(
            f"""
            <tr>
              <td>{bag_number}</td>
              <td>{page}</td>
              <td>{float(row.get("confidence", 0.0) or 0.0):.2f}</td>
              <td>{row.get("source", "-")}</td>
              <td><a href="{image_url}" target="_blank">View image</a></td>
            </tr>
            """
        )

    missing_rows_html = []
    for window in missing_windows:
        bag_number = int(window["bag_number"])
        gap_url = f"/debug/gap-table?set_num={set_num}&bag_number={bag_number}"
        missing_rows_html.append(
            f"""
            <tr>
              <td>{bag_number}</td>
              <td>{window.get("previous_confirmed_bag")}</td>
              <td>{window.get("previous_confirmed_page")}</td>
              <td>{window.get("next_confirmed_bag")}</td>
              <td>{window.get("next_confirmed_page")}</td>
              <td><a href="{gap_url}">Open gap table</a></td>
            </tr>
            """
        )

    confirmed_block = (
        "\n".join(confirmed_rows_html)
        if confirmed_rows_html
        else "<tr><td colspan='5'>No confirmed rows</td></tr>"
    )
    missing_block = (
        "\n".join(missing_rows_html)
        if missing_rows_html
        else "<tr><td colspan='6'>No missing bag windows</td></tr>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Set overview</title>
          <style>
            body {{
              font-family: Arial, sans-serif;
              margin: 16px;
              background: #f3f3f3;
            }}
            .card {{
              background: #fff;
              border: 1px solid #ddd;
              border-radius: 8px;
              padding: 12px;
              margin-bottom: 16px;
            }}
            table {{
              width: 100%;
              border-collapse: collapse;
              background: #fff;
              margin-bottom: 16px;
            }}
            th, td {{
              border: 1px solid #ddd;
              padding: 8px;
              text-align: left;
              vertical-align: top;
            }}
            th {{
              background: #f7f7f7;
            }}
            .btn {{
              display:inline-block;
              padding:8px 12px;
              background:#222;
              color:#fff;
              text-decoration:none;
              border-radius:6px;
            }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Set overview for {set_num}</h2>
            <p><strong>Mode:</strong> truth_guided</p>
            <p><strong>Missing numbers:</strong> {missing_numbers}</p>
            <p>
              <a class="btn" href="/api/set-workflow?set_num={set_num}" target="_blank">Workflow JSON</a>
              &nbsp;
              <a class="btn" href="/api/sequence-scan?set_num={set_num}" target="_blank">Sequence JSON</a>
            </p>
          </div>

          <div class="card">
            <h3>Confirmed rows</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Bag</th>
                <th>Page</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {confirmed_block}
            </tbody>
          </table>

          <div class="card">
            <h3>Missing bag windows</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Bag</th>
                <th>Prev bag</th>
                <th>Prev page</th>
                <th>Next bag</th>
                <th>Next page</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {missing_block}
            </tbody>
          </table>
        </body>
        </html>
        """
    )
