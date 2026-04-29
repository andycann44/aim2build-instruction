import json
from typing import Any, Dict, List, Optional, Tuple

from clean.services import debug_service, step_detector_service

STEP_CACHE_PATH = debug_service.DEBUG_ROOT / "step_cache.json"
page_step_cache: Dict[Tuple[str, int], List[int]] = {}


def _load_page_step_cache_from_disk() -> Dict[Tuple[str, int], List[int]]:
    if not STEP_CACHE_PATH.exists():
        return {}

    try:
        payload = json.loads(STEP_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    loaded: Dict[Tuple[str, int], List[int]] = {}
    for set_num, page_map in (payload or {}).items():
        if not isinstance(page_map, dict):
            continue
        for page, steps in page_map.items():
            try:
                page_num = int(page)
            except (TypeError, ValueError):
                continue
            normalized = sorted(
                {
                    int(value)
                    for value in (steps or [])
                    if int(value) > 0
                }
            )
            loaded[(str(set_num), int(page_num))] = list(normalized)
    return loaded


def _save_page_step_cache_to_disk() -> None:
    serialized: Dict[str, Dict[str, List[int]]] = {}
    for (set_num, page), steps in sorted(
        page_step_cache.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1])),
    ):
        set_key = str(set_num)
        page_key = str(int(page))
        serialized.setdefault(set_key, {})[page_key] = [int(value) for value in steps]

    STEP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEP_CACHE_PATH.write_text(
        json.dumps(serialized, indent=2, sort_keys=True),
        encoding="utf-8",
    )


page_step_cache.update(_load_page_step_cache_from_disk())


def get_cached_page_main_steps(set_num: str, page: int) -> Optional[List[int]]:
    cached = page_step_cache.get((str(set_num), int(page)))
    if cached is None:
        return None
    print(f"CACHE HIT {set_num}-{page}")
    return list(cached)


def store_cached_page_main_steps(
    set_num: str,
    page: int,
    main_steps: List[int],
) -> List[int]:
    normalized = sorted(
        {
            int(value)
            for value in (main_steps or [])
            if int(value) > 0
        }
    )
    page_step_cache[(str(set_num), int(page))] = list(normalized)
    _save_page_step_cache_to_disk()
    print(f"CACHE MISS STORE {set_num}-{page}")
    return list(normalized)


def _list_available_pages(set_num: str) -> List[int]:
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise RuntimeError("Rendered pages not found")

    page_numbers: List[int] = []
    for path in sorted(pages_dir.glob("page_*.png")):
        try:
            page_numbers.append(int(path.stem.replace("page_", "")))
        except ValueError:
            continue
    return page_numbers


def _build_step_sequence_row(
    page: int,
    main_step_values: List[int],
    previous_max_step: Optional[int],
    classified_step_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "page": int(page),
        "main_steps": [int(value) for value in sorted(set(main_step_values))],
        "main_step_min": None,
        "main_step_max": None,
        "previous_max_step": (
            int(previous_max_step) if previous_max_step is not None else None
        ),
        "step_drop": None,
        "step_sequence_role": "no_step_data",
        "step_role_reason": "no main_steps detected",
        "bag_start_allowed": True,
        "hero_transition_reset_candidate": False,
        "detected_step_boxes": list(classified_step_boxes or []),
    }

    if not main_step_values:
        return row

    main_step_min = int(min(main_step_values))
    main_step_max = int(max(main_step_values))
    step_drop = (
        int(previous_max_step) - int(main_step_min)
        if previous_max_step is not None
        else None
    )

    row["main_step_min"] = main_step_min
    row["main_step_max"] = main_step_max
    row["step_drop"] = step_drop

    if (
        previous_max_step is not None
        and int(previous_max_step) >= 20
        and int(main_step_min) <= 9
        and step_drop is not None
        and int(step_drop) >= 10
    ):
        row["hero_transition_reset_candidate"] = True

    if (
        previous_max_step is not None
        and int(previous_max_step) >= 20
        and int(main_step_min) <= 3
        and step_drop is not None
        and int(step_drop) >= 10
    ):
        row["step_sequence_role"] = "step_sequence_reset"
        row["step_role_reason"] = (
            "main steps reset from %d to %d"
            % (int(previous_max_step), int(main_step_min))
        )
        row["bag_start_allowed"] = True
        return row

    if (
        previous_max_step is not None
        and int(main_step_min) >= int(previous_max_step)
        and int(main_step_max) >= int(previous_max_step)
    ):
        row["step_sequence_role"] = "build_step_sequence_page"
        row["step_role_reason"] = (
            "main steps continue upward from previous max %d"
            % int(previous_max_step)
        )
        row["bag_start_allowed"] = False
        return row

    row["step_sequence_role"] = "ambiguous"
    if previous_max_step is None:
        row["step_role_reason"] = "no previous max step for comparison"
    else:
        row["step_role_reason"] = (
            "main steps neither continue upward nor reset cleanly from previous max %d"
            % int(previous_max_step)
        )
    row["bag_start_allowed"] = True
    return row


def build_step_sequence_prepass_for_pages(
    set_num: str,
    pages: List[int],
) -> List[Dict[str, Any]]:
    if not pages:
        return []

    available_pages = set(_list_available_pages(set_num))
    ordered_pages = sorted({int(page) for page in pages if int(page) > 0})
    previous_max_step: Optional[int] = None
    rows: List[Dict[str, Any]] = []

    for page in ordered_pages:
        if page not in available_pages:
            continue

        detected = step_detector_service.detect_steps(set_num, int(page))
        main_steps = detected.get("main_steps", []) or []
        classified_step_boxes = detected.get("classified_step_boxes", []) or []
        main_step_values = sorted(
            {
                int(item.get("value", 0) or 0)
                for item in main_steps
                if int(item.get("value", 0) or 0) > 0
            }
        )

        row = _build_step_sequence_row(
            page=int(page),
            main_step_values=main_step_values,
            previous_max_step=previous_max_step,
            classified_step_boxes=classified_step_boxes,
        )
        rows.append(row)

        if main_step_values:
            previous_max_step = int(max(main_step_values))

    return rows


def build_step_sequence_prepass(
    set_num: str,
    start_page: int,
    end_page: int,
) -> List[Dict[str, Any]]:
    start_page = int(start_page)
    end_page = int(end_page)
    if end_page < start_page:
        raise RuntimeError("end_page must be >= start_page")
    return build_step_sequence_prepass_for_pages(
        set_num=set_num,
        pages=list(range(start_page, end_page + 1)),
    )


def _score_step_reset(
    previous_max_step: int,
    current_min_step: int,
    current_max_step: int,
    step_values: List[int],
) -> float:
    drop = int(previous_max_step) - int(current_min_step)
    score = 0.0

    if current_min_step <= 2:
        score += 0.35
    if current_min_step == 1:
        score += 0.15

    if previous_max_step >= 8:
        score += 0.20
    elif previous_max_step >= 6:
        score += 0.10

    if drop >= 10:
        score += 0.25
    elif drop >= 6:
        score += 0.18
    elif drop >= 4:
        score += 0.08

    if len(step_values) >= 2 and current_max_step <= 4:
        score += 0.10
    if 1 in step_values and 2 in step_values:
        score += 0.10

    return min(1.0, round(score, 3))


def _filter_candidate_bag_starts(
    candidates: List[Dict[str, Any]],
    min_spacing_pages: int = 5,
    max_candidates: int = 30,
) -> List[Dict[str, Any]]:
    sorted_candidates = sorted(candidates, key=lambda item: int(item.get("page", 0) or 0))
    selected: List[Dict[str, Any]] = []

    for candidate in sorted_candidates:
        candidate_page = int(candidate.get("page", 0) or 0)
        candidate_prev = int(candidate.get("_previous_max_step", 0) or 0)
        candidate_drop = int(candidate.get("_step_drop", 0) or 0)
        candidate_conf = float(candidate.get("_confidence", 0.0) or 0.0)

        replaced = False
        keep_candidate = True

        for idx, existing in enumerate(selected):
            existing_page = int(existing.get("page", 0) or 0)
            if abs(candidate_page - existing_page) >= int(min_spacing_pages):
                continue

            existing_prev = int(existing.get("_previous_max_step", 0) or 0)
            existing_drop = int(existing.get("_step_drop", 0) or 0)
            existing_conf = float(existing.get("_confidence", 0.0) or 0.0)

            candidate_rank = (candidate_prev, candidate_drop, candidate_conf, -candidate_page)
            existing_rank = (existing_prev, existing_drop, existing_conf, -existing_page)

            if candidate_rank > existing_rank:
                selected[idx] = candidate
                replaced = True
            keep_candidate = False
            break

        if keep_candidate and not replaced:
            selected.append(candidate)

    selected.sort(key=lambda item: int(item.get("page", 0) or 0))
    selected = selected[: int(max_candidates)]

    output: List[Dict[str, Any]] = []
    for item in selected:
        output.append(
            {
                "page": int(item.get("page", 0) or 0),
                "steps": item.get("steps", []),
                "bag_start": True,
                "reason": str(item.get("reason", "")),
            }
        )
    return output


def scan_step_bag_sequence(
    set_num: str,
    start_page: int,
    end_page: int,
) -> Dict[str, Any]:
    start_page = int(start_page)
    end_page = int(end_page)
    if end_page < start_page:
        raise RuntimeError("end_page must be >= start_page")

    available_pages = set(_list_available_pages(set_num))
    page_steps: List[Dict[str, Any]] = []
    candidate_bag_starts: List[Dict[str, Any]] = []
    previous_max_step: Optional[int] = None
    for page in range(start_page, end_page + 1):
        if page not in available_pages:
            continue

        detected = step_detector_service.detect_steps(set_num, page)
        main_steps = detected.get("main_steps", []) or []
        sub_steps = detected.get("sub_steps", []) or []
        if main_steps:
            step_values = sorted(
                set(int(item.get("value", 0) or 0) for item in main_steps)
            )
        else:
            step_values = sorted(
                set(int(item.get("value", 0) or 0) for item in detected.get("step_candidates", []))
            )

        page_row: Dict[str, Any] = {
            "page": int(page),
            "steps": step_values,
            "bag_start": False,
            "reason": "",
        }

        if step_values:
            if any(int(v) > 200 for v in sub_steps):
                page_row["reason"] = "rejected implausible sub step value > 200"
                page_steps.append(page_row)
                continue
            if any(int(v) > 200 for v in step_values):
                page_row["reason"] = "rejected implausible step value > 200"
                page_steps.append(page_row)
                continue

            current_min_step = min(step_values)
            current_max_step = max(step_values)
            has_main_step_one = 1 in step_values
            has_sub_step_one = 1 in [int(v) for v in sub_steps]
            effective_reset_step: Optional[int] = None
            if has_main_step_one:
                effective_reset_step = 1
            elif current_min_step >= 10 and has_sub_step_one:
                effective_reset_step = 1

            reset_confidence = 0.0
            drop = None
            if previous_max_step is not None:
                if int(previous_max_step) > 200:
                    page_row["reason"] = "rejected previous max step > 200"
                    previous_max_step = current_max_step
                    page_steps.append(page_row)
                    continue
                compare_step = (
                    int(effective_reset_step)
                    if effective_reset_step is not None
                    else int(current_min_step)
                )
                drop = int(previous_max_step) - int(compare_step)
                reset_confidence = _score_step_reset(
                    previous_max_step=previous_max_step,
                    current_min_step=compare_step,
                    current_max_step=current_max_step,
                    step_values=step_values,
                )

            if (
                previous_max_step is not None
                and effective_reset_step == 1
                and previous_max_step >= 10
                and drop is not None
                and drop >= 10
            ):
                page_row["bag_start"] = True
                if has_main_step_one:
                    page_row["reason"] = "strong bag reset %d -> 1" % (previous_max_step,)
                else:
                    page_row["reason"] = "strong bag reset %d -> 1 (from sub_steps)" % (
                        previous_max_step,
                    )
                candidate_bag_starts.append(
                    {
                        "page": int(page),
                        "steps": step_values,
                        "bag_start": True,
                        "reason": page_row["reason"],
                        "_previous_max_step": int(previous_max_step),
                        "_step_drop": int(drop),
                        "_confidence": float(reset_confidence),
                    }
                )
            elif previous_max_step is not None and drop is not None and drop > 0:
                page_row["reason"] = "rejected reset from %d -> %d" % (
                    previous_max_step,
                    current_min_step,
                )

            previous_max_step = current_max_step

        page_steps.append(page_row)

    candidate_bag_starts = _filter_candidate_bag_starts(
        candidate_bag_starts,
        min_spacing_pages=5,
        max_candidates=30,
    )

    return {
        "set_num": str(set_num),
        "start_page": int(start_page),
        "end_page": int(end_page),
        "page_steps": page_steps,
        "candidate_bag_starts": candidate_bag_starts,
    }
