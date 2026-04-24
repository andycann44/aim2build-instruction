from typing import Any, Dict, List, Optional

from clean.services import debug_service, step_detector_service


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
            if any(int(v) > 200 for v in step_values):
                page_row["reason"] = "rejected implausible step value > 200"
                page_steps.append(page_row)
                continue

            current_min_step = min(step_values)
            current_max_step = max(step_values)
            reset_confidence = 0.0
            drop = None
            if previous_max_step is not None:
                if int(previous_max_step) > 200:
                    page_row["reason"] = "rejected previous max step > 200"
                    previous_max_step = current_max_step
                    page_steps.append(page_row)
                    continue
                drop = int(previous_max_step) - int(current_min_step)
                reset_confidence = _score_step_reset(
                    previous_max_step=previous_max_step,
                    current_min_step=current_min_step,
                    current_max_step=current_max_step,
                    step_values=step_values,
                )

            if (
                previous_max_step is not None
                and current_min_step == 1
                and previous_max_step >= 15
                and drop is not None
                and drop >= 10
            ):
                page_row["bag_start"] = True
                page_row["reason"] = "strong bag reset %d -> 1" % (
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
