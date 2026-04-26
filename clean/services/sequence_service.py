from clean.services import truth_service


def _build_confirmed_sequence(set_num: str):
    confirmed = truth_service.get_all_confirmed_bags(set_num)
    confirmed = [r for r in confirmed if "_error" not in r]
    confirmed_sorted = sorted(
        confirmed,
        key=lambda x: (x["bag_number"], x["start_page"])
    )
    return confirmed_sorted


def _find_missing_numbers(confirmed_rows):
    if not confirmed_rows:
        return []

    bag_numbers = sorted(set(int(r["bag_number"]) for r in confirmed_rows))
    if not bag_numbers:
        return []

    max_bag = max(bag_numbers)

    return [
        n for n in range(1, max_bag + 1)
        if n not in bag_numbers
    ]


def _build_bounds_for_missing(missing_number, confirmed_bag_map):
    n = int(missing_number)
    known_numbers = sorted(confirmed_bag_map.keys())

    prev_bag = None
    next_bag = None

    for k in known_numbers:
        if k < n:
            prev_bag = k
        elif k > n:
            next_bag = k
            break

    prev_page = None
    next_page = None

    if prev_bag is not None:
        prev_pages = confirmed_bag_map.get(prev_bag, [])
        if prev_pages:
            prev_page = prev_pages[0]

    if next_bag is not None:
        next_pages = confirmed_bag_map.get(next_bag, [])
        if next_pages:
            next_page = next_pages[0]

    return {
        "bag_number": n,
        "previous_confirmed_bag": prev_bag,
        "previous_confirmed_page": prev_page,
        "next_confirmed_bag": next_bag,
        "next_confirmed_page": next_page,
        "bounded": (prev_page is not None or next_page is not None),
    }


def get_missing_window(set_num: str, bag_number: int):
    confirmed = _build_confirmed_sequence(set_num)
    if not confirmed:
        return None

    confirmed_bag_map = truth_service.get_confirmed_bag_map(set_num)
    return _build_bounds_for_missing(int(bag_number), confirmed_bag_map)


def get_sequence_from_truth(set_num: str):
    confirmed = _build_confirmed_sequence(set_num)

    if not confirmed:
        raw_rows = truth_service.get_all_confirmed_bags(set_num)
        db_error = None
        if raw_rows and "_error" in raw_rows[0]:
            db_error = raw_rows[0]["_error"]

        return {
            "set_num": set_num,
            "confirmed": [],
            "missing_numbers": [],
            "missing_bag_windows": [],
            "note": "no truth data yet",
            "db_error": db_error,
        }

    missing = _find_missing_numbers(confirmed)
    confirmed_bag_map = truth_service.get_confirmed_bag_map(set_num)

    missing_windows = [
        _build_bounds_for_missing(n, confirmed_bag_map)
        for n in missing
    ]

    return {
        "set_num": set_num,
        "confirmed": confirmed,
        "missing_numbers": missing,
        "missing_bag_windows": missing_windows,
        "total_confirmed": len(confirmed),
        "max_bag_number": max(int(r["bag_number"]) for r in confirmed),
    }

def run_sequence_scan(set_num: str):
    return get_sequence_from_truth(set_num)