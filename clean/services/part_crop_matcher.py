from typing import List, Dict, Tuple
import math


def _rgb_to_tuple(rgb_str: str) -> Tuple[int, int, int]:
    if not rgb_str:
        return (0, 0, 0)
    try:
        return tuple(int(rgb_str[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 0, 0)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def match_by_colour(crop_rgb: Tuple[int, int, int], parts: List[Dict]) -> List[Dict]:
    results = []

    for p in parts:
        rgb = _rgb_to_tuple(p.get("rgb"))

        d = _dist(crop_rgb, rgb)

        color_score = 1.0 / (1.0 + d)

        results.append(
            {
                **p,
                "color_score": color_score,
                "image_score": None,
                "combined_score": color_score,
            }
        )

    results.sort(key=lambda x: x["combined_score"], reverse=True)

    return results[:5]
