from fastapi import APIRouter, Query

from clean.services import truth_service

router = APIRouter()


@router.get("/api/debug/truth")
def debug_truth(set_num: str = Query(...)):
    rows = truth_service.get_truth_for_set(set_num)

    return {
        "ok": True,
        "set_num": set_num,
        "count": len(rows),
        "rows": rows,
    }