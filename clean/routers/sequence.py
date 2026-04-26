from fastapi import APIRouter, Query

from clean.services import sequence_service, page_analyzer, debug_service

router = APIRouter()


@router.get("/api/sequence-scan")
def api_sequence_scan(set_num: str = Query(...)):
    return sequence_service.run_sequence_scan(set_num)


@router.get("/api/analyze-gap-page")
def api_analyze_gap_page(
    set_num: str = Query(...),
    bag_number: int = Query(..., ge=1),
    page: int = Query(..., ge=1),
):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "bag_number": int(bag_number),
            "page": int(page),
            "error": "no rendered pages found for set",
        }

    page_analyzer.configure_pages_dir(str(pages_dir))
    result = page_analyzer.analyze_page(int(page), include_image=False)

    return {
        "ok": True,
        "set_num": set_num,
        "bag_number": int(bag_number),
        "page": int(page),
        "analysis": result,
    }