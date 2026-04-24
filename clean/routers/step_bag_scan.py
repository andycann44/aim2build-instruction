from fastapi import APIRouter, HTTPException, Query

from clean.services import step_sequence_bag_service

router = APIRouter()


@router.get("/api/step-bag-scan")
def api_step_bag_scan(
    set_num: str = Query(...),
    start_page: int = Query(..., ge=1),
    end_page: int = Query(..., ge=1),
):
    try:
        return step_sequence_bag_service.scan_step_bag_sequence(
            set_num=set_num,
            start_page=start_page,
            end_page=end_page,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)
