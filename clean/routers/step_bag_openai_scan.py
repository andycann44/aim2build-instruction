from fastapi import APIRouter, HTTPException, Query

from clean.services import step_sequence_openai_bag_service

router = APIRouter()


@router.get("/api/step-bag-scan-openai-verify")
def api_step_bag_scan_openai_verify(
    set_num: str = Query(...),
    start_page: int = Query(..., ge=1),
    end_page: int = Query(..., ge=1),
    model: str = Query(step_sequence_openai_bag_service.DEFAULT_MODEL),
):
    try:
        return step_sequence_openai_bag_service.scan_step_bag_sequence_openai_verify(
            set_num=set_num,
            start_page=start_page,
            end_page=end_page,
            model=model,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)
