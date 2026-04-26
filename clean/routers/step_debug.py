from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from clean.services import step_detector_service

router = APIRouter()


@router.get("/debug/step-detect")
def debug_step_detect(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    try:
        return step_detector_service.detect_steps(set_num, page)
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.get("/debug/step-overlay")
def debug_step_overlay(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    try:
        overlay = step_detector_service.build_step_overlay(set_num, page)
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=500, detail=message)

    return Response(content=overlay, media_type="image/png")
