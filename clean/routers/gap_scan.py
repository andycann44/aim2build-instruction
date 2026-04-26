from typing import Optional

from fastapi import APIRouter, Query

from clean.services import gap_scan_service

router = APIRouter()


@router.get("/api/gap-scan")
def api_gap_scan(
    set_num: str = Query(...),
    bag_number: Optional[int] = Query(None, ge=1),
    fast: bool = Query(False),
):
    return gap_scan_service.scan_gaps(set_num, bag_number=bag_number, fast=fast)
