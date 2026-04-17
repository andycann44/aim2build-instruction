"""
FastAPI router to serve custom instruction-part images.

This router is designed to serve the PNGs created by the
instruction_part_catalog_bootstrapper script.

Expected folder layout (relative to this file by default):

    backend/app/data/instruction_catalog/
      assets/
        parts/
          3001-5.png
          3001-15.png
      parts_index.json
      sets/
        21330-1/
          manifest.json

You can override the root directory with the env var A2B_INSTRUCTION_ASSETS.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/images", tags=["images"])

DEFAULT_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "instruction_catalog"
)
ASSETS_ROOT = Path(os.getenv("A2B_INSTRUCTION_ASSETS", DEFAULT_ROOT)).resolve()


@router.get("/parts/{part_num}-{color_id}.png")
async def get_part_image(part_num: str, color_id: int):
    """
    Serve a part thumbnail for a given (part_num, color_id).

    The bootstrapper will create these (optionally as placeholders) in:
        ASSETS_ROOT / "assets" / "parts" / f"{part_num}-{color_id}.png"
    """
    img_path = ASSETS_ROOT / "assets" / "parts" / f"{part_num}-{color_id}.png"
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="Part image not found")
    return FileResponse(str(img_path))
