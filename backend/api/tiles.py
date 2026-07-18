"""Thin FastAPI wrapper serving DeepZoom tiles to the OpenSeadragon viewer.

All image reading/tiling lives in backend/io/pyramid.py (guardrail 3: endpoints
only validate + call + return; no image processing here). The frontend
addresses slides by `slide_id`, never by filesystem path (guardrail 2).

URL layout is the DeepZoom convention OpenSeadragon expects out of the box:
  GET /api/tiles/{slide_id}.dzi                       -> XML descriptor
  GET /api/tiles/{slide_id}_files/{level}/{col}_{row}.jpeg -> one tile
"""
from fastapi import APIRouter, HTTPException, Response

from backend import pyramid

router = APIRouter(prefix="/api/tiles")


@router.get("", response_model=list[str])
def list_slides() -> list[str]:
    """slide_ids of every slide available under SLIDES_DIR, for the picker.
    Returns ids only, never filesystem paths (guardrail 2)."""
    return pyramid.list_slides()


@router.get("/{slide_id}.dzi")
def get_dzi(slide_id: str) -> Response:
    try:
        xml = pyramid.get_dzi(slide_id)
    except pyramid.SlideNotFound:
        raise HTTPException(status_code=404, detail=f"slide not found: {slide_id}")
    return Response(content=xml, media_type="application/xml")


@router.get("/{slide_id}_files/{level}/{col}_{row}.jpeg")
def get_tile(slide_id: str, level: int, col: int, row: int) -> Response:
    try:
        data = pyramid.get_tile(slide_id, level, col, row)
    except pyramid.SlideNotFound:
        raise HTTPException(status_code=404, detail=f"slide not found: {slide_id}")
    except ValueError:
        raise HTTPException(status_code=404, detail="tile out of range")
    return Response(content=data, media_type="image/jpeg")
