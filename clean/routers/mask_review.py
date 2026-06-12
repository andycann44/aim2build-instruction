"""
Debug-only: /debug/mask-review

Presents every crop for a given bag as a visual card showing:
  1. The original crop image
  2. full_mask_overlay  (from create_full_crop_mask_debug)
  3. raw_master_mask    (binary, saved before any scrub)
  4. master_island_overlay (numbered connected-component bboxes)

PASS / FAIL buttons persist verdicts in a local JSON sidecar.
No part cropping, slot assignment, splitting, or fallback is triggered here.
"""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Paths  (mirror the constants in ai_snap_crop_service.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FULL_CROP_MASK_DIR       = _REPO_ROOT / "debug" / "ai_training" / "full_crop_masks"
_FULL_CROP_MASK_OVERLAY_DIR = _REPO_ROOT / "debug" / "ai_training" / "full_crop_mask_overlays"
_CROP_CACHE_DIR           = _REPO_ROOT / "debug" / "crop_cache"
_VERDICT_DIR              = _REPO_ROOT / "debug" / "ai_training" / "mask_review"

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_file(path: Path, max_w: int = 360) -> str:
    """Read an image file and return a JPEG data-URI, or '' if missing."""
    if not path or not path.exists():
        return ""
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _encode_crop_region(crop_image_path: str, crop_box: list,
                        max_w: int = 360) -> str:
    """Cut crop_box from a page image and return a JPEG data-URI."""
    p = Path(str(crop_image_path or ""))
    if not p.exists():
        return ""
    img = cv2.imread(str(p))
    if img is None:
        return ""
    if not crop_box or len(crop_box) < 4:
        return ""
    x, y, bw, bh = [int(v) for v in crop_box[:4]]
    ih, iw = img.shape[:2]
    x, y   = max(0, x),  max(0, y)
    bw, bh = min(bw, iw - x), min(bh, ih - y)
    if bw <= 0 or bh <= 0:
        return ""
    roi = img[y:y + bh, x:x + bw]
    if roi.size == 0:
        return ""
    if bw > max_w:
        scale = max_w / bw
        roi = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _load_bag_crops(set_num: str, bag: int) -> List[Dict[str, Any]]:
    path = _CROP_CACHE_DIR / f"{set_num}_bag{bag}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _find_mask_stem(set_num: str, bag: int, crop_id: str) -> Optional[str]:
    """
    Return the base stem (without _full_mask suffix) for the most recent mask
    files for this crop, or None if none exist yet.
    Example: '70618_bag2_p22_s26_c1_edf149117a2c'
    """
    safe_set  = "".join(ch for ch in str(set_num)  if ch.isalnum() or ch in "-_")
    safe_crop = "".join(ch for ch in str(crop_id)  if ch.isalnum() or ch in "-_")
    pattern   = f"{safe_set}_bag{bag}_{safe_crop}_*_full_mask.png"
    matches   = sorted(_FULL_CROP_MASK_DIR.glob(pattern))
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    return latest.stem.removesuffix("_full_mask")


def _load_verdicts(set_num: str, bag: int) -> Dict[str, str]:
    path = _VERDICT_DIR / f"{set_num}_bag{bag}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_verdict(set_num: str, bag: int, crop_id: str, verdict: str) -> None:
    _VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    path = _VERDICT_DIR / f"{set_num}_bag{bag}.json"
    try:
        existing: Dict[str, str] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
    except Exception:
        existing = {}
    existing[crop_id] = verdict
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _img_cell(uri: str, label: str) -> str:
    if uri:
        return (
            f'<div class="img-cell">'
            f'<div class="img-label">{escape(label)}</div>'
            f'<img src="{uri}" alt="{escape(label)}">'
            f'</div>'
        )
    return (
        f'<div class="img-cell">'
        f'<div class="img-label">{escape(label)}</div>'
        f'<div class="no-img">not generated yet</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/debug/mask-review", response_class=HTMLResponse)
def mask_review_page(
    set_num: str = Query("70618"),
    bag: int     = Query(2),
):
    crops    = _load_bag_crops(set_num, bag)
    verdicts = _load_verdicts(set_num, bag)
    crop_ids = {
        str(crop.get("crop_id") or "").strip()
        for crop in crops
        if str(crop.get("crop_id") or "").strip()
    }
    active_verdicts = {
        crop_id: verdict
        for crop_id, verdict in verdicts.items()
        if crop_id in crop_ids
    }

    n_total    = len(crop_ids)
    n_reviewed = sum(1 for crop_id in crop_ids if crop_id in active_verdicts)
    n_pass     = sum(
        1 for crop_id in crop_ids
        if active_verdicts.get(crop_id) == "pass"
    )
    n_fail     = sum(
        1 for crop_id in crop_ids
        if active_verdicts.get(crop_id) == "fail"
    )

    cards: List[str] = []
    for crop in crops:
        crop_id = str(crop.get("crop_id") or "").strip()
        if not crop_id:
            continue

        stem = _find_mask_stem(set_num, bag, crop_id)

        crop_uri    = _encode_crop_region(
            str(crop.get("crop_image_path") or ""),
            list(crop.get("crop_box") or []),
        )
        overlay_uri = _encode_file(
            _FULL_CROP_MASK_OVERLAY_DIR / f"{stem}_full_mask_overlay.png"
        ) if stem else ""
        raw_uri     = _encode_file(
            _FULL_CROP_MASK_DIR / f"{stem}_raw_master_mask.png"
        ) if stem else ""
        island_uri  = _encode_file(
            _FULL_CROP_MASK_OVERLAY_DIR / f"{stem}_master_island_overlay.png"
        ) if stem else ""

        verdict     = active_verdicts.get(crop_id, "")
        pass_sel    = "selected" if verdict == "pass" else ""
        fail_sel    = "selected" if verdict == "fail" else ""

        if verdict == "pass":
            badge = '<span class="badge pass">✓ PASS</span>'
        elif verdict == "fail":
            badge = '<span class="badge fail">✗ FAIL</span>'
        else:
            badge = ""

        mask_chip = (
            '<span class="chip ok">masks ready</span>'
            if stem else
            '<span class="chip missing">no masks yet — run auto-mask first</span>'
        )

        page_num = crop.get("page", "?")
        step_num = crop.get("step", "?")
        qty_lbl  = str(crop.get("qty_label") or "")

        img_grid = (
            _img_cell(crop_uri,    "crop")
            + _img_cell(overlay_uri, "full_mask_overlay")
            + _img_cell(raw_uri,     "raw_master_mask")
            + _img_cell(island_uri,  "master_island_overlay")
        )

        cards.append(f"""
<div class="card" id="card-{escape(crop_id)}" data-verdict="{escape(verdict)}">
  <div class="card-header">
    <span class="crop-id">{escape(crop_id)}</span>
    <span class="meta">p{page_num} s{step_num}</span>
    <span class="qty-lbl">{escape(qty_lbl)}</span>
    {mask_chip}
    {badge}
  </div>
  <div class="img-grid">{img_grid}</div>
  <div class="card-footer">
    <button class="btn pass-btn {pass_sel}"
            onclick="vote('{escape(crop_id)}','pass')">✓ PASS — mask is usable</button>
    <button class="btn fail-btn {fail_sel}"
            onclick="vote('{escape(crop_id)}','fail')">✗ FAIL — mask is bad</button>
  </div>
</div>""")

    body = "\n".join(cards) if cards else (
        '<p style="color:#888;padding:20px">No crops found. '
        'Check set_num / bag or rebuild the crop cache.</p>'
    )

    # ------------------------------------------------------------------
    # Inline CSS + HTML
    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full Mask Review — {escape(set_num)} bag {bag}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#181818;color:#ddd;padding:16px;line-height:1.4}}
h1{{font-size:1.15rem;color:#fff;margin-bottom:10px}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
.toolbar label{{font-size:.82rem;color:#aaa;display:flex;align-items:center;gap:5px}}
.toolbar input{{background:#2a2a2a;border:1px solid #444;color:#e0e0e0;
               padding:4px 8px;border-radius:4px;width:80px;font-size:.82rem}}
.toolbar .go{{background:#3a6fd8;border:none;color:#fff;padding:5px 14px;
              border-radius:4px;cursor:pointer;font-size:.82rem}}
.toolbar .go:hover{{background:#4a7fe8}}
.summary{{font-size:.82rem;color:#888;margin-bottom:16px}}
.summary b{{color:#fff}}
.card{{background:#222;border:1px solid #383838;border-radius:8px;
       margin-bottom:18px;overflow:hidden;transition:border-color .2s}}
.card[data-verdict="pass"]{{border-color:#2a6b2a}}
.card[data-verdict="fail"]{{border-color:#6b2a2a}}
.card-header{{display:flex;align-items:center;gap:9px;padding:9px 13px;
              background:#2a2a2a;border-bottom:1px solid #383838;flex-wrap:wrap}}
.crop-id{{font-weight:700;color:#fff;font-size:.92rem}}
.meta{{font-size:.74rem;color:#888}}
.qty-lbl{{font-size:.74rem;color:#aaa;font-style:italic}}
.chip{{font-size:.72rem;padding:2px 7px;border-radius:3px}}
.chip.ok{{background:#1b3a1b;color:#5ecf5e}}
.chip.missing{{background:#3a1b1b;color:#cf7070}}
.badge{{font-size:.74rem;padding:2px 8px;border-radius:3px;font-weight:700}}
.badge.pass{{background:#1b4a1b;color:#6eff6e}}
.badge.fail{{background:#4a1b1b;color:#ff7070}}
.img-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px 13px}}
@media(max-width:900px){{.img-grid{{grid-template-columns:repeat(2,1fr)}}}}
.img-cell{{display:flex;flex-direction:column;gap:3px}}
.img-label{{font-size:.68rem;color:#777;text-align:center;overflow:hidden;
            text-overflow:ellipsis;white-space:nowrap}}
.img-cell img{{width:100%;border-radius:4px;display:block;background:#111;
               object-fit:contain}}
.no-img{{min-height:70px;background:#111;border-radius:4px;display:flex;
         align-items:center;justify-content:center;color:#444;font-size:.7rem;
         text-align:center;padding:4px}}
.card-footer{{display:flex;gap:10px;padding:9px 13px;
              border-top:1px solid #383838}}
.btn{{padding:6px 18px;border:none;border-radius:5px;cursor:pointer;
      font-size:.84rem;font-weight:600;opacity:.5;
      transition:opacity .15s,transform .1s}}
.btn:hover{{opacity:.8}}
.btn.selected{{opacity:1;transform:scale(1.03)}}
.pass-btn{{background:#1e5c1e;color:#aaffaa}}
.pass-btn.selected{{background:#287228}}
.fail-btn{{background:#5c1e1e;color:#ffaaaa}}
.fail-btn.selected{{background:#8a2020}}
.toast{{position:fixed;bottom:18px;right:18px;background:#2a2a2a;color:#ddd;
        padding:9px 16px;border-radius:6px;font-size:.82rem;
        display:none;z-index:999;border:1px solid #444;pointer-events:none}}
</style>
</head>
<body>
<h1>Full Mask Review</h1>
<div class="toolbar">
  <label>set_num<input id="inp-set" value="{escape(set_num)}"></label>
  <label>bag<input id="inp-bag" type="number" min="1" value="{bag}" style="width:55px"></label>
  <button class="go" onclick="nav()">Load</button>
</div>
<div class="summary">
  <b>{n_total}</b> crops &nbsp;·&nbsp;
  <b>{n_reviewed}</b> reviewed &nbsp;·&nbsp;
  <b style="color:#6eff6e">{n_pass}</b> pass &nbsp;·&nbsp;
  <b style="color:#ff7070">{n_fail}</b> fail
</div>
{body}
<div class="toast" id="toast"></div>
<script>
function nav(){{
  const s=document.getElementById('inp-set').value.trim();
  const b=document.getElementById('inp-bag').value.trim();
  location.href='/debug/mask-review?set_num='+encodeURIComponent(s)+'&bag='+encodeURIComponent(b);
}}
async function vote(cropId,verdict){{
  const card=document.getElementById('card-'+cropId);
  const res=await fetch('/api/debug/mask-review/verdict',{{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{set_num:'{escape(set_num)}',bag:{bag},crop_id:cropId,verdict}})
  }});
  if(!res.ok){{toast('Error saving',true);return;}}
  card.dataset.verdict=verdict;
  // badge
  let b=card.querySelector('.badge');
  if(!b){{b=document.createElement('span');card.querySelector('.card-header').appendChild(b);}}
  b.className='badge '+verdict;
  b.textContent=verdict==='pass'?'✓ PASS':'✗ FAIL';
  // buttons
  card.querySelectorAll('.pass-btn').forEach(el=>el.classList.toggle('selected',verdict==='pass'));
  card.querySelectorAll('.fail-btn').forEach(el=>el.classList.toggle('selected',verdict==='fail'));
  toast(cropId+' → '+verdict);
}}
function toast(msg,err=false){{
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.borderColor=err?'#8a2020':'#287228';
  t.style.display='block';
  clearTimeout(t._t);
  t._t=setTimeout(()=>t.style.display='none',2000);
}}
</script>
</body>
</html>"""
    return HTMLResponse(html)


@router.post("/api/debug/mask-review/verdict")
async def mask_review_save_verdict(req: Request):
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "bad_json"}, status_code=400)
    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag     = int(data.get("bag") or 2)
    crop_id = str(data.get("crop_id") or "").strip()
    verdict = str(data.get("verdict") or "").strip().lower()
    if not crop_id or verdict not in ("pass", "fail"):
        return JSONResponse({"ok": False, "error": "invalid_params"}, status_code=400)
    _save_verdict(set_num, bag, crop_id, verdict)
    return JSONResponse({"ok": True, "crop_id": crop_id, "verdict": verdict})
