from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from clean.services import dataset_service

router = APIRouter()


def _render_load_set_page(error_message: str = "", set_num_value: str = ""):
    error_html = f"<p style='color:#b00020;'><strong>{error_message}</strong></p>" if error_message else ""

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Load set</title>
          <style>
            body {{
              font-family: Arial, sans-serif;
              margin: 24px;
              background: #f3f3f3;
            }}
            .card {{
              background: #fff;
              border: 1px solid #ddd;
              border-radius: 8px;
              padding: 16px;
              max-width: 520px;
            }}
            input {{
              padding: 8px;
              width: 220px;
              margin-right: 8px;
            }}
            button {{
              padding: 8px 12px;
              cursor: pointer;
            }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Load set from set number</h2>
            {error_html}
            <form method="post" action="/debug/load-set">
              <input type="text" name="set_num" value="{set_num_value}" placeholder="e.g. 21330" />
              <button type="submit">Load set</button>
            </form>
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/load-set")
def debug_load_set_page():
    return _render_load_set_page()


@router.post("/debug/load-set")
def debug_load_set_submit(set_num: str = Form(...)):
    result = dataset_service.load_set_from_number(set_num)

    if not result.get("ok"):
        return _render_load_set_page(
            error_message=result.get("error", "Unknown error"),
            set_num_value=set_num,
        )

    normalized_set_num = result["set_num"]
    return RedirectResponse(
        url=f"/api/sequence-scan?set_num={normalized_set_num}",
        status_code=303,
    )


@router.get("/api/load-set-status")
def api_load_set_status(set_num: str):
    return dataset_service.load_set_from_number(set_num)