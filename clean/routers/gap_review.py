from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from clean.services.gap_scan_service import scan_gap_for_bag

router = APIRouter()


@router.get("/debug/gap-review", response_class=HTMLResponse)
def gap_review(set_num: str = Query(...), bag: int = Query(...)):

    result = scan_gap_for_bag(set_num, bag)

    rows = result.get("analysis_rows", [])

    table_rows = ""

    for r in rows[:40]:
        table_rows += f"""
        <tr>
            <td>{r.get("page")}</td>
            <td>{r.get("bag_number")}</td>
            <td>{r.get("confidence")}</td>
            <td>{r.get("score")}</td>
            <td>{", ".join(r.get("why", []))}</td>
            <td>
                <form method="post" action="/api/debug/save-bag-truth">
                    <input type="hidden" name="set_num" value="{set_num}">
                    <input type="hidden" name="bag_number" value="{bag}">
                    <input type="hidden" name="start_page" value="{r.get("page")}">
                    <button type="submit">Confirm</button>
                </form>
            </td>
        </tr>
        """

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                margin: 0;
                padding: 20px;
                background: #f4f4f4;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}

            th {{
                background: #eee;
                position: sticky;
                top: 0;
            }}

            button {{
                padding: 6px 10px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>

        <h2>Gap Review — Bag {bag}</h2>

        <p>
            Pages {result.get("scan_start_page")} → {result.get("scan_end_page")}
        </p>

        <table>
            <tr>
                <th>Page</th>
                <th>Detected</th>
                <th>Confidence</th>
                <th>Score</th>
                <th>Why</th>
                <th>Action</th>
            </tr>
            {table_rows}
        </table>

    </body>
    </html>
    """
