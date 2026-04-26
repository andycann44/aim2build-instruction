from html import escape

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from clean.services import gap_scan_service, sequence_service

router = APIRouter()


def _window_text(window):
    if not window:
        return "unbounded"
    return f"pages {int(window[0])}-{int(window[1])}"


def _reasons_html(reasons):
    if not reasons:
        return ""
    items = "".join(f"<li>{escape(str(reason))}</li>" for reason in reasons)
    return f"<ul class=\"reasons\">{items}</ul>"


def _candidate_signal(candidate, key, truthy_label="Y", falsy_label=""):
    value = candidate.get(key)
    if isinstance(value, bool):
        return truthy_label if value else falsy_label
    if value is None:
        return ""
    return escape(str(value))


def _focus_debug_link(set_num: str, page: int):
    if str(set_num) == "70618" and int(page) == 81:
        return (
            f'<a href="/api/debug/page-analyzer-focus?set_num={escape(str(set_num))}&page={int(page)}" '
            f'target="_blank">Open focus debug</a>'
        )
    return '<span class="muted-inline">Focus debug unavailable</span>'


@router.get("/gap-review", response_class=HTMLResponse)
@router.get("/debug/gap-review", response_class=HTMLResponse)
def gap_review(set_num: str = Query(...)):
    sequence = sequence_service.run_sequence_scan(set_num)
    gap_payload = gap_scan_service.scan_gaps(set_num, fast=True)

    confirmed_rows = "".join(
        f"<li>Bag {int(row['bag_number'])} -&gt; page {int(row['start_page'])}</li>"
        for row in sequence.get("confirmed", [])
    ) or "<li>None saved yet.</li>"

    missing_rows = "".join(
        f"<li>Bag {int(window['bag_number'])} -&gt; pages {int(window['previous_confirmed_page']) + 1}-{int(window['next_confirmed_page']) - 1}</li>"
        if window.get("previous_confirmed_page") is not None and window.get("next_confirmed_page") is not None
        else f"<li>Bag {int(window['bag_number'])} -&gt; unbounded</li>"
        for window in sequence.get("missing_bag_windows", [])
    ) or "<li>No missing bags.</li>"

    gap_cards = []
    redirect_to = f"/gap-review?set_num={escape(str(set_num))}"

    for gap in gap_payload.get("gaps", []):
        gap_bag = int(gap.get("bag", 0) or 0)
        candidates = list(gap.get("candidates", []))
        matching = [
            candidate
            for candidate in candidates
            if candidate.get("bag_number") is None or int(candidate.get("bag_number")) == gap_bag
        ]
        conflicting = [
            candidate
            for candidate in candidates
            if candidate.get("bag_number") is not None and int(candidate.get("bag_number")) != gap_bag
        ]
        top5 = (matching + conflicting)[:5]
        top = top5[0] if top5 else None

        rows = "".join(
            f"""
            <tr>
                <td>{int(candidate.get('page', 0) or 0)}</td>
                <td>{float(candidate.get('score', 0.0) or 0.0):.3f}</td>
                <td>
                    {'' if candidate.get('bag_number') is None else escape(str(candidate.get('bag_number')))}
                    {'<span class="conflict-label">Conflicting bag number</span>' if candidate.get('bag_number') is not None and int(candidate.get('bag_number')) != gap_bag else ''}
                </td>
                <td>{escape(candidate.get('ocr_raw', '') or '')}</td>
                <td>{_candidate_signal(candidate, 'panel_found')}</td>
                <td>{escape(str(candidate.get('panel_source') or ''))}</td>
                <td>{_candidate_signal(candidate, 'number_box_found')}</td>
                <td>{float(candidate.get('confidence', 0.0) or 0.0):.2f}</td>
                <td>{escape(str(candidate.get('page_kind') or ''))}</td>
                <td>{_candidate_signal(candidate, 'strong_structure')}</td>
                <td>{_reasons_html(candidate.get('reasons', []))}</td>
                <td>
                    <div class="diag-links">
                        <a href="/api/debug/page-image?set_num={escape(str(set_num))}&page={int(candidate.get('page', 0) or 0)}" target="_blank">Open raw page</a>
                        <a href="/debug/page-numbers-overlay?set_num={escape(str(set_num))}&page={int(candidate.get('page', 0) or 0)}" target="_blank">Open OCR overlay</a>
                        <a href="/api/analyze-gap-page?set_num={escape(str(set_num))}&bag_number={gap_bag}&page={int(candidate.get('page', 0) or 0)}" target="_blank">Open analyzer JSON</a>
                        {_focus_debug_link(set_num, int(candidate.get('page', 0) or 0))}
                    </div>
                </td>
                <td>
                    <form method="post" action="/api/debug/save-bag-truth" class="save-form">
                        <input type="hidden" name="set_num" value="{escape(str(set_num))}">
                        <input type="hidden" name="bag_number" value="{gap_bag}">
                        <input type="hidden" name="start_page" value="{int(candidate.get('page', 0) or 0)}">
                        <input type="hidden" name="redirect_to" value="{redirect_to}">
                        <button type="submit">Save as truth</button>
                    </form>
                </td>
            </tr>
            """
            for candidate in top5
        ) or '<tr><td colspan="13" class="empty">No candidates</td></tr>'

        top_summary = (
            f"Top candidate: page {int(top['page'])}, score {float(top.get('score', 0.0) or 0.0):.3f}, "
            f"OCR {escape(top.get('ocr_raw', '') or '') or '-'}"
            if top
            else "Top candidate: none"
        )

        gap_cards.append(
            f"""
            <section class="card gap-card">
                <h2>Bag {gap_bag}</h2>
                <p class="meta">Window: {_window_text(gap.get('window'))}</p>
                <p class="meta">{top_summary}</p>
                <table>
                    <thead>
                        <tr>
                            <th>Page</th>
                            <th>Score</th>
                            <th>Bag Number</th>
                            <th>OCR Raw</th>
                            <th>Panel</th>
                            <th>Panel Source</th>
                            <th>Number Box</th>
                            <th>Confidence</th>
                            <th>Page Kind</th>
                            <th>Strong Structure</th>
                            <th>Reasons</th>
                            <th>Diagnostics</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </section>
            """
        )

    gap_cards_html = "".join(gap_cards) or '<section class="card"><p>No gap suggestions returned.</p></section>'

    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Gap Review {escape(str(set_num))}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 24px;
                    background: #f4f4f4;
                    color: #111;
                    font-family: Arial, sans-serif;
                }}
                .page {{
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .card {{
                    background: #fff;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                    padding: 20px;
                    margin-bottom: 18px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 18px;
                }}
                .meta {{
                    color: #666;
                }}
                .process-list {{
                    margin: 0;
                    padding-left: 22px;
                    line-height: 1.5;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 12px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    vertical-align: top;
                }}
                th {{
                    background: #f0f0f0;
                    font-size: 13px;
                }}
                .reasons {{
                    margin: 0;
                    padding-left: 18px;
                }}
                .diag-links {{
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }}
                .save-form {{
                    margin: 0;
                }}
                button {{
                    padding: 8px 12px;
                    cursor: pointer;
                    border: 0;
                    border-radius: 6px;
                    background: #111;
                    color: #fff;
                }}
                a {{ color: #111; }}
                .conflict-label {{
                    display: inline-block;
                    margin-left: 8px;
                    padding: 2px 6px;
                    border-radius: 999px;
                    background: #d7263d;
                    color: #fff;
                    font-size: 12px;
                    font-weight: 700;
                }}
                .warn {{
                    background: #fff3cd;
                    border: 1px solid #f0d98a;
                    border-radius: 8px;
                    padding: 12px;
                    margin-top: 12px;
                }}
                .muted-inline {{
                    color: #777;
                    font-size: 12px;
                }}
                .empty {{ color: #777; }}
            </style>
        </head>
        <body>
            <div class="page">
                <section class="card">
                    <h1>Sequence + Gap Review</h1>
                    <p class="meta">Set {escape(str(set_num))}</p>
                    <p><a href="/api/gap-scan?set_num={escape(str(set_num))}&fast=1" target="_blank">Open raw gap scan JSON</a></p>
                    <div class="warn">
                        Manual review only. Do not auto-save from rank alone. Page 32 is a known false positive case where OCR can read an instruction-step number instead of a bag-start number.
                    </div>
                </section>

                <section class="card">
                    <h2>Process</h2>
                    <ol class="process-list">
                        <li>Read saved truth anchors.</li>
                        <li>Build missing bag windows from anchors.</li>
                        <li>For each missing bag, scan only pages inside its window.</li>
                        <li>Reuse cached analyzer rows when possible.</li>
                        <li>For each candidate, show detected signals: <code>panel_found</code>, <code>panel_source</code>, <code>number_box_found</code>, <code>bag_number</code>, <code>confidence</code>, <code>page_kind</code>, <code>strong_structure</code>.</li>
                        <li>Rank candidates.</li>
                        <li>Human opens diagnostic overlay and saves truth.</li>
                    </ol>
                </section>

                <section class="summary-grid">
                    <div class="card">
                        <h2>Confirmed Bag Starts</h2>
                        <ul>{confirmed_rows}</ul>
                    </div>
                    <div class="card">
                        <h2>Missing Bags / Windows</h2>
                        <ul>{missing_rows}</ul>
                    </div>
                </section>

                {gap_cards_html}
            </div>
        </body>
        </html>
        """
    )
