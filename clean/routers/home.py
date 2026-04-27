from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Instruction Analyzer</title>
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    min-height: 100%;
                    overflow-y: auto;
                }

                body {
                    font-family: Arial, sans-serif;
                    background: #f4f4f4;
                    color: #111;
                }

                .page-wrap {
                    padding: 20px;
                }

                .card {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 700px;
                    margin: 0 auto 24px auto;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                    box-sizing: border-box;
                }

                h1, h2, h3, p {
                    margin-top: 0;
                }

                form {
                    margin-bottom: 14px;
                }

                input {
                    width: 100%;
                    padding: 10px;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    box-sizing: border-box;
                }

                button {
                    padding: 10px;
                    width: 100%;
                    margin-top: 5px;
                    cursor: pointer;
                    border: 0;
                    border-radius: 6px;
                    background: black;
                    color: white;
                    font-size: 14px;
                }

                button:hover,
                a.button:hover {
                    opacity: 0.92;
                }

                a.button {
                    display: block;
                    text-align: center;
                    padding: 10px;
                    background: black;
                    color: white;
                    text-decoration: none;
                    margin-top: 8px;
                    border-radius: 6px;
                }

                .muted {
                    color: #666;
                    font-size: 14px;
                }

                hr {
                    margin: 20px 0;
                    border: 0;
                    border-top: 1px solid #ddd;
                }

                .section-title {
                    margin-bottom: 8px;
                }
            </style>
        </head>
        <body>
            <div class="page-wrap">
                <div class="card">
                    <h2>LEGO Instruction Analyzer</h2>
                    <p class="muted">
                        Download instructions, scan likely bag-start pages, save confirmed truth,
                        then run sequence/gap review from anchors.
                    </p>

                    <hr>

                    <p class="section-title"><strong>Step 1: Download + render a set</strong></p>
                    <form action="/debug/load-set" method="post">
                        <input
                            name="set_num"
                            placeholder="Enter set number (e.g. 21330)"
                            required
                        >
                        <button type="submit">Download + Render</button>
                    </form>

                    <p class="section-title"><strong>Step 2: Analyzer scan</strong></p>
                    <form action="/debug/scan-set-with-analyzer" method="get">
                        <input
                            name="set_num"
                            placeholder="Set number for analyzer scan"
                            required
                        >
                        <input type="hidden" name="include_all" value="false">
                        <button type="submit">Run Analyzer Scan (find bag pages)</button>
                    </form>

                    <p class="section-title"><strong>Step 3: Sequence scan</strong></p>
                    <form action="/api/sequence-scan" method="get">
                        <input
                            name="set_num"
                            placeholder="Set number for sequence scan"
                            required
                        >
                        <button type="submit">Run Sequence Scan (needs saved bag truth)</button>
                    </form>

                    <p class="section-title"><strong>Step 4: Full bag scan review</strong></p>
                    <form action="/debug/full-bag-scan" method="get">
                        <input
                            name="set_num"
                            placeholder="Set number for full bag scan"
                            required
                        >
                        <input
                            name="start"
                            placeholder="Start page (optional)"
                        >
                        <input
                            name="end"
                            placeholder="End page (optional)"
                        >
                        <input
                            name="chunk_size"
                            placeholder="Chunk size (default 5)"
                        >
                        <input
                            name="max_chunks"
                            placeholder="Max chunks (default 3)"
                        >
                        <input
                            name="expected_next_bag"
                            placeholder="Expected next bag (optional)"
                        >
                        <button type="submit">Run Full Bag Scan Review</button>
                    </form>

                    <hr>

                    <p><strong>Quick links</strong></p>

                    <a class="button" href="/debug/scan-set-with-analyzer?set_num=21330&include_all=false">
                        Example: Analyzer scan for 21330
                    </a>

                    <a class="button" href="/api/sequence-scan?set_num=21330">
                        Example: Sequence scan for 21330
                    </a>

                    <a class="button" href="/api/debug/truth?set_num=21330">
                        Example: View saved truth for 21330
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
    )
