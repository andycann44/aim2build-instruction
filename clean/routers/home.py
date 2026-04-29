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
                    max-width: 960px;
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

                button[disabled] {
                    opacity: 0.7;
                    cursor: default;
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

                .grid-2 {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 18px;
                }

                .inline-grid {
                    display: grid;
                    grid-template-columns: 2fr 1fr 1fr;
                    gap: 12px;
                }

                .progress-shell {
                    width: 100%;
                    height: 16px;
                    border-radius: 999px;
                    background: #e6e6e6;
                    overflow: hidden;
                    margin: 12px 0 10px 0;
                }

                .progress-fill {
                    height: 100%;
                    width: 0%;
                    background: linear-gradient(90deg, #1b7f47 0%, #30b566 100%);
                    transition: width 0.2s ease;
                }

                .status-box,
                .results-box,
                .truth-box {
                    background: #fafafa;
                    border: 1px solid #e1e1e1;
                    border-radius: 8px;
                    padding: 14px;
                    margin-top: 12px;
                }

                .status-line {
                    margin: 6px 0;
                    color: #333;
                    font-size: 14px;
                }

                .pill {
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-right: 6px;
                    margin-bottom: 6px;
                }

                .pill-green {
                    background: #e7f7ed;
                    color: #15703d;
                }

                .pill-slate {
                    background: #ececec;
                    color: #555;
                }

                .list {
                    margin: 0;
                    padding-left: 18px;
                }

                .list li {
                    margin-bottom: 6px;
                }

                .small {
                    font-size: 13px;
                }

                .mono {
                    font-family: Menlo, Consolas, monospace;
                }

                @media (max-width: 820px) {
                    .grid-2,
                    .inline-grid {
                        grid-template-columns: 1fr;
                    }
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

                    <p class="section-title"><strong>Step 5: Process Full Book</strong></p>
                    <p class="muted">
                        Runs the chunked bag finder repeatedly, saving accepted bag starts after each chunk.
                    </p>

                    <div class="inline-grid">
                        <input
                            id="chunk-set-num"
                            value="70618"
                            placeholder="Set number"
                        >
                        <input
                            id="chunk-start-page"
                            type="number"
                            min="1"
                            value="1"
                            placeholder="Start page"
                        >
                        <input
                            id="chunk-size"
                            type="number"
                            min="1"
                            value="5"
                            placeholder="Chunk size"
                        >
                    </div>

                    <button id="chunk-scan-button" type="button">Start / Continue Scan</button>

                    <div class="progress-shell" aria-label="Scan progress">
                        <div id="chunk-progress-fill" class="progress-fill"></div>
                    </div>

                    <div id="chunk-status" class="status-box">
                        <p class="status-line"><strong>Status:</strong> Idle</p>
                        <p class="status-line">Scanning pages -</p>
                        <p class="status-line">Next start page: -</p>
                        <p class="status-line">Bags saved: -</p>
                    </div>

                    <div class="grid-2">
                        <div id="chunk-results" class="results-box">
                            <h3>Newly Found Bag Starts</h3>
                            <p class="muted small">No chunk run yet.</p>
                        </div>

                        <div id="saved-truth" class="truth-box">
                            <h3>Saved Truth</h3>
                            <p class="muted small">Run a chunk to load saved bag truth.</p>
                        </div>
                    </div>

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
            <script>
                (() => {
                    const setNumInput = document.getElementById("chunk-set-num");
                    const startInput = document.getElementById("chunk-start-page");
                    const chunkSizeInput = document.getElementById("chunk-size");
                    const button = document.getElementById("chunk-scan-button");
                    const progressFill = document.getElementById("chunk-progress-fill");
                    const statusBox = document.getElementById("chunk-status");
                    const resultsBox = document.getElementById("chunk-results");
                    const savedTruthBox = document.getElementById("saved-truth");

                    let scanRunning = false;

                    function setStatus(lines) {
                        statusBox.innerHTML = lines.map((line) => `<p class="status-line">${line}</p>`).join("");
                    }

                    function renderBagStarts(items, skippedRows) {
                        const bagStarts = Array.isArray(items) ? items : [];
                        const skipped = Array.isArray(skippedRows) ? skippedRows : [];

                        if (!bagStarts.length) {
                            resultsBox.innerHTML = `
                                <h3>Newly Found Bag Starts</h3>
                                <p class="muted small">No bag starts found in the latest chunk.</p>
                                <p class="muted small">Skipped rows: ${skipped.length}</p>
                            `;
                            return;
                        }

                        const rows = bagStarts.map((item) => `
                            <li>
                                <span class="pill pill-green">bag ${item.bag_number}</span>
                                <span class="mono">page ${item.page}</span>
                                <span class="pill pill-slate">score ${(item.bag_start_card_score ?? 0).toFixed ? item.bag_start_card_score.toFixed(1) : item.bag_start_card_score}</span>
                            </li>
                        `).join("");

                        resultsBox.innerHTML = `
                            <h3>Newly Found Bag Starts</h3>
                            <ul class="list small">${rows}</ul>
                            <p class="muted small">Skipped rows: ${skipped.length}</p>
                        `;
                    }

                    function renderSavedTruth(items) {
                        const saved = Array.isArray(items) ? items : [];
                        if (!saved.length) {
                            savedTruthBox.innerHTML = `
                                <h3>Saved Truth</h3>
                                <p class="muted small">No saved bag truth yet.</p>
                            `;
                            return;
                        }

                        const rows = saved.map((item) => `
                            <li>
                                <span class="pill pill-green">bag ${item.bag_number}</span>
                                <span class="mono">page ${item.start_page}</span>
                                <span class="pill pill-slate">${item.source || "unknown"}</span>
                            </li>
                        `).join("");

                        savedTruthBox.innerHTML = `
                            <h3>Saved Truth</h3>
                            <ul class="list small">${rows}</ul>
                        `;
                    }

                    async function runChunkScan() {
                        if (scanRunning) {
                            return;
                        }

                        scanRunning = true;
                        button.disabled = true;
                        button.textContent = "Scanning...";

                        const setNum = (setNumInput.value || "").trim();
                        let nextStart = Math.max(1, parseInt(startInput.value || "1", 10) || 1);
                        const chunkSize = Math.max(1, parseInt(chunkSizeInput.value || "5", 10) || 5);

                        if (!setNum) {
                            setStatus([
                                "<strong>Status:</strong> Please enter a set number.",
                                "Scanning pages -",
                                "Next start page: -",
                                "Bags saved: -"
                            ]);
                            button.disabled = false;
                            button.textContent = "Start / Continue Scan";
                            scanRunning = false;
                            return;
                        }

                        try {
                            while (scanRunning) {
                                const url = `/api/bag-find-chunked?set_num=${encodeURIComponent(setNum)}&start=${encodeURIComponent(nextStart)}&chunk_size=${encodeURIComponent(chunkSize)}`;
                                const response = await fetch(url);
                                const payload = await response.json();

                                if (!response.ok) {
                                    throw new Error(payload.detail || payload.error || "Chunk scan failed");
                                }

                                const totalPages = Number(payload.total_pages || 0);
                                const scannedStart = Number(payload.scanned_start || nextStart);
                                const scannedEnd = Number(payload.scanned_end || scannedStart);
                                const savedTruth = Array.isArray(payload.saved_truth) ? payload.saved_truth : [];
                                const nextStartPage = payload.next_start_page;
                                const done = Boolean(payload.done) || nextStartPage === null;
                                const progressPercent = totalPages > 0
                                    ? Math.max(0, Math.min(100, (scannedEnd / totalPages) * 100))
                                    : 0;

                                progressFill.style.width = `${progressPercent}%`;
                                setStatus([
                                    `<strong>Status:</strong> ${done ? "Complete" : "Scanning"}`,
                                    `Scanning pages ${scannedStart}-${scannedEnd} of ${totalPages || "?"}`,
                                    `Next start page: ${nextStartPage ?? "done"}`,
                                    `Bags saved: ${savedTruth.map((item) => `${item.bag_number}->${item.start_page}`).join(", ") || "none"}`
                                ]);

                                renderBagStarts(payload.bag_starts, payload.skipped_rows);
                                renderSavedTruth(savedTruth);

                                if (nextStartPage != null) {
                                    startInput.value = String(nextStartPage);
                                }

                                if (done) {
                                    break;
                                }

                                nextStart = Number(nextStartPage);
                                await new Promise((resolve) => setTimeout(resolve, 120));
                            }
                        } catch (error) {
                            setStatus([
                                `<strong>Status:</strong> Error`,
                                `${error && error.message ? error.message : "Unknown error"}`,
                                `Next start page: ${startInput.value || "-"}`,
                                "Bags saved: -"
                            ]);
                        } finally {
                            button.disabled = false;
                            button.textContent = "Start / Continue Scan";
                            scanRunning = false;
                        }
                    }

                    button.addEventListener("click", runChunkScan);
                })();
            </script>
        </body>
        </html>
        """
    )
