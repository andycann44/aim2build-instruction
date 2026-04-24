from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from clean.routers import home, analyzer_scan, sequence, debug, load_set, set_scan, workflow, debug_truth, step_debug, step_bag_scan, step_bag_openai_scan
from clean.routers import gap_review

app = FastAPI()

app.include_router(home.router)
app.include_router(analyzer_scan.router)
app.include_router(sequence.router)
app.include_router(debug.router)
app.include_router(load_set.router)
app.include_router(set_scan.router)
app.include_router(workflow.router)
app.include_router(debug_truth.router)
app.include_router(gap_review.router)
app.include_router(step_debug.router)
app.include_router(step_bag_scan.router)
app.include_router(step_bag_openai_scan.router)


@app.get("/health")
def health():
    return {"status": "ok", "load_set_ui": "/debug/load-set"}
