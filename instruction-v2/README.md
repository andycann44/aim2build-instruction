# instruction-v2 (isolated scaffold)

## Phase 1

Runs an isolated V2 Phase 1 flow:
1. Copy input PDF into `instruction-v2/pdfs/`
2. Render all pages into `instruction-v2/pages/<run_id>/`
3. Write page index to `instruction-v2/indexes/page_index.json`

### Run

```bash
bash instruction-v2/run_phase1.sh /path/to/instructions.pdf
```

Optional:

```bash
bash instruction-v2/run_phase1.sh /path/to/instructions.pdf --run-id my_run
```
