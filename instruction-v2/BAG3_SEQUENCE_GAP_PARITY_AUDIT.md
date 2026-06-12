# Bag 3 sequence gap parity audit

Set `70618`, bag `3`. Read-only. No promotion.

## Summary

| Classification | Count |
| --- | ---: |
| EMPTY_STEP | 4 |
| V2_FAILURE | 4 |

## Gap rows

### Step 46 (gap page 40)

- **Classification:** `V2_FAILURE`
- **Reason:** V2 sequence-gap full-page audit found step anchor but V1 raw detector missed it; no V2 callout box and no V1 callout probe result.
- **Previous:** step 45 page 40 (`p40_s45_c3`)
- **Next:** step 47 page 40 (`p40_s47_c2`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=False main_step=44 hits=[]
- **V2 step detection:** page 40, box [770, 142, 54, 36], source sequence_gap_full_page_audit
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, v2_step_from_sequence_gap_full_page_audit, step_anchor_without_callout_box

### Step 48 (gap page 41)

- **Classification:** `EMPTY_STEP`
- **Reason:** Both V1 and V2 detect the step number but neither pipeline found a callout box.
- **Previous:** step 47 page 40 (`p40_s47_c2`)
- **Next:** step 49 page 42 (`p42_s49_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=True main_step=48 hits=[48, 48, 48, 48]
- **V2 step detection:** page 41, box [115, 142, 54, 36], source step_map_primary
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, step_anchor_without_callout_box

### Step 63 (gap page 51)

- **Classification:** `EMPTY_STEP`
- **Reason:** Both V1 and V2 detect the step number but neither pipeline found a callout box.
- **Previous:** step 62 page 50 (`p50_s62_c3`)
- **Next:** step 64 page 52 (`p52_s64_c2`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=True main_step=63 hits=[63, 63, 63, 63]
- **V2 step detection:** page 51, box [116, 142, 54, 36], source step_map_primary
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, step_anchor_without_callout_box

### Step 70 (gap page 54)

- **Classification:** `V2_FAILURE`
- **Reason:** V2 step_map places step 70 on page 53 but sequence-gap review inferred page 54; gap page has no matching anchor.
- **Previous:** step 69 page 53 (`p53_s69_c2`)
- **Next:** step 72 page 55 (`p55_s72_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=False main_step=68 hits=[]
- **V2 step detection:** page 53, box [854, 142, 53, 36], source sequence_gap_full_page_audit
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, v2_step_from_sequence_gap_full_page_audit, v2_step_page_53_vs_gap_page_54, step_anchor_without_callout_box

### Step 71 (gap page 54)

- **Classification:** `EMPTY_STEP`
- **Reason:** Both V1 and V2 detect the step number but neither pipeline found a callout box.
- **Previous:** step 69 page 53 (`p53_s69_c2`)
- **Next:** step 72 page 55 (`p55_s72_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=True main_step=71 hits=[71, 71, 71, 71]
- **V2 step detection:** page 54, box [115, 294, 39, 35], source step_map_primary
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, step_anchor_without_callout_box

### Step 74 (gap page 55)

- **Classification:** `V2_FAILURE`
- **Reason:** V2 sequence-gap full-page audit found step anchor but V1 raw detector missed it; no V2 callout box and no V1 callout probe result.
- **Previous:** step 73 page 55 (`p55_s73_c2`)
- **Next:** step 76 page 56 (`p56_s76_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=False main_step=72 hits=[]
- **V2 step detection:** page 55, box [854, 142, 49, 36], source sequence_gap_full_page_audit
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, v2_step_from_sequence_gap_full_page_audit, step_anchor_without_callout_box

### Step 75 (gap page 55)

- **Classification:** `V2_FAILURE`
- **Reason:** V2 sequence-gap full-page audit found step anchor but V1 raw detector missed it; no V2 callout box and no V1 callout probe result.
- **Previous:** step 73 page 55 (`p55_s73_c2`)
- **Next:** step 76 page 56 (`p56_s76_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=False main_step=72 hits=[]
- **V2 step detection:** page 55, box [854, 727, 52, 36], source sequence_gap_full_page_audit
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, v2_step_from_sequence_gap_full_page_audit, step_anchor_without_callout_box

### Step 77 (gap page 56)

- **Classification:** `EMPTY_STEP`
- **Reason:** Both V1 and V2 detect the step number but neither pipeline found a callout box.
- **Previous:** step 76 page 56 (`p56_s76_c1`)
- **Next:** step 78 page 57 (`p57_s78_c1`)
- **V1 training crop:** no
- **V2 crop_cache:** no
- **Visible step number:** yes
- **Visible callout box:** no
- **V1 step detection:** detected=True main_step=76 hits=[77, 77, 77]
- **V2 step detection:** page 56, box [30, 449, 51, 35], source step_map_primary
- **Root-cause notes:** missing_from_v1_training_labels, missing_from_v2_crop_cache_export, step_anchor_without_callout_box

