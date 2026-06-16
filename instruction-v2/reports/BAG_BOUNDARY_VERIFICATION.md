# Bag Boundary Verification

Visual inspection of bag start pages against `indexes/04_bag_map.json`.

**Method:** Rendered page images only (`pages/70618_01/page_*.png`). No step map, crop cache, or pipeline rerun.

**Set:** 70618_01

---

## Page 6

| Field | Value |
|---|---|
| Visible bag number | Yes |
| Printed bag number | **1** |
| Bag intro page | Yes — top panel shows bag graphic, large **1**, red arrow to ship hull |
| Detected bag (`04_bag_map.json`) | **1** (start_page=6, end_page=21) |
| **Result** | **PASS** |

Visual bag: **1**  
Detected bag: **1**

**Bag marker crop:** `bag_boundary_crops/page_006_bag_marker.png`  
**Full intro panel:** `bag_boundary_crops/page_006_bag_panel.png`

---

## Page 22

| Field | Value |
|---|---|
| Visible bag number | Yes |
| Printed bag number | **2** |
| Bag intro page | Yes — top-left intro box with bag graphic and large **2** |
| Detected bag (`04_bag_map.json`) | **2** (start_page=22, end_page=38) |
| **Result** | **PASS** |

Visual bag: **2**  
Detected bag: **2**

**Bag marker crop:** `bag_boundary_crops/page_022_bag_marker.png`  
**Full intro panel:** `bag_boundary_crops/page_022_bag_panel.png`

---

## Page 39

| Field | Value |
|---|---|
| Visible bag number | Yes |
| Printed bag number | **3** |
| Bag intro page | Yes — top-left intro box with bag graphic and large **3** |
| Detected bag (`04_bag_map.json`) | **3** (start_page=39, end_page=57) |
| **Result** | **PASS** |

Visual bag: **3**  
Detected bag: **3**

**Bag marker crop:** `bag_boundary_crops/page_039_bag_marker.png`  
**Full intro panel:** `bag_boundary_crops/page_039_bag_panel.png`

---

## Page 58

| Field | Value |
|---|---|
| Visible bag number | Yes |
| Printed bag number | **4** |
| Bag intro page | Yes — top panel shows prior build + bag **4** + arrow to expanded ship |
| Detected bag (`04_bag_map.json`) | **4** (start_page=58, end_page=80) |
| **Result** | **PASS** |

Visual bag: **4**  
Detected bag: **4**

**Bag marker crop:** `bag_boundary_crops/page_058_bag_marker.png`  
**Full intro panel:** `bag_boundary_crops/page_058_bag_panel.png`

---

## Page 81

| Field | Value |
|---|---|
| Visible bag number | Yes |
| Printed bag number | **5** |
| Bag intro page | Yes — top panel shows brown base + bag **5** + arrow to ship build |
| Detected bag (`04_bag_map.json`) | **5** (start_page=81, end_page=103) |
| **Result** | **PASS** |

Visual bag: **5**  
Detected bag: **5**

**Bag marker crop:** `bag_boundary_crops/page_081_bag_marker.png`  
**Full intro panel:** `bag_boundary_crops/page_081_bag_panel.png`

---

## Summary

| Page | Visual bag | Detected bag | Bag intro | Result |
|---:|---:|---:|---|---|
| 6 | 1 | 1 | Yes | **PASS** |
| 22 | 2 | 2 | Yes | **PASS** |
| 39 | 3 | 3 | Yes | **PASS** |
| 58 | 4 | 4 | Yes | **PASS** |
| 81 | 5 | 5 | Yes | **PASS** |

**5 / 5 PASS** — All inspected bag boundary pages show a visible printed bag number matching `04_bag_map.json` start pages. Each page is a standard LEGO bag intro layout (bag graphic + large number + build arrow).

## Evidence files

```
reports/bag_boundary_crops/
  page_006_bag_marker.png
  page_006_bag_panel.png
  page_022_bag_marker.png
  page_022_bag_panel.png
  page_039_bag_marker.png
  page_039_bag_panel.png
  page_058_bag_marker.png
  page_058_bag_panel.png
  page_081_bag_marker.png
  page_081_bag_panel.png
```

Source images: `pages/70618_01/page_{006,022,039,058,081}.png`
