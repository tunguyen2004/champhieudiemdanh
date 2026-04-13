# AI Handoff Context (Detailed)

## 1) Why this file exists
This file is a direct handoff for another AI/engineer to understand:
- What this project does
- How code is structured
- How the main logic works in detail
- What was already done in this workspace/session

The goal is to remove re-discovery time before making further changes.

## 2) Project purpose
This is a Python OMR system for THPT-style answer sheets with 3 answer sections:
- FC: multiple choice (40 questions, A/B/C/D)
- TF: true/false style (8 questions, each has a/b/c/d, each sub-item chooses one of 2 columns)
- DG: short numeric/symbol answers via bubble grid (no OCR)

Main capabilities:
- Detect sheet from input photo/scan
- Warp to canonical size
- Extract answers from configured regions
- Return JSON data and annotated images
- Provide Flask web UI + API + batch CLI

## 3) Top-level structure
At repository root:
- `app.py`: Flask web app (upload/process/download)
- `config.json`: full detection/extraction configuration and region coordinates
- `process_batch.py`: CLI for processing many sheets
- `calibrate.py`: interactive coordinate calibration tool
- `core/`: main processing package
  - `core/__init__.py`: orchestration pipeline
  - `core/preprocessor.py`: image loading + preprocess functions
  - `core/detector.py`: sheet detection, warp, deskew, template alignment
  - `core/extractor.py`: SBD/MDT/FC/TF/DG extraction logic
  - `core/visualizer.py`: draw recognized marks and debug overlays
- `templates/` + `static/css/`: web UI
- `anh/`: sample/test images
- `uploads/`, `results/`, `debug/`: runtime output folders

## 4) Runtime data flow (real code path)
Primary pipeline is in `core.process_image(...)`:
1. Load image (Unicode-safe read in `load_image`)
2. Preprocess:
   - grayscale + CLAHE + Gaussian blur
   - binary generation (adaptive threshold for detection stage)
3. Find sheet corners (`find_corners`):
   - priority 1: ink-based page region
   - priority 2: corner markers
   - priority 3: contour fallback
4. Perspective warp to configured size (`warp_width`, `warp_height`)
5. Deskew by Hough lines (`deskew`)
6. Optional template alignment by ORB homography (`align_to_template`)
7. Build extraction binary (Otsu on enhanced grayscale)
8. Extract all fields (`extract_all`)
9. Draw annotation image (`visualize_results`)
10. Return structured output:
   - `res.fc`, `res.tf`, `res.dg`
   - `sbd`, `mdt`
   - warnings/errors
   - internals like detection method and skew angle

## 5) Detailed logic by module

### 5.1 `core/preprocessor.py`
Key points:
- `load_image(path)` uses `np.fromfile + cv2.imdecode` to support Unicode paths.
- `preprocess()` = grayscale -> CLAHE -> Gaussian blur.
- `preprocess_to_binary()` uses adaptive threshold (inverted).
- `preprocess_to_edges()` uses Canny on preprocessed grayscale.

Purpose:
- Normalize contrast/noise for robust sheet and bubble detection under variable lighting.

### 5.2 `core/detector.py`
Important functions:
- `order_points`: canonical corner order (tl, tr, br, bl).
- `_validate_quad`: filters invalid quadrilaterals (min area, edge ratio).
- `find_sheet_by_ink`:
  - Otsu invert, large morphological close, then `minAreaRect` on non-zero points.
  - This is first-priority detection in `find_corners`.
- `find_corner_markers`:
  - contour filtering by area/aspect/solidity + must be near image edges/corners.
- `find_sheet_contour`:
  - fallback contour strategy with multiple polygon approximation epsilons.
- `find_corners` fallback chain:
  - ink -> markers -> contour.
- `warp_perspective`: map to fixed target size.
- `deskew`:
  - estimate angle from Hough line segments (horizontal + vertical families),
  - weighted by line length,
  - only rotate if angle is meaningful and bounded.
- `TemplateAligner` (singleton):
  - loads template once,
  - warps template to canonical size,
  - extracts ORB features,
  - aligns warped candidate image via KNN ratio test + homography inlier check.

### 5.3 `core/extractor.py`
This module contains most domain logic and edge-case handling.

Shared helpers:
- `get_bubble_rect`: pixel ROI from normalized region coordinates.
- `compute_fill_ratio`: non-zero ratio in ROI.
- `auto_calibrate_grid_y`: scans local y-offsets to maximize confidence.

Robust mark handling:
- `detect_erased_mark`: detect erased traces via grayscale histogram zones.
- `detect_pencil_mark`: detect faint pencil marks by adaptive threshold + local contrast.
- `is_valid_mark`: remove noise using shape circularity + 3x3 concentration.
- `robust_bubble_detection`:
  - staged decision pipeline:
    - immediate fill acceptance above threshold,
    - early empty check,
    - noise filter,
    - erase detection,
    - pencil detection,
    - multi-threshold voting (Otsu/fixed/adaptive).

ID extraction:
- `extract_sbd` and `extract_mdt`:
  - column-wise best row by grayscale intensity gap,
  - fallback local-offset recovery if ambiguous,
  - returns `?` when unresolved.

FC extraction:
- `extract_fc`:
  - iterate configured FC groups,
  - evaluate each bubble using robust detector,
  - accept statuses `filled` and `pencil`,
  - warn on multi-fill, near-double, erased traces.

TF extraction:
- Uses specialized signal fusion rather than plain fill ratio:
  - core ellipse fill metric (`_tf_core_metrics`)
  - grayscale darkness
  - template delta vs clean template (`_tf_template_delta`)
- Decision logic resolves top vs alternative candidate with configurable thresholds:
  - strong/weak fill, min signal, signal diff, fill ratio, template confidence.
- Supports ambiguity warnings when both columns appear similarly strong.
- Output schema is grouped by question with subkeys `a,b,c,d`.

DG extraction (short numeric/symbol):
- `extract_dg` evaluates each column independently.
- Preferred path (if grayscale exists):
  - pick darkest row and require separation from second/median intensities.
- Binary fallback path:
  - use top/second/median fill gaps and noise cap.
- Includes row-11 guard (last row) to suppress bottom-edge false positives.

Aggregator:
- `extract_all` runs SBD + MDT + FC + TF + DG and merges warnings/errors.

### 5.4 `core/visualizer.py`
Responsibilities:
- Draw detected marks/labels back on warped image.
- Color conventions:
  - green: accepted mark
  - red: conflict (multi-mark)
  - gray: empty
- Handles FC/TF/DG + SBD/MDT overlays.
- `create_debug_image` draws region boxes to validate configuration.

## 6) Web layer behavior (`app.py`)
Routes:
- `GET /`: upload page.
- `POST /process`: process one/many files from form.
- `POST /api/process`: process one file via API and return JSON.
- `GET /result_image/<filename>` and `/debug_image/<filename>`: image serving.
- `GET /download/json/<filename>`: download result JSON.
- `GET /download/csv/<result_id>`: flatten output to CSV.

Notable implementation details:
- Upload extensions are explicitly allowlisted.
- Generated file names are UUID-prefixed.
- Result JSON is written for each batch form submission.
- CSV flattening includes compatibility logic for TF schema variants.

## 7) Config model (`config.json`)
Core controls:
- Geometry: `warp_width`, `warp_height`.
- Preprocess params: CLAHE, Gaussian, adaptive threshold, Canny.
- Detection params: marker filters and extraction thresholds.
- Region map:
  - `regions.sbd`, `regions.mdt`: ID grids.
  - `regions.fc.groups`: 4 blocks x 10 questions.
  - `regions.tf.groups`: 8 question groups (rows=4, cols=2).
  - `regions.dg.groups` + `rows`: short-answer columns and symbol rows.

Advanced sections:
- `tf_detection`: fused-signal thresholds.
- `dg_detection`: grayscale/binary separation guards.

## 8) CLI + calibration workflow
- `process_batch.py`:
  - reads all images in folder by extension,
  - runs full pipeline,
  - saves marked/debug outputs and summary stats.
- `calibrate.py`:
  - warps sample image,
  - interactive mouse coordinate display in normalized ratio,
  - helps update region coordinates in `config.json`.

## 9) Current testing assets
- There are many script-style test files (`test_*.py`) focused on:
  - detection robustness (`test_detection.py`, `test_ink_consistency.py`)
  - alignment experiments (`test_template_match.py`, `test_overlay.py`)
  - full pipeline checks (`test_full_pipeline.py`, `test_pipeline.py`, `test_calibrated.py`)
  - threshold diagnostics (`test_ratios.py`)
  - calibration checks (`test_calibrate.py`)
- These are mostly executable scripts for analysis/debugging, not formal pytest unit tests.

## 10) What was done by assistant in this environment
Practical environment setup actions already performed:
1. Verified Python command availability.
2. Verified/installed Python distribution via `winget` (reported installed package `Python.Python.3.14`).
3. Confirmed interpreter exists at:
   - `C:\\Users\\TUNA\\AppData\\Local\\Programs\\Python\\Python314\\python.exe`
4. Confirmed pip exists and works with that interpreter.
5. Updated user `PATH` to include:
   - `...\\Python314`
   - `...\\Python314\\Scripts`
6. Reordered user `PATH` so real Python path precedes `WindowsApps` alias.

Important note:
- In Codex sandbox shell, environment variables may not auto-refresh between calls.
- In a normal new terminal session, `python`, `pip`, and `py` should resolve correctly.

## 11) Suggested next actions for next AI
If continuing development, prioritize in this order:
1. Run a quick sanity pass on representative samples and capture failure cases.
2. Convert key script tests into repeatable automated tests (pytest-style).
3. Add explicit regression fixtures for TF ambiguous cases and DG row-11 false positives.
4. Add configuration profile support for multiple sheet templates if needed.

