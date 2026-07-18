# IntersectionCV

**Real-time, calibration-free traffic delay analysis from CCTV video.**

![IntersectionCV demo](video/intersection.gif)

IntersectionCV is a desktop computer vision application that measures how much vehicles are slowed down or stopped on a road, and compares that directly against a second road, using nothing but ordinary CCTV footage. It combines YOLOv8/YOLO26 object detection with ByteTrack multi-object tracking to follow every vehicle through two independently shaped zones and compute delay the way a manual floating-car survey would, automatically, for every vehicle, in real time.

Built and tested against footage from Saheb Bazar – Alupotti Road, Rajshahi, Bangladesh, a bustling, mixed-traffic bazar corridor.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Output Files](#output-files)
- [Batch Processing](#batch-processing)
- [Running on Google Colab](#running-on-google-colab)
- [Project Structure](#project-structure)
- [Supported Models and Classes](#supported-models-and-classes)
- [Performance Notes](#performance-notes)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Standard traffic delay surveys are manual: a floating car drives with traffic and logs stops and travel time, or an observer counts stopped vehicles at fixed intervals. Both are labour-intensive, cover a short window, and are hard to repeat consistently.

IntersectionCV automates this from a fixed camera. You draw two zones directly on the video preview, one over each road or approach you want to compare, and the system does the rest: detect every vehicle, track it through the zone, measure how long it took to cross versus how long it should have taken, and aggregate that into live dashboards and exportable data, all without measuring or calibrating anything by hand.

## Key Features

### Detection and Tracking
- YOLOv8 and YOLO26 object detection, with support for custom-trained weights (e.g. BNVD, a Bangladeshi Native Vehicle Dataset fine-tune)
- ByteTrack multi-object tracking for persistent per-vehicle identity across frames
- Rolling-window speed estimation instead of a frame-to-frame derivative, resistant to ordinary detection jitter
- A motion dead-zone so a stationary vehicle's box jitter is never misread as travel
- Ghost-track bridging: a vehicle that vanishes from occlusion or a ByteTrack ID switch is held in a short-lived buffer and re-linked to the next plausible detection instead of being split into two vehicles

### Calibration-Free Delay Measurement
- No calibration bars, no manual real-world speed input
- Each zone learns its own free-flow baseline live, from the fastest crossings it actually observes (a low percentile of real crossing times)
- Travel delay = actual crossing time − that zone's own learned baseline
- A live 0–100% flow index shows how close a zone is running to its own free-flow pace at any moment

### Dual-Zone Road Comparison
- Two independently shaped, six-point resizable zones ("Road A" / "Road B") positioned anywhere in the same camera frame
- A side-by-side comparison panel with a live, plain-language narrative (e.g. "Road A is currently experiencing 4.2s more delay per vehicle than Road B")
- Because both zones are scored against their own history using the same formula, roads of different length, shape, or vantage remain directly comparable

### Kinematic Delay Analysis
- Stall count: number of discrete stop-start cycles per vehicle, distinguishing a single long stop from repeated stop-and-go
- Acceleration noise: standard deviation of a vehicle's acceleration across its crossing, a classical car-following-theory measure of how smooth or jerky its motion is
- Both computed automatically from tracked video, no instrumented probe vehicle required

### Pedestrian and Hawker Activity
- A parallel, independent tracking channel for pedestrians in each zone
- Dwell-time-based "loitering" classification distinguishes informal vendor/hawker activity from people simply passing through
- Reported live per zone alongside vehicle delay

### Volume and Capacity
- Every completed vehicle is converted to a Passenger Car Unit (PCU) and tallied into a PCU/hour throughput figure per zone
- Optional user-supplied road capacity produces a volume/capacity ratio, a second, independent diagnosis of whether congestion is capacity-driven or friction-driven (parking, hawkers, geometry)

### Optional Physical Queue Length
- A single, lightweight reference line (two endpoints plus a known real-world length) can be enabled per zone
- When enabled, queue length is reported in metres, measured from the actual observed spread of currently queued vehicles

### Delay Distribution, Not Just the Mean
- Median and 85th-percentile delay, and percentage of vehicles meaningfully delayed, alongside the average
- Prevents a single mean from hiding a bimodal pattern where half of traffic moves freely and half is badly delayed

### Batch Processing
- Select multiple videos at once and process them back to back, unattended
- One summary CSV per video, same name as the video, saved in the same folder
- Resilient to individual failures: an error on one video does not halt the rest of the queue

### Session Logging
- Every completed run appends one row to an accumulating CSV log (date, time, both zones' key figures, an optional free-text note)
- Supports comparing multiple recording sessions across different times and days without manually collating separate exports

### Performance
- Explicit GPU device selection with a startup diagnostic reporting exactly which device and precision a run is using
- FP16 (half-precision) inference on supported GPUs
- Configurable inference resolution
- Live preview and UI refresh rate decoupled from the analysis rate: every frame is always fully analysed, only the (comparatively expensive) drawing and image transmission is rate-limited
- Detection restricted to only the classes actually used, reducing per-frame post-processing

### Cloud Deployment
- A companion script imports the same core analysis engine directly and runs it headlessly on Google Colab's free GPU tier, no reimplementation, same math, same CSV output

---

## How It Works

**Pipeline (repeats every video frame):**

1. **Video frame** — read frame by frame, none skipped
2. **Detect and track** — YOLO + ByteTrack, restricted to relevant classes
3. **Zone test** — point-in-polygon check against Road A and Road B
4. **Per-vehicle tracking** — rolling-window speed, stopped time, stall count, acceleration noise
5. **Zone aggregation** — live delay statistics and Road A vs. Road B comparison

**Per-vehicle, every frame it is inside a zone:**

```
speed          = distance over a rolling time window
is_stopped     = speed < zone_stop_threshold
stopped_time  += dt                          if is_stopped
stall_count   += 1                           on each moving -> stopped transition
distance      += segment_distance             if segment_distance > motion_floor
accel_noise    = stdev(per-frame acceleration samples)
```

**On exit (vehicle leaves the zone, or is confirmed lost after occlusion):**

```
time_in_zone     = exit_time - entry_time
free_flow_time   = 15th percentile of this zone's own historical crossing times
travel_delay     = max(0, time_in_zone - free_flow_time)
```

That `free_flow_time` baseline is what removes the need for calibration: as more vehicles complete a crossing, each zone continuously refines its own definition of "normal," and every subsequent vehicle's delay is measured against that. The baseline needs a handful of observed crossings (default: 5) before it is trusted; before that, delay is reported conservatively.

**Occlusion / ID-switch handling:** when a tracked vehicle is not seen in a frame (occluded, or ByteTrack assigns it a new ID on reappearance), its state is held as a "ghost" rather than finalized immediately. If a new detection appears nearby within a plausible time and distance (scaled to the zone's own learned pace), it is re-linked to the original vehicle's record. Ghosts that are not reclaimed within the bridge window are committed as genuinely finished.

---

## Installation

### Requirements

- Python 3.10+
- PyQt6
- ultralytics (YOLOv8 / YOLO26)
- opencv-python
- torch (with CUDA support recommended for real-time performance)
- psutil
- GPUtil (optional, for GPU usage display)

### Setup

```bash
git clone https://github.com/<your-username>/intersection-cv.git
cd intersection-cv

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install PyQt6 ultralytics opencv-python torch psutil GPUtil
```

For GPU acceleration, install a CUDA-enabled build of PyTorch matching your driver version from [pytorch.org](https://pytorch.org/get-started/locally/) **before** installing `ultralytics`, otherwise it may pull a CPU-only build.

Verify CUDA is actually available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run the app:

```bash
python intersection_cv_gui.py
```

---

## Usage

### Quick Start

1. **Configuration tab** — fill in survey metadata, pick a YOLOv8/YOLO26 variant or browse to custom weights, and review the Delay Parameters and Performance defaults.
2. **Analysis tab** — load a video (or select multiple for batch mode), drag the six corner handles of **Road A** and **Road B** onto the two roads/approaches you want to compare, then click **Run Analysis**.
3. Watch both zones' live dashboards and the comparison panel update in real time.
4. **Results tab** — export the summary CSV, the full per-vehicle log CSV, JSON, or preview a printable data sheet, at any time including mid-run.

### Configuring Zones

Each zone is a freely resizable six-point polygon, dragged directly on the video preview, or entered as exact percentage coordinates in Configuration → Comparison Zones. Layouts (both zones' shapes, plus any optional reference line) can be saved to and loaded from a JSON file, useful for reusing the same camera setup across multiple recording sessions.

### Delay Parameters

Tunable in Configuration → Delay Parameters:

| Setting | What it controls |
|---|---|
| Free-flow percentile | Which percentile of observed crossing times defines "free flow" for a zone (default 15%) |
| Stop sensitivity | Speed threshold (as % of zone size per second) below which a vehicle counts as stopped |
| Track-loss grace | How long an occluded/lost vehicle stays revivable before being finalized |
| Timeline bucket | Width of each bucket in the live flow-index chart |
| Speed averaging window | Rolling window length for the speed estimate; raise if readings look jumpy |
| Motion noise floor | Per-frame movement below this fraction of zone size is treated as detection jitter, not travel |

### Batch Mode

Click **Multiple Videos** instead of **Browse Video**, select as many files as needed, then **Run Analysis**. Each video is processed with the same zone layout and settings; a CSV is written next to each video with a matching filename, and the queue list shows live pass/fail status per video.

### Google Colab

`colab_batch_delay_analysis.py` imports `DelayWorker` and `write_summary_csv` directly from `intersection_cv_gui.py` and runs them headlessly (no display required) against a free T4 GPU:

1. Export a zone layout from the desktop app (Configuration → Save Layout).
2. Upload `intersection_cv_gui.py`, the layout JSON, and your videos to Colab / Google Drive.
3. Edit the config block at the top of `colab_batch_delay_analysis.py` (paths, model, confidence/IoU thresholds).
4. Run the script. One CSV is written next to each video, identical in format to the desktop app's output.

---

## Configuration Reference

| Group | Setting | Default | Notes |
|---|---|---|---|
| Model | Architecture | YOLOv8m | YOLOv8 or YOLO26, or custom `.pt` weights |
| Model | Confidence threshold | 0.40 | |
| Model | IoU threshold | 0.45 | |
| Performance | Compute device | Auto | Auto / GPU (CUDA) / CPU |
| Performance | Precision | Auto (FP16 on GPU) | |
| Performance | Inference resolution | 640 | 416 (fastest) to 1280 (most accurate) |
| Performance | Preview refresh rate | 15 fps | Analysis always runs on every frame regardless |
| Pedestrians & Capacity | Track pedestrians/hawkers | On | Requires a model with a person class |
| Pedestrians & Capacity | Loiter threshold | 20 s | Dwell time before a pedestrian reads as hawker-like |
| Pedestrians & Capacity | Zone capacity | Unset | PCU/hr; enables the volume/capacity ratio when set |
| Session Log | Auto-log each run | On | Appends to an accumulating CSV |

---

## Output Files

### Summary CSV
Per zone: vehicle count, total/average/median/85th-percentile delay, % vehicles delayed, stopped time, stall count, motion smoothness, PCU volume, volume/capacity ratio (if set), queue length (if a reference line is set), pedestrian/hawker activity, and a full per-vehicle-class breakdown. Leads with a "delay at a glance" table.

### Vehicle Log CSV
One row per tracked vehicle: zone, class, entry/exit time, time in zone, travel delay, stopped delay, pace percentage, stall count, acceleration noise.

### JSON
The complete result set (both zones, full class/timeline breakdowns, per-vehicle records) for downstream analysis.

### Session Log
One row appended per completed run: date, time, both zones' key figures, and any free-text condition notes (e.g. "market day," "heavy rain") — accumulates across multiple sessions for time/day comparisons.

---

## Project Structure

```
intersection_cv_gui.py             Desktop application (PyQt6) — the full analysis engine and GUI
colab_batch_delay_analysis.py      Headless batch runner for Google Colab, reuses the same engine
```

The analysis engine (`DelayWorker`) is a `QThread` subclass but has no other GUI dependency — it can be driven directly (bypassing `.start()`/the Qt event loop) for testing or headless/batch use, which is how the Colab script works.

---

## Supported Models and Classes

**Detection:** YOLOv8 (n/s/m/l/x) and YOLO26 (n/s/m/l/x), or any custom-trained `.pt` weights with a compatible class list.

**Vehicle classes:** Buses, Micro Buses, Trucks, Mini Trucks, Private Cars, Human Hollar, Bi-Cycles, Motor Cycles, Rickshaw, Auto Rickshaws, Van — mapped automatically from BNVD-style or COCO class names where possible.

**Pedestrians:** requires the loaded model to include a `person` class (present in COCO-trained weights; typically absent from vehicle-only custom weights such as BNVD, in which case pedestrian/hawker counts will read zero and the app will say so at startup).

---

## Performance Notes

- If a run reports "running on CPU" in the status line despite having a GPU, your PyTorch install most likely lacks CUDA support, reinstall it from [pytorch.org](https://pytorch.org/get-started/locally/) with the correct CUDA version for your driver.
- Lowering inference resolution and using a smaller model (`yolov8n`/`yolov8s`) meaningfully improves throughput on 4GB-class GPUs.
- A hardware-usage-monitoring call in early versions of this project was found to shell out to `nvidia-smi` as a subprocess on every frame, causing severe slowdowns; this is fixed in the current version (stats are cached and refreshed at most twice a second), but is worth knowing if profiling similar tools.

---

## Known Limitations

- LST/NDVI/road-density-style geospatial context is out of scope for this tool; it measures delay, volume, and pedestrian activity only.
- Each zone's free-flow baseline needs a handful of observed crossings before it is reliable; early readings from a short or unusually light-traffic recording should be treated with caution.
- Pedestrian/hawker detection depends on the loaded model including a person class.
- Occlusion handling uses position and elapsed time only, without visual re-identification, and can occasionally misattribute a track under prolonged, heavy occlusion.
- True batched GPU inference is not implemented; ByteTrack's sequential, online nature makes this a non-trivial extension rather than a simple config change.

## Roadmap

- Optional manually specified free-flow baseline for cross-validation against the self-learned one
- Visual re-identification for more robust occlusion handling
- Batched inference pipeline for further throughput gains
- Live RTSP/streaming camera support (frame-source abstraction is already in place for this)

---

## Acknowledgments

- [Ultralytics YOLOv8 / YOLO26](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BNVD (Bangladeshi Native Vehicle Dataset)](https://github.com/bipin-saha/BNVD)

## License

License to be determined.
