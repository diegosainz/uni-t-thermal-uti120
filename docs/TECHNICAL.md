# Technical Reference

This document covers implementation details for the UNI-T UTi120 thermal viewer.
For usage and installation see [README.md](../README.md). For reverse engineering
notes see [REVERSE_ENGINEERING.md](REVERSE_ENGINEERING.md).

## Image Processing Pipeline

- **Dark frame NUC** (Non-Uniformity Correction): subtracts a shutter-captured reference frame for per-pixel offset correction
- **Per-pixel gain correction** from factory K-buffer tables (Q13 fixed-point gain factors)
- **Bad pixel replacement**: factory-flagged bad pixels (gain table bit 15) are replaced via mask, plus any per-frame outlier deviating >8x MAD (Median Absolute Deviation) from its 3x3 median is unconditionally replaced — catching stuck, sluggish, and bad-gain pixels without detection delays
- **Vertical and horizontal stripe removal**: Ghidra-confirmed destriping algorithm using directional Gaussian smoothing and clamped residual correction
- **Temporal noise filter (TFF)**: motion-adaptive bilateral filter that smooths stationary pixels while preserving moving edges
- **Automatic shutter calibration**: FPA drift-triggered and periodic dark frame refresh (see below)

## Automatic Shutter Calibration

The camera performs automatic shutter calibration (dark frame refresh) to maintain
temperature measurement accuracy. Microbolometer sensors like the one in the UTi120
are subject to **FPA (Focal Plane Array) temperature drift**: as the sensor heats up
during operation, the baseline signal shifts, introducing measurement error. The shutter
briefly closes to capture a "dark frame" reference that is subtracted from live frames
to correct for this drift.

The auto-calibration logic (reconstructed from the APK's `ShutterHandler.java`) uses
three mechanisms:

1. **FPA drift → NUC + Shutter** (rare): When the FPA temperature drifts significantly
   from its baseline (≥0.6–1.2°C depending on warmup phase), a full hardware
   Non-Uniformity Correction is triggered along with a dark frame refresh.
2. **FPA drift → Shutter only** (frequent): On moderate drift (≥0.3–0.8°C), only the
   shutter closes for a quick dark frame update.
3. **Periodic timer** (after 6 min): Once the camera has been running for 6 minutes and
   is thermally stable, a shutter calibration is forced every 60 seconds regardless of
   drift, to prevent slow undetected error accumulation.

The drift thresholds tighten over time as the camera warms up:

| Time window | Shutter threshold | NUC threshold |
|-------------|-------------------|---------------|
| 0–3 min     | 0.80°C            | 1.20°C        |
| 3–6 min     | 0.50°C            | 1.00°C        |
| 6+ min      | 0.30°C            | 0.60°C        |

During calibration the image briefly freezes (~0.5s). This is normal and expected.

### Warmup Skip via Frame Counter

The warmup phase (graduated thresholds) exists because a cold microbolometer drifts
faster as it heats up. However, if the camera has already been powered on for a while
(e.g. reconnecting to an already-warm camera), repeating the warmup is unnecessary.

On the first frame received, the application reads the **frame counter** from the
frame header (uint16 at short offset 1, increments each frame). If the counter
exceeds 3600 (~6 minutes at 10 FPS), the warmup phase is skipped entirely: tight
thresholds (0.30°C/0.60°C) and the 60-second periodic timer are activated immediately.

Note: the frame counter is uint16 and wraps at 65535 (~109 min at 10 FPS). After a
wrap the counter could appear low even though the camera is warm, but this only
affects the initial threshold selection — calibration still works correctly with
wider thresholds during the first few minutes.

### Calibration Countdown

The Device stats panel shows a `Cal:` label indicating time until the next calibration:

- **`Cal: 45s`** — countdown to periodic calibration (after warmup, or skipped if camera already warm)
- **`Cal: drift 85%`** — FPA drift is approaching the shutter threshold (≥70%), calibration imminent
- **`Cal: warmup`** — camera is in the warmup phase, periodic timer not yet active

## Radiometric TIFF File Format

Each screenshot saves a radiometric TIFF alongside the PNG. TIFF is the most
portable cross-vendor radiometric raster format — vendor-specific formats like
FLIR's R-JPEG, DJI's R-JPG, and various CSV layouts don't open in other
ecosystems. Standard single-band TIFF, on the other hand, is read by
ImageJ/Fiji, QGIS, MATLAB, scientific Python, and most photo editors as image
data.

| File | Pixel type | Encoding | Decode |
|------|-----------|----------|--------|
| `thermal_{TS}.tif` | float32, single-band | `pixel = T_°C` (direct) | `T_°C = pixel` |

Opens natively in ImageJ/Fiji, QGIS, MATLAB, Metashape, and Pix4D with
cursor-hover showing °C — no decode step needed.

### Metadata layout

**TIFF tag 270 (ImageDescription)** — plain-text ImageJ-compatible key/value
block. Example:

```
ImageJ=1.53k
unit=C
min=20.13
max=37.85
emissivity=0.9500
ambient_c=22.00
distance_m=1.000
fpa_temp_c=37.250
shutter_temp_c=37.180
lens_temp_c=37.310
range=low
frame_counter=4218
device=UNI-T UTi120
software=uti120 v1.0.0
timestamp=2026-06-19T14:30:22.451+02:00
```

**TIFF tag 305 (Software)** — `uti120 v<version>`.

**Private tag 65000 (ASCII / JSON)** — full FrameProcessor state snapshot for
lossless round-trip: emissivity, ambient, distance, all four temperature
sensors (FPA / shutter / shutter-start / lens), calibration range,
frame_counter, dark-frame baseline temps, image transform state (flip /
rotation / brightness / contrast), and encoding formulas. Optional; non-essential
for third-party tools but lets uti120 reconstruct the capture context exactly.

### Loading in Python

```python
import tifffile

temp_map_c = tifffile.imread("thermal_20260619_143022.tif")  # float32, °C

# Read embedded metadata
with tifffile.TiffFile("thermal_20260619_143022.tif") as tf:
    desc = tf.pages[0].tags["ImageDescription"].value
    meta = dict(line.split("=", 1) for line in desc.splitlines() if "=" in line)
    print(f"Emissivity: {meta['emissivity']}, FPA: {meta['fpa_temp_c']} °C")
```

### Loading in ImageJ / Fiji

Drag the file into Fiji — cursor hover shows temperature in °C directly.

### Loading in QGIS

Drag the file in. It loads as a single-band raster with pixel values in °C;
the identify tool reads off temperatures directly.

### Opening in GIMP / Photoshop / other photo editors

These tools are general-purpose image editors, not thermal viewers — they will
show the file but probably **look blank**. GIMP assumes float TIFFs are
normalized to [0, 1]; our pixels are temperatures in °C (e.g. 20.0), all above
1.0, so GIMP clips everything to white. Use `Colors > Curves` and stretch the
input range to the actual temperature range to see the scene.

We embed the TIFF-standard `SMinSampleValue` / `SMaxSampleValue` tags (340/341)
hinting at the true value range. GDAL surfaces them as band statistics, which
QGIS uses for its default raster styling. Photo-editor support for these tags
is inconsistent — GIMP, in particular, doesn't auto-stretch based on them. For
visual inspection in a photo editor, open the PNG instead, or apply the
stretch manually.

**For visual inspection, open the `.png` file instead** — that's the display
image with overlays already in 8-bit RGB, which any viewer renders correctly.
The TIFF is for radiometric analysis tools.

### Compatibility notes

- **FLIR ResearchIR / FLIR Tools**: opens the file as 32-bit grayscale but
  treats it as raw counts — these tools assume FLIR's own encoding
  (`T_K = pixel × 0.04`) and will not show correct temperatures. Read pixels
  with the formula `T_°C = pixel`.
- **Hikmicro Analyzer / DJI Thermal Analysis Tool**: similar — proprietary
  encoding expected. Files open as raw grayscale.
- **MATLAB / scikit-image / Python tifffile**: read into numerical arrays
  directly; pixel values are already °C.

## GUI Stack

The viewer uses **PyQt6** for the GUI with a custom `QPainter`-based thermal display
widget and a dark Fusion theme. OpenCV is used internally for image processing
(resize, color conversion, palette application) but not for display. The camera USB
loop runs on a `QThread` with frame data delivered via signals to the main thread.
