# UNI-T UTi120 USB-C Thermal Camera Multi-Platform Desktop Application

> This was a one-off weekend project. I'm sharing it as-is in case it's useful to someone, but I don't intend to maintain it. Nevertheless, it provides a solid and future-proof alternative in case UNI-T stops maintaining the Android application, and being made in Python, it provides a very good base to improve and adapt for future use cases.

Python application to connect to the UNI-T UTi120Mobile thermal camera via USB-C,
display live thermal imagery, and read temperatures.

Product link: https://thermal.uni-trend.com/product/uti120m/

## Screenshots

<figure>
  <img src="docs/images/Multi-point%20measures.png" alt="Multi-point measures">
  <figcaption>Multi-point temperature probes with live readout, center crosshair, and min/max markers</figcaption>
</figure>

<figure>
  <img src="docs/images/Palette%20mosaic%20mode.png" alt="Palette mosaic mode">
  <figcaption>Mosaic view showing all 6 color palettes simultaneously for side-by-side comparison</figcaption>
</figure>

<figure>
  <img src="docs/images/Temp%20graph%20and%20region%20statistics.png" alt="Temperature graph and region statistics">
  <figcaption>Temperature-over-time graph with ROI selection showing region statistics (center, min, max, avg, ΔT)</figcaption>
</figure>

<figure>
  <img src="docs/images/3d%20visualization.png" alt="3D visualization">
  <figcaption>Interactive 3D surface plot where pixel height represents temperature</figcaption>
</figure>

## UNI-T Thermal Camera Model `UTi120M` Technical Specifications:

- Resolution: 120×90
- Temperature range: -20°C to 400°C
- FOV: 50°×38°
- Frame rate: 25 Hz
- Measurement accuracy: ±2% or ±2°C, whichever is greater
- Measurement resolution: 0.1°C
- Emissivity settings: 0.10-1.00
- Response band: 8-14 µm

## Reverse engineer approach and other possible compatible models

The USB protocol was reverse-engineered by decompiling the official Android APK
(`UT-Mobile_v1.1.18.apk`, version 1.1.18) using JADX, and by analyzing the native
library (`libImageProc.so`) with Ghidra. Results were validated and corrected
through empirical testing with actual hardware. Full reverse engineering details
are in [REVERSE_ENGINEERING.md](docs/REVERSE_ENGINEERING.md).

The Ghidra analysis of the native binary revealed that the image processing logic
is model-agnostic, so this application should work with other thermal camera models
that share the same Android application. The manufacturer APK supports the following
models, all using the same processing pipeline and calibration system — only
branding and firmware update eligibility differ:

| Product ID | Model | Vendor |
|------------|-------|--------|
| 0 | UTi120M | UNI-T |
| 1 | TI220 | Klein Tools (KT) |
| 2 | Thermal | OEM |
| 3 | LUTi120M | OEM |

The device model and vendor are logged at startup by reading hardware registers.

## AI disclaimer

The project was written mostly by an AI agent (Claude Code Opus 4.6) under my direction. 
I guided key decisions, provided the source materials (decompiled APK, Ghidra disassembly
output), and conducted live hardware testing, but almost all code, including the reverse
engineering analysis, protocol implementation, image processing pipeline, calibration system, 
and GUI, was written by the AI. I have not reviewed the code in detail for correctness or 
side effects. Review it carefully before using it for anything important.

## Features

### Temperature Measurement
- **Factory-calibrated temperature conversion** matching the manufacturer's native algorithm. Calibration data is downloaded from the device on first connection and cached locally
- **Multi-curve bilinear interpolation** with per-frame shutter temperature bias
- **Emissivity correction** (0.01–1.00) with material presets (human skin, water, concrete, metals, etc.)
- **Lens drift compensation** between shutter calibrations
- **Automatic low/high range switching** (−20–150°C / 120–400°C)

### Live Display
- **Real-time thermal overlay** with center crosshair, min/max markers, color scale bar, and emissivity value
- **Mouse cursor temperature**: hover anywhere to read the temperature at that point
- **Pinnable probe points**: left-click to place up to 10 persistent temperature probes with live readout
- **6 color palettes**: Iron, Rainbow, White Hot, Black Hot, Jet, and Inferno
- **Mosaic view**: 3×2 grid showing all palettes simultaneously (`M`)
- **3D surface plot**: interactive OpenGL 3D view where pixel height = temperature (`3`)
- **Auto-range lock**: lock color scale to a fixed range (`L`)
- **Isotherm overlay**: highlight pixels above/below a configurable temperature threshold
- **Audio alarm**: configurable high/low thresholds with hysteresis deadband, visual and audible alerts, selectable temperature source
- **Adjustable brightness, contrast, flip, and rotation**

### Region of Interest (ROI)
- **Drag-to-select** any rectangular region on the thermal image
- **Live ROI statistics**: center, min, max, average, and delta (ΔT) temperatures updated every frame
- Clear selection with right-click or Escape

### Recording and Export
- **Configurable save directory**: defaults to `~/Pictures/UTi120/` (Linux/macOS) or `~/Documents/UTi120/` (Windows)
- **Radiometric screenshots**: PNG + `.npz` file with raw temperature map, emissivity, and FPA temp ([format details](docs/TECHNICAL.md#radiometric-npz-file-format))
- **Video recording** (MP4, 720×540) with all overlays baked in, records any active view
- **Temperature-over-time graph** with CSV export

### Image Processing
Dark frame NUC, per-pixel gain correction, bad pixel replacement, stripe removal, temporal noise filter, and automatic shutter calibration with countdown timer. See [docs/TECHNICAL.md](docs/TECHNICAL.md) for details.

### Sidebar Controls
Palette selector, emissivity presets + slider, lock range toggle, and live stats (center/max/min/FPA temperatures, calibration countdown). Collapsible sections for image adjustments, isotherm, alarm, camera, and capture controls.

## Quick Start

### Linux (Working under Ubuntu 24.04.4)

```bash
# Install libusb (if not already present)
sudo apt install libusb-1.0-0-dev   # Debian/Ubuntu
# or: sudo dnf install libusb1-devel  # Fedora

# Set up and run
python3 -m venv .venv
source .venv/bin/activate
pip install .

# Plug in the camera, then:
python3 -m uti120
```

Or use the all-in-one launcher script which creates the venv, installs
dependencies, and runs the application automatically:

```bash
./uti120M.sh
```

```bash
# Install the udev rule for permanent non-root access:
sudo cp 99-uti120.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
# Then unplug and replug the camera
```

### Windows (NOT TESTED)

1. Install [Python 3](https://www.python.org/downloads/) (check "Add to PATH" during install).
2. Install a USB driver for the camera using [Zadig](https://zadig.akeo.ie/):
   - Plug in the camera
   - Open Zadig, select the device with VID `5656` / PID `1201`
   - Replace the driver with **WinUSB**   
3. Open a terminal (PowerShell or Command Prompt):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install .

# Plug in the camera, then:
python -m uti120
```

### macOS (NOT TESTED)

1. Install [Homebrew](https://brew.sh/) if not already installed.
2. Install libusb and Python:

```bash
brew install libusb python

python3 -m venv .venv
source .venv/bin/activate
pip install .

# Plug in the camera, then:
python3 -m uti120
```

See [docs/SHORTCUTS.md](docs/SHORTCUTS.md) for all keyboard shortcuts and mouse actions. Press `H` in the viewer for a quick reference.

## Troubleshooting

### Camera not detected
1. Check `lsusb` for VID:PID `5656:1201`
2. Try flipping the USB-C plug
3. Use a direct connection (not through a hub)

### Permission denied
```bash
sudo .venv/bin/python3 -m uti120
# Or install the udev rule (see Quick Start)
```

## License

This project is licensed under the [MIT License](LICENSE).

This software was developed through clean-room reverse engineering of the USB
protocol for interoperability purposes. It contains no code from the original
manufacturer. The protocol was determined by decompiling the official Android
APK and disassembling the native shared library using Ghidra, then validated
through empirical testing with actual hardware.
