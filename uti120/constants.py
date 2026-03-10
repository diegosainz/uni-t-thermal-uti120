"""Constants for UNI-T UTi120 thermal camera protocol and display."""

from __future__ import annotations

import sys
from pathlib import Path

# Camera VID:PID
USB_VID = 0x5656
USB_PID = 0x1201

# Frame layout (confirmed empirically):
#   25600 bytes = 12800 uint16 LE shorts
#   shorts[0:360]     = header/params (720 bytes)
#   shorts[360:11160] = 90 rows x 120 cols pixel data (21600 bytes)
#   shorts[11160:12800] = padding/zeros (3280 bytes)
FRAME_WIDTH = 120
FRAME_HEIGHT = 90
FRAME_PIXELS = FRAME_WIDTH * FRAME_HEIGHT  # 10800
FRAME_SIZE = 25600  # bytes per frame from device
PIXEL_OFFSET = 360  # short offset where pixel data begins
BULK_CHUNK_SIZE = 4096
BULK_TIMEOUT_MS = 13  # APK default per-chunk timeout
BULK_MAX_ITERS = (FRAME_SIZE // BULK_CHUNK_SIZE) + 2

# Frame header fields (short offsets)
HDR_FRAME_COUNTER = 1
HDR_SHUTTER_TEMP_START = 8
HDR_SHUTTER_TEMP_RT = 9
HDR_LENS_TEMP = 10
HDR_FP_TEMP = 11
# Command function codes
FUNC_WRITE_REG = 0x04
FUNC_READ_REG = 0x05
FUNC_SENSOR_CMD = 0x0A
FUNC_READ_SENSOR = 0x0B

# Register offsets
REG_RUN_STATUS = 0xF0

# Run status values
STATUS_IDLE = 0
STATUS_IMAGE_UPLOAD = 2

# Sensor command offsets
SENSOR_SHUTTER = 0x03
SENSOR_NUC = 0x04
SENSOR_MEASURE_RANGE = 0x09

# Frame request byte (0x81 = -127 signed, confirmed by testing)
CMD_REQUEST_FRAME = bytes([0x81])

# Display
DISPLAY_WIDTH = FRAME_WIDTH * 6   # 720
DISPLAY_HEIGHT = FRAME_HEIGHT * 6  # 540

# Transfer protocol (calibration package download)
FUNC_TRANSFER = 0x09
STATUS_PARAM_UPLOAD = 3
TRANSFER_BEGIN = 0x00
TRANSFER_CRC = 0x02
TRANSFER_END = 0x04
CALIB_CHUNK_SIZE = 4096

# Calibration package flash locations
CALIB_HIGH_FLASH_ADDR = 0x100000
CALIB_LOW_FLASH_ADDR = 0x132000
REG_PKG_LENGTH_HIGH = 0x0C
REG_PKG_LENGTH_LOW = 0x0D

# Application defaults
DEFAULT_PALETTE_IDX = 5           # Inferno
DEFAULT_EMISSIVITY = 0.95
DEFAULT_CONTRAST = 128            # unity multiplier (range 64-255)
DEFAULT_AMBIENT_TEMP = 22.0       # reflected / ambient temperature (°C)

# Processing parameters
FPA_SMOOTH_WINDOW = 15            # median-filter buffer for FPA temperature
DEFAULT_TFF_STD = 5               # temporal noise filter sigma
TEMP_MARGIN = 2                   # edge pixels excluded from min/max stats
RANGE_SWITCH_UP_C = 150.0         # switch to high range above this °C
RANGE_SWITCH_DOWN_C = 120.0       # switch back to low range below this °C
RANGE_SWITCH_COOLDOWN_S = 5.0     # seconds between range switches

# UI intervals
STATS_UPDATE_INTERVAL_MS = 250    # temperature label refresh
GRAPH_SAMPLE_INTERVAL_MS = 500    # graph data sampling rate

# Limits
MAX_PROBES = 10                   # maximum pinned temperature probes

# Alarm
ALARM_TEMP_MIN = -20
ALARM_TEMP_MAX = 400
ALARM_HYSTERESIS_DEFAULT = 0.5
ALARM_SOURCES = [
    ("Center", "center"),
    ("Global MAX", "global_max"),
    ("Global MIN", "global_min"),
    ("ROI Center", "roi_center"),
    ("ROI Avg", "roi_avg"),
    ("ROI MAX", "roi_max"),
    ("ROI MIN", "roi_min"),
    ("Mouse", "mouse"),
]
VIDEO_FPS = 25                    # recording framerate
RECONNECT_FAIL_THRESHOLD = 20     # consecutive failed frames before reconnect

# Emissivity presets — (display_name, emissivity_value)
# First entry must be "Custom" with None (used when slider is manually adjusted).
# Values sourced from ThermoWorks / Fluke industry reference tables.
EMISSIVITY_PRESETS = [
    ("Custom", None),
    ("Human Skin", 0.98),
    ("Water", 0.95),
    ("Ice / Snow", 0.97),
    ("Concrete", 0.95),
    ("Brick (red)", 0.93),
    ("Wood (planed)", 0.90),
    ("Glass", 0.92),
    ("Paper", 0.93),
    ("Rubber", 0.95),
    ("Plastic (black)", 0.95),
    ("Paint (flat)", 0.94),
    ("Fabric / Cloth", 0.90),
    ("Soil / Earth", 0.92),
    ("Asphalt", 0.95),
    ("Oxidized Steel", 0.79),
    ("Stainless Steel", 0.59),
    ("Oxidized Copper", 0.65),
    ("Anodized Aluminum", 0.77),
    ("Polished Metal", 0.10),
]


def default_save_dir() -> Path:
    """Return the default directory for saving screenshots, videos, and CSVs."""
    home = Path.home()
    if sys.platform == 'win32':
        return home / 'Documents' / 'UTi120'
    return home / 'Pictures' / 'UTi120'
