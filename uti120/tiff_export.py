"""Radiometric TIFF export for thermal frame data.

Writes a 32-bit IEEE float TIFF where each pixel value is the temperature in
**°C** directly. Opens natively in ImageJ/Fiji, QGIS, MATLAB, Metashape, and
Pix4D with cursor-hover showing °C — no decoding step.

There is no ISO/IEC standard for radiometric TIFF — only de-facto conventions.
Float32-Celsius is the most portable choice for the tools an end user is likely
to reach for.

All capture metadata (emissivity, ambient, distance, FPA/shutter/lens temps,
calibration range, frame counter, timestamp) is embedded in TIFF tag 270
(ImageDescription) as an ImageJ-compatible key/value block, plus a private tag
65000 carrying a JSON snapshot of the full FrameProcessor state for lossless
round-trip.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

from . import __version__

if TYPE_CHECKING:
    from .processor import FrameProcessor

__all__ = ["write_radiometric_tiff"]


def write_radiometric_tiff(processor: "FrameProcessor", base_path: Path) -> Path:
    """Write a float32-Celsius radiometric TIFF for the current frame.

    ``base_path`` is the path *without* extension (e.g. ``.../thermal_TS``).
    Returns the written path.
    """
    temp_map_c = np.asarray(processor._temp_map, dtype=np.float32)
    if temp_map_c.ndim != 2:
        raise ValueError(f"temp_map must be 2D, got shape {temp_map_c.shape}")

    timestamp = _dt.datetime.now().astimezone().isoformat()
    state_json = _state_json(processor, timestamp)
    path = base_path.with_suffix(".tif")

    vmin, vmax = float(temp_map_c.min()), float(temp_map_c.max())
    description = _imagej_description(processor, vmin, vmax, timestamp)

    # TIFF-standard SMinSampleValue/SMaxSampleValue (tags 340/341) — TIFF FLOAT
    # (type 'f') for float32 data. GDAL exposes these as band statistics, which
    # QGIS uses for its default raster styling. Photo-editor support is
    # inconsistent (GIMP ignores them).
    extratags = [
        (65000, "s", 0, state_json, True),
        (340, "f", 1, vmin, True),
        (341, "f", 1, vmax, True),
    ]
    tifffile.imwrite(
        str(path),
        temp_map_c,
        photometric="minisblack",
        description=description,
        software=f"uti120 v{__version__}",
        extratags=extratags,
        compression="zlib",
        # metadata=None suppresses tifffile's auto-added second ImageDescription
        # tag carrying its own {"shape": [...]} JSON, which otherwise overrides
        # our description for any reader that returns the last tag (PIL/libtiff).
        metadata=None,
    )
    return path


def _imagej_description(
    processor: "FrameProcessor",
    vmin: float,
    vmax: float,
    timestamp: str,
) -> str:
    lines = [
        "ImageJ=1.53k",
        "unit=C",
        f"min={vmin}",
        f"max={vmax}",
        f"emissivity={processor.emissivity:.4f}",
        f"ambient_c={processor.ambient_temp:.2f}",
        f"distance_m={processor.distance:.3f}",
        f"fpa_temp_c={processor.fpa_temp:.3f}",
        f"shutter_temp_c={processor.shutter_temp:.3f}",
        f"lens_temp_c={processor.lens_temp:.3f}",
        f"range={'high' if processor._active_range else 'low'}",
        f"frame_counter={processor.frame_counter}",
        "device=UNI-T UTi120",
        f"software=uti120 v{__version__}",
        f"timestamp={timestamp}",
    ]
    return "\n".join(lines)


def _state_json(processor: "FrameProcessor", timestamp: str) -> bytes:
    state = {
        "device": "UNI-T UTi120",
        "software": f"uti120 v{__version__}",
        "timestamp": timestamp,
        "frame_counter": int(processor.frame_counter),
        "emissivity": float(processor.emissivity),
        "ambient_c": float(processor.ambient_temp),
        "distance_m": float(processor.distance),
        "fpa_temp_c": float(processor.fpa_temp),
        "fpa_temp_raw": int(processor.fpa_temp_raw),
        "shutter_temp_c": float(processor.shutter_temp),
        "shutter_temp_start_c": float(processor.shutter_temp_start),
        "lens_temp_c": float(processor.lens_temp),
        "range": "high" if processor._active_range else "low",
        "dark_shutter_temp_c": float(processor._dark_shutter_temp),
        "dark_lens_temp_c": float(processor._dark_lens_temp),
        "dark_fpa_temp_c": float(processor._dark_fpa_temp),
        "flip": bool(processor.flip),
        "rotation_deg": int(processor.rotation),
        "brightness": int(processor.brightness),
        "contrast": int(processor.contrast),
        "encoding": {
            "formula": "pixel = T_C",
            "decode": "T_C = pixel",
        },
    }
    return json.dumps(state, separators=(",", ":")).encode("ascii")
