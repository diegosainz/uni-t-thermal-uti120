"""Frame processing for thermal image data."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import cv2
from scipy.ndimage import median_filter

from .constants import (
    FRAME_SIZE, FRAME_WIDTH, FRAME_HEIGHT, FRAME_PIXELS, PIXEL_OFFSET,
    HDR_FRAME_COUNTER, HDR_SHUTTER_TEMP_RT, HDR_LENS_TEMP, HDR_FP_TEMP, HDR_SHUTTER_TEMP_START,
    DEFAULT_PALETTE_IDX, DEFAULT_EMISSIVITY, DEFAULT_CONTRAST,
    DEFAULT_AMBIENT_TEMP, FPA_SMOOTH_WINDOW, DEFAULT_TFF_STD,
    TEMP_MARGIN, RANGE_SWITCH_UP_C, RANGE_SWITCH_DOWN_C,
    RANGE_SWITCH_COOLDOWN_S,
)
from .palettes import apply_palette
from .calibration import (
    get_curve_segments, y16_to_temperature_array,
    y16_to_temperature_interpolated, lens_drift_correct_zx01c,
    emiss_correct,
)

if TYPE_CHECKING:
    from .calibration import CalibrationPackage

__all__ = ["FrameProcessor"]

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Process raw frame data into thermal images."""

    def __init__(self) -> None:
        self.palette_idx = DEFAULT_PALETTE_IDX
        self._last_normalized = None
        self.flip = False
        self.rotation = 0  # 0, 90, 180, 270 degrees clockwise
        self.min_temp = 0.0
        self.max_temp = 0.0
        self.center_temp = 0.0
        self.min_pos = (0, 0)
        self.max_pos = (0, 0)
        # Time-gated max/min updates (once per second)
        self._last_maxmin_update = 0.0
        self.brightness = 0
        self.contrast = DEFAULT_CONTRAST

        self.auto_range = True          # True = auto-scale, False = locked
        self._locked_vmin = None        # locked Y16 percentile min
        self._locked_vmax = None        # locked Y16 percentile max
        self.emissivity = DEFAULT_EMISSIVITY
        self.ambient_temp = DEFAULT_AMBIENT_TEMP
        self.distance = 1.0  # object distance in metres (correction disabled: MP[0xa4]=0)
        # Mouse cursor temperature
        self.mouse_temp = None  # None when cursor is outside window
        self.mouse_pos = None  # display pixel coords, None until mouse enters
        self._temp_map = None   # last computed temperature map (90x120)
        # Header info from last frame
        self.frame_counter = 0
        self.shutter_temp = 0.0
        self.shutter_temp_start = 0.0
        self.lens_temp = 0.0
        self.fpa_temp = 0.0
        self.fpa_temp_raw = 0
        # Calibration curve (piecewise-linear correction, fallback)
        self._cal_native = None
        self._cal_true = None
        # Factory calibration (curve-based, accurate)
        self._calib_pkg = None
        self._dark_frame = None
        self._dark_shutter_temp = 25.0  # shutter RT temp when dark was captured
        self._dark_lens_temp = 0.0      # lens temp when dark was captured
        self._dark_fpa_temp = 0.0       # FPA temp when dark was captured
        self._curve_buf = None
        self._curve_segments = None  # all 4 curve segments for interpolation
        self._fpa_weight = 1.0       # FPA interpolation weight
        self._curve_steps = 0
        self._core_body_temp = 0.0
        self._nuc_gain = None  # per-pixel NUC gain from Section 2 K-buffer
        self._bad_pixel_mask = None  # bool mask from gain table bit 15
        # FPA temperature smoothing (Ghidra: smoothFocusTemp, median)
        self._fpa_temps = deque(maxlen=FPA_SMOOTH_WINDOW)
        # Range switching state
        self._active_range = 0  # 0=low, 1=high
        self._calib_low = None
        self._calib_high = None
        self._last_range_switch = 0.0  # time.time() of last switch
        # Flat-field correction map (per-pixel FPN offsets)
        self.fpn_map = None
        self._load_flatfield()
        # Temporal Noise Filter (TFF) state
        # Matches native TimeNoiseFliter at 0x118a70
        self.tff_std = DEFAULT_TFF_STD
        self._tff_prev = None
        self._tff_weights = None
        self._build_tff_weights()

    def _load_flatfield(self) -> None:
        """Load flat-field FPN correction map if available."""
        fpn_path = Path(__file__).resolve().parent.parent / 'flatfield_fpn.npy'
        if fpn_path.exists():
            self.fpn_map = np.load(str(fpn_path)).astype(np.float32)
            if self.fpn_map.shape == (FRAME_HEIGHT, FRAME_WIDTH):
                logger.info("  Loaded flat-field correction (FPN std=%.1f)", self.fpn_map.std())
            else:
                logger.warning("  flatfield_fpn.npy shape mismatch, ignoring")
                self.fpn_map = None

    def _build_tff_weights(self) -> None:
        """Precompute 128-entry TFF weight lookup table.

        Matches native put_tff_std(): Gaussian decay exp(-d^2/std^2)
        scaled to [0, 4096]. weight[d] = prev frame contribution when
        pixel difference is d Y16 counts. std=5 → diffs >~10 get little smoothing.
        """
        std_sq = float(self.tff_std * self.tff_std)
        weights = np.zeros(128, dtype=np.int32)
        for d in range(128):
            weights[d] = int(round(4096.0 * math.exp(-(d * d) / std_sq)))
        self._tff_weights = weights

    def _apply_tff(self, y16_display: np.ndarray) -> np.ndarray:
        """Apply motion-adaptive bilateral temporal noise filter.

        Matches native TimeNoiseFliter (0x118a70). Blends current frame with
        previous using per-pixel weight based on absolute difference. Large
        differences (>= 128 counts) pass through unfiltered (real scene change).
        """
        if self._tff_prev is None:
            self._tff_prev = y16_display.copy()
            return y16_display

        prev = self._tff_prev
        diff = np.abs(y16_display - prev)
        diff_idx = np.clip(diff.astype(np.int32), 0, 127)
        w = self._tff_weights[diff_idx]

        blended = ((y16_display * (4096 - w) + prev * w) / 4096.0).astype(np.float32)
        output = np.where(diff >= 128.0, y16_display, blended)
        self._tff_prev = output.copy()
        return output

    def set_tff_param(self, std: int) -> None:
        """Set temporal noise filter strength (APK default: 5, range 1-200)."""
        std = max(1, min(200, int(std)))
        if std != self.tff_std:
            self.tff_std = std
            self._build_tff_weights()
            self._tff_prev = None

    def set_calibration(self, points: list[tuple[float, float]]) -> None:
        """Set calibration from device-read points (fallback method).

        Args:
            points: list of (reference_temp, camera_reading) tuples in °C,
                    as returned by UTi120Camera.read_calibration_points().
        """
        # Filter out zero/zero pairs and sort by camera reading
        valid = [(ref, cam) for ref, cam in points if ref != 0.0 or cam != 0.0]
        valid.sort(key=lambda p: p[1])
        if len(valid) >= 2:
            self._cal_native = np.array([p[1] for p in valid])
            self._cal_true = np.array([p[0] for p in valid])

    def set_calibration_package(self, calib_pkg: CalibrationPackage) -> None:
        """Set factory calibration package for accurate temperature conversion.

        Args:
            calib_pkg: CalibrationPackage instance loaded from device flash data
        """
        self._calib_pkg = calib_pkg
        self._curve_steps = calib_pkg.curve_steps
        self._core_body_temp = calib_pkg.core_body_temp
        # Force immediate max/min update on next frame after calibration change
        self._last_maxmin_update = 0.0

        # Extract bad pixel mask from gain table bit 15 (Ghidra-confirmed)
        # gain bit 15 set = bad pixel, used by ReplaceBadPoint
        if calib_pkg.corrections.shape[0] > 0:
            gain_table = calib_pkg.corrections[0]
            self._bad_pixel_mask = (gain_table & 0x8000) != 0
            n_bad = np.count_nonzero(self._bad_pixel_mask)
            if n_bad > 0:
                logger.info("  Bad pixel mask: %d pixels flagged", n_bad)

        # Initial curve selection (will be updated per-frame based on FPA temp)
        self._select_curve(self.fpa_temp_raw)

    def set_calibration_packages(self, low_pkg: CalibrationPackage | None, high_pkg: CalibrationPackage | None) -> None:
        """Store both calibration packages for range switching.

        Args:
            low_pkg: CalibrationPackage for low-temp range (may be None)
            high_pkg: CalibrationPackage for high-temp range (may be None)
        """
        self._calib_low = low_pkg
        self._calib_high = high_pkg
        # Start in low range
        if low_pkg is not None:
            self._active_range = 0
            self.set_calibration_package(low_pkg)

    def check_range_switch(self) -> int | None:
        """Check if temperature range should switch based on max_temp.

        Returns:
            New range_id (0 or 1) if switch needed, None otherwise.
        """
        if self._calib_low is None or self._calib_high is None:
            return None

        now = time.time()
        if now - self._last_range_switch < RANGE_SWITCH_COOLDOWN_S:
            return None

        if self._active_range == 0 and self.max_temp > RANGE_SWITCH_UP_C:
            return 1
        if self._active_range == 1 and self.max_temp < RANGE_SWITCH_DOWN_C:
            return 0
        return None

    def switch_range(self, new_range: int) -> CalibrationPackage:
        """Switch to the specified calibration range.

        Args:
            new_range: 0=low, 1=high

        Returns:
            The CalibrationPackage for the new range.
        """
        pkg = self._calib_high if new_range == 1 else self._calib_low
        self._active_range = new_range
        self._last_range_switch = time.time()
        self.set_calibration_package(pkg)
        return pkg

    @property
    def active_range(self) -> int:
        return self._active_range

    def set_dark_frame(self, dark_frame: np.ndarray | None, shutter_temp: float | None = None,
                       lens_temp: float | None = None, fpa_temp: float | None = None) -> None:
        """Set dark frame reference for NUC correction.

        Args:
            dark_frame: float32 array (90x120) captured during shutter close,
                       or None to clear
            shutter_temp: shutter RT temp at dark capture time (°C)
            lens_temp: lens temp at dark capture time (°C)
            fpa_temp: FPA temp at dark capture time (°C)
        """
        self._dark_frame = dark_frame
        if shutter_temp is not None:
            self._dark_shutter_temp = shutter_temp
        if lens_temp is not None:
            self._dark_lens_temp = lens_temp
        if fpa_temp is not None:
            self._dark_fpa_temp = fpa_temp
        if dark_frame is not None:
            # Reset FPA smoothing buffer on dark capture (Ghidra: ResetFocusTempState)
            self._fpa_temps.clear()
            # Reset TFF buffer: NUC offset changed, prev frame no longer valid
            self._tff_prev = None
            logger.info("  Dark frame set: mean=%.0f, std=%.0f", dark_frame.mean(), dark_frame.std())


    def _replace_bad_pixels(self, y16_frame: np.ndarray) -> np.ndarray:
        """Replace bad pixels with 3x3 median of valid neighbors.

        Two-pass approach:
        1. Replace pixels flagged in the bad pixel mask (factory + runtime detected)
        2. Replace any remaining outlier that deviates >5x MAD from its 3x3 median
        """
        med = median_filter(y16_frame, size=3)
        result = y16_frame.copy()
        # Pass 1: mask-based replacement
        if self._bad_pixel_mask is not None and np.any(self._bad_pixel_mask):
            result[self._bad_pixel_mask] = med[self._bad_pixel_mask]
        # Pass 2: unconditional outlier replacement
        deviation = np.abs(result - med)
        mad = np.median(deviation)
        threshold = max(mad * 8.0, 20.0)
        outliers = deviation > threshold
        if np.any(outliers):
            result[outliers] = med[outliers]
        return result

    def _select_curve(self, fpa_temp_raw: int) -> None:
        """Select curve segments and pixel offset based on FPA temperature."""
        if self._calib_pkg is None:
            return
        segments, vi, fpa_weight = get_curve_segments(
            self._calib_pkg, fpa_temp_raw)
        self._curve_buf = segments[0]  # kept for has_calibration check
        self._curve_segments = segments
        self._fpa_weight = fpa_weight
        self._nuc_gain = self._calib_pkg.get_nuc_gain(vi)

    @property
    def has_calibration(self) -> bool:
        """True if factory calibration + dark frame are available."""
        return (self._calib_pkg is not None and
                self._dark_frame is not None and
                self._curve_buf is not None)

    def parse_frame(self, raw_data: bytes | None) -> np.ndarray | None:
        """Parse raw frame into pixel array and header info.

        Frame layout (25600 bytes = 12800 uint16 LE):
          shorts[0:360]     - header (magic, counter, dimensions, params)
          shorts[360:11160] - 120x90 pixel data (raw sensor values)
          shorts[11160:]    - padding (zeros)
        """
        if raw_data is None or len(raw_data) < FRAME_SIZE:
            return None

        shorts = np.frombuffer(raw_data, dtype='<u2')

        # Frame alignment check: trailing padding should be mostly zeros.
        # If we started reading mid-frame, pixel data spills into the
        # padding region and this check rejects the misaligned frame.
        padding = shorts[11160:]
        if np.count_nonzero(padding) > len(padding) // 4:
            return None

        # Parse header params
        self.frame_counter = int(shorts[HDR_FRAME_COUNTER])
        # Temperatures in raw units / 100 = Celsius
        if shorts[HDR_SHUTTER_TEMP_START] > 0:
            self.shutter_temp_start = shorts[HDR_SHUTTER_TEMP_START] / 100.0
        if shorts[HDR_SHUTTER_TEMP_RT] > 0:
            self.shutter_temp = shorts[HDR_SHUTTER_TEMP_RT] / 100.0
        if shorts[HDR_LENS_TEMP] > 0:
            self.lens_temp = shorts[HDR_LENS_TEMP] / 100.0
        if shorts[HDR_FP_TEMP] > 0:
            self.fpa_temp = shorts[HDR_FP_TEMP] / 100.0
            self.fpa_temp_raw = int(shorts[HDR_FP_TEMP])
            # Smooth FPA temp via 15-sample median (Ghidra: smoothFocusTemp)
            self._fpa_temps.append(self.fpa_temp_raw)
            smoothed_fpa = int(np.median(list(self._fpa_temps)))
            # Update curve selection with smoothed value
            if self._calib_pkg is not None:
                self._select_curve(smoothed_fpa)

        # Extract pixel data
        pixels = shorts[PIXEL_OFFSET:PIXEL_OFFSET + FRAME_PIXELS]
        if len(pixels) < FRAME_PIXELS:
            return None

        return pixels.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    def _nuc_to_celsius(self, nuc_y16: np.ndarray) -> np.ndarray:
        """Convert NUC-corrected Y16 to Celsius (drift + curve + emissivity).

        Expects Y16 that has already been through NUC subtraction, gain
        correction, bad pixel replacement, and TFF. This matches the native
        pipeline where temperature measurement reads from the post-TFF Y16
        snapshot buffer.
        """
        # LensDriftCorrectZX01C: lens temperature drift correction
        l_drift = lens_drift_correct_zx01c(
            self.lens_temp, self._dark_lens_temp, self._dark_fpa_temp)

        # Use current frame's realtime shutter temp for bias calculation,
        # matching native updateMeasureParam: MP[0x1c] = realtimeShutterTemp/100
        shutter_t = self.shutter_temp

        # Multi-curve FPA-weighted interpolation (Ghidra-confirmed)
        if self._curve_segments is not None:
            temps = y16_to_temperature_interpolated(
                nuc_y16, self._curve_segments, self._fpa_weight,
                self._curve_steps, self._core_body_temp,
                shutter_temp=shutter_t,
                lens_drift=l_drift,
                distance=self.distance,
                focus_distance_params=self._calib_pkg.focus_distance_params)
        else:
            temps = y16_to_temperature_array(
                nuc_y16, self._curve_buf, self._curve_steps,
                self._core_body_temp, shutter_temp=shutter_t,
                lens_drift=l_drift)

        # Emissivity correction (Ghidra EmissCor: skip when e >= 0.98)
        if self.emissivity < 0.98 and self._curve_buf is not None:
            temps = emiss_correct(
                temps, self.ambient_temp, self.emissivity,
                self._curve_buf, self._curve_steps,
                self._core_body_temp)
        return temps

    def _apply_nuc(self, y16_raw: np.ndarray) -> np.ndarray:
        """NUC subtraction + gain correction + bad pixel replacement."""
        nuc_y16 = y16_raw - self._dark_frame
        if self._nuc_gain is not None:
            nuc_y16 = nuc_y16 * self._nuc_gain
        return self._replace_bad_pixels(nuc_y16)

    def raw_to_celsius(self, raw_value: np.ndarray) -> np.ndarray:
        """Convert raw sensor value to calibrated Celsius.

        If factory calibration + dark frame are available, uses NUC
        subtraction + manufacturer drift corrections + curve lookup.
        Otherwise falls back to approximate formula + piecewise correction.
        """
        if self.has_calibration:
            return self._nuc_to_celsius(self._apply_nuc(raw_value))

        # Fallback: approximate formula + piecewise correction
        approx = raw_value / 100.0 - 273.15 + 240.0
        if self._cal_native is not None and self._cal_true is not None:
            return np.interp(approx, self._cal_native, self._cal_true)
        return approx

    def process(self, raw_data: bytes | None) -> np.ndarray | None:
        """Process raw frame into displayable image."""
        y16 = self.parse_frame(raw_data)
        if y16 is None:
            return None

        y16_raw = y16.astype(np.float32)

        # --- Unified NUC pipeline (matches native InfraredImageProcess order):
        #     NUCbyTwoPoint → ReplaceBadPoint → TimeNoiseFilter → Y16 snapshot
        #     Same filtered Y16 is used for both temperature AND display.
        if self.has_calibration:
            nuc_y16 = self._apply_nuc(y16_raw)
            nuc_y16 = self._apply_tff(nuc_y16)

            temp_map = self._nuc_to_celsius(nuc_y16)
            y16_display = nuc_y16
        else:
            temp_map = self.raw_to_celsius(y16_raw)
            y16_display = y16_raw.copy()
            y16_display = self._apply_tff(y16_display)

        self._temp_map = temp_map  # retained for mouse lookup
        # Use edge margin to exclude noisy border pixels from statistics
        m = TEMP_MARGIN
        temp_roi = temp_map[m:-m, m:-m] if m > 0 else temp_map
        # Only update max/min once per second to reduce jumpiness
        now = time.time()
        if now - self._last_maxmin_update >= 1.0:
            self._last_maxmin_update = now
            self.min_temp = float(np.min(temp_roi))
            self.max_temp = float(np.max(temp_roi))
            min_idx = np.unravel_index(np.argmin(temp_roi), temp_roi.shape)
            max_idx = np.unravel_index(np.argmax(temp_roi), temp_roi.shape)
            self.min_pos = (int(min_idx[1]) + m, int(min_idx[0]) + m)
            self.max_pos = (int(max_idx[1]) + m, int(max_idx[0]) + m)
        cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        self.center_temp = float(temp_map[cy, cx])

        # Normalize using percentiles for better contrast
        if self.auto_range or self._locked_vmin is None:
            vmin = np.percentile(y16_display, 1)
            vmax = np.percentile(y16_display, 99)
            if not self.auto_range:
                # First frame after lock — capture current range
                self._locked_vmin = vmin
                self._locked_vmax = vmax
        else:
            vmin = self._locked_vmin
            vmax = self._locked_vmax

        if vmax > vmin:
            normalized = np.clip((y16_display - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.full_like(y16_display, 128, dtype=np.uint8)

        # Brightness/contrast (native Ghidra formula)
        if self.brightness != 0 or self.contrast != 128:
            img = normalized.astype(np.int16)
            mean = int(np.mean(normalized))
            img = ((img - mean) * self.contrast >> 7) + mean + self.brightness * 5
            normalized = np.clip(img, 0, 255).astype(np.uint8)

        if self.flip:
            normalized = cv2.flip(normalized, 1)   # horizontal mirror
        if self.rotation == 90:
            normalized = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            normalized = cv2.rotate(normalized, cv2.ROTATE_180)
        elif self.rotation == 270:
            normalized = cv2.rotate(normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self._last_normalized = normalized
        return apply_palette(normalized, self.palette_idx)

    def unlock_range(self) -> None:
        """Unlock color scale to resume auto-scaling."""
        self.auto_range = True
        self._locked_vmin = None
        self._locked_vmax = None

    @property
    def display_width(self) -> int:
        """Width of displayed frame after rotation."""
        return FRAME_HEIGHT if self.rotation in (90, 270) else FRAME_WIDTH

    @property
    def display_height(self) -> int:
        """Height of displayed frame after rotation."""
        return FRAME_WIDTH if self.rotation in (90, 270) else FRAME_HEIGHT

    def update_mouse_temp(self, display_x: int, display_y: int, display_w: int, display_h: int) -> None:
        """Update mouse temperature from display pixel coordinates."""
        if self._temp_map is None:
            self.mouse_temp = None
            self.mouse_pos = None
            return
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        dw, dh = self.display_width, self.display_height
        # Map display coords to rotated-frame coords
        dx = int(display_x * dw / display_w)
        dy = int(display_y * dh / display_h)
        # Inverse rotation (un-rotate then un-flip)
        if self.rotation == 0:
            fx, fy = dx, dy
        elif self.rotation == 90:
            fx, fy = dy, H - 1 - dx
        elif self.rotation == 180:
            fx, fy = W - 1 - dx, H - 1 - dy
        else:  # 270
            fx, fy = W - 1 - dy, dx
        # Inverse flip
        if self.flip:
            fx = W - 1 - fx
        if 0 <= fx < W and 0 <= fy < H:
            self.mouse_temp = float(self._temp_map[fy, fx])
            self.mouse_pos = (display_x, display_y)
        else:
            self.mouse_temp = None
            self.mouse_pos = None

    def get_region_temps(self, fx1: int, fy1: int, fx2: int, fy2: int) -> np.ndarray | None:
        """Return temp_map slice for the given frame-coordinate rectangle."""
        if self._temp_map is None:
            return None
        fx1, fx2 = max(0, fx1), min(FRAME_WIDTH, fx2)
        fy1, fy2 = max(0, fy1), min(FRAME_HEIGHT, fy2)
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        return self._temp_map[fy1:fy2, fx1:fx2]
