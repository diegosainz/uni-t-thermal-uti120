"""APK-matching auto-recalibration based on FPA temperature drift.

Reconstructed from ShutterHandler.java and UsbCameraHelper.java in the
decompiled APK. Monitors FPA sensor temperature every frame and triggers
shutter/NUC operations when drift exceeds time-scaled thresholds.
"""

from __future__ import annotations

import logging
import time

__all__ = ["ShutterHandler"]

logger = logging.getLogger(__name__)


class ShutterHandler:
    """Auto-recalibration state machine matching APK ShutterHandler.java.

    Three mechanisms:
    1. FPA delta → NUC+Shutter (rare): full hardware NUC when large drift detected
    2. FPA delta → Shutter-only (frequent): dark frame refresh on moderate drift
    3. Periodic timer (after 6 min): 60s shutter-only refresh
    """

    # Time-scaled threshold table: (seconds_cutoff, shutter_delta_C, nuc_delta_C)
    # Thresholds tighten as camera warms up (from UsbCameraHelper.java)
    THRESHOLDS = [
        (180,  0.80, 1.20),   # 0-3 min: warmup, wide thresholds
        (360,  0.50, 1.00),   # 3-6 min: tightening
        (None, 0.30, 0.60),   # 6+ min: stable, tight thresholds
    ]

    NUC_COOLDOWN = 30.0         # seconds between NUC operations
    PERIODIC_INTERVAL = 60.0    # seconds for periodic shutter (after 6 min)
    PERIODIC_ENABLE_TIME = 360.0  # enable periodic timer after 6 min
    WARMUP_FRAME_THRESHOLD = 3600  # ~6 min at 10 FPS: camera already warm

    def __init__(self) -> None:
        self._start_time = time.time()
        self._base_fpa_nuc = None       # FPA baseline for NUC triggers
        self._base_fpa_shutter = None   # FPA baseline for shutter triggers
        self._last_nuc_time = 0.0
        self._last_shutter_time = 0.0

    def _get_thresholds(self) -> tuple[float, float]:
        """Get current (shutter_delta, nuc_delta) based on elapsed time."""
        elapsed = time.time() - self._start_time
        for cutoff, shutter_delta, nuc_delta in self.THRESHOLDS:
            if cutoff is None or elapsed < cutoff:
                return shutter_delta, nuc_delta
        # Should not reach here, but fallback to tightest
        return self.THRESHOLDS[-1][1], self.THRESHOLDS[-1][2]

    def check(self, fpa_temp_celsius: float, frame_counter: int = 0) -> str | None:
        """Check if recalibration is needed based on FPA temperature drift.

        Args:
            fpa_temp_celsius: Current FPA temperature in °C (from processor.fpa_temp)
            frame_counter: Frame counter from camera header (uint16, wraps at 65535)

        Returns:
            'nuc' for full hardware NUC + dark frame refresh,
            'shutter' for dark frame refresh only,
            or None if no action needed.
        """
        now = time.time()

        # First call: set baselines, no action
        if self._base_fpa_nuc is None:
            self._base_fpa_nuc = fpa_temp_celsius
            self._base_fpa_shutter = fpa_temp_celsius
            self._last_shutter_time = now
            self._last_nuc_time = now
            # If the camera has been running long enough, skip warmup
            if frame_counter >= self.WARMUP_FRAME_THRESHOLD:
                self._start_time = now - self.PERIODIC_ENABLE_TIME
                logger.info("Camera already warm (frame_counter=%d), skipping warmup — "
                            "using tight thresholds and periodic timer immediately",
                            frame_counter)
            else:
                logger.info("Camera warmup phase (frame_counter=%d, threshold=%d)",
                            frame_counter, self.WARMUP_FRAME_THRESHOLD)
            return None

        shutter_delta, nuc_delta = self._get_thresholds()

        # Check NUC trigger (large drift, with cooldown)
        nuc_drift = abs(fpa_temp_celsius - self._base_fpa_nuc)
        if nuc_drift >= nuc_delta and (now - self._last_nuc_time) >= self.NUC_COOLDOWN:
            return 'nuc'

        # Check shutter trigger (moderate drift, no cooldown)
        shutter_drift = abs(fpa_temp_celsius - self._base_fpa_shutter)
        if shutter_drift >= shutter_delta:
            return 'shutter'

        # Periodic timer (after 6 min, every 60s)
        elapsed = now - self._start_time
        if elapsed >= self.PERIODIC_ENABLE_TIME:
            if (now - self._last_shutter_time) >= self.PERIODIC_INTERVAL:
                return 'shutter'

        return None

    def did_nuc(self, fpa_temp_celsius: float) -> None:
        """Call after executing a NUC action to reset baselines."""
        now = time.time()
        self._base_fpa_nuc = fpa_temp_celsius
        self._base_fpa_shutter = fpa_temp_celsius
        self._last_nuc_time = now
        self._last_shutter_time = now

    def did_shutter(self, fpa_temp_celsius: float) -> None:
        """Call after executing a shutter-only action to reset shutter baseline."""
        now = time.time()
        self._base_fpa_shutter = fpa_temp_celsius
        self._last_shutter_time = now

    def time_until_next(self, fpa_temp_celsius: float) -> dict:
        """Estimate time/proximity to next calibration trigger.

        Returns dict with:
            periodic_remaining: seconds until periodic trigger (None if warmup)
            shutter_drift_pct: current drift as % of shutter threshold
            nuc_drift_pct: current drift as % of NUC threshold
        """
        now = time.time()
        elapsed = now - self._start_time
        shutter_delta, nuc_delta = self._get_thresholds()

        # Drift percentages
        if self._base_fpa_shutter is not None:
            shutter_drift = abs(fpa_temp_celsius - self._base_fpa_shutter)
            shutter_pct = (shutter_drift / shutter_delta) * 100
        else:
            shutter_pct = 0.0

        if self._base_fpa_nuc is not None:
            nuc_drift = abs(fpa_temp_celsius - self._base_fpa_nuc)
            nuc_pct = (nuc_drift / nuc_delta) * 100
        else:
            nuc_pct = 0.0

        # Periodic countdown
        if elapsed >= self.PERIODIC_ENABLE_TIME:
            since_last = now - self._last_shutter_time
            periodic_remaining = max(0.0, self.PERIODIC_INTERVAL - since_last)
        else:
            periodic_remaining = None

        return {
            'periodic_remaining': periodic_remaining,
            'shutter_drift_pct': shutter_pct,
            'nuc_drift_pct': nuc_pct,
        }
