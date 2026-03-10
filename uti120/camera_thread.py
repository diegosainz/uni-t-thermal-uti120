"""Background thread for USB camera acquisition and recording."""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path

import cv2
import usb.core
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

from .constants import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    STATUS_IDLE, STATUS_IMAGE_UPLOAD, default_save_dir,
    RECONNECT_FAIL_THRESHOLD,
)
from .camera import UTi120Camera
from .processor import FrameProcessor
from .calibration import (
    load_calibration_cache, save_calibration_cache,
    CalibrationPackage,
)
from .shutter_handler import ShutterHandler

__all__ = ["CameraThread"]

logger = logging.getLogger(__name__)


class CameraThread(QThread):
    """USB camera loop running on a background thread."""

    frame_ready = pyqtSignal(object, object)  # (colored_bgr ndarray, processor ref)
    status_message = pyqtSignal(str)
    camera_ready = pyqtSignal()
    init_failed = pyqtSignal(str)
    def __init__(self, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self.camera = UTi120Camera()
        self.processor = FrameProcessor()
        self.shutter_handler = ShutterHandler()
        self.running = False
        self._do_shutter = False
        self._do_nuc = False
        self.save_dir = default_save_dir()

    def run(self) -> None:
        self.running = True

        # --- Init camera ---
        if not self.camera.find_and_connect():
            self.init_failed.emit("Failed to connect. Is the camera plugged in?")
            return

        info = self.camera.get_device_info()
        info_str = ", ".join(f"{k}: {v}" for k, v in info.items())
        logger.info("Device info: %s", info_str)
        self.status_message.emit(f"Connected: {info_str}")

        # Calibration points (fallback)
        cal_points = self.camera.read_calibration_points()
        if not cal_points:
            self.init_failed.emit("Failed to read calibration points from device.")
            return
        self.processor.set_calibration(cal_points)

        # Factory calibration packages (serial-aware cache)
        serial = info.get('serial', '')
        logger.info("Device serial: %s", serial)
        calib_pkgs = load_calibration_cache(serial)
        if calib_pkgs:
            logger.info("Calibration cache valid for serial %s", serial)
            self.status_message.emit("Loaded calibration from cache")
        else:
            logger.info("Calibration cache miss — downloading from device "
                        "(serial=%s)", serial)
            calib_pkgs = {}
            raw_data: dict[int, bytes] = {}
            for range_id, label in [(0, "low-temp"), (1, "high-temp")]:
                pkg_data = self.camera.download_calibration_package(range_id)
                if pkg_data:
                    try:
                        calib_pkgs[range_id] = CalibrationPackage(data=pkg_data)
                        raw_data[range_id] = pkg_data
                        self.status_message.emit(f"Downloaded {label} calibration")
                    except (struct.error, ValueError, AssertionError) as e:
                        self.status_message.emit(
                            f"WARNING: {label} parse failed: {e}")
            if raw_data:
                save_calibration_cache(
                    serial, raw_data.get(0), raw_data.get(1))

        low_pkg = calib_pkgs.get(0)
        high_pkg = calib_pkgs.get(1)
        if low_pkg:
            self.processor.set_calibration_packages(low_pkg, high_pkg)

        # Set image upload mode
        self.camera.set_run_status(STATUS_IDLE)
        time.sleep(0.2)
        # Drain stale bulk data to prevent frame misalignment on startup
        drained = self.camera._drain_bulk()
        if drained:
            logger.debug("Drained %d stale bytes from bulk endpoint", drained)
        self.camera.set_run_status(STATUS_IMAGE_UPLOAD)
        time.sleep(0.3)

        # Initial NUC
        self.status_message.emit("Initial NUC...")
        self.camera.trigger_shutter()
        time.sleep(0.5)
        # Drain again after shutter (trigger_shutter re-enables streaming)
        self.camera._drain_bulk()
        for _ in range(10):
            self.camera.request_frame()
            time.sleep(0.02)

        # Initial dark frame
        self._do_shutter_calibration()

        # Warmup
        for _ in range(5):
            self.camera.request_frame()

        self.camera_ready.emit()
        self.status_message.emit("Streaming")

        # --- Main frame loop ---
        fail_count = 0

        while self.running:
            # Handle pending commands
            if self._do_shutter:
                self._do_shutter = False
                self.status_message.emit("Shutter calibration...")
                self._do_shutter_calibration()
                self.status_message.emit("Streaming")
                continue

            if self._do_nuc:
                self._do_nuc = False
                self.status_message.emit("NUC calibration...")
                self.camera.trigger_shutter()
                time.sleep(0.5)
                self._do_shutter_calibration()
                self.shutter_handler.did_nuc(self.processor.fpa_temp)
                self.status_message.emit("Streaming")
                continue

            # Read frame
            try:
                raw = self.camera.request_frame()
            except usb.core.USBError:
                if not self.camera.reconnect():
                    self.status_message.emit("Connection lost")
                    break
                fail_count = 0
                continue

            if raw is None:
                fail_count += 1
                if fail_count > RECONNECT_FAIL_THRESHOLD:
                    self.status_message.emit("Reconnecting...")
                    if not self.camera.reconnect():
                        self.status_message.emit("Connection lost")
                        break
                    fail_count = 0
                continue

            fail_count = 0

            # Process
            colored = self.processor.process(raw)
            if colored is None:
                continue

            # Auto-recalibration
            action = self.shutter_handler.check(self.processor.fpa_temp, self.processor.frame_counter)
            if action == 'nuc':
                self.status_message.emit(f"Auto-NUC (FPA={self.processor.fpa_temp:.2f}°C)")
                self.camera.trigger_shutter()
                time.sleep(0.5)
                self._do_shutter_calibration()
                self.shutter_handler.did_nuc(self.processor.fpa_temp)
                continue
            elif action == 'shutter':
                self.status_message.emit(f"Auto-shutter (FPA={self.processor.fpa_temp:.2f}°C)")
                self._do_shutter_calibration()
                self.shutter_handler.did_shutter(self.processor.fpa_temp)
                continue

            # Range switching
            new_range = self.processor.check_range_switch()
            if new_range is not None:
                label = "HIGH" if new_range == 1 else "LOW"
                self.status_message.emit(f"Switching to {label} range...")
                pkg = self.processor.switch_range(new_range)
                self.camera.set_measure_range(
                    pkg.sensor_gain, pkg.sensor_int, pkg.sensor_res)
                time.sleep(0.3)
                self._do_shutter_calibration()
                self.status_message.emit("Streaming")
                continue

            # Resize for display
            display = cv2.resize(colored, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                 interpolation=cv2.INTER_NEAREST)

            self.frame_ready.emit(display, self.processor)

        self.camera.close()

    def _do_shutter_calibration(self) -> None:
        result = self.camera.trigger_shutter_with_dark_capture()
        dark, shutter_temp, lens_temp, fpa_temp = result
        if dark is not None:
            self.processor.set_dark_frame(dark, shutter_temp, lens_temp, fpa_temp)
        else:
            self.camera.trigger_shutter()
        time.sleep(0.3)
        for _ in range(3):
            self.camera.request_frame()

    @pyqtSlot()
    def request_shutter(self) -> None:
        self._do_shutter = True

    @pyqtSlot()
    def request_nuc(self) -> None:
        self._do_nuc = True

    def stop(self) -> None:
        self.running = False
        self.wait(5000)
