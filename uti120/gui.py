"""PyQt6 thermal viewer GUI — MainWindow and application entry point."""

from __future__ import annotations

import sys
import time
import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import (
    QFont, QShortcut, QKeySequence, QPalette, QColor, QCloseEvent,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QGroupBox, QFrame,
    QMessageBox, QCheckBox, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialogButtonBox, QFileDialog,
    QDoubleSpinBox, QSplitter,
)

from pathlib import Path

from .constants import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT, EMISSIVITY_PRESETS, default_save_dir,
    STATS_UPDATE_INTERVAL_MS, DEFAULT_PALETTE_IDX,
    ALARM_TEMP_MIN, ALARM_TEMP_MAX, ALARM_SOURCES, ALARM_HYSTERESIS_DEFAULT,
)
from .palettes import PALETTES
from .camera_thread import CameraThread
from .widgets import ThermalWidget, MosaicWidget, CollapsibleSection
from .graph import TemperatureGraphPanel

if TYPE_CHECKING:
    from .processor import FrameProcessor

try:
    from .surface3d import ThermalSurface3D
    HAS_3D = True
except ImportError:
    HAS_3D = False


try:
    from PyQt6.QtMultimedia import QSoundEffect
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False


def _generate_alarm_wav() -> str:
    """Generate a two-tone alert WAV (beep-beep-pause pattern) and return its path."""
    import atexit
    import os
    import tempfile
    import wave
    sample_rate = 22050
    # Two short beeps with a pause: beep(0.15s) + gap(0.08s) + beep(0.15s) + pause(0.4s)
    beep_dur = 0.15
    gap_dur = 0.08
    pause_dur = 0.40
    t_beep = np.linspace(0, beep_dur, int(sample_rate * beep_dur), endpoint=False)
    # Use 880 Hz (A5) with a gentle fade-in/out envelope
    envelope = np.minimum(t_beep / 0.01, 1.0) * np.minimum((beep_dur - t_beep) / 0.01, 1.0)
    beep = (np.sin(2 * np.pi * 880 * t_beep) * envelope * 12000).astype(np.int16)
    gap = np.zeros(int(sample_rate * gap_dur), dtype=np.int16)
    pause = np.zeros(int(sample_rate * pause_dur), dtype=np.int16)
    samples = np.concatenate([beep, gap, beep, pause])
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = f.name
    f.close()
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    atexit.register(os.unlink, path)
    return path


__all__ = ["MainWindow", "run_gui"]


class MainWindow(QMainWindow):
    """Main application window with thermal display and control sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UTi120 Thermal Viewer")
        self._save_dir = default_save_dir()

        # --- Central layout: thermal widget + graph panel + sidebar ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)

        # Left column: thermal display + graph panel (splitter for resizing)
        left_col = QSplitter(Qt.Orientation.Vertical)
        left_col.setChildrenCollapsible(False)

        self.thermal = ThermalWidget()
        left_col.addWidget(self.thermal)

        self.mosaic = MosaicWidget(self.thermal)
        self.mosaic.setVisible(False)
        left_col.addWidget(self.mosaic)

        self._mosaic_active = False

        self._surface3d_active = False
        self.surface3d = None
        if HAS_3D:
            self.surface3d = ThermalSurface3D()
            self.surface3d.setVisible(False)
            left_col.addWidget(self.surface3d)

        self.graph_panel = TemperatureGraphPanel(self.thermal)
        self.graph_panel.setMinimumHeight(60)
        left_col.addWidget(self.graph_panel)
        left_col.setStretchFactor(0, 1)  # thermal gets stretch
        self._left_splitter = left_col

        layout.addWidget(left_col, stretch=1)

        # Sidebar
        sidebar = self._build_sidebar()
        layout.addWidget(sidebar, stretch=0)

        # Status bar
        self.statusBar().showMessage("Initializing...")

        # Camera thread
        self.cam_thread = CameraThread()
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.status_message.connect(self._on_status)
        self.cam_thread.camera_ready.connect(self._on_camera_ready)
        self.cam_thread.init_failed.connect(self._on_init_failed)
        self.cam_thread.save_dir = self._save_dir
        self.graph_panel.save_dir = self._save_dir

        # Stats update timer (don't update labels every frame)
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(STATS_UPDATE_INTERVAL_MS)

        # Recording state
        self._video_writer: cv2.VideoWriter | None = None
        self._recording_active: bool = False
        self._recording_start: float | None = None
        self._recording_file: str | None = None

        # Window-wide keyboard shortcuts (work even when GL widget has focus)
        self._setup_shortcuts()

        # Alarm state
        self._alarm_enabled: bool = False
        self._alarm_source: str = "center"
        self._alarm_high: float = float(ALARM_TEMP_MAX)
        self._alarm_low: float = float(ALARM_TEMP_MIN)
        self._alarm_hysteresis: float = ALARM_HYSTERESIS_DEFAULT
        self._alarm_active: bool = False
        self._alarm_sound: QSoundEffect | None = None
        if _HAS_SOUND:
            from PyQt6.QtCore import QUrl
            self._alarm_sound = QSoundEffect()
            self._alarm_sound.setSource(QUrl.fromLocalFile(_generate_alarm_wav()))
            self._alarm_sound.setLoopCount(QSoundEffect.Loop.Infinite.value)

        self.resize(DISPLAY_WIDTH + 180, DISPLAY_HEIGHT)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(170)
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(4, 8, 4, 8)
        vbox.setSpacing(6)

        # --- Palette (always visible) ---
        vbox.addWidget(QLabel("Palette"))
        self.palette_combo = QComboBox()
        self.palette_combo.setToolTip("Color map used to render temperatures (shortcut: P)")
        for name, _ in PALETTES:
            self.palette_combo.addItem(name)
        self.palette_combo.setCurrentIndex(DEFAULT_PALETTE_IDX)
        self.palette_combo.currentIndexChanged.connect(self._on_palette_changed)
        vbox.addWidget(self.palette_combo)

        # --- Emissivity (always visible) ---
        grp_emiss = QGroupBox("Emissivity")
        gl = QVBoxLayout(grp_emiss)
        self.emiss_combo = QComboBox()
        self.emiss_combo.setToolTip("Select a material preset or use Custom for manual control")
        for name, val in EMISSIVITY_PRESETS:
            label = name if val is None else f"{name}  ({val:.2f})"
            self.emiss_combo.addItem(label)
        self.emiss_combo.currentIndexChanged.connect(self._on_emiss_preset_changed)
        gl.addWidget(self.emiss_combo)
        self.emiss_label = QLabel("0.95")
        self.emiss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gl.addWidget(self.emiss_label)
        self.emiss_slider = QSlider(Qt.Orientation.Horizontal)
        self.emiss_slider.setToolTip("Surface emissivity — lower for shiny/metallic surfaces (shortcut: E/Shift+E)")
        self.emiss_slider.setRange(10, 100)
        self.emiss_slider.setValue(95)
        self.emiss_slider.valueChanged.connect(self._on_emissivity_changed)
        gl.addWidget(self.emiss_slider)
        vbox.addWidget(grp_emiss)

        # --- Lock Range (always visible) ---
        btn_lock = QPushButton("[L] Lock Range")
        btn_lock.setToolTip("Lock the color scale to the current temperature range (shortcut: L)")
        btn_lock.setCheckable(True)
        btn_lock.clicked.connect(self._on_lock_range)
        self.btn_lock = btn_lock
        vbox.addWidget(btn_lock)

        # === Section: Image (collapsed) ===
        sec_image = CollapsibleSection("Image", expanded=False)
        lay = sec_image.content_layout()

        grp_bright = QGroupBox("Brightness")
        bl = QVBoxLayout(grp_bright)
        self.bright_label = QLabel("0")
        self.bright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self.bright_label)
        self.bright_slider = QSlider(Qt.Orientation.Horizontal)
        self.bright_slider.setToolTip("Adjust image brightness offset (shortcut: +/-)")
        self.bright_slider.setRange(-20, 20)
        self.bright_slider.setValue(0)
        self.bright_slider.valueChanged.connect(self._on_brightness_changed)
        bl.addWidget(self.bright_slider)
        lay.addWidget(grp_bright)

        grp_contrast = QGroupBox("Contrast")
        ctl = QVBoxLayout(grp_contrast)
        self.contrast_label = QLabel("128")
        self.contrast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctl.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setToolTip("Image contrast multiplier (shortcut: A/Shift+A)")
        self.contrast_slider.setRange(64, 255)
        self.contrast_slider.setValue(128)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        ctl.addWidget(self.contrast_slider)
        lay.addWidget(grp_contrast)

        btn_flip = QPushButton("[F] Flip H")
        btn_flip.setToolTip("Mirror the image horizontally (shortcut: F)")
        btn_flip.setCheckable(True)
        btn_flip.clicked.connect(self._on_flip)
        self.btn_flip = btn_flip
        lay.addWidget(btn_flip)

        btn_rotate = QPushButton("[R] Rotate")
        btn_rotate.setToolTip("Rotate the image 90° clockwise (shortcut: R)")
        btn_rotate.clicked.connect(self._on_rotate)
        self.btn_rotate = btn_rotate
        lay.addWidget(btn_rotate)

        vbox.addWidget(sec_image)

        # === Section: Isotherm (collapsed) ===
        sec_iso = CollapsibleSection("Isotherm", expanded=False)
        lay = sec_iso.content_layout()
        self.iso_check = QCheckBox("Enabled")
        self.iso_check.toggled.connect(self._on_isotherm_toggled)
        lay.addWidget(self.iso_check)
        self.iso_mode = QComboBox()
        self.iso_mode.addItems(["Above", "Below"])
        self.iso_mode.currentTextChanged.connect(self._on_isotherm_mode)
        lay.addWidget(self.iso_mode)
        self.iso_label = QLabel("50°C")
        self.iso_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.iso_label)
        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(-20, 400)
        self.iso_slider.setValue(50)
        self.iso_slider.valueChanged.connect(self._on_isotherm_threshold)
        lay.addWidget(self.iso_slider)
        vbox.addWidget(sec_iso)

        # === Section: Alarm (collapsed) ===
        self._sec_alarm = CollapsibleSection("Alarm", expanded=False)
        lay = self._sec_alarm.content_layout()

        self.alarm_check = QCheckBox("Enabled")
        self.alarm_check.setToolTip("Enable audio alarm when temperature crosses thresholds")
        self.alarm_check.toggled.connect(self._on_alarm_toggled)
        lay.addWidget(self.alarm_check)

        self.alarm_source_combo = QComboBox()
        self.alarm_source_combo.setToolTip("Temperature source to monitor")
        for label, _ in ALARM_SOURCES:
            self.alarm_source_combo.addItem(label)
        self.alarm_source_combo.currentIndexChanged.connect(self._on_alarm_source_changed)
        lay.addWidget(self.alarm_source_combo)

        lay.addWidget(QLabel("High ≤"))
        self.alarm_high_spin = QDoubleSpinBox()
        self.alarm_high_spin.setRange(ALARM_TEMP_MIN, ALARM_TEMP_MAX)
        self.alarm_high_spin.setValue(ALARM_TEMP_MAX)
        self.alarm_high_spin.setSingleStep(0.5)
        self.alarm_high_spin.setSuffix(" °C")
        self.alarm_high_spin.setToolTip("Alarm triggers when source temp exceeds this value")
        self.alarm_high_spin.valueChanged.connect(self._on_alarm_high_changed)
        lay.addWidget(self.alarm_high_spin)

        lay.addWidget(QLabel("Low ≥"))
        self.alarm_low_spin = QDoubleSpinBox()
        self.alarm_low_spin.setRange(ALARM_TEMP_MIN, ALARM_TEMP_MAX)
        self.alarm_low_spin.setValue(ALARM_TEMP_MIN)
        self.alarm_low_spin.setSingleStep(0.5)
        self.alarm_low_spin.setSuffix(" °C")
        self.alarm_low_spin.setToolTip("Alarm triggers when source temp drops below this value")
        self.alarm_low_spin.valueChanged.connect(self._on_alarm_low_changed)
        lay.addWidget(self.alarm_low_spin)

        lay.addWidget(QLabel("Hysteresis"))
        self.alarm_hyst_spin = QDoubleSpinBox()
        self.alarm_hyst_spin.setRange(0.0, 20.0)
        self.alarm_hyst_spin.setValue(ALARM_HYSTERESIS_DEFAULT)
        self.alarm_hyst_spin.setSingleStep(0.5)
        self.alarm_hyst_spin.setSuffix(" °C")
        self.alarm_hyst_spin.setToolTip("Deadband to prevent alarm toggling near threshold")
        self.alarm_hyst_spin.valueChanged.connect(self._on_alarm_hysteresis_changed)
        lay.addWidget(self.alarm_hyst_spin)

        btn_test_alarm = QPushButton("Test Sound")
        btn_test_alarm.setToolTip("Play the alarm sound for 2 seconds")
        btn_test_alarm.clicked.connect(self._on_test_alarm)
        lay.addWidget(btn_test_alarm)

        vbox.addWidget(self._sec_alarm)

        # === Section: Camera (collapsed) ===
        sec_cam = CollapsibleSection("Camera", expanded=False)
        lay = sec_cam.content_layout()

        btn_shutter = QPushButton("[C] Shutter Cal")
        btn_shutter.setToolTip("Close the shutter and recalibrate the sensor (shortcut: C)")
        btn_shutter.clicked.connect(self._on_shutter)
        lay.addWidget(btn_shutter)

        btn_nuc = QPushButton("[N] NUC")
        btn_nuc.setToolTip("Non-Uniformity Correction — fix pixel-to-pixel drift (shortcut: N)")
        btn_nuc.clicked.connect(self._on_nuc)
        lay.addWidget(btn_nuc)

        vbox.addWidget(sec_cam)

        # === Section: Capture (expanded) ===
        sec_cap = CollapsibleSection("Capture", expanded=True)
        lay = sec_cap.content_layout()

        self.btn_record = QPushButton("[Ctrl+R] Record")
        self.btn_record.setToolTip("Record video with overlays at 720×540 (shortcut: Ctrl+R)")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._on_record_toggled)
        lay.addWidget(self.btn_record)

        btn_screenshot = QPushButton("[S] Screenshot")
        btn_screenshot.setToolTip("Save the current view as a PNG file (shortcut: S)")
        btn_screenshot.clicked.connect(self._save_screenshot)
        lay.addWidget(btn_screenshot)

        self._save_dir_btn = QPushButton(self._save_dir_label())
        self._save_dir_btn.setToolTip(f"Save directory: {self._save_dir}")
        self._save_dir_btn.clicked.connect(self._choose_save_dir)
        lay.addWidget(self._save_dir_btn)

        vbox.addWidget(sec_cap)

        # --- Utility buttons (always visible) ---
        self.btn_mosaic = QPushButton("[M] Mosaic")
        self.btn_mosaic.setToolTip("Toggle 6-palette mosaic view (shortcut: M)")
        self.btn_mosaic.setCheckable(True)
        self.btn_mosaic.clicked.connect(self._toggle_mosaic)
        vbox.addWidget(self.btn_mosaic)

        if HAS_3D:
            self.btn_3d = QPushButton("[3] 3D Surface")
            self.btn_3d.setToolTip("Toggle 3D surface plot (shortcut: 3)")
            self.btn_3d.setCheckable(True)
            self.btn_3d.clicked.connect(self._toggle_3d)
            vbox.addWidget(self.btn_3d)

        btn_help = QPushButton("[H] Shortcuts")
        btn_help.setToolTip("Show keyboard shortcuts (shortcut: H)")
        btn_help.clicked.connect(self._show_help)
        vbox.addWidget(btn_help)

        btn_cal = QPushButton("Calibration Points")
        btn_cal.setToolTip("View/edit blackbody calibration points stored on device")
        btn_cal.clicked.connect(self._on_calibration_points)
        vbox.addWidget(btn_cal)

        # --- Stats (always visible) ---
        vbox.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        vbox.addWidget(sep)

        lbl_header = QLabel("Device")
        lbl_header.setFont(QFont("monospace", 9, QFont.Weight.Bold))
        vbox.addWidget(lbl_header)

        self.lbl_fpa = QLabel("FPA: --")
        self.lbl_shutter = QLabel("Shutter: --")
        self.lbl_lens = QLabel("Lens: --")
        self.lbl_range = QLabel("Range: LOW")
        self.lbl_cal = QLabel("Cal: --")

        for lbl in [self.lbl_fpa, self.lbl_shutter, self.lbl_lens,
                     self.lbl_range, self.lbl_cal]:
            lbl.setFont(QFont("monospace", 9))
            vbox.addWidget(lbl)

        vbox.addStretch()
        return sidebar

    def _setup_shortcuts(self) -> None:
        """Register all keyboard shortcuts as QShortcut (window-wide scope)."""
        def sc(key, slot):
            QShortcut(QKeySequence(key), self, slot)

        proc = self.cam_thread.processor

        sc("Q", self.close)
        sc("S", self._save_screenshot)
        sc("Ctrl+R", self._toggle_recording)
        sc("H", self._show_help)
        sc("Escape", lambda: (self.thermal.clear_selection(), self.thermal.clear_probes()))
        sc("P", self._shortcut_cycle_palette)
        sc("F", self._shortcut_flip)
        sc("R", self._shortcut_rotate)
        sc("C", self.cam_thread.request_shutter)
        sc("N", self.cam_thread.request_nuc)
        sc("L", self._shortcut_lock_range)
        sc("G", self.graph_panel.toggle_collapse)
        sc("M", self._toggle_mosaic)
        sc("3", self._toggle_3d)
        sc("D", self._shortcut_delete_probe)
        sc("+", self._shortcut_brightness_up)
        sc("=", self._shortcut_brightness_up)
        sc("-", self._shortcut_brightness_down)
        sc("A", self._shortcut_contrast_up)
        sc("Shift+A", self._shortcut_contrast_down)
        sc("E", self._shortcut_emissivity_up)
        sc("Shift+E", self._shortcut_emissivity_down)

    def _shortcut_cycle_palette(self) -> None:
        proc = self.cam_thread.processor
        idx = (proc.palette_idx + 1) % len(PALETTES)
        proc.palette_idx = idx
        self.palette_combo.setCurrentIndex(idx)

    def _shortcut_flip(self) -> None:
        proc = self.cam_thread.processor
        proc.flip = not proc.flip
        self.btn_flip.setChecked(proc.flip)
        self.thermal.clear_selection()

    def _shortcut_rotate(self) -> None:
        proc = self.cam_thread.processor
        proc.rotation = (proc.rotation + 90) % 360
        self.thermal.clear_selection()

    def _shortcut_lock_range(self) -> None:
        checked = not self.btn_lock.isChecked()
        self.btn_lock.setChecked(checked)
        self._on_lock_range(checked)

    def _shortcut_delete_probe(self) -> None:
        if self.thermal._probes and self.thermal._mouse_pos:
            mx, my = self.thermal._mouse_pos
            self.thermal.remove_nearest_probe(mx, my)
        elif self.thermal._probes:
            self.thermal._probes.pop()

    def _shortcut_brightness_up(self) -> None:
        proc = self.cam_thread.processor
        val = min(proc.brightness + 1, 20)
        proc.brightness = val
        self.bright_slider.setValue(val)

    def _shortcut_brightness_down(self) -> None:
        proc = self.cam_thread.processor
        val = max(proc.brightness - 1, -20)
        proc.brightness = val
        self.bright_slider.setValue(val)

    def _shortcut_contrast_up(self) -> None:
        proc = self.cam_thread.processor
        val = min(proc.contrast + 8, 255)
        proc.contrast = val
        self.contrast_slider.setValue(val)

    def _shortcut_contrast_down(self) -> None:
        proc = self.cam_thread.processor
        val = max(proc.contrast - 8, 64)
        proc.contrast = val
        self.contrast_slider.setValue(val)

    def _shortcut_emissivity_up(self) -> None:
        proc = self.cam_thread.processor
        e = min(proc.emissivity + 0.05, 1.0)
        proc.emissivity = e
        self.emiss_slider.setValue(int(e * 100))

    def _shortcut_emissivity_down(self) -> None:
        proc = self.cam_thread.processor
        e = max(proc.emissivity - 0.05, 0.10)
        proc.emissivity = e
        self.emiss_slider.setValue(int(e * 100))

    def start(self) -> None:
        """Start the camera thread."""
        self.cam_thread.start()

    @pyqtSlot(object, object)
    def _on_frame(self, display_bgr: np.ndarray, processor: FrameProcessor) -> None:
        if self._mosaic_active:
            self.mosaic.update_frame(display_bgr, processor)
        elif self._surface3d_active and HAS_3D:
            self.surface3d.update_frame(display_bgr, processor)
        else:
            self.thermal.update_frame(display_bgr, processor)
        self.graph_panel.set_processor(processor)

        # Write composited frame to video (overlays baked in)
        if self._recording_active and self._video_writer is not None:
            if self._mosaic_active:
                composited = self.mosaic.render_composited_frame()
            elif self._surface3d_active and HAS_3D:
                composited = self.surface3d.render_composited_frame()
            else:
                composited = self.thermal.render_composited_frame(display_bgr, processor)
            self._video_writer.write(composited)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    @pyqtSlot()
    def _on_camera_ready(self) -> None:
        self.statusBar().showMessage("Streaming")

    @pyqtSlot(str)
    def _on_init_failed(self, msg: str) -> None:
        self.statusBar().showMessage(f"ERROR: {msg}")

    def _update_stats(self) -> None:
        proc = self.cam_thread.processor
        self.lbl_fpa.setText(f"FPA:     {proc.fpa_temp:.1f}°C")
        self.lbl_shutter.setText(f"Shutter: {proc.shutter_temp:.1f}°C")
        self.lbl_lens.setText(f"Lens:    {proc.lens_temp:.1f}°C")
        rng = "HIGH" if proc.active_range == 1 else "LOW"
        self.lbl_range.setText(f"Range:  {rng}")
        # Calibration countdown
        info = self.cam_thread.shutter_handler.time_until_next(proc.fpa_temp)
        drift_pct = info['shutter_drift_pct']
        periodic = info['periodic_remaining']
        if drift_pct >= 70:
            self.lbl_cal.setText(f"Cal:    drift {drift_pct:.0f}%")
        elif periodic is not None:
            self.lbl_cal.setText(f"Cal:    {periodic:.0f}s")
        else:
            self.lbl_cal.setText(f"Cal:    warmup")
        # Alarm check (with hysteresis deadband)
        if self._alarm_enabled:
            temp = self._get_alarm_temp()
            if temp is not None:
                src_label = {k: lbl for lbl, k in ALARM_SOURCES}.get(self._alarm_source, self._alarm_source)
                hyst = self._alarm_hysteresis
                if self._alarm_active:
                    # Already alarming — clear only when temp returns within deadband
                    over_high = temp > (self._alarm_high - hyst)
                    under_low = temp < (self._alarm_low + hyst)
                    if over_high or under_low:
                        if temp > self._alarm_high or over_high:
                            msg = f"{src_label}: {temp:.1f}°C > {self._alarm_high:.1f}°C"
                        else:
                            msg = f"{src_label}: {temp:.1f}°C < {self._alarm_low:.1f}°C"
                        self.thermal._alarm_message = msg
                    else:
                        self.thermal._alarm_message = ""
                        self._set_alarm_active(False)
                else:
                    # Not alarming — trigger at exact thresholds
                    if temp > self._alarm_high or temp < self._alarm_low:
                        if temp > self._alarm_high:
                            msg = f"{src_label}: {temp:.1f}°C > {self._alarm_high:.1f}°C"
                        else:
                            msg = f"{src_label}: {temp:.1f}°C < {self._alarm_low:.1f}°C"
                        self.thermal._alarm_message = msg
                        self._set_alarm_active(True)
            elif self._alarm_active:
                self.thermal._alarm_message = ""
                self._set_alarm_active(False)

        # Update recording elapsed time in status bar
        if self._recording_start is not None:
            elapsed = int(time.time() - self._recording_start)
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            ts = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            self.statusBar().showMessage(f"REC {ts} — {self._recording_file}")

    def _on_palette_changed(self, idx: int) -> None:
        self.cam_thread.processor.palette_idx = idx

    def _on_emiss_preset_changed(self, idx: int) -> None:
        if idx <= 0:
            return
        _, val = EMISSIVITY_PRESETS[idx]
        if val is not None:
            self.emiss_slider.setValue(round(val * 100))

    def _on_emissivity_changed(self, val: int) -> None:
        e = val / 100.0
        self.cam_thread.processor.emissivity = e
        self.emiss_label.setText(f"{e:.2f}")
        # Switch to "Custom" if slider doesn't match current preset
        idx = self.emiss_combo.currentIndex()
        if idx > 0:
            _, preset_val = EMISSIVITY_PRESETS[idx]
            if preset_val is not None and round(preset_val * 100) != val:
                self.emiss_combo.blockSignals(True)
                self.emiss_combo.setCurrentIndex(0)
                self.emiss_combo.blockSignals(False)

    def _on_brightness_changed(self, val: int) -> None:
        self.cam_thread.processor.brightness = val
        self.bright_label.setText(str(val))

    def _on_contrast_changed(self, val: int) -> None:
        self.cam_thread.processor.contrast = val
        self.contrast_label.setText(str(val))

    def _on_lock_range(self, checked: bool) -> None:
        proc = self.cam_thread.processor
        if checked:
            proc.auto_range = False
            proc._locked_vmin = None  # will capture on next frame
            proc._locked_vmax = None
        else:
            proc.unlock_range()

    def _on_isotherm_toggled(self, checked: bool) -> None:
        self.thermal._isotherm_enabled = checked

    def _on_isotherm_mode(self, text: str) -> None:
        self.thermal._isotherm_mode = text.lower()

    def _on_isotherm_threshold(self, val: int) -> None:
        self.thermal._isotherm_threshold = float(val)
        self.iso_label.setText(f"{val}°C")

    # --- Alarm ---

    def _on_test_alarm(self) -> None:
        """Play alarm sound for 2 seconds as a preview."""
        if not self._alarm_sound:
            return
        if self._alarm_sound.isPlaying():
            self._alarm_sound.stop()
            return
        self._alarm_sound.play()
        QTimer.singleShot(2000, self._alarm_sound.stop)

    def _on_alarm_toggled(self, checked: bool) -> None:
        self._alarm_enabled = checked
        if not checked:
            self._set_alarm_active(False)

    def _on_alarm_source_changed(self, idx: int) -> None:
        _, key = ALARM_SOURCES[idx]
        self._alarm_source = key

    def _on_alarm_high_changed(self, val: float) -> None:
        self._alarm_high = val

    def _on_alarm_low_changed(self, val: float) -> None:
        self._alarm_low = val

    def _on_alarm_hysteresis_changed(self, val: float) -> None:
        self._alarm_hysteresis = val

    def _get_alarm_temp(self) -> float | None:
        """Return the current temperature for the selected alarm source."""
        proc = self.cam_thread.processor
        source = self._alarm_source
        if source == "center":
            return proc.center_temp
        if source == "global_max":
            return proc.max_temp
        if source == "global_min":
            return proc.min_temp
        if source == "mouse":
            return proc.mouse_temp
        if source.startswith("roi_"):
            stats = self.thermal._region_stats
            if not stats:
                return None
            return stats.get(source[4:])  # 'center', 'avg', 'max', 'min'
        return None

    def _set_alarm_active(self, active: bool) -> None:
        """Update alarm active state, sound, and visual feedback."""
        if active == self._alarm_active:
            return
        self._alarm_active = active
        self.thermal._alarm_active = active
        if active:
            if self._alarm_sound and not self._alarm_sound.isPlaying():
                self._alarm_sound.play()
            self._sec_alarm._toggle_btn.setStyleSheet(
                "QToolButton { color: #ff4444; font-weight: bold; }")
        else:
            if self._alarm_sound and self._alarm_sound.isPlaying():
                self._alarm_sound.stop()
            self._sec_alarm._toggle_btn.setStyleSheet("")

    def _on_shutter(self) -> None:
        self.cam_thread.request_shutter()

    def _on_nuc(self) -> None:
        self.cam_thread.request_nuc()

    def _on_flip(self, checked: bool) -> None:
        self.cam_thread.processor.flip = checked
        self.thermal.clear_selection()

    def _on_rotate(self) -> None:
        proc = self.cam_thread.processor
        proc.rotation = (proc.rotation + 90) % 360
        self.thermal.clear_selection()

    def _toggle_recording(self) -> None:
        if self._recording_active:
            self._stop_video_recording()
        else:
            self._start_video_recording()

    def _on_record_toggled(self, checked: bool) -> None:
        if checked:
            self._start_video_recording()
        else:
            self._stop_video_recording()

    def _start_video_recording(self) -> None:
        save_dir = self._ensure_save_dir()
        fname = str(save_dir /
            f"thermal_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            fname, fourcc, 25, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self._recording_active = True
        self._recording_file = fname
        self._recording_start = time.time()
        self.thermal._recording = True
        if HAS_3D:
            self.surface3d._recording = True
        self.btn_record.setChecked(True)
        self.btn_record.setText("[Ctrl+R] Stop")
        self.statusBar().showMessage(f"Recording: {fname}")

    def _stop_video_recording(self) -> None:
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._recording_active = False
        fname = self._recording_file
        self._recording_start = None
        self._recording_file = None
        self.thermal._recording = False
        if HAS_3D:
            self.surface3d._recording = False
        self.btn_record.setChecked(False)
        self.btn_record.setText("[Ctrl+R] Record")
        if fname:
            self.statusBar().showMessage(f"Recording saved: {fname}")

    def _save_dir_label(self) -> str:
        """Short label for the save-directory button."""
        try:
            rel = self._save_dir.relative_to(Path.home())
            return f"Save to: ~/{rel}"
        except ValueError:
            return f"Save to: {self._save_dir}"

    def _choose_save_dir(self) -> None:
        """Open a dialog to choose the save directory."""
        d = QFileDialog.getExistingDirectory(self, "Choose save directory", str(self._save_dir))
        if d:
            self._save_dir = Path(d)
            self._save_dir_btn.setText(self._save_dir_label())
            self._save_dir_btn.setToolTip(f"Save directory: {self._save_dir}")
            self.cam_thread.save_dir = self._save_dir
            self.graph_panel.save_dir = self._save_dir

    def _ensure_save_dir(self) -> Path:
        """Create the save directory if it doesn't exist and return the path."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        return self._save_dir

    def _save_screenshot(self) -> None:
        if self._mosaic_active:
            ready = self.mosaic._qimages[0] is not None
            widget = self.mosaic
        else:
            ready = self.thermal._qimage is not None
            widget = self.thermal
        if ready:
            save_dir = self._ensure_save_dir()
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = str(save_dir / f"thermal_{ts}.png")
            # Grab the widget with overlay painted on it
            pixmap = widget.grab()
            pixmap.save(fname)
            # Save radiometric data (raw temperature map)
            proc = self.cam_thread.processor
            msg = f"Screenshot: {fname}"
            if proc._temp_map is not None:
                npz_fname = str(save_dir / f"thermal_{ts}.npz")
                np.savez_compressed(npz_fname,
                                    temp_map=proc._temp_map,
                                    emissivity=proc.emissivity,
                                    fpa_temp=proc.fpa_temp)
                msg += f" + {npz_fname}"
            self.statusBar().showMessage(msg)

    def _on_calibration_points(self) -> None:
        """Show dialog for viewing blackbody calibration points (read-only)."""
        camera = self.cam_thread.camera
        points = camera.read_calibration_points()
        if points is None:
            QMessageBox.warning(self, "Calibration", "Failed to read calibration points.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Blackbody Calibration Points")
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("9 blackbody reference points stored on device."))

        table = QTableWidget(9, 2)
        table.setHorizontalHeaderLabels(["Reference (°C)", "Camera (°C)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i, (ref, cam) in enumerate(points):
            table.setItem(i, 0, QTableWidgetItem(f"{ref:.4f}"))
            table.setItem(i, 1, QTableWidgetItem(f"{cam:.4f}"))
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        dlg.resize(400, 350)
        dlg.exec()

    def _show_help(self) -> None:
        QMessageBox.information(self, "Keyboard Shortcuts",
            "Q\tQuit\n"
            "S\tSave screenshot\n"
            "P\tCycle palette\n"
            "F\tFlip image horizontally\n"
            "R\tRotate image 90° clockwise\n"
            "Ctrl+R\tToggle video recording\n"
            "C\tShutter calibration\n"
            "N\tNon-Uniformity Correction\n"
            "+/-\tAdjust brightness\n"
            "A/Shift+A\tIncrease/decrease contrast\n"
            "E\tIncrease emissivity\n"
            "Shift+E\tDecrease emissivity\n"
            "G\tToggle temperature graph\n"
            "M\tToggle mosaic view (all palettes)\n"
            "3\tToggle 3D surface plot\n"
            "L\tLock/unlock color range\n"
            "D\tDelete nearest probe point\n"
            "Esc\tClear selection & probes\n"
            "H\tShow this help\n"
            "\nLeft-click to pin a probe point.\n"
            "Double-click to remove nearest probe.\n"
            "Left-drag to select a region.\n"
            "Right-click to clear selection.\n"
            "Screenshots save PNG + radiometric .npz\n"
            "\n3D Surface Controls:\n"
            "Left-drag\tRotate view\n"
            "Ctrl+drag\tPan view\n"
            "Middle-drag\tPan view\n"
            "Scroll wheel\tZoom in/out")

    def _toggle_mosaic(self) -> None:
        self._mosaic_active = not self._mosaic_active
        if self._mosaic_active and self._surface3d_active:
            self._surface3d_active = False
            if HAS_3D:
                self.surface3d.setVisible(False)
                self.btn_3d.setChecked(False)
        # Hide old view before showing new to avoid both being visible
        if self._mosaic_active:
            self.thermal.setVisible(False)
            self.mosaic.setVisible(True)
        else:
            self.mosaic.setVisible(False)
            self.thermal.setVisible(not self._surface3d_active)
        self.btn_mosaic.setChecked(self._mosaic_active)
        self._refresh_view_layout()

    def _toggle_3d(self) -> None:
        if not HAS_3D:
            return
        self._surface3d_active = not self._surface3d_active
        if self._surface3d_active and self._mosaic_active:
            self._mosaic_active = False
            self.mosaic.setVisible(False)
            self.btn_mosaic.setChecked(False)
        # Hide old view before showing new to avoid both being visible
        if self._surface3d_active:
            self.thermal.setVisible(False)
            self.surface3d.setVisible(True)
        else:
            self.surface3d.setVisible(False)
            self.thermal.setVisible(not self._mosaic_active)
        self.btn_3d.setChecked(self._surface3d_active)
        self._refresh_view_layout()

    def _refresh_view_layout(self) -> None:
        """Force the layout to recalculate after switching view widgets."""
        self.centralWidget().layout().invalidate()
        self.centralWidget().layout().activate()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._recording_active:
            self._stop_video_recording()
        if self._alarm_sound and self._alarm_sound.isPlaying():
            self._alarm_sound.stop()
        self.cam_thread.stop()
        event.accept()


def run_gui() -> None:
    """Launch the PyQt6 thermal viewer."""
    import signal
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.ColorRole.Link, QColor(80, 160, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 120, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    window.start()

    # Handle Ctrl+C gracefully (avoid core dump from interrupted QPainter)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # Timer keeps the Python event loop alive so signal handlers fire
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec())
