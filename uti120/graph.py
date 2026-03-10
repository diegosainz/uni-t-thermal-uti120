"""Temperature-over-time graph panel with QPainter rendering."""

from __future__ import annotations

import math
import time
import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPainterPath, QPaintEvent
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QSplitter

from .constants import FRAME_WIDTH, FRAME_HEIGHT, GRAPH_SAMPLE_INTERVAL_MS, default_save_dir

if TYPE_CHECKING:
    from .processor import FrameProcessor
    from .widgets import ThermalWidget


__all__ = ["TemperatureGraphPanel"]


class TemperatureGraphPanel(QWidget):
    """Collapsible panel that graphs temperature over time using QPainter."""

    GRAPH_HEIGHT = 250
    WINDOW_SECONDS = 60  # show last 60 seconds of data

    # Distinct colors for up to 10 probe lines
    PROBE_COLORS = [
        QColor(255, 165, 0),    # orange
        QColor(0, 200, 255),    # cyan
        QColor(255, 80, 80),    # red
        QColor(100, 255, 100),  # green
        QColor(200, 100, 255),  # purple
        QColor(255, 255, 80),   # yellow
        QColor(255, 128, 200),  # pink
        QColor(128, 220, 200),  # teal
        QColor(200, 200, 100),  # olive
        QColor(180, 180, 255),  # lavender
    ]

    # Distinct colors for ROI stat series
    ROI_COLORS = {
        "ROI Avg": QColor(255, 204, 0),      # yellow
        "ROI Min": QColor(0, 180, 255),       # blue
        "ROI Max": QColor(255, 80, 80),       # red
        "ROI Center": QColor(100, 255, 100),  # green
    }

    def __init__(self, thermal_widget: ThermalWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thermal = thermal_widget
        self._processor_ref = None
        self._data = {}  # dict[str, list[tuple[float, float]]] — series_name → [(elapsed, temp)]
        self._emissivity_log: dict[float, float] = {}  # elapsed → emissivity
        self._running = False
        self._collapsed = True
        self._start_time = 0.0
        self.save_dir = default_save_dir()

        # --- Header bar ---
        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(4)

        self._toggle_btn = QPushButton("\u25b6 Temperature Graph")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self._toggle_btn)

        header_layout.addStretch()

        self._time_combo = QComboBox()
        self._time_combo.addItems(["30s", "1m", "2m", "5m", "10m"])
        self._time_combo.setCurrentIndex(1)  # default 1m
        self._time_combo.setFixedWidth(55)
        self._time_combo.currentIndexChanged.connect(self._on_time_range_changed)
        header_layout.addWidget(self._time_combo)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["Center", "Mouse", "Selection (ROI)", "Probes"])
        self._source_combo.setFixedWidth(100)
        header_layout.addWidget(self._source_combo)

        self._start_stop_btn = QPushButton("Start")
        self._start_stop_btn.setFixedWidth(50)
        self._start_stop_btn.clicked.connect(self._toggle_recording)
        header_layout.addWidget(self._start_stop_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setFixedWidth(80)
        self._export_btn.setStyleSheet(
            "QPushButton:disabled { color: #555; }"
        )
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        header_layout.addWidget(self._export_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedWidth(50)
        self._clear_btn.clicked.connect(self._clear_data)
        header_layout.addWidget(self._clear_btn)

        # --- Graph canvas ---
        self._canvas = QWidget()
        self._canvas.setMinimumHeight(80)
        self._canvas.paintEvent = self._paint_graph
        self._canvas.setVisible(False)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._canvas, stretch=1)

        # Sample timer
        self._timer = QTimer()
        self._timer.setInterval(GRAPH_SAMPLE_INTERVAL_MS)
        self._timer.timeout.connect(self._sample)

        # Fonts
        self._font = QFont("monospace", 8)

    def set_processor(self, processor: FrameProcessor) -> None:
        self._processor_ref = processor

    def toggle_collapse(self) -> None:
        self._toggle_collapse()

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self._canvas.setVisible(not self._collapsed)
        arrow = "\u25b6" if self._collapsed else "\u25bc"
        self._toggle_btn.setText(f"{arrow} Temperature Graph")
        # Ask parent splitter to give us space when expanding
        if not self._collapsed:
            splitter = self.parent()
            if isinstance(splitter, QSplitter):
                sizes = splitter.sizes()
                total = sum(sizes)
                graph_h = max(self.GRAPH_HEIGHT, int(total * 0.35))
                my_idx = splitter.indexOf(self)
                # Shrink the first visible widget to make room
                new_sizes = list(sizes)
                new_sizes[0] = total - graph_h
                new_sizes[my_idx] = graph_h
                splitter.setSizes(new_sizes)

    _TIME_RANGE_MAP = [30, 60, 120, 300, 600]

    def _on_time_range_changed(self, index: int) -> None:
        self.WINDOW_SECONDS = self._TIME_RANGE_MAP[index]
        self._canvas.update()

    def _toggle_recording(self) -> None:
        if self._running:
            self._running = False
            self._timer.stop()
            self._start_stop_btn.setText("Start")
            self._export_btn.setEnabled(bool(self._data))
        else:
            self._data = {}
            self._start_time = time.time()
            self._running = True
            self._export_btn.setEnabled(False)
            self._start_stop_btn.setText("Stop")
            # Expand if collapsed
            if self._collapsed:
                self._toggle_collapse()
            self._timer.start()

    def _sample(self) -> None:
        if self._processor_ref is None:
            return
        source = self._source_combo.currentText()
        elapsed = time.time() - self._start_time
        self._emissivity_log[elapsed] = self._processor_ref.emissivity
        if source == "Probes":
            probes = self._thermal._probes
            proc = self._processor_ref
            if proc._temp_map is None:
                return
            for i, (fx, fy) in enumerate(probes):
                key = f"P{i+1}"
                if 0 <= fx < FRAME_WIDTH and 0 <= fy < FRAME_HEIGHT:
                    temp = float(proc._temp_map[fy, fx])
                    self._data.setdefault(key, []).append((elapsed, temp))
        elif source == "Selection (ROI)":
            stats = self._thermal._region_stats
            if stats is not None:
                for stat_key, data_key in [('avg', 'ROI Avg'), ('min', 'ROI Min'),
                                            ('max', 'ROI Max'), ('center', 'ROI Center')]:
                    self._data.setdefault(data_key, []).append((elapsed, stats[stat_key]))
        else:
            temp = None
            if source == "Center":
                temp = self._processor_ref.center_temp
            elif source == "Mouse":
                temp = self._processor_ref.mouse_temp
            if temp is not None:
                self._data.setdefault(source, []).append((elapsed, temp))
        self._canvas.update()

    def _clear_data(self) -> None:
        self._data = {}
        self._emissivity_log = {}
        self._export_btn.setEnabled(False)
        self._canvas.update()

    def _export_csv(self) -> None:
        if not self._data:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fname = self.save_dir / f"thermal_graph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        keys = sorted(self._data.keys())
        start_dt = datetime.datetime.fromtimestamp(self._start_time)
        with open(fname, 'w') as f:
            all_times = sorted(set(
                t for series in self._data.values() for t, _ in series
            ))
            lookups = {}
            for key in keys:
                lookups[key] = {t: v for t, v in self._data[key]}
            f.write("timestamp,emissivity," + ",".join(keys) + "\n")
            for t in all_times:
                ts = (start_dt + datetime.timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
                emiss = self._emissivity_log.get(t)
                emiss_str = f"{emiss:.2f}" if emiss is not None else ""
                vals = []
                for k in keys:
                    v = lookups[k].get(t)
                    vals.append(f"{v:.1f}" if v is not None else "")
                f.write(f"{ts},{emiss_str},{','.join(vals)}\n")
        win = self.window()
        if hasattr(win, 'statusBar'):
            win.statusBar().showMessage(f"Graph exported: {fname}")
        self._export_btn.setEnabled(False)

    def _series_color(self, key: str) -> QColor:
        """Return the color for a data series key."""
        # Probe keys: P1, P2, ...
        if key.startswith("P") and key[1:].isdigit():
            idx = int(key[1:]) - 1
            return self.PROBE_COLORS[idx % len(self.PROBE_COLORS)]
        # ROI stat series
        if key in self.ROI_COLORS:
            return self.ROI_COLORS[key]
        # Single-source keys (Center, Mouse)
        return QColor(255, 204, 0)  # yellow

    def _paint_graph(self, event: QPaintEvent) -> None:
        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self._canvas.width(), self._canvas.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(42, 42, 42))

        margin_left = 50
        margin_right = 10
        margin_top = 10
        margin_bottom = 20
        gw = w - margin_left - margin_right
        gh = h - margin_top - margin_bottom

        if gw < 10 or gh < 10:
            painter.end()
            return

        painter.setFont(self._font)

        # Collect all data points across series
        total_points = sum(len(v) for v in self._data.values())
        if total_points < 2:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(margin_left, margin_top, gw, gh,
                             Qt.AlignmentFlag.AlignCenter,
                             "No data" if total_points == 0 else "Collecting...")
            painter.end()
            return

        # Determine visible time window from all series
        all_t_max = max(s[-1][0] for s in self._data.values() if s)
        t_min = max(0, all_t_max - self.WINDOW_SECONDS)
        t_max = all_t_max

        # Compute Y range across all visible series
        all_temps = []
        visible_series = {}  # key → [(t, v)]
        for key, series in self._data.items():
            vis = [(t, v) for t, v in series if t >= t_min]
            if len(vis) < 2 and len(series) >= 2:
                vis = series[-2:]
            if vis:
                visible_series[key] = vis
                all_temps.extend(v for _, v in vis)

        if not all_temps:
            painter.end()
            return

        v_min = min(all_temps) - 1.0
        v_max = max(all_temps) + 1.0
        if v_max - v_min < 2.0:
            mid = (v_max + v_min) / 2
            v_min = mid - 1.0
            v_max = mid + 1.0

        # Choose a Y-axis step that keeps labels ≥20 px apart
        px_per_deg = gh / (v_max - v_min)
        step = 0.2
        for s in [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]:
            if s * px_per_deg >= 20:
                step = s
                break
        else:
            step = 100

        first_grid = math.ceil(v_min / step) * step
        last_grid = math.floor(v_max / step) * step

        # Grid lines
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DotLine))
        deg = first_grid
        while deg <= last_grid + step * 0.01:
            y = margin_top + gh - (deg - v_min) / (v_max - v_min) * gh
            painter.drawLine(int(margin_left), int(y),
                             int(margin_left + gw), int(y))
            deg += step

        # Y-axis labels
        _fmt = "{:.0f}°C" if step >= 1 else "{:.1f}°C"
        painter.setPen(QColor(160, 160, 160))
        deg = first_grid
        while deg <= last_grid + step * 0.01:
            y = margin_top + gh - (deg - v_min) / (v_max - v_min) * gh
            painter.drawText(2, int(y + 4), _fmt.format(round(deg, 1)))
            deg += step

        # X-axis labels (every 10s)
        if t_max > t_min:
            t_start_label = int(t_min / 10) * 10 + 10
            for ts in range(t_start_label, int(t_max) + 1, 10):
                if ts < t_min:
                    continue
                x = margin_left + (ts - t_min) / (t_max - t_min) * gw
                painter.drawText(int(x - 10), h - 2, f"{ts:.0f}s")

        # Axes border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(margin_left, margin_top,
                         margin_left, margin_top + gh)
        painter.drawLine(margin_left, margin_top + gh,
                         margin_left + gw, margin_top + gh)

        # Draw each data series
        t_range = t_max - t_min if t_max > t_min else 1.0
        v_range = v_max - v_min
        legend_y = margin_top + 12
        for key in sorted(visible_series.keys()):
            vis = visible_series[key]
            color = self._series_color(key)

            # Data line
            path = QPainterPath()
            for i, (t, v) in enumerate(vis):
                x = margin_left + (t - t_min) / t_range * gw
                y = margin_top + gh - (v - v_min) / v_range * gh
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Legend entry (top-right)
            last_v = vis[-1][1]
            label = key
            legend_text = f"{label}: {last_v:.1f}°C"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(legend_text)
            lx = margin_left + gw - tw - 4
            painter.fillRect(QRectF(lx - 3, legend_y - 10, tw + 6, 13),
                             QColor(0, 0, 0, 140))
            painter.setPen(color)
            painter.drawText(lx, legend_y, legend_text)
            legend_y += 14

        painter.end()
