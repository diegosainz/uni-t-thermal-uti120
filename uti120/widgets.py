"""Thermal display widgets — ThermalWidget, MosaicWidget, CollapsibleSection."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import cv2
from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import (QImage, QMouseEvent, QPaintEvent, QPainter, QColor,
                         QFont, QPen, QPainterPath)
from PyQt6.QtWidgets import QWidget, QSizePolicy, QToolButton, QVBoxLayout

from .constants import FRAME_WIDTH, FRAME_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT, MAX_PROBES
from .palettes import PALETTES, apply_palette

__all__ = ["ThermalWidget", "MosaicWidget", "CollapsibleSection"]

if TYPE_CHECKING:
    from .processor import FrameProcessor


def _compute_region_stats(region: np.ndarray | None) -> dict | None:
    """Compute min/max/avg/center/delta stats from a temperature region array."""
    if region is None or region.size == 0:
        return None
    cy, cx = region.shape[0] // 2, region.shape[1] // 2
    stats = {
        'min': float(np.min(region)),
        'max': float(np.max(region)),
        'avg': float(np.mean(region)),
        'center': float(region[cy, cx]),
        'w': region.shape[1],
        'h': region.shape[0],
    }
    stats['delta'] = stats['max'] - stats['min']
    return stats


def draw_text_with_bg(painter: QPainter, x: int, y: int, text: str, color: QColor) -> None:
    """Draw text with a semi-transparent dark background."""
    fm = painter.fontMetrics()
    rect = fm.boundingRect(text)
    pad = 3
    bg_rect = QRectF(x - pad, y - rect.height() - pad + 2,
                     rect.width() + pad * 2, rect.height() + pad * 2)
    painter.fillRect(bg_rect, QColor(0, 0, 0, 160))
    painter.setPen(color)
    painter.drawText(x, y, text)


class ThermalWidget(QWidget):
    """Custom widget that displays the thermal image with overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._qimage: QImage | None = None
        self._processor: FrameProcessor | None = None
        self._mouse_pos: tuple[int, int] | None = None  # widget coords
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_time: float = time.time()

        # Region selection
        self._selection_start: tuple[int, int] | None = None  # (x, y) drag origin
        self._selection_rect: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2) widget coords
        self._selection_rect_frame: tuple[int, int, int, int] | None = None  # (fx1, fy1, fx2, fy2) frame coords
        self._region_stats: dict[str, float | int] | None = None     # dict with min, max, avg, delta, w, h
        self._recording: bool = False       # set by MainWindow when recording

        # Pinned probe points — list of (fx, fy) in frame coordinates
        self._probes: list[tuple[int, int]] = []
        self.MAX_PROBES: int = MAX_PROBES

        self._double_click_pending: bool = False  # suppress probe placement after double-click

        # Isotherm overlay
        self._isotherm_enabled: bool = False
        self._isotherm_threshold: float = 50.0  # °C
        self._isotherm_mode: str = 'above'    # 'above' or 'below'

        # Alarm visual feedback
        self._alarm_active: bool = False
        self._alarm_message: str = ""

        # Fonts
        self._font_normal: QFont = QFont("monospace", 10)
        self._font_small: QFont = QFont("monospace", 8)
        self._font_maxmin: QFont = QFont("monospace", 10, QFont.Weight.Bold)
        self._font_temp: QFont = QFont("monospace", 11, QFont.Weight.Bold)

    def update_frame(self, display_bgr: np.ndarray, processor: FrameProcessor) -> None:
        """Called from main thread when a new frame arrives."""
        self._processor = processor

        # Update mouse temp if tracking
        if self._mouse_pos is not None:
            mx, my = self._mouse_pos
            ix, iy, iw, ih = self._image_rect()
            processor.update_mouse_temp(mx - ix, my - iy, iw, ih)

        # BGR → RGB → QImage
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._qimage = QImage(rgb.data, w, h, ch * w,
                              QImage.Format.Format_RGB888).copy()

        # Recompute region stats live
        if self._selection_rect_frame is not None and self._selection_start is None:
            self._recompute_region_stats_from_frame()
        elif self._selection_rect is not None and self._selection_start is None:
            self._recompute_region_stats()

        # FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()

        self.update()

    def render_composited_frame(self, display_bgr: np.ndarray,
                                processor: FrameProcessor) -> np.ndarray:
        """Render thermal image + overlays onto an offscreen QImage.

        Returns a BGR numpy array suitable for cv2.VideoWriter.  Transient
        elements (mouse cursor, FPS, REC indicator) are excluded.
        """
        h, w = display_bgr.shape[:2]
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, w, h, 3 * w,
                        QImage.Format.Format_RGB888).copy()

        painter = QPainter(qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_overlay(painter, processor, w, h, for_recording=True)
        painter.end()

        # QImage (RGB888) → numpy BGR
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._qimage is None:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(self._font_normal)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Waiting for camera...")
            painter.end()
            return

        # Draw thermal image preserving 4:3 aspect ratio (letterbox/pillarbox)
        ix, iy, iw, ih = self._image_rect()
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        target = QRectF(ix, iy, iw, ih)
        source = QRectF(0, 0, self._qimage.width(), self._qimage.height())
        painter.drawImage(target, self._qimage, source)

        # Draw overlays in image-relative coordinate space
        painter.save()
        painter.translate(ix, iy)
        if self._processor is not None:
            self._draw_overlay(painter, self._processor, iw, ih,
                               img_offset=(ix, iy))

        # FPS
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(self._font_small)
        painter.drawText(8, 15, f"FPS: {self._fps:.1f}")

        # Recording indicator
        if self._recording:
            rec_x = iw - 60
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rec_x, 6, 10, 10)
            painter.setPen(QColor(255, 60, 60))
            painter.setFont(self._font_small)
            painter.drawText(rec_x + 14, 15, "REC")

        # Alarm border + flashing banner
        if self._alarm_active:
            # Flash: 500ms on / 500ms off
            flash_on = (int(time.monotonic() * 2) % 2) == 0
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(1, 1, int(iw) - 2, int(ih) - 2)
            if flash_on:
                banner_font = QFont("monospace", 12, QFont.Weight.Bold)
                painter.setFont(banner_font)
                fm = painter.fontMetrics()
                text = f"ALARM  {self._alarm_message}" if self._alarm_message else "ALARM"
                tw = fm.horizontalAdvance(text) + 16
                th = fm.height() + 8
                bx = (int(iw) - tw) // 2
                by = 4
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(200, 0, 0, 200))
                painter.drawRoundedRect(bx, by, tw, th, 4, 4)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(bx + 8, by + fm.ascent() + 4, text)
        painter.restore()

        painter.end()

    def _draw_overlay(self, painter: QPainter, proc: FrameProcessor,
                      w: int | None = None, h: int | None = None,
                      for_recording: bool = False,
                      img_offset: tuple[int, int] = (0, 0)) -> None:
        """Draw temperature overlay elements using QPainter.

        When *w*/*h* are given they override self.width()/height() (used for
        off-screen rendering at a fixed resolution, e.g. video recording).
        When *for_recording* is True, transient elements (mouse cursor) are
        skipped.  *img_offset* is the (x, y) offset of the image rect within
        the widget, used to adjust widget-coord elements (selection rect).
        """
        if w is None:
            w = self.width()
        if h is None:
            h = self.height()
        ox, oy = img_offset

        # Isotherm overlay (drawn first, beneath other markers)
        self._draw_isotherm(painter, proc, w, h)

        # --- Center crosshair ---
        cx, cy = w // 2, h // 2
        cross = 15
        pen_white = QPen(QColor(255, 255, 255), 1)
        painter.setPen(pen_white)
        painter.drawLine(cx - cross, cy, cx + cross, cy)
        painter.drawLine(cx, cy - cross, cx, cy + cross)

        painter.setFont(self._font_temp)
        draw_text_with_bg(painter, cx + 12, cy - 8,
                                f"{proc.center_temp:.1f}°C",
                                QColor(255, 255, 255))

        # --- Max marker (red triangle + label) ---
        mx, my = self._frame_to_widget(proc.max_pos[0], proc.max_pos[1], proc, w, h)
        pen_red = QPen(QColor(255, 60, 60), 2)
        painter.setPen(pen_red)
        painter.drawLine(mx, my - 6, mx - 5, my + 4)
        painter.drawLine(mx - 5, my + 4, mx + 5, my + 4)
        painter.drawLine(mx + 5, my + 4, mx, my - 6)
        painter.setFont(self._font_maxmin)
        draw_text_with_bg(painter, mx + 10, my + 4,
                                f"MAX {proc.max_temp:.1f}°C",
                                QColor(255, 60, 60))

        # --- Min marker (blue triangle + label) ---
        nx, ny = self._frame_to_widget(proc.min_pos[0], proc.min_pos[1], proc, w, h)
        pen_blue = QPen(QColor(80, 160, 255), 2)
        painter.setPen(pen_blue)
        painter.drawLine(nx, ny + 6, nx - 5, ny - 4)
        painter.drawLine(nx - 5, ny - 4, nx + 5, ny - 4)
        painter.drawLine(nx + 5, ny - 4, nx, ny + 6)
        painter.setFont(self._font_maxmin)
        draw_text_with_bg(painter, nx + 10, ny + 12,
                                f"MIN {proc.min_temp:.1f}°C",
                                QColor(80, 160, 255))

        # --- Color scale bar ---
        bar_w = 20
        bar_x = w - bar_w - 45
        bar_top = 30
        bar_bottom = h - 30
        bar_h = bar_bottom - bar_top

        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
        bar_colored = apply_palette(gradient, proc.palette_idx)
        bar_rgb = cv2.cvtColor(bar_colored, cv2.COLOR_BGR2RGB)
        bar_strip = np.repeat(bar_rgb, bar_w, axis=1)
        bar_img = QImage(bar_strip.data, bar_w, bar_h, bar_w * 3,
                         QImage.Format.Format_RGB888).copy()
        painter.drawImage(bar_x, bar_top, bar_img)

        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(bar_x, bar_top, bar_w, bar_h)

        painter.setFont(self._font_small)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(bar_x - 5, bar_top - 5, f"{proc.max_temp:.0f}°C")
        painter.drawText(bar_x - 5, bar_bottom + 14, f"{proc.min_temp:.0f}°C")
        painter.drawText(bar_x - 5, bar_bottom + 28, f"\u03b5: {proc.emissivity:.2f}")

        # --- Mouse cursor temperature (skip for recording) ---
        if not for_recording and proc.mouse_temp is not None and proc.mouse_pos is not None:
            mx, my = proc.mouse_pos
            pen_cyan = QPen(QColor(0, 255, 255), 1)
            painter.setPen(pen_cyan)
            painter.drawLine(mx - 8, my, mx + 8, my)
            painter.drawLine(mx, my - 8, mx, my + 8)

            label = f"{proc.mouse_temp:.1f}°C"
            lx = mx + 14 if mx < w - 100 else mx - 90
            ly = my - 6 if my > 20 else my + 20
            draw_text_with_bg(painter, lx, ly, label, QColor(0, 255, 255))

        # --- Pinned probe points ---
        if self._probes and proc._temp_map is not None:
            probe_color = QColor(255, 165, 0)  # orange
            painter.setFont(self._font_small)
            for i, (fx, fy) in enumerate(self._probes):
                # Look up live temperature
                if 0 <= fx < FRAME_WIDTH and 0 <= fy < FRAME_HEIGHT:
                    temp = float(proc._temp_map[fy, fx])
                else:
                    continue
                # Convert frame coords to widget coords (accounting for flip)
                px, py = self._frame_to_widget(fx, fy, proc, w, h)
                # Draw diamond marker
                d = 5
                painter.setPen(QPen(probe_color, 2))
                painter.drawLine(px, py - d, px + d, py)
                painter.drawLine(px + d, py, px, py + d)
                painter.drawLine(px, py + d, px - d, py)
                painter.drawLine(px - d, py, px, py - d)
                # Label
                label = f"P{i+1}: {temp:.1f}°C"
                lx = px + 10 if px < w - 120 else px - 110
                ly = py - 6 if py > 20 else py + 20
                draw_text_with_bg(painter, lx, ly, label, probe_color)

        # --- Region selection rectangle + stats ---
        has_roi = False
        if self._selection_rect_frame is not None and self._processor is not None:
            # Derive widget coords from canonical frame coords (works for both modes)
            fx1, fy1, fx2, fy2 = self._selection_rect_frame
            sx1, sy1 = self._frame_to_widget(fx1, fy1, self._processor, w, h)
            sx2, sy2 = self._frame_to_widget(fx2, fy2, self._processor, w, h)
            x1, y1 = min(sx1, sx2), min(sy1, sy2)
            x2, y2 = max(sx1, sx2), max(sy1, sy2)
            has_roi = True
        elif self._selection_rect is not None:
            # In-progress drag (before frame coords are stored)
            sx1, sy1, sx2, sy2 = self._selection_rect
            x1, y1, x2, y2 = sx1 - ox, sy1 - oy, sx2 - ox, sy2 - oy
            has_roi = True
        if has_roi:
            pen_sel = QPen(QColor(0, 255, 100), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen_sel)
            painter.setBrush(QColor(0, 255, 100, 30))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if self._region_stats is not None:
                s = self._region_stats
                lines = [
                    f"ROI: {s['w']}x{s['h']} px  Ctr: {s['center']:.1f}°C",
                    f"Min: {s['min']:.1f}°C  Max: {s['max']:.1f}°C",
                    f"Avg: {s['avg']:.1f}°C  \u0394T: {s['delta']:.1f}°C",
                ]
                painter.setFont(self._font_small)
                color = QColor(0, 255, 100)
                # Position label below selection, or above if near bottom
                lx = x1
                ly = y2 + 16
                if ly + 40 > h:
                    ly = y1 - 36
                for i, line in enumerate(lines):
                    draw_text_with_bg(painter, lx, ly + i * 14, line, color)

    def _draw_isotherm(self, painter: QPainter, proc: FrameProcessor,
                       w: int | None = None, h: int | None = None) -> None:
        """Draw semi-transparent overlay on pixels above/below isotherm threshold."""
        if not self._isotherm_enabled or proc._temp_map is None:
            return
        if w is None:
            w = self.width()
        if h is None:
            h = self.height()
        temp_map = proc._temp_map

        if self._isotherm_mode == 'above':
            mask = temp_map >= self._isotherm_threshold
        else:
            mask = temp_map <= self._isotherm_threshold

        if not np.any(mask):
            return

        # Apply flip then rotate to match display
        if proc.flip:
            mask = np.flip(mask, axis=1)
        if proc.rotation == 90:
            mask = np.rot90(mask, k=-1)
        elif proc.rotation == 180:
            mask = np.rot90(mask, k=2)
        elif proc.rotation == 270:
            mask = np.rot90(mask, k=1)

        # Create RGBA overlay at rotated resolution, then scale
        oh, ow = mask.shape
        overlay = np.zeros((oh, ow, 4), dtype=np.uint8)
        overlay[mask] = [0, 255, 100, 120]  # green fill, semi-transparent

        # Draw cyan contour around isotherm boundary for visibility on all palettes
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255, 220), 1)

        overlay_resized = cv2.resize(overlay, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
        qimg = QImage(overlay_resized.data, w, h, w * 4,
                      QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(0, 0, qimg)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._probes:
            # Undo the probe placed by the first click of the double-click first,
            # so remove_nearest_probe targets the intended probe, not the just-placed one.
            self._probes.pop()
            pos = event.position()
            if self._probes:
                self.remove_nearest_probe(int(pos.x()), int(pos.y()))
            self._double_click_pending = True

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._selection_start = (int(pos.x()), int(pos.y()))
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_selection()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())
        self._mouse_pos = (x, y)

        if (self._selection_start is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            sx, sy = self._selection_start
            self._selection_rect = (min(sx, x), min(sy, y),
                                    max(sx, x), max(sy, y))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._double_click_pending:
                self._double_click_pending = False
                self._selection_start = None
                return
            if self._selection_start is not None:
                pos = event.position()
                x, y = int(pos.x()), int(pos.y())
                sx, sy = self._selection_start
                x1, y1 = min(sx, x), min(sy, y)
                x2, y2 = max(sx, x), max(sy, y)
                # Only keep selection if it has meaningful size
                if x2 - x1 > 3 and y2 - y1 > 3:
                    self._selection_rect = (x1, y1, x2, y2)
                    # Store frame-coord version for mosaic rendering
                    ffx1, ffy1 = self._widget_to_frame(x1, y1)
                    ffx2, ffy2 = self._widget_to_frame(x2, y2)
                    self._selection_rect_frame = (
                        min(ffx1, ffx2), min(ffy1, ffy2),
                        max(ffx1, ffx2), max(ffy1, ffy2))
                    self._recompute_region_stats()
                else:
                    # Click (not drag) → place probe (keep existing ROI)
                    if len(self._probes) < self.MAX_PROBES:
                        fx, fy = self._widget_to_frame(sx, sy)
                        self._probes.append((fx, fy))
                self._selection_start = None

    def leaveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = None
        if self._processor:
            self._processor.mouse_temp = None
            self._processor.mouse_pos = None

    def clear_selection(self) -> None:
        self._selection_start = None
        self._selection_rect = None
        self._selection_rect_frame = None
        self._region_stats = None

    def clear_probes(self) -> None:
        self._probes.clear()

    def remove_nearest_probe(self, wx: int, wy: int) -> None:
        """Remove the probe closest to widget coords (wx, wy)."""
        if not self._probes or self._processor is None:
            return
        fx, fy = self._widget_to_frame(wx, wy)
        best_i, best_d = 0, float('inf')
        for i, (px, py) in enumerate(self._probes):
            d = (px - fx) ** 2 + (py - fy) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        self._probes.pop(best_i)

    def _image_rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) of the aspect-ratio-preserving image area."""
        ww, wh = self.width(), self.height()
        proc = self._processor
        if proc and proc.rotation in (90, 270):
            fw, fh = FRAME_HEIGHT, FRAME_WIDTH
        else:
            fw, fh = FRAME_WIDTH, FRAME_HEIGHT
        img_aspect = fw / fh
        widget_aspect = ww / wh if wh > 0 else img_aspect
        if widget_aspect > img_aspect:
            # Widget wider than 4:3 — pillarbox
            h = wh
            w = int(h * img_aspect)
        else:
            # Widget taller than 4:3 — letterbox
            w = ww
            h = int(w / img_aspect)
        x = (ww - w) // 2
        y = (wh - h) // 2
        return x, y, w, h

    def _frame_to_widget(self, fx: int, fy: int, proc: FrameProcessor,
                         w: int | None = None, h: int | None = None) -> tuple[int, int]:
        """Convert frame coords to image-relative coords, accounting for flip and rotation."""
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        # Flip first
        dx = (W - 1 - fx) if proc.flip else fx
        dy = fy
        # Then rotate
        if proc.rotation == 90:
            dx, dy = H - 1 - dy, dx
            dw, dh = H, W
        elif proc.rotation == 180:
            dx, dy = W - 1 - dx, H - 1 - dy
            dw, dh = W, H
        elif proc.rotation == 270:
            dx, dy = dy, W - 1 - dx
            dw, dh = H, W
        else:
            dw, dh = W, H
        if w is None:
            _, _, w, h = self._image_rect()
        wx = int(dx * w / dw)
        wy = int(dy * h / dh)
        return wx, wy

    def _widget_to_frame(self, wx: int, wy: int) -> tuple[int, int]:
        """Convert widget coords to frame coords, accounting for flip and rotation."""
        ix, iy, iw, ih = self._image_rect()
        proc = self._processor
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        if proc and proc.rotation in (90, 270):
            dw, dh = H, W
        else:
            dw, dh = W, H
        dx = int((wx - ix) * dw / iw)
        dy = int((wy - iy) * dh / ih)
        dx = max(0, min(dw - 1, dx))
        dy = max(0, min(dh - 1, dy))
        # Inverse rotation
        if proc is None or proc.rotation == 0:
            fx, fy = dx, dy
        elif proc.rotation == 90:
            fx, fy = dy, H - 1 - dx
        elif proc.rotation == 180:
            fx, fy = W - 1 - dx, H - 1 - dy
        else:  # 270
            fx, fy = W - 1 - dy, dx
        # Inverse flip
        if proc and proc.flip:
            fx = W - 1 - fx
        return max(0, min(fx, W - 1)), max(0, min(fy, H - 1))

    def _recompute_region_stats(self) -> None:
        """Compute temperature statistics for the selected region."""
        if self._selection_rect is None or self._processor is None:
            self._region_stats = None
            return
        x1, y1, x2, y2 = self._selection_rect
        fx1, fy1 = self._widget_to_frame(x1, y1)
        fx2, fy2 = self._widget_to_frame(x2, y2)
        # Ensure proper ordering after flip
        if fx1 > fx2:
            fx1, fx2 = fx2, fx1
        if fy1 > fy2:
            fy1, fy2 = fy2, fy1
        region = self._processor.get_region_temps(fx1, fy1, fx2, fy2)
        self._region_stats = _compute_region_stats(region)

    def _recompute_region_stats_from_frame(self) -> None:
        """Compute region stats directly from frame-coord selection."""
        if self._selection_rect_frame is None or self._processor is None:
            self._region_stats = None
            return
        fx1, fy1, fx2, fy2 = self._selection_rect_frame
        region = self._processor.get_region_temps(fx1, fy1, fx2, fy2)
        self._region_stats = _compute_region_stats(region)

    def sizeHint(self) -> QSize:
        return QSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)


class MosaicWidget(QWidget):
    """Displays the thermal image with all 6 palettes in a 3×2 grid.

    All interaction state (probes, selection, isotherm) lives in the
    ThermalWidget passed at construction — this widget only reads from it.
    """

    COLS, ROWS = 3, 2

    def __init__(self, thermal_widget: ThermalWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thermal: ThermalWidget = thermal_widget  # single source of truth
        self.setMinimumSize(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)
        self.setMouseTracking(True)
        self._qimages: list[QImage | None] = [None] * len(PALETTES)
        self._palette_names: list[str] = [name for name, _ in PALETTES]
        self._processor: FrameProcessor | None = None

        # Fonts (smaller for ~240×270 panels)
        self._font_label: QFont = QFont("monospace", 9, QFont.Weight.Bold)
        self._font_small: QFont = QFont("monospace", 7)
        self._font_maxmin: QFont = QFont("monospace", 8, QFont.Weight.Bold)
        self._font_temp: QFont = QFont("monospace", 8, QFont.Weight.Bold)

        # Drag state (mirrors ThermalWidget logic but in frame coords)
        self._drag_start_frame: tuple[int, int] | None = None  # (fx, fy) while dragging
        self._double_click_pending: bool = False

    # --- geometry helpers ---

    def _panel_geom(self) -> tuple[float, float]:
        """Return (panel_w, panel_h) in widget pixels."""
        return self.width() / self.COLS, self.height() / self.ROWS

    def _hit_panel(self, wx: int, wy: int) -> tuple[int, float, float] | None:
        """Return (panel_idx, local_x, local_y) or None if outside."""
        pw, ph = self._panel_geom()
        col = int(wx / pw)
        row = int(wy / ph)
        if 0 <= col < self.COLS and 0 <= row < self.ROWS:
            return row * self.COLS + col, wx - col * pw, wy - row * ph
        return None

    def _local_to_frame(self, lx: float, ly: float, pw: float, ph: float) -> tuple[int, int]:
        """Convert panel-local coords to frame coords, respecting flip and rotation."""
        proc = self._processor
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        if proc and proc.rotation in (90, 270):
            dw, dh = H, W
        else:
            dw, dh = W, H
        dx = int(lx * dw / pw)
        dy = int(ly * dh / ph)
        # Inverse rotation
        if proc is None or proc.rotation == 0:
            fx, fy = dx, dy
        elif proc.rotation == 90:
            fx, fy = dy, H - 1 - dx
        elif proc.rotation == 180:
            fx, fy = W - 1 - dx, H - 1 - dy
        else:  # 270
            fx, fy = W - 1 - dy, dx
        # Inverse flip
        if proc and proc.flip:
            fx = W - 1 - fx
        return max(0, min(fx, W - 1)), max(0, min(fy, H - 1))

    def _frame_to_panel(self, fx: int, fy: int, pw: float, ph: float) -> tuple[float, float]:
        """Convert frame coords to panel-local coords, respecting flip and rotation."""
        proc = self._processor
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        dx = (W - 1 - fx) if (proc and proc.flip) else fx
        dy = fy
        if proc and proc.rotation == 90:
            dx, dy = H - 1 - dy, dx
            dw, dh = H, W
        elif proc and proc.rotation == 180:
            dx, dy = W - 1 - dx, H - 1 - dy
            dw, dh = W, H
        elif proc and proc.rotation == 270:
            dx, dy = dy, W - 1 - dx
            dw, dh = H, W
        else:
            dw, dh = W, H
        return dx * pw / dw, dy * ph / dh

    # --- frame updates ---

    def update_frame(self, display_bgr: np.ndarray, processor: FrameProcessor) -> None:
        """Generate 6 palette views from the processor's normalized frame."""
        self._processor = processor
        normalized = getattr(processor, '_last_normalized', None)
        if normalized is None:
            return

        # Update mouse temp if tracking
        if self._thermal._mouse_pos is not None:
            mx, my = self._thermal._mouse_pos
            hit = self._hit_panel(mx, my)
            if hit is not None:
                _, lx, ly = hit
                pw, ph = self._panel_geom()
                fx, fy = self._local_to_frame(lx, ly, pw, ph)
                if 0 <= fx < FRAME_WIDTH and 0 <= fy < FRAME_HEIGHT and processor._temp_map is not None:
                    processor.mouse_temp = float(processor._temp_map[fy, fx])
                    processor.mouse_pos = (mx, my)

        for i in range(len(PALETTES)):
            colored = apply_palette(normalized, i)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self._qimages[i] = QImage(
                rgb.data, w, h, ch * w,
                QImage.Format.Format_RGB888
            ).copy()

        # Recompute region stats live
        if self._thermal._selection_rect_frame is not None and self._drag_start_frame is None:
            self._recompute_region_stats_frame()

        self.update()

    def render_composited_frame(self) -> np.ndarray:
        """Render the 6-palette mosaic to an offscreen image for recording.

        Returns a BGR numpy array (DISPLAY_WIDTH × DISPLAY_HEIGHT) suitable
        for cv2.VideoWriter.
        """
        w, h = DISPLAY_WIDTH, DISPLAY_HEIGHT
        qimage = QImage(w, h, QImage.Format.Format_RGB888)
        qimage.fill(QColor(0, 0, 0))

        painter = QPainter(qimage)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pw = w / self.COLS
        ph = h / self.ROWS
        proc = self._processor

        for i, qimg in enumerate(self._qimages):
            col = i % self.COLS
            row = i // self.COLS
            x = col * pw
            y = row * ph

            if qimg is not None:
                target = QRectF(x, y, pw, ph)
                source = QRectF(0, 0, qimg.width(), qimg.height())
                painter.drawImage(target, qimg, source)

            if proc is not None:
                self._draw_panel_overlay(painter, x, y, pw, ph, i, proc, for_recording=True)

            # Palette name label with shadow
            painter.setFont(self._font_label)
            label = self._palette_names[i]
            tx, ty = x + 4, y + 14
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(int(tx + 1), int(ty + 1), label)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(int(tx), int(ty), label)

        # Grid lines
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        for c in range(1, self.COLS):
            gx = int(c * pw)
            painter.drawLine(gx, 0, gx, h)
        for r in range(1, self.ROWS):
            gy = int(r * ph)
            painter.drawLine(0, gy, w, gy)

        painter.end()

        # QImage (RGB888) → numpy BGR
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # --- painting ---

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pw, ph = self._panel_geom()
        proc = self._processor

        for i, qimg in enumerate(self._qimages):
            col = i % self.COLS
            row = i // self.COLS
            x = col * pw
            y = row * ph

            if qimg is not None:
                target = QRectF(x, y, pw, ph)
                source = QRectF(0, 0, qimg.width(), qimg.height())
                painter.drawImage(target, qimg, source)

            if proc is not None:
                self._draw_panel_overlay(painter, x, y, pw, ph, i, proc)

            # Palette name label with shadow
            painter.setFont(self._font_label)
            label = self._palette_names[i]
            tx, ty = x + 4, y + 14
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(int(tx + 1), int(ty + 1), label)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(int(tx), int(ty), label)

        # Grid lines
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        for c in range(1, self.COLS):
            gx = int(c * pw)
            painter.drawLine(gx, 0, gx, self.height())
        for r in range(1, self.ROWS):
            gy = int(r * ph)
            painter.drawLine(0, gy, self.width(), gy)

        # Recording indicator
        if self._thermal._recording:
            rec_x = self.width() - 60
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rec_x, 6, 10, 10)
            painter.setPen(QColor(255, 60, 60))
            painter.setFont(self._font_small)
            painter.drawText(rec_x + 14, 15, "REC")

        painter.end()

    def _draw_panel_overlay(self, painter: QPainter, ox: float, oy: float, pw: float, ph: float, palette_idx: int, proc: FrameProcessor, for_recording: bool = False) -> None:
        """Draw all overlay elements within one panel at offset (ox, oy)."""
        if proc.rotation in (90, 270):
            sx = pw / FRAME_HEIGHT
            sy = ph / FRAME_WIDTH
        else:
            sx = pw / FRAME_WIDTH
            sy = ph / FRAME_HEIGHT

        def f2p(fx: int, fy: int) -> tuple[float, float]:
            """Frame coords to panel-absolute widget coords."""
            lx, ly = self._frame_to_panel(fx, fy, pw, ph)
            return ox + lx, oy + ly

        # --- Isotherm ---
        self._draw_panel_isotherm(painter, ox, oy, pw, ph, proc)

        # --- Center crosshair ---
        cx, cy = f2p(FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        cross = 8
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawLine(int(cx - cross), int(cy), int(cx + cross), int(cy))
        painter.drawLine(int(cx), int(cy - cross), int(cx), int(cy + cross))
        painter.setFont(self._font_temp)
        draw_text_with_bg(painter, int(cx + 8), int(cy - 5),
                          f"{proc.center_temp:.1f}°C", QColor(255, 255, 255))

        # --- Max marker ---
        mx, my = f2p(proc.max_pos[0], proc.max_pos[1])
        painter.setPen(QPen(QColor(255, 60, 60), 1))
        painter.drawLine(int(mx), int(my - 4), int(mx - 3), int(my + 3))
        painter.drawLine(int(mx - 3), int(my + 3), int(mx + 3), int(my + 3))
        painter.drawLine(int(mx + 3), int(my + 3), int(mx), int(my - 4))
        painter.setFont(self._font_maxmin)
        draw_text_with_bg(painter, int(mx + 6), int(my + 3),
                          f"MAX {proc.max_temp:.1f}°C", QColor(255, 60, 60))

        # --- Min marker ---
        nx, ny = f2p(proc.min_pos[0], proc.min_pos[1])
        painter.setPen(QPen(QColor(80, 160, 255), 1))
        painter.drawLine(int(nx), int(ny + 4), int(nx - 3), int(ny - 3))
        painter.drawLine(int(nx - 3), int(ny - 3), int(nx + 3), int(ny - 3))
        painter.drawLine(int(nx + 3), int(ny - 3), int(nx), int(ny + 4))
        painter.setFont(self._font_maxmin)
        draw_text_with_bg(painter, int(nx + 6), int(ny + 8),
                          f"MIN {proc.min_temp:.1f}°C", QColor(80, 160, 255))

        # --- Color scale bar ---
        bar_w = 12
        bar_x = int(ox + pw - bar_w - 30)
        bar_top = int(oy + 20)
        bar_bottom = int(oy + ph - 20)
        bar_h = bar_bottom - bar_top
        if bar_h > 10:
            gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
            bar_colored = apply_palette(gradient, palette_idx)
            bar_rgb = cv2.cvtColor(bar_colored, cv2.COLOR_BGR2RGB)
            bar_strip = np.repeat(bar_rgb, bar_w, axis=1)
            bar_img = QImage(bar_strip.data, bar_w, bar_h, bar_w * 3,
                             QImage.Format.Format_RGB888).copy()
            painter.drawImage(bar_x, bar_top, bar_img)
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRect(bar_x, bar_top, bar_w, bar_h)
            painter.setFont(self._font_small)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(bar_x - 3, bar_top - 3, f"{proc.max_temp:.0f}°")
            painter.drawText(bar_x - 3, bar_bottom + 10, f"{proc.min_temp:.0f}°")

        # --- Mouse cursor temperature (skip for recording) ---
        if not for_recording and proc.mouse_temp is not None and self._thermal._mouse_pos is not None:
            # Show cursor crosshair in every panel at the same frame position
            hit = self._hit_panel(*self._thermal._mouse_pos)
            if hit is not None:
                _, lx, ly = hit
                mfx, mfy = self._local_to_frame(lx, ly, pw, ph)
                mpx, mpy = f2p(mfx, mfy)
                painter.setPen(QPen(QColor(0, 255, 255), 1))
                painter.drawLine(int(mpx - 5), int(mpy), int(mpx + 5), int(mpy))
                painter.drawLine(int(mpx), int(mpy - 5), int(mpx), int(mpy + 5))
                lbl = f"{proc.mouse_temp:.1f}°C"
                lx_t = int(mpx + 8) if mpx < ox + pw - 60 else int(mpx - 60)
                ly_t = int(mpy - 4) if mpy > oy + 14 else int(mpy + 14)
                draw_text_with_bg(painter, lx_t, ly_t, lbl, QColor(0, 255, 255))

        # --- Probes ---
        if self._thermal._probes and proc._temp_map is not None:
            probe_color = QColor(255, 165, 0)
            painter.setFont(self._font_small)
            for i, (fx, fy) in enumerate(self._thermal._probes):
                if not (0 <= fx < FRAME_WIDTH and 0 <= fy < FRAME_HEIGHT):
                    continue
                temp = float(proc._temp_map[fy, fx])
                ppx, ppy = f2p(fx, fy)
                d = 3
                painter.setPen(QPen(probe_color, 1))
                painter.drawLine(int(ppx), int(ppy - d), int(ppx + d), int(ppy))
                painter.drawLine(int(ppx + d), int(ppy), int(ppx), int(ppy + d))
                painter.drawLine(int(ppx), int(ppy + d), int(ppx - d), int(ppy))
                painter.drawLine(int(ppx - d), int(ppy), int(ppx), int(ppy - d))
                lbl = f"P{i+1}:{temp:.1f}°"
                lx_t = int(ppx + 6) if ppx < ox + pw - 60 else int(ppx - 60)
                ly_t = int(ppy - 4) if ppy > oy + 14 else int(ppy + 14)
                draw_text_with_bg(painter, lx_t, ly_t, lbl, probe_color)

        # --- ROI selection ---
        sr = self._thermal._selection_rect_frame
        if sr is not None:
            fx1, fy1, fx2, fy2 = sr
            p1x, p1y = f2p(fx1, fy1)
            p2x, p2y = f2p(fx2, fy2)
            rx1, ry1 = min(p1x, p2x), min(p1y, p2y)
            rx2, ry2 = max(p1x, p2x), max(p1y, p2y)
            painter.setPen(QPen(QColor(0, 255, 100), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(0, 255, 100, 30))
            painter.drawRect(int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if self._thermal._region_stats is not None:
                s = self._thermal._region_stats
                lines = [
                    f"ROI {s['w']}x{s['h']}  Ctr:{s['center']:.1f}°",
                    f"Min:{s['min']:.1f}° Max:{s['max']:.1f}°",
                    f"Avg:{s['avg']:.1f}° \u0394T:{s['delta']:.1f}°",
                ]
                painter.setFont(self._font_small)
                color = QColor(0, 255, 100)
                lx_t = int(rx1)
                ly_t = int(ry2 + 10)
                if ly_t + 30 > oy + ph:
                    ly_t = int(ry1 - 24)
                for idx, line in enumerate(lines):
                    draw_text_with_bg(painter, lx_t, ly_t + idx * 10, line, color)

    def _draw_panel_isotherm(self, painter: QPainter, ox: float, oy: float, pw: float, ph: float, proc: FrameProcessor) -> None:
        """Draw isotherm overlay within one panel."""
        tw = self._thermal
        if not tw._isotherm_enabled or proc._temp_map is None:
            return

        temp_map = proc._temp_map
        if tw._isotherm_mode == 'above':
            mask = temp_map >= tw._isotherm_threshold
        else:
            mask = temp_map <= tw._isotherm_threshold

        if not np.any(mask):
            return

        if proc.flip:
            mask = np.flip(mask, axis=1)
        if proc.rotation == 90:
            mask = np.rot90(mask, k=-1)
        elif proc.rotation == 180:
            mask = np.rot90(mask, k=2)
        elif proc.rotation == 270:
            mask = np.rot90(mask, k=1)

        oh, ow = mask.shape
        overlay = np.zeros((oh, ow, 4), dtype=np.uint8)
        overlay[mask] = [0, 255, 100, 120]

        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255, 220), 1)

        ipw, iph = int(pw), int(ph)
        overlay_resized = cv2.resize(overlay, (ipw, iph),
                                     interpolation=cv2.INTER_NEAREST)
        qimg = QImage(overlay_resized.data, ipw, iph, ipw * 4,
                      QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(int(ox), int(oy), qimg)

    # --- mouse events → write to ThermalWidget state ---

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        wx, wy = int(pos.x()), int(pos.y())
        self._thermal._mouse_pos = (wx, wy)

        # Handle drag for selection
        if self._drag_start_frame is not None and event.buttons() & Qt.MouseButton.LeftButton:
            hit = self._hit_panel(wx, wy)
            if hit is not None:
                _, lx, ly = hit
                pw, ph = self._panel_geom()
                fx, fy = self._local_to_frame(lx, ly, pw, ph)
                sfx, sfy = self._drag_start_frame
                x1, y1 = min(sfx, fx), min(sfy, fy)
                x2, y2 = max(sfx, fx), max(sfy, fy)
                # Store frame-coord selection on ThermalWidget
                self._thermal._selection_rect_frame = (x1, y1, x2, y2)
                # Also set widget-coord selection_rect so ThermalWidget's
                # _recompute_region_stats can work (it converts from widget)
                # We'll use _selection_rect_frame directly instead.
                self._thermal._selection_rect = (x1, y1, x2, y2)  # marker that selection exists

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            hit = self._hit_panel(int(pos.x()), int(pos.y()))
            if hit is not None:
                _, lx, ly = hit
                pw, ph = self._panel_geom()
                fx, fy = self._local_to_frame(lx, ly, pw, ph)
                self._drag_start_frame = (fx, fy)
        elif event.button() == Qt.MouseButton.RightButton:
            self._thermal.clear_selection()
            self._thermal._selection_rect_frame = None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._double_click_pending:
                self._double_click_pending = False
                self._drag_start_frame = None
                return
            if self._drag_start_frame is not None:
                pos = event.position()
                hit = self._hit_panel(int(pos.x()), int(pos.y()))
                if hit is not None:
                    _, lx, ly = hit
                    pw, ph = self._panel_geom()
                    fx, fy = self._local_to_frame(lx, ly, pw, ph)
                    sfx, sfy = self._drag_start_frame
                    x1, y1 = min(sfx, fx), min(sfy, fy)
                    x2, y2 = max(sfx, fx), max(sfy, fy)
                    if x2 - x1 > 1 and y2 - y1 > 1:
                        # Selection drag
                        self._thermal._selection_rect_frame = (x1, y1, x2, y2)
                        self._thermal._selection_rect = (x1, y1, x2, y2)
                        self._recompute_region_stats_frame()
                    else:
                        # Click → place probe (keep existing ROI)
                        if len(self._thermal._probes) < self._thermal.MAX_PROBES:
                            self._thermal._probes.append((sfx, sfy))
                self._drag_start_frame = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._thermal._probes:
            # Undo the probe placed by the first click of the double-click first
            self._thermal._probes.pop()
            pos = event.position()
            hit = self._hit_panel(int(pos.x()), int(pos.y()))
            if hit is not None and self._thermal._probes:
                _, lx, ly = hit
                pw, ph = self._panel_geom()
                fx, fy = self._local_to_frame(lx, ly, pw, ph)
                # Remove nearest probe (in frame coords)
                best_i, best_d = 0, float('inf')
                for i, (px, py) in enumerate(self._thermal._probes):
                    d = (px - fx) ** 2 + (py - fy) ** 2
                    if d < best_d:
                        best_d = d
                        best_i = i
                self._thermal._probes.pop(best_i)
            self._double_click_pending = True

    def leaveEvent(self, event: QMouseEvent) -> None:
        self._thermal._mouse_pos = None
        if self._processor:
            self._processor.mouse_temp = None
            self._processor.mouse_pos = None

    def _recompute_region_stats_frame(self) -> None:
        """Compute region stats from frame-coord selection."""
        sr = getattr(self._thermal, '_selection_rect_frame', None)
        proc = self._processor
        if sr is None or proc is None:
            self._thermal._region_stats = None
            return
        fx1, fy1, fx2, fy2 = sr
        region = proc.get_region_temps(fx1, fy1, fx2, fy2)
        self._thermal._region_stats = _compute_region_stats(region)

    def sizeHint(self) -> QSize:
        return QSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)


class CollapsibleSection(QWidget):
    """A section with a clickable header that toggles content visibility."""

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle_btn = QToolButton()
        self._toggle_btn.setText(title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(expanded)
        self._toggle_btn.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }"
        )
        self._toggle_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle_btn.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(4)
        self._content.setVisible(expanded)

        sp = self._content.sizePolicy()
        sp.setRetainSizeWhenHidden(False)
        self._content.setSizePolicy(sp)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle_btn)
        layout.addWidget(self._content)

        sp = self.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.setSizePolicy(sp)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
