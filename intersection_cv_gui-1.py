"""
IntersectionCV v5 — Two-Road Delay Comparison
Saheb Bazar – Alupotti Rd, Rajshahi · YOLOv8 / YOLO26 + ByteTrack · PyQt6
"""
import sys, csv, json, time, random, math, os
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QProgressBar, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QGroupBox, QScrollArea, QLineEdit,
    QTextEdit, QTabWidget, QCheckBox, QStatusBar, QSizePolicy,
    QStackedWidget, QDialog, QDialogButtonBox, QMessageBox,
    QAbstractItemView, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import (
    Qt, QThread, QTimer, QRect, QRectF, QPointF, pyqtSignal,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont,
    QAction, QKeySequence, QLinearGradient, QCursor,
)

# ──────────────────────────────────────────────────────────────────────
# TOKENS
# ──────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#090B0F",
    "s0":      "#0E1117",
    "s1":      "#141921",
    "s2":      "#1A2030",
    "s3":      "#202638",
    "bd":      "#232840",
    "bd2":     "#2E3450",
    "acc":     "#00C8EE",   # cyan signature
    "acc_d":   "#007A99",
    "ok":      "#22D06E",
    "warn":    "#F5A623",
    "danger":  "#F04060",
    "tp":      "#E6EAF8",
    "ts":      "#5E6A8A",
    "td":      "#303650",
    "dir0":    "#B06EF0",   # left   – purple
    "dir1":    "#F5882A",   # right  – amber
    "dir2":    "#00C8EE",   # straight – cyan
    "v0":  "#00C8EE", "v1": "#8B7EF8", "v2": "#F46060",
    "v3":  "#F5A623", "v4": "#28D09A", "v5": "#F472C0",
    "v6":  "#38BCFA", "v7": "#A0E040", "v8": "#F5882A",
    "v9":  "#B06EF0", "v10":"#60D8F0",
}
VEHICLE_CLASSES = [
    "Buses","Micro Buses","Trucks","Mini Trucks",
    "Private Cars","Human Hollar","Bi-Cycles",
    "Motor Cycles","Rickshaw","Auto Rickshaws","Van",
]
MODEL_SIZES = [
    # YOLOv8 (stable, BNVD weights available)
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    # YOLO26 (retrain target — use custom .pt once trained)
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
]

# ──────────────────────────────────────────────────────────────────────
# TRAFFIC DELAY PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# No calibration bars, no manual km/h baseline. Each ROI zone learns its
# own free-flow baseline from the fastest crossings it actually observes
# (a rolling low percentile of real crossing times — the same idea
# traffic engineers use to estimate free-flow speed empirically). Motion/
# stop thresholds scale to each zone's own on-screen size (its bounding
# diagonal in normalized coordinates), so nothing needs real-world units.
DEFAULT_STOP_FRAC_PER_SEC   = 0.035  # below this fraction of zone-diagonal/sec = stopped
DEFAULT_MOTION_FLOOR_FRAC   = 0.004  # per-frame motion below this fraction = box jitter, not travel
DEFAULT_GRACE_SEC           = 1.5    # track-loss grace before an in-zone occupant is finalised
DEFAULT_BUCKET_SEC          = 15.0   # timeline chart bucket width, in video-seconds
DEFAULT_SPEED_WINDOW_SEC    = 0.4    # rolling window for speed estimation — bigger = smoother
DEFAULT_FREEFLOW_PERCENTILE = 0.15   # bottom 15% of crossing times = this zone's free-flow baseline
MIN_BASELINE_SAMPLES        = 5      # crossings needed before the percentile is trusted
DELAY_FLAG_SEC               = 5.0    # travel delay above this counts toward "% vehicles delayed"
DEFAULT_LOITER_SEC           = 20.0   # a pedestrian dwelling this long reads as a hawker/vendor, not a passer-by

# Commonly-cited approximate Passenger Car Unit (PCU) factors for mixed
# South-Asian traffic — a starting point for converting a mixed vehicle
# stream into one comparable throughput number, NOT a precise locally-
# calibrated standard. Adjust these if you have locally-derived values.
PCU_FACTORS = {
    "Buses": 3.0, "Micro Buses": 1.5, "Trucks": 3.0, "Mini Trucks": 2.0,
    "Private Cars": 1.0, "Human Hollar": 2.0, "Bi-Cycles": 0.5,
    "Motor Cycles": 0.5, "Rickshaw": 1.0, "Auto Rickshaws": 0.75, "Van": 1.5,
}

def percentile(sorted_values, pct):
    """pct in [0,100]. sorted_values must already be sorted ascending."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

def stdev(values):
    """Population standard deviation — used as 'acceleration noise': how
    much a vehicle's acceleration fluctuates over its crossing. Near zero
    for steady travel; large for stop-and-go, jerky motion. Classic
    car-following-theory measure of traffic stream disruption."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5

# Congestion bands by average travel delay per vehicle (seconds). Rough,
# HCM-inspired bands for an unsignalised mixed-traffic approach — treat
# these as a starting point; they adapt automatically as each zone's own
# free-flow baseline is learned from real traffic.
CONGESTION_LEVELS = [
    (5.0,          "FREE FLOW", C["ok"]),
    (15.0,         "LIGHT",     C["acc"]),
    (30.0,         "MODERATE",  C["warn"]),
    (60.0,         "HEAVY",     C["dir1"]),
    (float("inf"), "SEVERE",    C["danger"]),
]

def classify_congestion(avg_delay_s):
    for limit, label, color in CONGESTION_LEVELS:
        if avg_delay_s <= limit:
            return label, color
    return CONGESTION_LEVELS[-1][1], CONGESTION_LEVELS[-1][2]

# Volume/capacity bands — a SEPARATE diagnostic from the delay-based
# congestion level above: this one only fires when a capacity estimate is
# supplied, and tells you whether a road is jammed because it's genuinely
# over capacity vs. jammed despite modest volume (friction — parking,
# pedestrians, encroachment). HCM-style V/C thresholds.
VC_LEVELS = [
    (0.60, "UNDER CAPACITY", C["ok"]),
    (0.75, "MODERATE LOAD",  C["acc"]),
    (0.85, "HIGH LOAD",      C["warn"]),
    (0.95, "NEAR CAPACITY",  C["dir1"]),
    (float("inf"), "OVER CAPACITY", C["danger"]),
]

def classify_vc(vc_ratio):
    if vc_ratio is None:
        return None, C["ts"]
    for limit, label, color in VC_LEVELS:
        if vc_ratio <= limit:
            return label, color
    return VC_LEVELS[-1][1], VC_LEVELS[-1][2]

# ──────────────────────────────────────────────────────────────────────
# STYLESHEET
# ──────────────────────────────────────────────────────────────────────
QSS = f"""
* {{
    font-family: 'Segoe UI', 'Inter', 'SF Pro Text', sans-serif;
    font-size: 12px;
    color: {C['tp']};
}}
QMainWindow, QDialog {{ background: {C['bg']}; }}
QWidget {{ background: transparent; }}
QScrollArea {{ border: none; }}

/* scrollbars */
QScrollBar:vertical   {{ background: transparent; width: 5px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 5px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {C['bd']}; border-radius: 2px; min-height: 20px; }}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {C['bd2']}; }}
QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {{ height:0; width:0; }}

QStatusBar {{
    background: {C['bg']}; border-top: 1px solid {C['s1']};
    color: {C['td']}; font-size: 10px; padding: 0 12px; }}
QToolTip {{
    background: {C['s2']}; border: 1px solid {C['bd']};
    color: {C['tp']}; padding: 5px 8px; border-radius: 6px; }}

/* inputs */
QLineEdit, QSpinBox {{
    background: {C['s0']}; border: 1px solid {C['bd']};
    border-radius: 7px; padding: 7px 10px; color: {C['tp']}; }}
QLineEdit:hover, QSpinBox:hover  {{ border-color: {C['bd2']}; }}
QLineEdit:focus, QSpinBox:focus  {{ border-color: {C['acc']}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width:0; border:none; }}

QComboBox {{
    background: {C['s0']}; border: 1px solid {C['bd']};
    border-radius: 7px; padding: 7px 28px 7px 10px; color: {C['tp']}; }}
QComboBox:hover  {{ border-color: {C['bd2']}; }}
QComboBox:focus  {{ border-color: {C['acc']}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: none; width: 0; }}
QComboBox QAbstractItemView {{
    background: {C['s2']}; border: 1px solid {C['bd']};
    color: {C['tp']}; outline: none;
    selection-background-color: {C['s3']}; }}

/* sliders */
QSlider::groove:horizontal {{
    background: {C['s2']}; height: 3px; border-radius: 1px; }}
QSlider::handle:horizontal {{
    background: {C['acc']}; width: 13px; height: 13px;
    margin: -5px 0; border-radius: 6px; }}
QSlider::handle:horizontal:hover {{ background: #33D8FF; }}
QSlider::sub-page:horizontal {{ background: {C['acc']}; border-radius: 1px; }}

/* progress */
QProgressBar {{
    background: {C['s1']}; border: none; border-radius: 3px;
    height: 4px; color: transparent; }}
QProgressBar::chunk {{ background: {C['acc']}; border-radius: 3px; }}

/* table */
QTableWidget {{
    background: {C['s0']}; border: none;
    gridline-color: {C['s1']}; color: {C['tp']}; }}
QTableWidget::item {{ padding: 4px 8px; border: none; }}
QTableWidget::item:selected {{ background: {C['s2']}; color: {C['acc']}; }}
QHeaderView::section {{
    background: {C['s0']}; border: none;
    border-bottom: 1px solid {C['s1']};
    border-right: 1px solid {C['s1']};
    padding: 6px 8px; color: {C['ts']};
    font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }}

/* group box */
QGroupBox {{
    border: 1px solid {C['s2']}; border-radius: 10px;
    margin-top: 14px; padding-top: 10px;
    color: {C['ts']}; font-size: 10px;
    font-weight: 600; letter-spacing: 0.8px; }}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 5px;
    background: {C['bg']}; text-transform: uppercase; }}

/* tabs */
QTabWidget::pane {{
    background: {C['s0']}; border: 1px solid {C['bd']};
    border-radius: 10px; margin-top: 0; }}
QTabBar::tab {{
    background: transparent; border: none;
    padding: 8px 18px; color: {C['ts']};
    border-bottom: 2px solid transparent; margin-right: 4px; }}
QTabBar::tab:selected  {{ color: {C['acc']}; border-bottom-color: {C['acc']}; }}
QTabBar::tab:hover:!selected {{ color: {C['tp']}; }}

/* checkboxes */
QCheckBox {{ color: {C['ts']}; spacing: 7px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {C['bd']};
    border-radius: 4px; background: {C['s0']}; }}
QCheckBox::indicator:checked {{ background: {C['acc']}; border-color: {C['acc']}; }}
QCheckBox:hover {{ color: {C['tp']}; }}

/* ── buttons ── */
QPushButton {{
    border-radius: 8px; font-size: 12px; font-weight: 600;
    padding: 8px 16px; border: none; }}
QPushButton:disabled {{ color: {C['td']}; background: {C['s1']}; border-color: transparent; }}

QPushButton#btn_primary {{
    background: {C['acc']}; color: {C['bg']}; }}
QPushButton#btn_primary:hover  {{ background: #33D8FF; }}
QPushButton#btn_primary:pressed {{ background: {C['acc_d']}; }}

QPushButton#btn_ghost {{
    background: transparent; border: 1px solid {C['bd']}; color: {C['ts']}; }}
QPushButton#btn_ghost:hover {{
    border-color: {C['acc']}44; color: {C['tp']}; background: {C['s2']}; }}

QPushButton#btn_danger {{
    background: {C['s1']}; border: 1px solid {C['danger']}55; color: {C['danger']}; }}
QPushButton#btn_danger:hover {{ background: {C['s2']}; border-color: {C['danger']}; }}

QPushButton#btn_icon {{
    background: {C['s1']}; border: 1px solid {C['bd']};
    color: {C['ts']}; padding: 6px 10px; border-radius: 7px; }}
QPushButton#btn_icon:hover {{
    border-color: {C['acc']}55; color: {C['acc']}; background: {C['s2']}; }}

QPushButton#btn_nav {{
    background: transparent; border: none; border-radius: 8px;
    text-align: left; color: {C['ts']};
    padding: 10px 14px; font-weight: 500; font-size: 13px; }}
QPushButton#btn_nav:hover {{ background: {C['s1']}; color: {C['tp']}; }}
QPushButton#btn_nav[active=true] {{
    background: {C['s1']}; color: {C['acc']};
    border-left: 2px solid {C['acc']};
    border-radius: 0 8px 8px 0; padding-left: 12px; }}
"""

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _lbl(text):
    """Tiny uppercase section label."""
    w = QLabel(text.upper())
    w.setStyleSheet(
        f"color:{C['ts']}; font-size:9px; font-weight:700; "
        f"letter-spacing:1px; padding:0; margin:0;")
    return w

def _card():
    """Standard dark card frame."""
    f = QFrame()
    f.setStyleSheet(f"""
        QFrame {{
            background: {C['s0']};
            border: 1px solid {C['bd']};
            border-radius: 11px;
        }}
    """)
    return f

def _sep():
    s = QFrame()
    s.setFrameShape(QFrame.Shape.HLine)
    s.setStyleSheet(f"background:{C['s1']}; border:none;")
    s.setFixedHeight(1)
    return s

# ──────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ──────────────────────────────────────────────────────────────────────

class UsageMeter(QWidget):
    """
    Arc usage meter – three concentric half-arcs for CPU / RAM / GPU.
    FPS shown in the centre. Lives above the stat cards.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(92)
        self._cpu = self._ram = self._gpu = 0.0
        self._tcpu = self._tram = self._tgpu = 0.0
        self._fps  = 0.0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_values(self, cpu, ram, gpu, fps=0.0):
        self._tcpu = cpu; self._tram = ram
        self._tgpu = gpu; self._fps  = fps

    def _tick(self):
        s = 0.10
        self._cpu += (self._tcpu - self._cpu) * s
        self._ram += (self._tram - self._ram) * s
        self._gpu += (self._tgpu - self._gpu) * s
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()
        cx = w // 2; cy = h - 8

        rings = [
            (self._cpu, C["acc"],  44, "CPU"),
            (self._ram, C["dir0"], 30, "MEM"),
            (self._gpu, C["ok"],   16, "GPU"),
        ]
        for val, color, r, tag in rings:
            qc    = QColor(color)
            track = QColor(qc.red(), qc.green(), qc.blue(), 22)
            rect  = QRectF(cx-r, cy-r, r*2, r*2)
            p.setPen(QPen(track, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 180*16, -180*16)
            span = int(-180*16 * max(0, min(1, val/100)))
            if span:
                p.setPen(QPen(qc, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawArc(rect, 180*16, span)
            # tag at far right of arc
            p.setPen(QPen(QColor(C["ts"])))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(int(cx+r+6), int(cy-r+6), tag)

        # FPS centre
        p.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        p.setPen(QPen(QColor(C["tp"])))
        fps_s = f"{self._fps:.0f}"
        fm    = p.fontMetrics()
        p.drawText(cx - fm.horizontalAdvance(fps_s)//2, cy - 2, fps_s)
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QPen(QColor(C["ts"])))
        p.drawText(cx - 8, cy + 14, "fps")

        # bottom % labels
        for val, color, _, offset in [
            (self._cpu, C["acc"],  None, -34),
            (self._ram, C["dir0"], None,   0),
            (self._gpu, C["ok"],  None,  34),
        ]:
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.setPen(QPen(QColor(color)))
            txt = f"{val:.0f}%"
            p.drawText(int(cx + offset - 14), int(cy + 26), txt)


class StatCard(QFrame):
    """Compact stat card with coloured accent line."""
    def __init__(self, title, value="0", subtitle="", color=C["acc"], parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedHeight(80)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(3)

        top = QHBoxLayout()
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(
            f"color:{C['ts']}; font-size:9px; font-weight:700; letter-spacing:0.7px;")
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{color}; font-size:7px;")
        top.addWidget(dot); top.addWidget(lbl); top.addStretch()
        lay.addLayout(top)

        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(
            f"color:{color}; font-size:24px; font-weight:700;")
        lay.addWidget(self.val_lbl)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color:{C['td']}; font-size:10px;")
            lay.addWidget(sub)

    def set_value(self, v):
        self.val_lbl.setText(str(v))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        c = QColor(self._color)
        bg = QColor(c.red(), c.green(), c.blue(), 14)
        bd = QColor(c.red(), c.green(), c.blue(), 45)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(bd, 1))
        p.drawRoundedRect(r.adjusted(0, 0, -1, -1), 10, 10)
        # accent left bar
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawRoundedRect(QRectF(0, 18, 3, r.height()-36), 1.5, 1.5)


class PulsingDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = QColor(C["td"])
        self._alpha = 255; self._dir = -7
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_color(self, hex_c):
        self._color = QColor(hex_c)

    def _tick(self):
        self._alpha += self._dir
        if self._alpha <= 60 or self._alpha >= 255: self._dir = -self._dir
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self._color.red(), self._color.green(),
                   self._color.blue(), self._alpha)
        p.setBrush(QBrush(c)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(1, 1, 8, 8))


class TimelineChart(QWidget):
    """
    Line chart of a zone's live "flow index" (0-100%, where 100% = as
    fast as that zone's own learned free-flow pace) per time bucket.
    Because it's a percentage of each zone's own baseline rather than an
    absolute speed, Zone A's and Zone B's charts stay directly
    comparable even if the two roads are completely different lengths
    or shapes. Queue depth is shown as faint bars underneath the line.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._buckets = []

    def update_data(self, buckets):
        self._buckets = buckets or []
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(C["s0"]))
        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 8, 8, 14, 16
        plot_w = max(1, w - pad_l - pad_r)
        plot_h = max(1, h - pad_t - pad_b)

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(C["ts"])))
        p.drawText(QRectF(pad_l, 0, 240, 12), Qt.AlignmentFlag.AlignLeft,
                   "100% = this zone's own free-flow pace")

        if len(self._buckets) < 2:
            p.setPen(QPen(QColor(C["td"])))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Timeline fills in once analysis starts…")
            return

        max_val = 100.0
        max_queue = max(1.0, max(b.get("avg_queue", 0) for b in self._buckets))
        n = len(self._buckets)

        def xf(i): return pad_l + (i / (n - 1)) * plot_w
        def yf(v): return pad_t + plot_h - (min(v, max_val) / max_val) * plot_h

        # queue bars (faint, behind the line)
        bw = plot_w / n
        for i, b in enumerate(self._buckets):
            qv = b.get("avg_queue", 0)
            bh = (qv / max_queue) * plot_h * 0.55
            bx = pad_l + i * bw
            qc = QColor(C["danger"])
            qc.setAlpha(24 if qv <= 3 else 44)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qc))
            p.drawRect(QRectF(bx, pad_t + plot_h - bh, max(1, bw - 1), bh))

        # 100%-of-baseline reference line (dashed)
        ffy = yf(100.0)
        pen = QPen(QColor(C["ts"]), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(pad_l, ffy), QPointF(pad_l + plot_w, ffy))

        # flow-index line, colour-graded per segment
        for i in range(n - 1):
            v0 = self._buckets[i]["flow_index"]
            v1 = self._buckets[i + 1]["flow_index"]
            ratio = ((v0 + v1) / 2) / 100.0
            if ratio >= 0.7:   col = QColor(C["ok"])
            elif ratio >= 0.4: col = QColor(C["warn"])
            else:              col = QColor(C["danger"])
            p.setPen(QPen(col, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(xf(i), yf(v0)), QPointF(xf(i + 1), yf(v1)))

        # axis time labels
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(C["ts"])))
        p.drawText(QRectF(pad_l, pad_t + plot_h + 2, 80, 12),
                   Qt.AlignmentFlag.AlignLeft, "0:00")
        total_s = self._buckets[-1]["t1"]
        mm, ss = int(total_s // 60), int(total_s % 60)
        p.drawText(QRectF(w - pad_r - 60, pad_t + plot_h + 2, 60, 12),
                   Qt.AlignmentFlag.AlignRight, f"{mm}:{ss:02d}")


class ComparisonPanel(QWidget):
    """Side-by-side Zone A vs Zone B snapshot with a plain-language read
    on which road is currently more delayed — the headline comparison
    view, always visible above the per-zone boards."""
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        row = QHBoxLayout(); row.setSpacing(10)
        self.col_a = self._make_col("A")
        self.col_b = self._make_col("B")
        row.addLayout(self.col_a["layout"], 1)
        row.addLayout(self.col_b["layout"], 1)
        lay.addLayout(row)

        self.narrative = QLabel("Waiting for data from both roads…")
        self.narrative.setStyleSheet(f"color:{C['tp']}; font-size:12px; font-weight:600;")
        self.narrative.setWordWrap(True)
        lay.addWidget(self.narrative)

        self.ped_note = QLabel("")
        self.ped_note.setStyleSheet(f"color:{C['ts']}; font-size:11px;")
        self.ped_note.setWordWrap(True)
        lay.addWidget(self.ped_note)

    def _make_col(self, zone):
        color = VideoCanvas.ZONE_COLORS.get(zone, C["acc"])
        v = QVBoxLayout(); v.setSpacing(2)
        name = QLabel(VideoCanvas.ZONE_LABELS.get(zone, zone))
        name.setStyleSheet(f"color:{color}; font-size:12px; font-weight:700;")
        v.addWidget(name)
        delay_lbl = QLabel("— s avg delay")
        delay_lbl.setStyleSheet(f"color:{C['tp']}; font-size:18px; font-weight:700;")
        v.addWidget(delay_lbl)
        sub_lbl = QLabel("0 vehicles · queue 0")
        sub_lbl.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        v.addWidget(sub_lbl)
        return {"layout": v, "delay": delay_lbl, "sub": sub_lbl, "name": name}

    def update_data(self, snap):
        zones = snap.get("zones", {})
        for z, col in (("A", self.col_a), ("B", self.col_b)):
            zi = zones.get(z)
            if not zi:
                continue
            col["delay"].setText(f'{zi["avg_delay"]:.1f}s avg delay')
            ped = zi.get("pedestrians", {})
            col["sub"].setText(
                f'{zi["vehicles"]} vehicles · queue {zi["queue_now"]} · {zi["congestion"][0]}  ·  '
                f'{ped.get("loitering_now", 0)} hawkers')

        cmp = snap.get("comparison", {})
        more = cmp.get("more_delayed_zone")
        diff = abs(cmp.get("delay_diff_s", 0.0))
        if more and zones.get(more):
            other = "B" if more == "A" else "A"
            more_label  = zones[more]["label"]
            other_label = zones.get(other, {}).get("label", other)
            self.narrative.setText(
                f"{more_label} is currently experiencing about {diff:.1f}s more "
                f"delay per vehicle than {other_label}.")
        elif zones.get("A") and zones.get("B"):
            self.narrative.setText("Both roads are running about the same right now.")
        else:
            self.narrative.setText("Waiting for data from both roads…")

        ped_diff = cmp.get("pedestrian_diff")
        if ped_diff is not None and zones.get("A") and zones.get("B"):
            if abs(ped_diff) < 1:
                self.ped_note.setText("Hawker/pedestrian activity is similar on both roads.")
            else:
                busier = "A" if ped_diff > 0 else "B"
                self.ped_note.setText(
                    f"{zones[busier]['label']} also has more hawker/pedestrian "
                    f"activity right now ({abs(ped_diff)} more loitering).")
        else:
            self.ped_note.setText("")

class ClassDelayTable(QTableWidget):
    """Rows = vehicle class, columns = count / avg travel delay / avg
    stopped time — replaces the old per-direction volume grid."""
    COLS = ["Count", "Avg Delay (s)", "Avg Stopped (s)"]

    def __init__(self, parent=None):
        super().__init__(len(VEHICLE_CLASSES), len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.setVerticalHeaderLabels(VEHICLE_CLASSES)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setDefaultSectionSize(27)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._reset()

    def _reset(self):
        for r, cls in enumerate(VEHICLE_CLASSES):
            for c in range(len(self.COLS)):
                item = QTableWidgetItem("0" if c == 0 else "—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QBrush(QColor(C[f"v{r}"])))
                item.setFont(QFont("Consolas", 10))
                self.setItem(r, c, item)

    def update_data(self, class_summary):
        class_summary = class_summary or {}
        for r, cls in enumerate(VEHICLE_CLASSES):
            st = class_summary.get(cls, {})
            count = st.get("count", 0)
            avg_delay   = (st.get("delay", 0.0) / count) if count else 0.0
            avg_stopped = (st.get("stopped", 0.0) / count) if count else 0.0
            vals = [str(count),
                    f"{avg_delay:.1f}" if count else "—",
                    f"{avg_stopped:.1f}" if count else "—"]
            for c, v in enumerate(vals):
                item = self.item(r, c)
                if item:
                    item.setText(v)
                    item.setFont(QFont("Consolas", 10,
                        QFont.Weight.Bold if count else QFont.Weight.Normal))

    def get_totals(self):
        totals = {}
        for r in range(self.rowCount()):
            cls = VEHICLE_CLASSES[r]
            totals[cls] = self.item(r, 0).text() if self.item(r, 0) else "0"
        return totals

    def clear_all(self):
        self._reset()


ROI_HANDLE_R = 7   # px radius for a grabbable ROI corner handle


class VideoCanvas(QWidget):
    """
    Custom widget that owns its entire paint loop.
    Draws: background → video frame (or idle text) → two independently
    draggable/resizable 6-point ROI zones (always on top).

    Both zones are stored in NORMALIZED coordinates (0..1 relative to the
    video frame), so they stay correct across window resizes and
    different video resolutions. Zone "A" and zone "B" are meant to frame
    two different roads/approaches in the same camera view, so their
    delay can be measured and compared against each other.
    """
    file_dropped   = pyqtSignal(str)
    layout_changed = pyqtSignal()   # emitted whenever a zone is dragged

    ZONE_COLORS = {"A": "#00C8EE", "B": "#F0A030"}
    ZONE_LABELS = {"A": "Road A", "B": "Road B"}

    # Generic starting hexagons on opposite sides of frame — drag the 6
    # corners of each to fit whichever two roads/approaches you want to
    # compare in your own camera view.
    ROI_A_DEFAULT = [
        (0.04, 0.55), (0.14, 0.20), (0.34, 0.12),
        (0.46, 0.32), (0.38, 0.80), (0.16, 0.95),
    ]
    ROI_B_DEFAULT = [
        (0.62, 0.30), (0.72, 0.12), (0.92, 0.18),
        (0.97, 0.55), (0.84, 0.92), (0.60, 0.82),
    ]

    # Optional single reference line per zone — NOT required. Enable it
    # only if you want queue length reported in metres instead of just a
    # vehicle count: drag its 2 endpoints onto something of known real
    # length in that zone (a lane width, a shopfront, a parked vehicle of
    # known length) and enter that length. Leave disabled and everything
    # else still works exactly the same, purely in relative/normalized terms.
    REF_DEFAULT = {
        "A": {"enabled": False, "p1": (0.10, 0.85), "p2": (0.30, 0.85), "length_m": 5.0},
        "B": {"enabled": False, "p1": (0.68, 0.85), "p2": (0.88, 0.85), "length_m": 5.0},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self._pixmap  = None          # current video frame
        self._label   = "Drop a video here  ·  or click Browse"
        self._sublabel= ""

        self._rois = {
            "A": [tuple(p) for p in self.ROI_A_DEFAULT],
            "B": [tuple(p) for p in self.ROI_B_DEFAULT],
        }
        self._refs = {z: dict(v) for z, v in self.REF_DEFAULT.items()}
        self._drag   = None    # ("roi", zone, idx) | ("ref", zone, "p1"/"p2")
        self._hover  = None
        self._editable = True  # allow dragging handles on this canvas

        self.setStyleSheet("")        # no Qt stylesheet interference

    def set_loaded(self, name):
        self._pixmap   = None
        self._label    = f"📹  {name}"
        self._sublabel = "Click  ▶ Run Analysis  to start"
        self.update()

    def set_frame(self, px: QPixmap):
        self._pixmap = px
        self.update()

    # ── ROI state ────────────────────────────────────────────────────
    def set_roi(self, zone, corners):
        self._rois[zone] = [tuple(p) for p in corners]
        self.update()

    def set_rois(self, rois):
        for zone, corners in rois.items():
            self._rois[zone] = [tuple(p) for p in corners]
        self.update()

    def get_roi(self, zone):
        return [tuple(p) for p in self._rois[zone]]

    def get_rois(self):
        return {z: [tuple(p) for p in pts] for z, pts in self._rois.items()}

    # ── optional reference-line state (physical queue length) ──────────
    def set_ref(self, zone, ref):
        self._refs[zone] = dict(ref)
        self.update()

    def set_refs(self, refs):
        for zone, ref in refs.items():
            self._refs[zone] = dict(ref)
        self.update()

    def get_ref(self, zone):
        return dict(self._refs[zone])

    def get_refs(self):
        return {z: dict(r) for z, r in self._refs.items()}

    # ── frame ↔ widget coordinate mapping ───────────────────────────────
    def _frame_geom(self):
        """Returns (ox, oy, w, h) — the top-left offset and size of the
        drawable video area within this widget (accounts for letterboxing
        when the video's aspect ratio differs from the widget's)."""
        w, h = self.width(), self.height()
        if self._pixmap is not None and not self._pixmap.isNull():
            pw, ph = self._pixmap.width(), self._pixmap.height()
            if pw > 0 and ph > 0:
                scale = min(w / pw, h / ph)
                sw, sh = pw * scale, ph * scale
            else:
                sw, sh = w, h
        else:
            sw, sh = w, h
        ox = (w - sw) / 2
        oy = (h - sh) / 2
        return ox, oy, sw, sh

    def _to_widget(self, nx, ny):
        ox, oy, w, h = self._frame_geom()
        return ox + nx * w, oy + ny * h

    def _to_norm(self, px, py):
        ox, oy, w, h = self._frame_geom()
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        nx = (px - ox) / w
        ny = (py - oy) / h
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ── mouse handling: drag ROI corners or reference-line endpoints ────
    def _hit_test(self, pos):
        px, py = pos.x(), pos.y()
        for zone, pts in self._rois.items():
            for i, (nx, ny) in enumerate(pts):
                wx, wy = self._to_widget(nx, ny)
                if (px - wx) ** 2 + (py - wy) ** 2 <= (ROI_HANDLE_R + 5) ** 2:
                    return ("roi", zone, i)
        for zone, ref in self._refs.items():
            if not ref.get("enabled"):
                continue
            for ep in ("p1", "p2"):
                nx, ny = ref[ep]
                wx, wy = self._to_widget(nx, ny)
                if (px - wx) ** 2 + (py - wy) ** 2 <= (ROI_HANDLE_R + 5) ** 2:
                    return ("ref", zone, ep)
        return None

    def mousePressEvent(self, e):
        if self._editable and e.button() == Qt.MouseButton.LeftButton:
            pos = e.position().toPoint()
            hit = self._hit_test(pos)
            if hit:
                self._drag = hit
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if self._editable and self._drag:
            nx, ny = self._to_norm(pos.x(), pos.y())
            kind, zone, key = self._drag
            if kind == "roi":
                self._rois[zone][key] = (nx, ny)
            else:
                self._refs[zone][key] = (nx, ny)
            self.update()
            self.layout_changed.emit()
        elif self._editable:
            hit = self._hit_test(pos)
            if hit != self._hover:
                self._hover = hit
                self.setCursor(
                    QCursor(Qt.CursorShape.OpenHandCursor) if hit
                    else QCursor(Qt.CursorShape.ArrowCursor))
                self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.layout_changed.emit()
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        if self._hover is not None:
            self._hover = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.update()
        super().leaveEvent(e)

    # ── drag-drop (files) ────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm')):
                self.file_dropped.emit(path)

    # ── paint ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = QRectF(0, 0, w, h)

        # 1. Background
        p.fillRect(r, QColor(C["s0"]))

        # 2. Border
        if self._pixmap is None:
            pen = QPen(QColor(C["bd2"]), 1.5, Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])
        else:
            pen = QPen(QColor(C["bd"]), 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r.adjusted(1,1,-1,-1), 11, 11)

        # 3. Video frame or idle text
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            ox = (w - scaled.width())  // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)
            self._draw_rois(p, ox, oy, scaled.width(), scaled.height())
        else:
            # idle / loaded state — draw centred text
            p.setPen(QPen(QColor(C["ts"])))
            p.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
            p.drawText(QRectF(0, h//2 - 28, w, 32),
                       Qt.AlignmentFlag.AlignCenter, self._label)
            if self._sublabel:
                p.setFont(QFont("Segoe UI", 11))
                p.setPen(QPen(QColor(C["td"])))
                p.drawText(QRectF(0, h//2 + 8, w, 24),
                           Qt.AlignmentFlag.AlignCenter, self._sublabel)
            # always draw the zones, even on idle/loaded state, so the
            # user can position them before a video is even loaded
            self._draw_rois(p, 0, 0, w, h)

    def _draw_rois(self, p, ox, oy, w, h):
        """Draw both 6-point ROI zones (translucent fill + border +
        draggable corner handles). ox,oy = top-left offset of the
        drawable video area within the widget; w,h = its size."""
        from PyQt6.QtGui import QPolygonF

        for zone, pts in self._rois.items():
            color = QColor(self.ZONE_COLORS[zone])
            poly_pts = [QPointF(ox + nx * w, oy + ny * h) for nx, ny in pts]
            poly = QPolygonF(poly_pts)
            fill = QColor(color.red(), color.green(), color.blue(), 26)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPolygon(poly)

            for i, pt in enumerate(poly_pts):
                hovered  = self._hover == ("roi", zone, i)
                dragging = self._drag  == ("roi", zone, i)
                r = ROI_HANDLE_R + (3 if (hovered or dragging) else 0)
                p.setPen(QPen(QColor("#08090C"), 1.5))
                p.setBrush(QBrush(QColor("#FFFFFF") if dragging else color))
                p.drawEllipse(pt, r, r)
                p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                p.setPen(QPen(color))
                p.drawText(QRectF(pt.x() + r + 3, pt.y() - 8, 26, 16),
                           Qt.AlignmentFlag.AlignLeft, f"{zone}{i+1}")

            # zone label badge at its centroid
            cxs = [pt.x() for pt in poly_pts]; cys = [pt.y() for pt in poly_pts]
            cx, cy = sum(cxs) / len(cxs), sum(cys) / len(cys)
            label = self.ZONE_LABELS.get(zone, zone)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 14
            th = fm.height() + 8
            bg = QColor(color.red(), color.green(), color.blue(), 210)
            p.setBrush(QBrush(bg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(cx - tw / 2, cy - th / 2, tw, th), 5, 5)
            p.setPen(QPen(QColor("#08090C")))
            p.drawText(QRectF(cx - tw / 2, cy - th / 2, tw, th),
                       Qt.AlignmentFlag.AlignCenter, label)

            # optional reference line (only if the user enabled one, for
            # reporting physical queue length in metres)
            ref = self._refs.get(zone, {})
            if ref.get("enabled"):
                p1n, p2n = ref["p1"], ref["p2"]
                x1, y1 = ox + p1n[0] * w, oy + p1n[1] * h
                x2, y2 = ox + p2n[0] * w, oy + p2n[1] * h
                pen = QPen(color, 2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                for ep, (ex, ey) in (("p1", (x1, y1)), ("p2", (x2, y2))):
                    hovered  = self._hover == ("ref", zone, ep)
                    dragging = self._drag  == ("ref", zone, ep)
                    r = ROI_HANDLE_R - 2 + (3 if (hovered or dragging) else 0)
                    p.setPen(QPen(QColor("#08090C"), 1.5))
                    p.setBrush(QBrush(QColor("#FFFFFF") if dragging else color))
                    p.drawEllipse(QPointF(ex, ey), r, r)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                rlabel = f'{ref.get("length_m", 0):.1f} m ref'
                p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                p.setPen(QPen(color))
                p.drawText(QRectF(mx - 30, my + 6, 60, 14),
                           Qt.AlignmentFlag.AlignCenter, rlabel)


# ──────────────────────────────────────────────────────────────────────
# BNVD CLASS → DATA SHEET COLUMN MAPPING
# ──────────────────────────────────────────────────────────────────────
# BNVD has 17 classes; map each index to the 11-column data sheet.
# Classes with no match (Pedestrian, PowerTiller, Wheelbarrow) → None.
BNVD_TO_SHEET = {
    0:  "Bi-Cycles",       # Bicycle
    1:  "Buses",           # Bus
    2:  "Auto Rickshaws",  # Bhotbhoti  (3-wheeler variant)
    3:  "Private Cars",    # Car
    4:  "Auto Rickshaws",  # CNG        (3-wheeler / auto-rickshaw)
    5:  "Auto Rickshaws",  # Easybike   (battery auto)
    6:  "Human Hollar",    # Leguna     (human hauler / covered van)
    7:  "Motor Cycles",    # Motorbike
    8:  "Micro Buses",     # MPV
    9:  None,              # Pedestrian — not counted
    10: "Mini Trucks",     # Pickup
    11: None,              # PowerTiller — not counted
    12: "Rickshaw",        # Rickshaw
    13: "Van",             # ShoppingVan
    14: "Trucks",          # Truck
    15: "Van",             # Van
    16: None,              # Wheelbarrow — not counted
}

# When using a generic COCO-trained YOLOv8 (no custom weights),
# map COCO class indices to data sheet columns where possible.
COCO_TO_SHEET = {
    1:  "Motor Cycles",    # motorcycle
    2:  "Private Cars",    # car
    3:  "Trucks",          # truck
    5:  "Buses",           # bus
    7:  "Trucks",          # truck (heavy)
}


# ──────────────────────────────────────────────────────────────────────
# GEOMETRY
# Point-in-polygon zone membership, plus normalized-coordinate distance
# helpers — no real-world calibration needed (see WORKER below for how
# delay gets measured without it).
# ──────────────────────────────────────────────────────────────────────

def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon test. pt=(x,y), poly=[(x,y), ...],
    both in the same coordinate space (normalized 0..1 here)."""
    x, y = pt
    inside = False
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_int:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def dist_norm(p1, p2):
    """Euclidean distance between two normalized (0..1) points. There's
    no metric calibration any more — every distance/speed/threshold in
    the worker is expressed in this same normalized-coordinate unit, and
    delay is measured by comparing crossing *time* against each zone's
    own learned baseline rather than against a real-world speed."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def polygon_diag(poly):
    """Bounding-box diagonal of a polygon, in normalized units — used to
    scale each zone's own motion/stop thresholds to its own on-screen
    size and shape, so nothing needs manual real-world calibration."""
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    d = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return d if d > 1e-6 else 0.05


# ──────────────────────────────────────────────────────────────────────
# WORKER
# ──────────────────────────────────────────────────────────────────────

class DelayWorker(QThread):
    """
    Runs YOLO + ByteTrack over TWO independent, user-drawn ROI zones
    (e.g. two different roads/approaches visible in the same camera) and
    turns per-vehicle tracks into directly-comparable delay metrics for
    each zone — with no real-world calibration required:

      • Each zone learns its own free-flow baseline live, from the
        fastest crossings it actually observes (a rolling low percentile
        of real crossing times — self-calibrating, and naturally
        accounts for that zone's own length/shape).
      • travel delay = actual crossing time − that zone's own baseline.
      • stopped delay = time spent below a stop-speed threshold that's
        scaled to the zone's own on-screen size.
      • Because both zones report delay in the same units (seconds,
        relative to each one's own learned "normal"), Zone A and Zone B
        are directly comparable — that comparison is the point.

    Also tracked per zone, all optional/best-effort:
      • Pedestrians/hawkers — live count, how many are "loitering" (a
        proxy for vendor/hawker activity vs. people just passing through).
      • PCU-adjusted volume (throughput) and, if you supply a capacity
        estimate, a volume/capacity ratio — a second, independent
        diagnostic of whether a road is jammed because it's genuinely
        over capacity vs. jammed by friction despite modest volume.
      • Physical queue length in metres, IF an optional single reference
        line has been calibrated for that zone (otherwise queue is just
        a vehicle count, as before).
      • Delay distribution — median, 85th percentile, and % of vehicles
        meaningfully delayed, not just the mean.
    """
    frame_update  = pyqtSignal(dict, float, float, float, float)
    frame_signal  = pyqtSignal(QImage)
    prog_update   = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_signal  = pyqtSignal(str)
    finished      = pyqtSignal(dict)

    def __init__(self, video_path, model, conf, iou, weights="",
                 rois=None, refs=None, capacities=None,
                 stop_frac_per_sec=DEFAULT_STOP_FRAC_PER_SEC,
                 motion_floor_frac=DEFAULT_MOTION_FLOOR_FRAC,
                 grace_sec=DEFAULT_GRACE_SEC,
                 bucket_sec=DEFAULT_BUCKET_SEC,
                 speed_window_sec=DEFAULT_SPEED_WINDOW_SEC,
                 freeflow_percentile=DEFAULT_FREEFLOW_PERCENTILE,
                 track_pedestrians=True,
                 loiter_sec=DEFAULT_LOITER_SEC,
                 active_classes=None,
                 device="auto", half="auto", imgsz=640,
                 preview_fps=15.0,
                 ema=0.35, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model      = model
        self.conf       = conf
        self.iou        = iou
        self.weights    = weights
        self.rois = {z: [tuple(p) for p in pts] for z, pts in
                     (rois or {"A": VideoCanvas.ROI_A_DEFAULT,
                               "B": VideoCanvas.ROI_B_DEFAULT}).items()}
        self.zone_diag = {z: polygon_diag(pts) for z, pts in self.rois.items()}
        self.stop_frac_per_sec   = max(0.001, stop_frac_per_sec)
        self.motion_floor_frac   = max(0.0, motion_floor_frac)
        self.grace_sec           = max(0.1, grace_sec)
        self.bucket_sec          = max(2.0, bucket_sec)
        self.speed_window_sec    = max(0.1, speed_window_sec)
        self.freeflow_percentile = min(0.9, max(0.02, freeflow_percentile))
        self.track_pedestrians   = track_pedestrians
        self.loiter_sec          = max(1.0, loiter_sec)
        # how long a vehicle that's vanished (occlusion, ID switch, or a
        # brief flicker across a zone's edge) stays "revivable" before
        # we give up and commit it as genuinely finished
        self.bridge_sec = self.grace_sec + 1.0
        self.active_classes = active_classes    # None = all classes active
        self.ema        = ema
        self.fps        = 25.0
        self._run       = True
        self._paused    = False
        self._t_now     = 0.0

        # ── performance: GPU device, precision, inference resolution ──
        # these change HOW FAST every frame is analysed, never WHICH
        # frames — analysis always runs on every single frame regardless
        self.device_pref  = device    # "auto" | "cuda" | "cpu"
        self.half_pref    = half      # "auto" | True | False
        self.imgsz        = int(imgsz) if imgsz else 640
        self.preview_interval_s = 1.0 / max(1.0, preview_fps)
        self._last_preview_t    = -999.0
        self._cached_cpu = self._cached_ram = self._cached_gpu = 0.0
        self._last_stats_t      = -999.0

        self.capacities = dict(capacities or {})   # zone -> PCU/hr, optional
        refs = refs or {}
        self.ref_scale = {}                        # zone -> metres/norm-unit, or None
        for z in self.rois:
            r = refs.get(z)
            if r and r.get("enabled"):
                d = dist_norm(r["p1"], r["p2"])
                self.ref_scale[z] = (r.get("length_m", 1.0) / d) if d > 1e-6 else None
            else:
                self.ref_scale[z] = None

        self.zones = {z: self._new_zone() for z in self.rois}

    def _new_zone(self):
        return {
            "active": {}, "ghost": [], "completed": [],
            "class_summary": {c: {"count": 0, "delay": 0.0, "stopped": 0.0}
                               for c in VEHICLE_CLASSES},
            "crossing_times": [], "delay_samples": [],
            "total_delay": 0.0, "total_stopped": 0.0, "vehicle_count": 0,
            "pcu_total": 0.0, "total_stops": 0, "accel_noise_samples": [],
            "max_queue": 0, "timeline": [], "bucket_start": 0.0,
            "bucket_flow": [], "bucket_queue": [], "_queue_now": 0,
            "ped_active": {}, "ped_completed": [],
        }

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._run = False; self._paused = False

    def _stop_threshold(self, z):
        return self.stop_frac_per_sec * self.zone_diag[z]

    def _motion_floor(self, z):
        return self.motion_floor_frac * self.zone_diag[z]

    # ── performance: device/precision resolution, throttled stats ──────
    def _resolve_device_and_precision(self):
        """Figure out what hardware we're actually about to run on, and
        report it plainly — 'is it even using the GPU' is the single
        most common silent cause of a video taking hours instead of
        minutes, so this is surfaced as a status message, not buried."""
        gpu_name = None
        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_ok = False

        if self.device_pref == "cpu":
            device = "cpu"
        elif self.device_pref == "cuda":
            device = 0 if cuda_ok else "cpu"
        else:  # auto
            device = 0 if cuda_ok else "cpu"

        if self.half_pref == "auto":
            half = (device != "cpu")
        else:
            half = bool(self.half_pref) and (device != "cpu")

        return device, half, cuda_ok, gpu_name

    def _get_system_stats(self):
        """CPU/RAM/GPU usage for the live meter — throttled to a few
        times a second. GPUtil in particular shells out to nvidia-smi as
        a subprocess on every call; polling it once per frame at 25-30fps
        can, on its own, turn a 16-minute video into an hours-long job."""
        now = time.time()
        if now - self._last_stats_t < 0.5:
            return self._cached_cpu, self._cached_ram, self._cached_gpu
        self._last_stats_t = now
        try:
            import psutil
            self._cached_cpu = psutil.cpu_percent()
            self._cached_ram = psutil.virtual_memory().percent
        except Exception:
            pass
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            self._cached_gpu = gpus[0].load * 100 if gpus else 0.0
        except Exception:
            self._cached_gpu = 0.0
        return self._cached_cpu, self._cached_ram, self._cached_gpu

    # ── class mapping (unchanged BNVD/COCO logic, now class-filterable) ──
    def _get_class_map(self, model_names: dict) -> dict:
        SHEET_COLS  = list(VEHICLE_CLASSES)
        SHEET_LOWER = {c.lower(): c for c in SHEET_COLS}
        IGNORE = {
            "pedestrian", "pedestrians", "person", "people",
            "ambulance", "powertiller", "wheelbarrow", "animal",
        }
        BNVD_LEGACY = {
            "bicycle": "Bi-Cycles", "bus": "Buses", "bhotbhoti": "Auto Rickshaws",
            "car": "Private Cars", "cng": "Auto Rickshaws", "easybike": "Auto Rickshaws",
            "leguna": "Human Hollar", "motorbike": "Motor Cycles", "mpv": "Micro Buses",
            "pickup": "Mini Trucks", "shoppingvan": "Van", "truck": "Trucks", "van": "Van",
        }
        mapping = {}
        for idx, name in model_names.items():
            nl = name.lower().strip()
            if nl in SHEET_LOWER:      mapping[idx] = SHEET_LOWER[nl]
            elif nl in IGNORE:         mapping[idx] = None
            elif nl in BNVD_LEGACY:    mapping[idx] = BNVD_LEGACY[nl]
            elif idx in COCO_TO_SHEET: mapping[idx] = COCO_TO_SHEET[idx]
            else:                      mapping[idx] = None
        if self.active_classes is not None:
            mapping = {idx: (cls if cls in self.active_classes else None)
                       for idx, cls in mapping.items()}
        return mapping

    def _get_pedestrian_ids(self, model_names: dict) -> set:
        PED_NAMES = {"pedestrian", "pedestrians", "person", "people"}
        return {idx for idx, name in model_names.items()
                if name.lower().strip() in PED_NAMES}

    # ── self-learning free-flow baseline, per zone ─────────────────────
    def _free_flow_time(self, z):
        times = self.zones[z]["crossing_times"]
        if not times:
            return None
        if len(times) < MIN_BASELINE_SAMPLES:
            return min(times)
        s = sorted(times)
        idx = max(0, int(self.freeflow_percentile * len(s)) - 1)
        return s[idx]

    def _zone_flow_index(self, z, t):
        """0-100 live 'how close to this zone's own free-flow pace' —
        the same units for both zones regardless of their size/shape, so
        A and B are always directly comparable."""
        zst = self.zones[z]
        ff = self._free_flow_time(z)
        if not zst["active"] or ff is None:
            return 100.0
        total = 0.0
        for st in zst["active"].values():
            elapsed = max(t - st["entry_time"], 1e-6)
            total += 1.0 if elapsed <= ff else ff / elapsed
        return (total / len(zst["active"])) * 100.0

    def _queue_length_m(self, z):
        """Physical extent of the current queue, in metres — only
        available if a reference line was calibrated for this zone.
        Measured as the largest gap between currently-queued vehicles'
        actual detected positions, not an assumption about vehicle
        length, so it reflects what's really on screen."""
        scale = self.ref_scale.get(z)
        if scale is None:
            return None
        thresh = self._stop_threshold(z)
        positions = [st["last_pos"] for st in self.zones[z]["active"].values()
                     if st["speed"] < thresh]
        if len(positions) < 2:
            return 0.0
        best = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = dist_norm(positions[i], positions[j])
                if d > best:
                    best = d
        return best * scale

    # ── shared delay bookkeeping ────────────────────────────────────────
    def _commit_vehicle(self, z, cls, entry_time, exit_time,
                         distance_norm, stopped_time, track_id=None,
                         stops_count=0, accel_noise=0.0):
        zst = self.zones[z]
        time_in_zone = max(0.0, exit_time - entry_time)
        ff = self._free_flow_time(z)
        travel_delay = max(0.0, time_in_zone - ff) if ff is not None else 0.0
        pace_pct = (ff / time_in_zone * 100.0) if (ff is not None and time_in_zone > 0) else 100.0
        rec = dict(track_id=track_id, cls=cls, zone=z,
                   entry_time=entry_time, exit_time=exit_time,
                   time_in_zone=time_in_zone, distance_norm=distance_norm,
                   travel_delay=travel_delay, stopped_delay=stopped_time,
                   pace_pct=pace_pct, stops_count=stops_count, accel_noise=accel_noise)
        zst["completed"].append(rec)
        cs = zst["class_summary"].setdefault(cls, {"count": 0, "delay": 0.0, "stopped": 0.0})
        cs["count"] += 1; cs["delay"] += travel_delay; cs["stopped"] += stopped_time
        zst["total_delay"]   += travel_delay
        zst["total_stopped"] += stopped_time
        zst["vehicle_count"] += 1
        zst["crossing_times"].append(time_in_zone)
        zst["delay_samples"].append(travel_delay)
        zst["pcu_total"]     += PCU_FACTORS.get(cls, 1.0)
        zst["total_stops"]   += stops_count
        zst["accel_noise_samples"].append(accel_noise)

    def _finalize_track(self, z, track_id, st, exit_time):
        accel_noise = stdev(st.get("accel_samples", []))
        self._commit_vehicle(z, st["cls"], st["entry_time"], exit_time,
                              st["dist_norm"], st["stopped_time"], track_id=track_id,
                              stops_count=st.get("stops_count", 0), accel_noise=accel_noise)

    def _revive_ghost(self, z, track_id, pos, t):
        """
        Try to re-attach a just-vanished occupant of zone z to a fresh
        detection — covers ByteTrack ID switches after occlusion and
        momentary boxes that flicker just outside the zone's edge.
        Prefer the same original track id; otherwise fall back to the
        nearest still-fresh ghost, accepting the match only if the jump
        is plausible for how long it's been gone (scaled to this zone's
        own learned pace) — so a single physical vehicle doesn't get
        chopped into several short, noisy "vehicles", while genuinely
        separate vehicles nearby don't get merged into one.
        """
        zst = self.zones[z]
        for i, g in enumerate(zst["ghost"]):
            if g.get("orig_track_id") == track_id and t - g["last_time"] <= self.bridge_sec:
                return zst["ghost"].pop(i)
        ff = self._free_flow_time(z)
        diag = self.zone_diag[z]
        typical_speed = diag / max(ff or 2.0, 0.5)   # normalized units / sec
        best_i, best_d = None, None
        for i, g in enumerate(zst["ghost"]):
            elapsed = t - g["last_time"]
            if elapsed > self.bridge_sec:
                continue
            d = dist_norm(g["last_pos"], pos)
            limit = max(diag * 0.15, typical_speed * 1.6 * elapsed)
            if d <= limit and (best_d is None or d < best_d):
                best_d, best_i = d, i
        if best_i is not None:
            return zst["ghost"].pop(best_i)
        return None

    def _sample_frame(self, z, t, queue_now, flow_index):
        zst = self.zones[z]
        zst["bucket_flow"].append(flow_index)
        zst["bucket_queue"].append(queue_now)
        zst["max_queue"] = max(zst["max_queue"], queue_now)
        if t - zst["bucket_start"] >= self.bucket_sec:
            self._flush_bucket(z, t)

    def _flush_bucket(self, z, t_end):
        zst = self.zones[z]
        if not zst["bucket_flow"]:
            return
        zst["timeline"].append({
            "t0": zst["bucket_start"], "t1": t_end,
            "flow_index": sum(zst["bucket_flow"]) / len(zst["bucket_flow"]),
            "avg_queue":  sum(zst["bucket_queue"]) / len(zst["bucket_queue"]),
        })
        zst["bucket_start"] = t_end
        zst["bucket_flow"]  = []
        zst["bucket_queue"] = []

    def _snapshot(self):
        zones_out = {}
        for z, zst in self.zones.items():
            vc_count = zst["vehicle_count"]
            avg_delay   = (zst["total_delay"] / vc_count) if vc_count else 0.0
            avg_stopped = (zst["total_stopped"] / vc_count) if vc_count else 0.0
            level, color = classify_congestion(avg_delay)

            samples = sorted(zst["delay_samples"])
            median_delay = percentile(samples, 50)
            p85_delay    = percentile(samples, 85)
            pct_delayed  = (sum(1 for d in samples if d > DELAY_FLAG_SEC) / len(samples) * 100.0
                             ) if samples else 0.0

            elapsed_hr = max(self._t_now, 1.0) / 3600.0
            pcu_per_hour = zst["pcu_total"] / elapsed_hr
            capacity = self.capacities.get(z) or None
            vc_ratio = (pcu_per_hour / capacity) if capacity else None
            vc_level, vc_color = classify_vc(vc_ratio)

            ped_dwells = zst["ped_completed"]
            avg_ped_dwell = (sum(ped_dwells) / len(ped_dwells)) if ped_dwells else 0.0
            live_loitering = sum(1 for pst in zst["ped_active"].values()
                                  if (self._t_now - pst["entry_time"]) >= self.loiter_sec)
            ped_current   = zst.get("_ped_current", len(zst["ped_active"]))
            loitering_now = zst.get("_ped_loitering_now", live_loitering)

            avg_stops = (zst["total_stops"] / vc_count) if vc_count else 0.0
            noise_samples = zst["accel_noise_samples"]
            avg_accel_noise = (sum(noise_samples) / len(noise_samples)) if noise_samples else 0.0

            zones_out[z] = {
                "label":         VideoCanvas.ZONE_LABELS.get(z, z),
                "vehicles":      vc_count + len(zst["active"]) + len(zst["ghost"]),
                "completed":     vc_count,
                "total_delay":   zst["total_delay"],
                "avg_delay":     avg_delay,
                "median_delay":  median_delay,
                "p85_delay":     p85_delay,
                "pct_delayed":   pct_delayed,
                "total_stopped": zst["total_stopped"],
                "avg_stopped":   avg_stopped,
                "avg_stops":     avg_stops,
                "avg_accel_noise": avg_accel_noise,
                "queue_now":     zst.get("_queue_now", 0),
                "max_queue":     zst["max_queue"],
                "queue_length_m": self._queue_length_m(z),
                "congestion":    (level, color),
                "free_flow_time": self._free_flow_time(z),
                "pcu_per_hour":  pcu_per_hour,
                "capacity_pcu_hr": capacity,
                "vc_ratio":      vc_ratio,
                "vc_level":      (vc_level, vc_color),
                "pedestrians": {
                    "current":       ped_current,
                    "loitering_now": loitering_now,
                    "total_seen":    len(zst["ped_completed"]) + len(zst["ped_active"]),
                    "avg_dwell_s":   avg_ped_dwell,
                },
                "class_summary": {k: dict(v) for k, v in zst["class_summary"].items()},
                "timeline":      list(zst["timeline"]),
            }
        comparison = {}
        if "A" in zones_out and "B" in zones_out:
            a, b = zones_out["A"], zones_out["B"]
            diff = a["avg_delay"] - b["avg_delay"]
            more = None
            if abs(diff) >= 0.5:
                more = "A" if diff > 0 else "B"
            ped_diff = a["pedestrians"]["loitering_now"] - b["pedestrians"]["loitering_now"]
            comparison = {
                "delay_diff_s": diff,
                "more_delayed_zone": more,
                "pedestrian_diff": ped_diff,
            }
        return {"zones": zones_out, "comparison": comparison}

    # ── main run loop ───────────────────────────────────────────────
    def run(self):
        try:
            self._run_inference()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _run_inference(self):
        try:
            from ultralytics import YOLO
            HAS_ULTRALYTICS = True
        except ImportError:
            HAS_ULTRALYTICS = False

        if not HAS_ULTRALYTICS:
            self._run_mock()
            return

        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.fps = fps if fps and fps > 1 else 25.0
        except Exception:
            self.fps = 25.0

        weights = self.weights.strip() if self.weights else ""
        model_name = weights if weights else f"{self.model}.pt"
        self.status_update.emit(f"Loading {Path(model_name).name}…")
        try:
            model = YOLO(model_name)
        except Exception as e:
            self.error_signal.emit(
                f"Could not load model '{model_name}':\n{e}\n\n"
                "Make sure ultralytics is installed and the .pt path is correct.")
            return

        device, half, cuda_ok, gpu_name = self._resolve_device_and_precision()

        cls_map = self._get_class_map(model.names)
        ped_ids = self._get_pedestrian_ids(model.names) if self.track_pedestrians else set()
        arch = "YOLO26" if "26" in model_name else "YOLOv8"

        if device == "cpu":
            hw_note = ("⚠ running on CPU — this will be much slower than a GPU run; "
                       "check that a CUDA-enabled PyTorch build is installed" if not cuda_ok
                       else "running on CPU by request")
        else:
            hw_note = f"running on GPU: {gpu_name or 'CUDA device 0'}{'  ·  FP16' if half else ''}"
        ped_note = (" · ⚠ no person class — hawker/pedestrian counts will stay 0"
                    if (self.track_pedestrians and not ped_ids) else "")
        self.status_update.emit(
            f"Running {arch} + ByteTrack  ·  {hw_note}  ·  imgsz={self.imgsz}"
            f"  ·  {self.fps:.1f} fps source{ped_note}")

        smooth_box = {}
        track_cls  = {}
        EMA        = self.ema
        frame_idx  = 0

        try:
            results = model.track(
                source=self.video_path, conf=self.conf, iou=self.iou,
                tracker="bytetrack.yaml", stream=True, persist=True, verbose=False,
                device=device, half=half, imgsz=self.imgsz,
            )
        except Exception as e:
            self.error_signal.emit(f"Inference error:\n{e}")
            return

        for r in results:
            if not self._run: break
            while self._paused:
                time.sleep(0.04)
                if not self._run: break

            t = frame_idx / self.fps
            self._t_now = t
            active_ids = set()
            zone_seen  = {z: set() for z in self.rois}
            ped_seen   = {z: set() for z in self.rois}
            ped_tids_this_frame = set()

            if r.boxes is not None and r.boxes.id is not None:
                h_fr, w_fr = r.orig_shape

                for box in r.boxes:
                    if box.id is None: continue
                    cls_id    = int(box.cls[0])
                    track_id  = int(box.id[0])
                    sheet_cls = cls_map.get(cls_id)
                    is_ped    = sheet_cls is None and cls_id in ped_ids
                    if sheet_cls is None and not is_ped:
                        continue
                    active_ids.add(track_id)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if track_id not in smooth_box:
                        smooth_box[track_id] = [x1, y1, x2, y2]
                    else:
                        s = smooth_box[track_id]
                        smooth_box[track_id] = [
                            s[0] + EMA * (x1 - s[0]), s[1] + EMA * (y1 - s[1]),
                            s[2] + EMA * (x2 - s[2]), s[3] + EMA * (y2 - s[3])]
                    if sheet_cls is not None and track_id not in track_cls:
                        track_cls[track_id] = sheet_cls

                    sx1, sy1, sx2, sy2 = smooth_box[track_id]
                    cx = ((sx1 + sx2) / 2) / w_fr
                    cy = ((sy1 + sy2) / 2) / h_fr

                    if sheet_cls is not None:
                        for z, roi_pts in self.rois.items():
                            zst    = self.zones[z]
                            inside = point_in_polygon((cx, cy), roi_pts)
                            if inside:
                                zone_seen[z].add(track_id)
                                st = zst["active"].get(track_id)
                                if st is None:
                                    st = self._revive_ghost(z, track_id, (cx, cy), t)
                                    if st is not None:
                                        zst["active"][track_id] = st

                                if st is None:
                                    zst["active"][track_id] = {
                                        "cls": track_cls[track_id],
                                        "entry_time": t, "last_time": t,
                                        "last_pos": (cx, cy), "dist_norm": 0.0,
                                        "stopped_time": 0.0, "speed": 0.0,
                                        "pos_hist": deque([(t, cx, cy)]),
                                        "orig_track_id": track_id,
                                        "stops_count": 0, "was_stopped": False,
                                        "_settled": False,
                                        "accel_samples": [],
                                    }
                                else:
                                    dt = max(1e-6, t - st["last_time"])
                                    seg = dist_norm(st["last_pos"], (cx, cy))
                                    floor = self._motion_floor(z)
                                    if seg > floor:
                                        st["dist_norm"] += seg
                                    st["pos_hist"].append((t, cx, cy))
                                    while (len(st["pos_hist"]) > 1 and
                                           t - st["pos_hist"][0][0] > self.speed_window_sec):
                                        st["pos_hist"].popleft()
                                    t0, x0, y0 = st["pos_hist"][0]
                                    span  = max(1e-3, t - t0)
                                    win_d = dist_norm((x0, y0), (cx, cy))
                                    # the window hasn't necessarily filled yet (start of a
                                    # crossing) — scale the floor down proportionally, or a
                                    # genuinely-moving vehicle reads as falsely stopped for
                                    # its first fraction of a second, on every single vehicle
                                    win_floor = floor * min(1.0, span / self.speed_window_sec)
                                    prev_speed = st["speed"]
                                    st["speed"] = (win_d / span) if win_d > win_floor else 0.0
                                    if st["speed"] < self._stop_threshold(z):
                                        st["stopped_time"] += dt

                                    # both the box EMA smoothing and the speed window itself
                                    # take a moment to settle after a track first appears, so
                                    # the apparent speed reads artificially low right at entry
                                    # — count stall cycles / sample acceleration noise only
                                    # once things have actually settled, or every vehicle picks
                                    # up one phantom "stall" purely from that ramp-up
                                    if (t - st["entry_time"]) >= self.speed_window_sec:
                                        if st["_settled"]:
                                            st["accel_samples"].append((st["speed"] - prev_speed) / dt)
                                            is_stopped_now = st["speed"] < self._stop_threshold(z)
                                            if is_stopped_now and not st["was_stopped"]:
                                                st["stops_count"] += 1
                                            st["was_stopped"] = is_stopped_now
                                        else:
                                            # first settled frame: establish the moving/stopped
                                            # baseline without counting a transition into it
                                            st["_settled"]    = True
                                            st["was_stopped"] = st["speed"] < self._stop_threshold(z)
                                    st["last_time"] = t
                                    st["last_pos"]  = (cx, cy)
                            else:
                                st = zst["active"].get(track_id)
                                if st is not None:
                                    st["orig_track_id"] = track_id
                                    st["last_time"] = t
                                    zst["ghost"].append(st)
                                    del zst["active"][track_id]
                    elif is_ped:
                        for z, roi_pts in self.rois.items():
                            zst    = self.zones[z]
                            inside = point_in_polygon((cx, cy), roi_pts)
                            if inside:
                                ped_seen[z].add(track_id)
                                ped_tids_this_frame.add(track_id)
                                pst = zst["ped_active"].get(track_id)
                                if pst is None:
                                    zst["ped_active"][track_id] = {
                                        "entry_time": t, "last_time": t, "last_pos": (cx, cy)}
                                else:
                                    pst["last_time"] = t
                                    pst["last_pos"]  = (cx, cy)
                            else:
                                pst = zst["ped_active"].get(track_id)
                                if pst is not None:
                                    dwell = pst["last_time"] - pst["entry_time"]
                                    zst["ped_completed"].append(dwell)
                                    del zst["ped_active"][track_id]

                gone = set(smooth_box.keys()) - active_ids
                for tid in gone:
                    smooth_box.pop(tid, None); track_cls.pop(tid, None)

            for z, zst in self.zones.items():
                missing = set(zst["active"].keys()) - zone_seen[z]
                for tid in list(missing):
                    st = zst["active"][tid]
                    st["orig_track_id"] = st.get("orig_track_id", tid)
                    zst["ghost"].append(st)
                    del zst["active"][tid]

                still = []
                for g in zst["ghost"]:
                    if t - g["last_time"] > self.bridge_sec:
                        self._finalize_track(z, g.get("orig_track_id"), g, g["last_time"])
                    else:
                        still.append(g)
                zst["ghost"] = still

                ped_missing = set(zst["ped_active"].keys()) - ped_seen[z]
                for tid in list(ped_missing):
                    pst = zst["ped_active"][tid]
                    dwell = pst["last_time"] - pst["entry_time"]
                    zst["ped_completed"].append(dwell)
                    del zst["ped_active"][tid]

                thresh = self._stop_threshold(z)
                queue_now = sum(1 for st in zst["active"].values() if st["speed"] < thresh)
                zst["_queue_now"] = queue_now
                zst["_ped_current"] = len(zst["ped_active"])
                zst["_ped_loitering_now"] = sum(
                    1 for pst in zst["ped_active"].values()
                    if (t - pst["entry_time"]) >= self.loiter_sec)
                self._sample_frame(z, t, queue_now, self._zone_flow_index(z, t))

            # everything above this line runs on every single frame, no
            # exceptions — that's the actual analysis. Everything below
            # (building a display snapshot, drawing the annotated frame,
            # polling CPU/GPU usage) is purely for the live UI and is
            # throttled to a target refresh rate — a human doesn't need
            # 30 redraws/sec, and skipping the redraw (not the analysis)
            # on most frames is where most of the real time goes
            now_wall = time.time()
            if now_wall - self._last_preview_t >= self.preview_interval_s:
                self._last_preview_t = now_wall
                snap = self._snapshot()
                self._draw_frame(r, smooth_box, track_cls, ped_tids_this_frame, snap)
                inf_ms = r.speed.get("inference", 33)
                fps_live = 1000 / max(inf_ms, 1)
                cpu, ram, gpu = self._get_system_stats()
                self.frame_update.emit(snap, fps_live, cpu, ram, gpu)

            total_frames = getattr(r, "total_frames", None)
            if total_frames and total_frames > 0:
                self.prog_update.emit(int(frame_idx / total_frames * 100))
            else:
                self.prog_update.emit(min(frame_idx % 200, 99))
            frame_idx += 1

        t_end = frame_idx / self.fps if self.fps else 0.0
        self._t_now = t_end
        for z, zst in self.zones.items():
            self._flush_bucket(z, t_end)
            for tid, st in list(zst["active"].items()):
                self._finalize_track(z, tid, st, st["last_time"])
            zst["active"].clear()
            for g in zst["ghost"]:
                self._finalize_track(z, g.get("orig_track_id"), g, g["last_time"])
            zst["ghost"].clear()
            for tid, pst in list(zst["ped_active"].items()):
                dwell = pst["last_time"] - pst["entry_time"]
                zst["ped_completed"].append(dwell)
            zst["ped_active"].clear()

        final = self._snapshot()
        final["per_vehicle"] = {z: list(self.zones[z]["completed"]) for z in self.zones}
        self.finished.emit(final)

    # ── frame annotation: both zones, boxes, pedestrians, dual-zone HUD ──
    def _draw_frame(self, r, smooth_box, track_cls, ped_tids, snap):
        try:
            import cv2, numpy as np
        except Exception:
            try:
                annotated = r.plot()
                h2, w2, ch = annotated.shape
                rgb = annotated[:, :, ::-1].copy()
                qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format.Format_RGB888)
                self.frame_signal.emit(qimg.copy())
            except Exception:
                pass
            return

        frame = r.orig_img.copy()
        h_fr, w_fr = frame.shape[:2]

        CLASS_COLORS = {
            "Buses": (255, 200, 50), "Micro Buses": (180, 130, 255),
            "Trucks": (80, 100, 255), "Mini Trucks": (80, 200, 255),
            "Private Cars": (50, 210, 140), "Human Hollar": (255, 100, 200),
            "Bi-Cycles": (255, 180, 50), "Motor Cycles": (130, 230, 80),
            "Rickshaw": (50, 180, 255), "Auto Rickshaws": (200, 100, 255),
            "Van": (100, 230, 230),
        }
        DEFAULT_COLOR = (0, 200, 230)
        STOP_COLOR    = (60, 60, 240)
        PED_COLOR     = (120, 255, 120)
        # BGR versions of VideoCanvas.ZONE_COLORS (#00C8EE, #F0A030)
        ZONE_BGR = {"A": (238, 200, 0), "B": (48, 160, 240)}

        for tid, (sx1, sy1, sx2, sy2) in smooth_box.items():
            ix1, iy1, ix2, iy2 = int(sx1), int(sy1), int(sx2), int(sy2)

            if tid in ped_tids:
                cx_p, cy_p = (ix1 + ix2) // 2, (iy1 + iy2) // 2
                rad = max(4, min(10, (ix2 - ix1) // 4))
                cv2.circle(frame, (cx_p, cy_p), rad, PED_COLOR, 2, cv2.LINE_AA)
                continue

            label = track_cls.get(tid, "")
            stopped = False
            for z, zst in self.zones.items():
                st = zst["active"].get(tid)
                if st is not None and st["speed"] < self._stop_threshold(z):
                    stopped = True
                    break
            color = STOP_COLOR if stopped else CLASS_COLORS.get(label, DEFAULT_COLOR)

            cx1, cy1 = max(0, ix1), max(0, iy1)
            cx2, cy2 = min(w_fr, ix2), min(h_fr, iy2)
            if cx2 > cx1 and cy2 > cy1:
                sub  = frame[cy1:cy2, cx1:cx2]
                fill = np.full_like(sub, color)
                cv2.addWeighted(fill, 0.12, sub, 0.88, 0, sub)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

            clen = max(8, min(20, (ix2 - ix1) // 5))
            for cx_c, cy_c, dx, dy in [(ix1, iy1, 1, 1), (ix2, iy1, -1, 1),
                                        (ix1, iy2, 1, -1), (ix2, iy2, -1, -1)]:
                cv2.line(frame, (cx_c, cy_c), (cx_c + dx * clen, cy_c), color, 3)
                cv2.line(frame, (cx_c, cy_c), (cx_c, cy_c + dy * clen), color, 3)

            lbl_text = f"{label} #{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX; fscale = 0.42; thick = 1
            (tw, th), _ = cv2.getTextSize(lbl_text, font, fscale, thick)
            pad = 4
            lx1, ly1 = ix1, max(0, iy1 - th - pad * 2)
            lx2, ly2 = ix1 + tw + pad * 2, iy1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(frame, lbl_text, (lx1 + pad, ly2 - pad // 2 - 1),
                        font, fscale, (10, 10, 10), thick, cv2.LINE_AA)

        # ── both zones ──
        for z, pts in self.rois.items():
            bgr = ZONE_BGR.get(z, (200, 200, 200))
            abs_pts = [(int(x * w_fr), int(y * h_fr)) for x, y in pts]
            poly = np.array(abs_pts, dtype=np.int32)
            cv2.polylines(frame, [poly], True, bgr, 2, cv2.LINE_AA)
            xs = [p[0] for p in abs_pts]; ys = [p[1] for p in abs_pts]
            bx1, by1 = max(0, min(xs)), max(0, min(ys))
            bx2, by2 = min(w_fr, max(xs)), min(h_fr, max(ys))
            if bx2 > bx1 and by2 > by1:
                sub = frame[by1:by2, bx1:bx2]
                overlay_sub = sub.copy()
                local_poly = (poly - [bx1, by1]).astype(np.int32)
                cv2.fillPoly(overlay_sub, [local_poly], bgr)
                cv2.addWeighted(overlay_sub, 0.06, sub, 0.94, 0, sub)

        # ── dual-zone delay HUD (+ hawker/loitering activity) ──
        y0 = 8
        for z in self.rois:
            zi = snap["zones"].get(z)
            if not zi: continue
            bgr = ZONE_BGR.get(z, (40, 40, 40))
            hud = (f'{zi["label"]}: Q{zi["queue_now"]}  Delay {zi["avg_delay"]:.1f}s  '
                   f'{zi["congestion"][0]}  ·  Hawkers {zi["pedestrians"]["loitering_now"]}')
            (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (8, y0), (18 + tw, y0 + th + 10), bgr, -1)
            cv2.putText(frame, hud, (14, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (10, 10, 10), 1, cv2.LINE_AA)
            y0 += th + 16

        more = snap.get("comparison", {}).get("more_delayed_zone")
        if more:
            label = snap["zones"][more]["label"]
            diff  = abs(snap["comparison"]["delay_diff_s"])
            cmp_txt = f"{label} is +{diff:.1f}s more delayed"
            (tw, th), _ = cv2.getTextSize(cmp_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (8, y0), (18 + tw, y0 + th + 10), (10, 10, 10), -1)
            cv2.putText(frame, cmp_txt, (14, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        rgb  = frame[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w_fr, h_fr, 3 * w_fr, QImage.Format.Format_RGB888)
        self.frame_signal.emit(qimg.copy())

    # ── demo fallback (no ultralytics installed) — two contrasting
    #    synthetic congestion patterns, plus synthetic pedestrian/hawker
    #    activity, so the comparison is visible even without a model ──
    def _run_mock(self):
        self.status_update.emit("⚠ ultralytics not found — running demo mode")
        total_ticks = 1400
        dt = 0.15
        t  = 0.0
        wts = [5, 2, 2, 1, 12, 1, 3, 9, 7, 5, 2]
        zone_keys = list(self.rois.keys())

        for i in range(total_ticks):
            if not self._run: break
            while self._paused:
                time.sleep(0.03)
                if not self._run: break
            t += dt
            self._t_now = t
            phase = i / total_ticks

            for zi, z in enumerate(zone_keys):
                # first zone: a congestion wave rising then falling.
                # second zone (if present): milder, steadier traffic —
                # so the comparison view has something to actually show.
                if zi == 0:
                    congestion = 0.15 + 0.85 * (math.sin(phase * math.pi) ** 2)
                else:
                    congestion = 0.05 + 0.20 * (math.sin(phase * math.pi * 2 + 1.0) ** 2)

                zst = self.zones[z]
                if (i + zi * 3) % 6 == 0:
                    cls = random.choices(VEHICLE_CLASSES, weights=wts, k=1)[0]
                    base_time = random.uniform(3.0, 7.0)
                    stretch = 1 + congestion * random.uniform(1.2, 4.0)
                    time_in_zone = base_time * stretch
                    stopped = max(0.0, time_in_zone - base_time) * random.uniform(0.2, 0.7) * congestion
                    # more congestion -> more stop-and-go cycles and jerkier motion
                    stops_count = int(round(congestion * random.uniform(0, 4)))
                    accel_noise = congestion * random.uniform(0.02, 0.08)
                    self._commit_vehicle(z, cls, t - time_in_zone, t, 0.3, stopped,
                                         stops_count=stops_count, accel_noise=accel_noise)

                queue_now  = int(congestion * random.uniform(0, 8))
                flow_index = max(5.0, 100.0 * (1 - 0.9 * congestion))
                zst["_queue_now"] = queue_now
                self._sample_frame(z, t, queue_now, flow_index)

                # fake pedestrian/hawker presence: busier road -> more
                # foot traffic and more vendors setting up shop
                ped_target = int(2 + congestion * 10)
                while len(zst["ped_active"]) < ped_target:
                    fake_id = f"mockped_{z}_{i}_{len(zst['ped_active'])}"
                    zst["ped_active"][fake_id] = {
                        "entry_time": t - random.uniform(0, 40), "last_time": t,
                        "last_pos": (0.5, 0.5)}
                while len(zst["ped_active"]) > ped_target:
                    zst["ped_active"].pop(next(iter(zst["ped_active"])))
                zst["_ped_current"] = len(zst["ped_active"])
                zst["_ped_loitering_now"] = sum(
                    1 for pst in zst["ped_active"].values()
                    if (t - pst["entry_time"]) >= self.loiter_sec)

            fps = random.uniform(22, 30)
            cpu = min(95, 30 + i / total_ticks * 45 + random.gauss(0, 3))
            ram = min(88, 38 + i / total_ticks * 25 + random.gauss(0, 2))
            self.frame_update.emit(self._snapshot(), fps, cpu, ram, 0.0)
            self.prog_update.emit(int(i / total_ticks * 100))
            time.sleep(0.008)

        for z in self.zones:
            self._flush_bucket(z, t)
            zst = self.zones[z]
            for pst in zst["ped_active"].values():
                zst["ped_completed"].append(pst["last_time"] - pst["entry_time"])
            zst["ped_active"].clear()
        final = self._snapshot()
        final["per_vehicle"] = {z: list(self.zones[z]["completed"]) for z in self.zones}
        self.finished.emit(final)


class ZoneBoard(QWidget):
    """One zone's self-contained delay board: stat cards, a live
    congestion badge, extended stats (distribution, volume/capacity,
    queue length, pedestrian activity), a flow timeline, and a
    per-class delay table — each zone is analysed completely
    independently."""
    def __init__(self, zone, parent=None):
        super().__init__(parent)
        self.zone = zone
        color = VideoCanvas.ZONE_COLORS.get(zone, C["acc"])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(10)

        cards_row = QHBoxLayout(); cards_row.setSpacing(8)
        self.card_vehicles = StatCard("Vehicles",    color=color)
        self.card_delay    = StatCard("Total Delay", subtitle="veh·min",     color=C["warn"])
        self.card_avg      = StatCard("Avg Delay",   subtitle="sec/vehicle", color=C["dir1"])
        self.card_queue    = StatCard("Queue Now",   color=C["danger"])
        for card in (self.card_vehicles, self.card_delay, self.card_avg, self.card_queue):
            cards_row.addWidget(card)
        lay.addLayout(cards_row)

        self.level_lbl = QLabel("")
        self.level_lbl.setStyleSheet(f"color:{C['ts']}; font-size:11px; font-weight:700;")
        lay.addWidget(self.level_lbl)

        extra_card = _card()
        ec = QVBoxLayout(extra_card)
        ec.setContentsMargins(14, 10, 14, 10); ec.setSpacing(6)
        ec.addWidget(_lbl("Extended Stats"))
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(4)
        self.stat_median      = QLabel("—")
        self.stat_p85         = QLabel("—")
        self.stat_pctdelayed  = QLabel("—")
        self.stat_stops       = QLabel("—")
        self.stat_smoothness  = QLabel("—")
        self.stat_pcu         = QLabel("—")
        self.stat_vc          = QLabel("—")
        self.stat_queue_m     = QLabel("—")
        self.stat_ped         = QLabel("—")
        rows = [
            ("Median delay",         self.stat_median),
            ("85th %ile delay",      self.stat_p85),
            ("% vehicles delayed",   self.stat_pctdelayed),
            ("Stalls / vehicle",     self.stat_stops),
            ("Motion smoothness",    self.stat_smoothness),
            ("PCU volume",           self.stat_pcu),
            ("Volume / Capacity",    self.stat_vc),
            ("Queue length",         self.stat_queue_m),
            ("Pedestrians / hawkers",self.stat_ped),
        ]
        for i, (caption, lbl) in enumerate(rows):
            cap = QLabel(caption)
            cap.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
            lbl.setStyleSheet(f"color:{C['tp']}; font-size:10px; font-weight:700;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(cap, i, 0)
            grid.addWidget(lbl, i, 1)
        ec.addLayout(grid)
        lay.addWidget(extra_card)

        chart_card = _card()
        cc = QVBoxLayout(chart_card)
        cc.setContentsMargins(14, 10, 14, 10); cc.setSpacing(6)
        cc.addWidget(_lbl("Flow Timeline"))
        self.timeline_chart = TimelineChart()
        cc.addWidget(self.timeline_chart)
        lay.addWidget(chart_card)

        tbl_card = _card()
        tc = QVBoxLayout(tbl_card)
        tc.setContentsMargins(0, 10, 0, 0); tc.setSpacing(6)
        th = QHBoxLayout(); th.setContentsMargins(14, 0, 14, 0)
        th.addWidget(_lbl("Delay by Vehicle Class"))
        tc.addLayout(th)
        self.class_tbl = ClassDelayTable()
        tc.addWidget(self.class_tbl)
        lay.addWidget(tbl_card, 1)

    def clear_all(self):
        self.class_tbl.clear_all()
        self.timeline_chart.update_data([])
        self.card_vehicles.set_value(0)
        self.card_delay.set_value("0.0")
        self.card_avg.set_value("0.0")
        self.card_queue.set_value(0)
        self.level_lbl.setText("")
        for lbl in (self.stat_median, self.stat_p85, self.stat_pctdelayed,
                    self.stat_stops, self.stat_smoothness,
                    self.stat_pcu, self.stat_vc, self.stat_queue_m, self.stat_ped):
            lbl.setText("—")
            lbl.setStyleSheet(f"color:{C['tp']}; font-size:10px; font-weight:700;")

    def update_data(self, zi):
        if not zi:
            return
        self.card_vehicles.set_value(zi.get("vehicles", 0))
        self.card_delay.set_value(f'{zi.get("total_delay", 0.0) / 60:.1f}')
        self.card_avg.set_value(f'{zi.get("avg_delay", 0.0):.1f}')
        self.card_queue.set_value(zi.get("queue_now", 0))
        label, color = zi.get("congestion", ("—", C["ts"]))
        self.level_lbl.setText(f"●  {label}")
        self.level_lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:700;")

        median = zi.get("median_delay"); p85 = zi.get("p85_delay")
        self.stat_median.setText(f"{median:.1f}s" if median is not None else "—")
        self.stat_p85.setText(f"{p85:.1f}s" if p85 is not None else "—")
        self.stat_pctdelayed.setText(f'{zi.get("pct_delayed", 0.0):.0f}%')
        self.stat_stops.setText(f'{zi.get("avg_stops", 0.0):.1f}')

        noise = zi.get("avg_accel_noise", 0.0)
        if noise < 0.010:
            noise_label, noise_color = "smooth", C["ok"]
        elif noise < 0.030:
            noise_label, noise_color = "uneven", C["warn"]
        else:
            noise_label, noise_color = "stop-and-go", C["danger"]
        self.stat_smoothness.setText(f"{noise_label}  ({noise:.3f})")
        self.stat_smoothness.setStyleSheet(f"color:{noise_color}; font-size:10px; font-weight:700;")

        self.stat_pcu.setText(f'{zi.get("pcu_per_hour", 0.0):.0f} PCU/hr')

        vc = zi.get("vc_ratio")
        vc_level, vc_color = zi.get("vc_level", (None, C["ts"]))
        if vc is not None:
            self.stat_vc.setText(f"{vc:.2f}  ({vc_level})")
            self.stat_vc.setStyleSheet(f"color:{vc_color}; font-size:10px; font-weight:700;")
        else:
            self.stat_vc.setText("— (set capacity)")
            self.stat_vc.setStyleSheet(f"color:{C['ts']}; font-size:10px; font-weight:700;")

        qlen = zi.get("queue_length_m")
        self.stat_queue_m.setText(f"{qlen:.1f} m" if qlen is not None else "— (no ref line)")

        ped = zi.get("pedestrians", {})
        self.stat_ped.setText(f'{ped.get("current", 0)} now · {ped.get("loitering_now", 0)} loitering')

        self.timeline_chart.update_data(zi.get("timeline", []))
        self.class_tbl.update_data(zi.get("class_summary", {}))


class AnalysisPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path   = None
        self.worker       = None
        self.running      = False
        self.paused       = False
        self.batch_paths  = []      # queued video paths for batch mode
        self.batch_index  = 0
        self.batch_active = False
        self._build()

    # ─── layout ──────────────────────────────────────────────────────

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 10, 16)
        root.setSpacing(12)

        # ── LEFT: video + transport ───────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        self.canvas = VideoCanvas()
        self.canvas.file_dropped.connect(self.load_video)
        self.canvas.setMinimumHeight(280)
        left.addWidget(self.canvas, 1)
        # both ROI zones initialise to their own default hexagons

        # transport
        tb = QHBoxLayout(); tb.setSpacing(8)

        self.btn_browse = QPushButton("📂  Browse Video")
        self.btn_browse.setObjectName("btn_ghost")
        self.btn_browse.setFixedHeight(38)
        self.btn_browse.setFixedWidth(150)
        self.btn_browse.clicked.connect(self._browse)
        tb.addWidget(self.btn_browse)

        self.btn_browse_multi = QPushButton("📂  Multiple Videos")
        self.btn_browse_multi.setObjectName("btn_ghost")
        self.btn_browse_multi.setFixedHeight(38)
        self.btn_browse_multi.setFixedWidth(160)
        self.btn_browse_multi.clicked.connect(self._browse_multi)
        tb.addWidget(self.btn_browse_multi)

        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setFixedWidth(170)
        self.btn_run.setEnabled(True)
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #00C8EE;
                color: #090B0F;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #33D8FF; }
            QPushButton:pressed { background-color: #007A99; }
            QPushButton:disabled {
                background-color: #1A2030;
                color: #303650;
            }
        """)
        tb.addWidget(self.btn_run)

        tb.addStretch(1)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btn_ghost")
        self.btn_pause.setFixedHeight(38)
        self.btn_pause.setFixedWidth(100)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        tb.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        tb.addWidget(self.btn_stop)

        left.addLayout(tb)

        # progress + status
        self.prog = QProgressBar(); self.prog.setValue(0)
        self.prog.setFixedHeight(4)
        left.addWidget(self.prog)

        status_row = QHBoxLayout(); status_row.setSpacing(6)
        self.dot = PulsingDot()
        self.status_lbl = QLabel("Ready — load a video to begin")
        self.status_lbl.setStyleSheet(f"color:{C['tp']}; font-size:11px; opacity:0.7;")
        self.fps_lbl = QLabel("")
        self.fps_lbl.setStyleSheet(f"color:{C['acc']}; font-size:11px; font-weight:600;")
        status_row.addWidget(self.dot)
        status_row.addWidget(self.status_lbl)
        status_row.addStretch()
        status_row.addWidget(self.fps_lbl)
        left.addLayout(status_row)

        # batch queue — hidden until multiple videos are selected
        self.batch_list = QListWidget()
        self.batch_list.setFixedHeight(110)
        self.batch_list.setVisible(False)
        left.addWidget(self.batch_list)

        left_w = QWidget(); left_w.setLayout(left)
        root.addWidget(left_w, 4)

        # ── RIGHT: usage + stats + chart + table ──────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(6, 0, 0, 0)
        right.setSpacing(10)

        # Usage meter card
        meter_card = _card()
        mc = QVBoxLayout(meter_card)
        mc.setContentsMargins(14, 10, 14, 6)
        mc.setSpacing(4)
        mh = QHBoxLayout()
        mh.addWidget(_lbl("System Usage"))
        mh.addStretch()
        mc.addLayout(mh)
        self.meter = UsageMeter()
        mc.addWidget(self.meter)
        right.addWidget(meter_card)

        # Road A vs Road B comparison — the headline view
        cmp_card = _card()
        cmpl = QVBoxLayout(cmp_card)
        cmpl.setContentsMargins(14, 10, 14, 10)
        cmpl.setSpacing(6)
        cmpl.addWidget(_lbl("Road A vs Road B"))
        self.comparison = ComparisonPanel()
        cmpl.addWidget(self.comparison)
        right.addWidget(cmp_card)

        # Each zone gets its own full board (timeline + class delay table)
        self.tabs = QTabWidget()
        self.board_a = ZoneBoard("A")
        self.board_b = ZoneBoard("B")
        self.tabs.addTab(self.board_a, VideoCanvas.ZONE_LABELS["A"])
        self.tabs.addTab(self.board_b, VideoCanvas.ZONE_LABELS["B"])
        right.addWidget(self.tabs, 1)

        right_w = QWidget(); right_w.setLayout(right)
        root.addWidget(right_w, 3)

    # ─── slots ───────────────────────────────────────────────────────

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)")
        if p: self.load_video(p)

    def load_video(self, path):
        self.video_path  = path
        self.batch_paths = []
        self.batch_list.setVisible(False)
        self.canvas.set_loaded(Path(path).name)
        self.btn_run.setEnabled(True)
        self.status_lbl.setText(f"Loaded: {Path(path).name}")
        self.dot.set_color(C["warn"])

    def _browse_multi(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple Videos", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)")
        if not paths:
            return
        self.batch_paths = paths
        self.batch_index = 0
        self.video_path  = None
        self.batch_list.clear()
        for p in paths:
            self.batch_list.addItem(QListWidgetItem(f"⏳  {Path(p).name}"))
        self.batch_list.setVisible(True)
        self.canvas.set_loaded(f"{len(paths)} videos queued for batch")
        self.btn_run.setEnabled(True)
        self.status_lbl.setText(
            f"{len(paths)} videos queued — click Run Analysis to start the batch")
        self.dot.set_color(C["warn"])

    def _toggle_run(self):
        if not self.running: self._start()
        else:                self._stop()

    def _build_worker(self, video_path):
        """Construct a fully-configured DelayWorker from current Config
        settings for a given video — shared by single-video and batch runs
        so both always use identical settings."""
        w = self.window()
        model   = w.cfg.model_combo.currentText()
        conf    = w.cfg.conf_slider.value() / 100
        iou     = w.cfg.iou_slider.value()  / 100
        weights = w.cfg.weights_path.text()
        dp      = w.cfg.get_delay_params()
        pp      = w.cfg.get_perf_params()
        return DelayWorker(
            video_path          = video_path or "",
            model               = model,
            conf                = conf,
            iou                 = iou,
            weights             = weights,
            rois                = self.canvas.get_rois(),
            refs                = self.canvas.get_refs(),
            capacities          = w.cfg.get_capacities(),
            stop_frac_per_sec   = dp["stop_frac_per_sec"],
            motion_floor_frac   = dp["motion_floor_frac"],
            grace_sec           = dp["grace_sec"],
            bucket_sec          = dp["bucket_sec"],
            speed_window_sec    = dp["speed_window_sec"],
            freeflow_percentile = dp["freeflow_percentile"],
            track_pedestrians   = dp["track_pedestrians"],
            loiter_sec          = dp["loiter_sec"],
            active_classes      = w.cfg.get_active_classes(),
            ema                 = w.cfg.smooth_slider.value()  / 100,
            device              = pp["device"],
            half                = pp["half"],
            imgsz               = pp["imgsz"],
            preview_fps         = pp["preview_fps"],
        )

    def _style_run_button_stop(self):
        self.btn_run.setText("■  Stop")
        self.btn_run.setObjectName("btn_danger")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #1A1020; color: #F04060;
                border: 1px solid #F04060; border-radius: 8px;
                font-size: 13px; font-weight: 700; padding: 0 16px;
            }
            QPushButton:hover { background-color: #2A1530; }
        """)

    def _start(self):
        if self.batch_paths:
            self._start_batch()
            return
        if not self.video_path:
            self._browse()
            if not self.video_path:
                return
        self.running = True; self.paused = False
        self.board_a.clear_all(); self.board_b.clear_all()
        self.prog.setValue(0)
        self._style_run_button_stop()
        self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
        self.dot.set_color(C["ok"]); self.status_lbl.setText("Analysing…")

        self.worker = self._build_worker(self.video_path)
        self.worker.frame_update.connect(self._on_frame)
        self.worker.frame_signal.connect(self._on_canvas_frame)
        self.worker.prog_update.connect(self.prog.setValue)
        self.worker.status_update.connect(self.status_lbl.setText)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    # ── batch mode: run several videos back-to-back, one CSV per video ──
    def _start_batch(self):
        self.running = True; self.paused = False; self.batch_active = True
        self.batch_index = 0
        self.prog.setValue(0)
        self._style_run_button_stop()
        self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
        self.btn_browse.setEnabled(False); self.btn_browse_multi.setEnabled(False)
        self._run_next_batch_item()

    def _run_next_batch_item(self):
        if self.batch_index >= len(self.batch_paths):
            self._finish_batch()
            return
        path = self.batch_paths[self.batch_index]
        n    = len(self.batch_paths)
        self.video_path = path

        item = self.batch_list.item(self.batch_index)
        if item:
            item.setText(f"▶  {Path(path).name}  — processing…")
        self.canvas.set_loaded(f"{Path(path).name}  ({self.batch_index + 1}/{n})")
        self.board_a.clear_all(); self.board_b.clear_all()
        self.prog.setValue(0)
        self.dot.set_color(C["ok"])
        self.status_lbl.setText(f"Batch {self.batch_index + 1}/{n}: {Path(path).name}")

        idx = self.batch_index
        self.worker = self._build_worker(path)
        self.worker.frame_update.connect(self._on_frame)
        self.worker.frame_signal.connect(self._on_canvas_frame)
        self.worker.prog_update.connect(self.prog.setValue)
        self.worker.status_update.connect(
            lambda s, idx=idx, n=n: self.status_lbl.setText(f"[{idx + 1}/{n}] {s}"))
        self.worker.error_signal.connect(self._on_batch_item_error)
        self.worker.finished.connect(self._on_batch_item_done)
        self.worker.start()

    def _on_batch_item_done(self, final):
        if not self.batch_active or self.batch_index >= len(self.batch_paths):
            return  # batch was stopped before this video's worker actually finished
        path     = self.batch_paths[self.batch_index]
        csv_path = str(Path(path).with_suffix(".csv"))
        cfg      = self.window().cfg
        item     = self.batch_list.item(self.batch_index)
        try:
            write_summary_csv(csv_path, final, cfg.get_meta(), cfg.site_name.text())
            if item:
                item.setText(f"✓  {Path(path).name}  →  {Path(csv_path).name}")
        except Exception as e:
            if item:
                item.setText(f"✗  {Path(path).name}  — CSV write failed: {e}")

        self.window().results.set_results(final)
        self.batch_index += 1
        self._run_next_batch_item()

    def _on_batch_item_error(self, msg):
        if not self.batch_active or self.batch_index >= len(self.batch_paths):
            return  # batch was stopped before this video's worker actually finished
        path = self.batch_paths[self.batch_index]
        item = self.batch_list.item(self.batch_index)
        if item:
            short = msg.splitlines()[0][:80] if msg else "error"
            item.setText(f"✗  {Path(path).name}  — {short}")
        self.batch_index += 1
        self._run_next_batch_item()

    def _finish_batch(self):
        n = len(self.batch_paths)
        self.batch_active = False
        self._reset()
        self.btn_browse.setEnabled(True); self.btn_browse_multi.setEnabled(True)
        self.status_lbl.setText(f"Batch complete — {n} video(s) processed")
        self.dot.set_color(C["ok"])
        QMessageBox.information(self, "Batch complete",
            f"Processed {n} video(s).\n\nA CSV was saved next to each video "
            "(same name, .csv extension).")

    def _toggle_pause(self):
        if not self.worker: return
        if self.paused:
            self.worker.resume(); self.paused = False
            self.btn_pause.setText("⏸  Pause")
            self.dot.set_color(C["ok"]); self.status_lbl.setText("Analysing…")
        else:
            self.worker.pause(); self.paused = True
            self.btn_pause.setText("▶  Resume")
            self.dot.set_color(C["warn"]); self.status_lbl.setText("Paused")

    def _stop(self):
        if self.worker: self.worker.stop()
        if self.batch_active:
            self.batch_active = False
            remaining = len(self.batch_paths) - self.batch_index
            self.status_lbl.setText(
                f"Batch stopped — {self.batch_index} processed, {remaining} skipped")
            self.batch_paths = []
            self.btn_browse.setEnabled(True); self.btn_browse_multi.setEnabled(True)
        self._reset()

    def _reset(self):
        self.running = False; self.paused = False
        self.btn_run.setText("▶  Run Analysis")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #00C8EE; color: #090B0F;
                border: none; border-radius: 8px;
                font-size: 13px; font-weight: 700; padding: 0 16px;
            }
            QPushButton:hover { background-color: #33D8FF; }
            QPushButton:pressed { background-color: #007A99; }
        """)
        self.btn_pause.setEnabled(False); self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)
        self.dot.set_color(C["td"])

    def _on_frame(self, snap, fps, cpu, ram, gpu):
        self.fps_lbl.setText(f"{fps:.1f} fps")
        self.meter.set_values(cpu, ram, gpu, fps)
        zones = snap.get("zones", {})
        self.board_a.update_data(zones.get("A"))
        self.board_b.update_data(zones.get("B"))
        self.comparison.update_data(snap)

    def _on_done(self, final):
        self._reset()
        self.dot.set_color(C["ok"])
        self.status_lbl.setText("Analysis complete ✓")
        self.prog.setValue(100)
        self.window().results.set_results(final)

    def _on_canvas_frame(self, qimg):
        px = QPixmap.fromImage(qimg)
        self.canvas.set_frame(px)

    def _on_error(self, msg):
        self._reset()
        self.dot.set_color(C["danger"])
        self.status_lbl.setText("Error — see details")
        QMessageBox.critical(self, "IntersectionCV — Error", msg)


# ─────────────────────────────────────────────────────────────────────

class ConfigPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setStyleSheet(f"background:{C['bg']};")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(24, 24, 24, 32)
        lay.setSpacing(16)

        t = QLabel("Configuration")
        t.setStyleSheet("font-size:19px; font-weight:700;")
        lay.addWidget(t)
        s = QLabel("Survey metadata, model, inference and tracker settings.")
        s.setStyleSheet(f"color:{C['ts']}; font-size:12px; margin-bottom:4px;")
        lay.addWidget(s)

        # ── Survey metadata
        g1 = QGroupBox("Survey Metadata")
        gl = QGridLayout(g1)
        gl.setColumnStretch(1, 1); gl.setColumnStretch(3, 1)
        gl.setHorizontalSpacing(10); gl.setVerticalSpacing(8)

        gl.addWidget(QLabel("Intersection"), 0, 0)
        self.intersection = QLineEdit()
        self.intersection.setPlaceholderText("e.g., Rajshahi Medical College Mor")
        gl.addWidget(self.intersection, 0, 1, 1, 3)

        gl.addWidget(QLabel("Date"), 1, 0)
        self.date = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        gl.addWidget(self.date, 1, 1)

        gl.addWidget(QLabel("Time from"), 1, 2)
        self.time_from = QLineEdit(); self.time_from.setPlaceholderText("08:00")
        gl.addWidget(self.time_from, 1, 3)

        gl.addWidget(QLabel("Time to"), 2, 0)
        self.time_to = QLineEdit(); self.time_to.setPlaceholderText("09:00")
        gl.addWidget(self.time_to, 2, 1)

        gl.addWidget(QLabel("Observer"), 2, 2)
        self.observer = QLineEdit(); self.observer.setPlaceholderText("Your name")
        gl.addWidget(self.observer, 2, 3)
        lay.addWidget(g1)

        # ── Model
        g2 = QGroupBox("Model")
        g2l = QGridLayout(g2)
        g2l.setColumnStretch(1, 1)
        g2l.setHorizontalSpacing(10); g2l.setVerticalSpacing(8)

        g2l.addWidget(QLabel("Model variant"), 0, 0)
        self.model_combo = QComboBox()
        for m in MODEL_SIZES:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentIndex(2)
        g2l.addWidget(self.model_combo, 0, 1)

        # PT file browser — first class citizen
        g2l.addWidget(QLabel("Custom weights (.pt)"), 1, 0)
        pt_row = QHBoxLayout(); pt_row.setSpacing(6)
        self.weights_path = QLineEdit()
        self.weights_path.setPlaceholderText(
            "Optional — browse a .pt file to override the variant above "
            "(e.g. BNVD fine-tuned weights)")
        pt_row.addWidget(self.weights_path)
        btn_browse_pt = QPushButton("Browse .pt")
        btn_browse_pt.setObjectName("btn_icon")
        btn_browse_pt.setFixedWidth(84)
        btn_browse_pt.setFixedHeight(34)
        btn_browse_pt.clicked.connect(self._browse_pt)
        pt_row.addWidget(btn_browse_pt)
        btn_clr = QPushButton("✕")
        btn_clr.setObjectName("btn_icon")
        btn_clr.setFixedWidth(30); btn_clr.setFixedHeight(34)
        btn_clr.setToolTip("Clear custom weights")
        btn_clr.clicked.connect(lambda: self.weights_path.clear())
        pt_row.addWidget(btn_clr)
        g2l.addLayout(pt_row, 1, 1)

        # hint label
        hint = QLabel(
            "BNVD pretrained weights (Bangladesh-native classes) → "
            "github.com/bipin-saha/BNVD  ·  Checkpoints folder")
        hint.setStyleSheet(f"color:{C['ts']}; font-size:10px; font-style:italic;")
        hint.setWordWrap(True)
        g2l.addWidget(hint, 2, 1)
        lay.addWidget(g2)

        # ── Inference
        g3 = QGroupBox("Inference Parameters")
        g3l = QGridLayout(g3)
        g3l.setColumnStretch(1,1); g3l.setColumnStretch(3,1)
        g3l.setHorizontalSpacing(20); g3l.setVerticalSpacing(10)

        def sl_row(lo, hi, val, fmt=lambda v: f"{v/100:.2f}"):
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi); sl.setValue(val)
            lbl = QLabel(fmt(val))
            lbl.setFixedWidth(36)
            lbl.setStyleSheet(f"color:{C['acc']}; font-weight:600;")
            sl.valueChanged.connect(lambda v,l=lbl,f=fmt: l.setText(f(v)))
            row = QHBoxLayout(); row.addWidget(sl); row.addWidget(lbl)
            return sl, row

        g3l.addWidget(QLabel("Confidence"), 0, 0)
        self.conf_slider, conf_row = sl_row(10, 95, 40)
        g3l.addLayout(conf_row, 0, 1)

        g3l.addWidget(QLabel("IOU threshold"), 1, 0)
        self.iou_slider, iou_row = sl_row(10, 90, 45)
        g3l.addLayout(iou_row, 1, 1)

        g3l.addWidget(QLabel("Image size (px)"), 0, 2)
        self.img_size = QComboBox()
        for sz in ["320","416","512","640","768","1024","1280"]:
            self.img_size.addItem(sz)
        self.img_size.setCurrentIndex(3)
        g3l.addWidget(self.img_size, 0, 3)

        g3l.addWidget(QLabel("Device"), 1, 2)
        self.device = QComboBox()
        for d in ["CPU","CUDA:0 (GPU)","CUDA:1","MPS (Apple)"]:
            self.device.addItem(d)
        g3l.addWidget(self.device, 1, 3)
        lay.addWidget(g3)

        # ── ByteTrack
        g4 = QGroupBox("Tracker — ByteTrack")
        g4l = QGridLayout(g4)
        g4l.setColumnStretch(1,1); g4l.setColumnStretch(3,1)
        g4l.setHorizontalSpacing(20); g4l.setVerticalSpacing(10)

        g4l.addWidget(QLabel("Track threshold"), 0, 0)
        self.track_thresh, tt_row = sl_row(20, 80, 50)
        g4l.addLayout(tt_row, 0, 1)

        g4l.addWidget(QLabel("Max lost frames"), 0, 2)
        self.max_lost = QSpinBox(); self.max_lost.setRange(5,90); self.max_lost.setValue(30)
        g4l.addWidget(self.max_lost, 0, 3)

        g4l.addWidget(QLabel("Min track length"), 1, 0)
        self.min_track = QSpinBox(); self.min_track.setRange(1,30); self.min_track.setValue(3)
        g4l.addWidget(self.min_track, 1, 1)

        g4l.addWidget(QLabel("Buffer frames"), 1, 2)
        self.buffer = QSpinBox(); self.buffer.setRange(10,200); self.buffer.setValue(30)
        g4l.addWidget(self.buffer, 1, 3)

        # Box smoothing
        g4l.addWidget(QLabel("Box smoothing (EMA)"), 2, 0)
        smooth_row = QHBoxLayout()
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(5, 80); self.smooth_slider.setValue(35)
        smooth_lbl = QLabel("0.35")
        smooth_lbl.setFixedWidth(36)
        smooth_lbl.setStyleSheet(f"color:{C['acc']}; font-weight:600;")
        self.smooth_slider.valueChanged.connect(
            lambda v: smooth_lbl.setText(f"{v/100:.2f}"))
        smooth_row.addWidget(self.smooth_slider); smooth_row.addWidget(smooth_lbl)
        g4l.addLayout(smooth_row, 2, 1)

        smooth_note = QLabel(
            "Lower = smoother boxes (more lag) · Higher = snappier (more flicker)")
        smooth_note.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        g4l.addWidget(smooth_note, 3, 0, 1, 4)
        lay.addWidget(g4)

        # ── Delay Parameters
        g4b = QGroupBox("Delay Parameters")
        g4bl = QGridLayout(g4b)
        g4bl.setColumnStretch(1, 1); g4bl.setColumnStretch(3, 1)
        g4bl.setHorizontalSpacing(20); g4bl.setVerticalSpacing(10)

        g4bl.addWidget(QLabel("Free-flow percentile"), 0, 0)
        self.freeflow_pct = QSpinBox()
        self.freeflow_pct.setRange(5, 50); self.freeflow_pct.setValue(int(DEFAULT_FREEFLOW_PERCENTILE * 100))
        self.freeflow_pct.setSuffix(" %")
        g4bl.addWidget(self.freeflow_pct, 0, 1)

        g4bl.addWidget(QLabel("Stop sensitivity"), 0, 2)
        self.stop_thresh = QDoubleSpinBox()
        self.stop_thresh.setRange(0.5, 20.0); self.stop_thresh.setDecimals(1)
        self.stop_thresh.setValue(DEFAULT_STOP_FRAC_PER_SEC * 100); self.stop_thresh.setSuffix(" %/s")
        g4bl.addWidget(self.stop_thresh, 0, 3)

        g4bl.addWidget(QLabel("Track-loss grace"), 1, 0)
        self.grace_sec = QDoubleSpinBox()
        self.grace_sec.setRange(0.2, 6.0); self.grace_sec.setDecimals(1)
        self.grace_sec.setValue(DEFAULT_GRACE_SEC); self.grace_sec.setSuffix(" s")
        g4bl.addWidget(self.grace_sec, 1, 1)

        g4bl.addWidget(QLabel("Timeline bucket"), 1, 2)
        self.bucket_sec = QSpinBox()
        self.bucket_sec.setRange(5, 120); self.bucket_sec.setValue(int(DEFAULT_BUCKET_SEC))
        self.bucket_sec.setSuffix(" s")
        g4bl.addWidget(self.bucket_sec, 1, 3)

        g4bl.addWidget(QLabel("Speed averaging window"), 2, 0)
        self.speed_window = QDoubleSpinBox()
        self.speed_window.setRange(0.1, 2.0); self.speed_window.setDecimals(1)
        self.speed_window.setValue(DEFAULT_SPEED_WINDOW_SEC); self.speed_window.setSuffix(" s")
        g4bl.addWidget(self.speed_window, 2, 1)

        g4bl.addWidget(QLabel("Motion noise floor"), 2, 2)
        self.motion_floor = QDoubleSpinBox()
        self.motion_floor.setRange(0.0, 5.0); self.motion_floor.setDecimals(2)
        self.motion_floor.setValue(DEFAULT_MOTION_FLOOR_FRAC * 100); self.motion_floor.setSuffix(" %")
        g4bl.addWidget(self.motion_floor, 2, 3)

        note_smooth = QLabel(
            "If speed/queue readings look jumpy: raise the averaging window "
            "(speed is measured across this rolling span instead of frame-to-"
            "frame, so detection jitter gets averaged out) and/or raise the "
            "motion floor (per-frame movement below this % of a zone's own "
            "size is treated as box jitter, not travel). Raise stop "
            "sensitivity too if queued vehicles flicker in and out of the "
            "queue count.")
        note_smooth.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        note_smooth.setWordWrap(True)
        g4bl.addWidget(note_smooth, 3, 0, 1, 4)

        note_delay = QLabel(
            "No manual speed calibration needed — each zone learns its own "
            "free-flow baseline live, from the fastest crossings it actually "
            "observes (the bottom N% of real crossing times). Travel delay is "
            "then measured against that zone's own baseline, so Road A and "
            "Road B stay directly comparable even if their shapes/lengths "
            "differ. The baseline needs a handful of crossings before it's "
            "trusted — expect early readings to under-report delay.")
        note_delay.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        note_delay.setWordWrap(True)
        g4bl.addWidget(note_delay, 4, 0, 1, 4)
        lay.addWidget(g4b)

        # ── Video Source
        g4c = QGroupBox("Video Source")
        g4cl = QGridLayout(g4c)
        g4cl.setColumnStretch(1, 1)
        g4cl.setHorizontalSpacing(10); g4cl.setVerticalSpacing(8)

        g4cl.addWidget(QLabel("Source"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Recorded file", "Live stream (RTSP) — coming soon"])
        g4cl.addWidget(self.source_combo, 0, 1)

        g4cl.addWidget(QLabel("RTSP URL"), 1, 0)
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://…  —  live ingestion arrives in a later update")
        self.rtsp_url.setEnabled(False)
        g4cl.addWidget(self.rtsp_url, 1, 1)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(g4c)

        # ── Performance
        g4e = QGroupBox("Performance")
        g4el = QGridLayout(g4e)
        g4el.setColumnStretch(1, 1); g4el.setColumnStretch(3, 1)
        g4el.setHorizontalSpacing(20); g4el.setVerticalSpacing(10)

        g4el.addWidget(QLabel("Compute device"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (use GPU if available)", "GPU (CUDA)", "CPU"])
        g4el.addWidget(self.device_combo, 0, 1)

        g4el.addWidget(QLabel("Precision"), 0, 2)
        self.half_combo = QComboBox()
        self.half_combo.addItems(["Auto (FP16 on GPU)", "FP16 (GPU only)", "FP32"])
        g4el.addWidget(self.half_combo, 0, 3)

        g4el.addWidget(QLabel("Inference resolution"), 1, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["416 (fastest)", "480", "640 (default)", "960", "1280 (most accurate)"])
        self.imgsz_combo.setCurrentIndex(2)
        g4el.addWidget(self.imgsz_combo, 1, 1)

        g4el.addWidget(QLabel("Preview refresh rate"), 1, 2)
        self.preview_fps = QSpinBox()
        self.preview_fps.setRange(2, 30); self.preview_fps.setValue(15)
        self.preview_fps.setSuffix(" fps")
        g4el.addWidget(self.preview_fps, 1, 3)

        note_perf = QLabel(
            "Every frame is always fully analysed for delay — none of this "
            "skips frames. It only changes how fast each frame is processed "
            "(device/precision/resolution) and how often the live preview "
            "redraws. If a 16-minute video is taking hours: check the status "
            "line when a run starts — it reports the actual device in use. "
            "'running on CPU' when you have a GPU usually means PyTorch isn't "
            "installed with CUDA support. Lowering inference resolution and "
            "using a smaller model (e.g. yolov8n/s instead of l/x) also helps "
            "a lot on a 4GB-class GPU.")
        note_perf.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        note_perf.setWordWrap(True)
        g4el.addWidget(note_perf, 2, 0, 1, 4)
        lay.addWidget(g4e)

        # ── Pedestrians & Capacity
        g4d = QGroupBox("Pedestrians & Capacity")
        g4dl = QGridLayout(g4d)
        g4dl.setColumnStretch(1, 1); g4dl.setColumnStretch(3, 1)
        g4dl.setHorizontalSpacing(20); g4dl.setVerticalSpacing(10)

        self.track_peds = QCheckBox("Track pedestrians / hawkers")
        self.track_peds.setChecked(True)
        g4dl.addWidget(self.track_peds, 0, 0, 1, 2)

        g4dl.addWidget(QLabel("Loiter threshold"), 0, 2)
        self.loiter_sec = QDoubleSpinBox()
        self.loiter_sec.setRange(3.0, 120.0); self.loiter_sec.setDecimals(0)
        self.loiter_sec.setValue(DEFAULT_LOITER_SEC); self.loiter_sec.setSuffix(" s")
        g4dl.addWidget(self.loiter_sec, 0, 3)

        g4dl.addWidget(QLabel(f"{VideoCanvas.ZONE_LABELS['A']} capacity"), 1, 0)
        self.capacity_a = QSpinBox()
        self.capacity_a.setRange(0, 20000); self.capacity_a.setValue(0)
        self.capacity_a.setSuffix(" PCU/hr"); self.capacity_a.setSpecialValueText("unknown")
        g4dl.addWidget(self.capacity_a, 1, 1)

        g4dl.addWidget(QLabel(f"{VideoCanvas.ZONE_LABELS['B']} capacity"), 1, 2)
        self.capacity_b = QSpinBox()
        self.capacity_b.setRange(0, 20000); self.capacity_b.setValue(0)
        self.capacity_b.setSuffix(" PCU/hr"); self.capacity_b.setSpecialValueText("unknown")
        g4dl.addWidget(self.capacity_b, 1, 3)

        note_ped = QLabel(
            "A pedestrian/hawker dwelling in a zone past the loiter threshold reads "
            "as vendor-like activity rather than someone just passing through. "
            "Capacity (in PCU/hr — see the note below) is optional: leave at 0/"
            "'unknown' to skip the volume/capacity ratio and just see raw PCU "
            "throughput; set it (e.g. from a lane-count/width estimate) to get "
            "an over-capacity vs. friction-caused-congestion diagnosis.")
        note_ped.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        note_ped.setWordWrap(True)
        g4dl.addWidget(note_ped, 2, 0, 1, 4)
        lay.addWidget(g4d)

        # ── Comparison Zones
        g5 = QGroupBox("Comparison Zones (ROI A / ROI B)")
        g5l = QVBoxLayout(g5)
        g5l.setSpacing(10)

        note = QLabel(
            "Drag the 6 corner handles of each zone directly on the video "
            "preview (Analysis tab) to fit the two roads/approaches you want "
            "to compare — Road A and Road B are analysed and reported "
            "completely independently, then compared against each other.")
        note.setStyleSheet(f"color:{C['ts']}; font-size:11px;")
        note.setWordWrap(True)
        g5l.addWidget(note)

        site_row = QHBoxLayout(); site_row.setSpacing(8)
        site_row.addWidget(QLabel("Site name"))
        self.site_name = QLineEdit("Saheb Bazar – Alupotti Rd")
        site_row.addWidget(self.site_name, 1)
        btn_save = QPushButton("💾 Save Layout")
        btn_save.setObjectName("btn_ghost"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_layout)
        site_row.addWidget(btn_save)
        btn_load = QPushButton("📂 Load Layout")
        btn_load.setObjectName("btn_ghost"); btn_load.setFixedHeight(32)
        btn_load.clicked.connect(self._load_layout)
        site_row.addWidget(btn_load)
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setObjectName("btn_icon")
        btn_reset.setFixedHeight(32); btn_reset.setFixedWidth(70)
        btn_reset.clicked.connect(self._reset_layout)
        site_row.addWidget(btn_reset)
        g5l.addLayout(site_row)

        # ROI corners — one 6-row box per zone, plus an optional
        # reference line for physical queue-length reporting
        self.roi_spins = {"A": [], "B": []}
        self.ref_widgets = {}
        for zone in ("A", "B"):
            zbox = QGroupBox(f"{VideoCanvas.ZONE_LABELS[zone]}  ·  Zone {zone} corners (% of frame)")
            zbox.setStyleSheet(zbox.styleSheet() +
                f"QGroupBox {{ color: {VideoCanvas.ZONE_COLORS[zone]}; }}")
            zgrid = QGridLayout(zbox)
            zgrid.setHorizontalSpacing(14); zgrid.setVerticalSpacing(6)
            for i in range(6):
                zgrid.addWidget(QLabel(f"{zone}{i+1}"), i, 0)
                xs = QSpinBox(); xs.setRange(0, 100); xs.setSuffix(" % x")
                ys = QSpinBox(); ys.setRange(0, 100); ys.setSuffix(" % y")
                zgrid.addWidget(xs, i, 1)
                zgrid.addWidget(ys, i, 2)
                self.roi_spins[zone].append((xs, ys))

            ref_enable = QCheckBox("Reference line (optional — enables queue length in metres)")
            zgrid.addWidget(ref_enable, 6, 0, 1, 3)
            rx1 = QSpinBox(); rx1.setRange(0, 100); rx1.setSuffix(" % x1")
            ry1 = QSpinBox(); ry1.setRange(0, 100); ry1.setSuffix(" % y1")
            rx2 = QSpinBox(); rx2.setRange(0, 100); rx2.setSuffix(" % x2")
            ry2 = QSpinBox(); ry2.setRange(0, 100); ry2.setSuffix(" % y2")
            rlen = QDoubleSpinBox(); rlen.setRange(0.1, 200.0); rlen.setDecimals(1)
            rlen.setValue(5.0); rlen.setSuffix(" m")
            zgrid.addWidget(rx1, 7, 0); zgrid.addWidget(ry1, 7, 1); zgrid.addWidget(rx2, 7, 2)
            ref_row2 = QHBoxLayout(); ref_row2.addWidget(ry2); ref_row2.addWidget(rlen)
            zgrid.addLayout(ref_row2, 8, 0, 1, 3)
            self.ref_widgets[zone] = dict(enable=ref_enable, x1=rx1, y1=ry1, x2=rx2, y2=ry2, length=rlen)

            g5l.addWidget(zbox)
        lay.addWidget(g5)

        self._syncing = False
        for zone in ("A", "B"):
            for xs, ys in self.roi_spins[zone]:
                xs.valueChanged.connect(lambda _v: self._push_layout_to_canvas())
                ys.valueChanged.connect(lambda _v: self._push_layout_to_canvas())
            rw = self.ref_widgets[zone]
            rw["enable"].toggled.connect(lambda _v: self._push_layout_to_canvas())
            for key in ("x1", "y1", "x2", "y2", "length"):
                rw[key].valueChanged.connect(lambda _v: self._push_layout_to_canvas())

        # ── Session Log
        g5b = QGroupBox("Session Log")
        g5bl = QGridLayout(g5b)
        g5bl.setColumnStretch(1, 1)
        g5bl.setHorizontalSpacing(10); g5bl.setVerticalSpacing(8)

        self.auto_log = QCheckBox("Auto-log each completed run")
        self.auto_log.setChecked(True)
        g5bl.addWidget(self.auto_log, 0, 0, 1, 3)

        g5bl.addWidget(QLabel("Log file"), 1, 0)
        log_row = QHBoxLayout(); log_row.setSpacing(6)
        self.log_path = QLineEdit("traffic_delay_sessions.csv")
        log_row.addWidget(self.log_path)
        btn_log_browse = QPushButton("Browse")
        btn_log_browse.setObjectName("btn_icon"); btn_log_browse.setFixedWidth(70); btn_log_browse.setFixedHeight(32)
        btn_log_browse.clicked.connect(self._browse_log_path)
        log_row.addWidget(btn_log_browse)
        g5bl.addLayout(log_row, 1, 1, 1, 2)

        g5bl.addWidget(QLabel("Conditions / notes"), 2, 0)
        self.session_notes = QLineEdit()
        self.session_notes.setPlaceholderText(
            "e.g. haat/market day, heavy rain, regular weekday…")
        g5bl.addWidget(self.session_notes, 2, 1, 1, 2)

        note_log = QLabel(
            "Every completed run appends one row here (date, time, both zones' "
            "delay/volume/pedestrian summary, and this note) — run multiple "
            "sessions across different times/days and you'll have a growing "
            "dataset to compare, instead of stitching separate exports together.")
        note_log.setStyleSheet(f"color:{C['ts']}; font-size:10px;")
        note_log.setWordWrap(True)
        g5bl.addWidget(note_log, 3, 0, 1, 3)
        lay.addWidget(g5b)

        # ── Classes
        g6 = QGroupBox("Active Vehicle Classes")
        g6l = QGridLayout(g6)
        g6l.setHorizontalSpacing(12); g6l.setVerticalSpacing(8)
        self.class_checks = {}
        for i, cls in enumerate(VEHICLE_CLASSES):
            cb = QCheckBox(cls); cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color:{C[f'v{i}']}; }}")
            self.class_checks[cls] = cb
            g6l.addWidget(cb, i // 4, i % 4)
        lay.addWidget(g6)
        lay.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)

    def _browse_pt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select PyTorch Weights", "",
            "PyTorch Weights (*.pt *.pth);;All Files (*)")
        if p: self.weights_path.setText(p)

    # ── ROI / calibration sync with the video canvas ────────────────────

    def _get_canvas(self):
        w = self.window()
        return getattr(getattr(w, "analysis", None), "canvas", None)

    def _push_layout_to_canvas(self):
        """Spinboxes → canvas (user typed/edited a value)."""
        if self._syncing:
            return
        canvas = self._get_canvas()
        if canvas is None:
            return
        for zone in ("A", "B"):
            pts = [(xs.value() / 100, ys.value() / 100) for xs, ys in self.roi_spins[zone]]
            canvas.set_roi(zone, pts)
            rw = self.ref_widgets[zone]
            canvas.set_ref(zone, {
                "enabled": rw["enable"].isChecked(),
                "p1": (rw["x1"].value() / 100, rw["y1"].value() / 100),
                "p2": (rw["x2"].value() / 100, rw["y2"].value() / 100),
                "length_m": rw["length"].value(),
            })

    def _pull_layout_from_canvas(self):
        """Canvas (user dragged a handle) → spinboxes."""
        canvas = self._get_canvas()
        if canvas is None:
            return
        self._syncing = True
        try:
            for zone in ("A", "B"):
                pts = canvas.get_roi(zone)
                for (xs, ys), (nx, ny) in zip(self.roi_spins[zone], pts):
                    xs.setValue(round(nx * 100)); ys.setValue(round(ny * 100))
                ref = canvas.get_ref(zone)
                rw = self.ref_widgets[zone]
                rw["enable"].setChecked(ref.get("enabled", False))
                rw["x1"].setValue(round(ref["p1"][0] * 100)); rw["y1"].setValue(round(ref["p1"][1] * 100))
                rw["x2"].setValue(round(ref["p2"][0] * 100)); rw["y2"].setValue(round(ref["p2"][1] * 100))
                rw["length"].setValue(ref.get("length_m", 5.0))
        finally:
            self._syncing = False

    def _current_layout_dict(self):
        canvas = self._get_canvas()
        if canvas:
            rois, refs = canvas.get_rois(), canvas.get_refs()
        else:
            rois = {"A": [tuple(p) for p in VideoCanvas.ROI_A_DEFAULT],
                    "B": [tuple(p) for p in VideoCanvas.ROI_B_DEFAULT]}
            refs = {z: dict(v) for z, v in VideoCanvas.REF_DEFAULT.items()}
        return {"site_name": self.site_name.text(), "rois": rois, "refs": refs}

    def _save_layout(self):
        data = self._current_layout_dict()
        default_name = (self.site_name.text() or "site").strip().replace(" ", "_") + "_layout.json"
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Zone Layout", default_name, "JSON Files (*.json)")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Layout saved to:\n{p}")

    def _load_layout(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load Zone Layout", "", "JSON Files (*.json)")
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.site_name.setText(data.get("site_name", self.site_name.text()))
        canvas = self._get_canvas()
        if canvas:
            rois = data.get("rois", {
                "A": VideoCanvas.ROI_A_DEFAULT, "B": VideoCanvas.ROI_B_DEFAULT})
            canvas.set_rois(rois)
            refs = data.get("refs")
            if refs:
                canvas.set_refs(refs)
            self._pull_layout_from_canvas()

    def _reset_layout(self):
        canvas = self._get_canvas()
        if canvas:
            canvas.set_rois({
                "A": [tuple(p) for p in VideoCanvas.ROI_A_DEFAULT],
                "B": [tuple(p) for p in VideoCanvas.ROI_B_DEFAULT],
            })
            canvas.set_refs({z: dict(v) for z, v in VideoCanvas.REF_DEFAULT.items()})
            self._pull_layout_from_canvas()

    def get_meta(self):
        return dict(
            intersection=self.intersection.text(),
            date=self.date.text(),
            time_from=self.time_from.text(),
            time_to=self.time_to.text(),
            observer=self.observer.text(),
        )

    def _on_source_changed(self, idx):
        if idx == 1:
            QMessageBox.information(self, "Coming soon",
                "Live CCTV/RTSP ingestion isn't wired up yet — this release "
                "analyses recorded video files. Switching back to Recorded file.")
            self.source_combo.blockSignals(True)
            self.source_combo.setCurrentIndex(0)
            self.source_combo.blockSignals(False)

    def get_delay_params(self):
        return dict(
            freeflow_percentile=self.freeflow_pct.value() / 100.0,
            stop_frac_per_sec=self.stop_thresh.value() / 100.0,
            grace_sec=self.grace_sec.value(),
            bucket_sec=float(self.bucket_sec.value()),
            speed_window_sec=self.speed_window.value(),
            motion_floor_frac=self.motion_floor.value() / 100.0,
            track_pedestrians=self.track_peds.isChecked(),
            loiter_sec=self.loiter_sec.value(),
        )

    def get_perf_params(self):
        device = ["auto", "cuda", "cpu"][self.device_combo.currentIndex()]
        half   = ["auto", True, False][self.half_combo.currentIndex()]
        imgsz  = [416, 480, 640, 960, 1280][self.imgsz_combo.currentIndex()]
        return dict(device=device, half=half, imgsz=imgsz,
                    preview_fps=float(self.preview_fps.value()))

    def get_active_classes(self):
        return {cls for cls, cb in self.class_checks.items() if cb.isChecked()}

    def get_capacities(self):
        return {
            "A": self.capacity_a.value() or None,
            "B": self.capacity_b.value() or None,
        }

    def _browse_log_path(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Session Log File", self.log_path.text() or "traffic_delay_sessions.csv",
            "CSV Files (*.csv)")
        if p: self.log_path.setText(p)

    def get_session_log_config(self):
        return dict(
            enabled=self.auto_log.isChecked(),
            path=self.log_path.text().strip() or "traffic_delay_sessions.csv",
            notes=self.session_notes.text().strip(),
        )


# ─────────────────────────────────────────────────────────────────────

def write_summary_csv(path, data, meta, site_name=""):
    """
    Write the standard delay-summary CSV — shared by the interactive
    Export button and the multi-video batch runner so both always
    produce identical output. Delay is the headline metric: a
    quick-glance table comes first, before the fuller per-zone/
    per-class breakdown.
    """
    zones = data.get("zones", {})
    cmp   = data.get("comparison", {})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["IntersectionCV — Traffic Delay Summary"])
        if site_name:
            w.writerow(["Site", site_name])
        for k, v in meta.items():
            w.writerow([k.capitalize(), v])
        w.writerow([])

        # ── delay at a glance — the main thing this file is for ──
        w.writerow(["DELAY AT A GLANCE"])
        w.writerow(["Zone", "Vehicles", "Avg Delay (s)", "Median Delay (s)",
                    "Total Delay (veh·min)", "Stalls/Vehicle", "Congestion"])
        for z, zi in zones.items():
            median = zi.get("median_delay")
            w.writerow([
                zi.get("label", z), zi.get("vehicles", 0),
                f'{zi.get("avg_delay", 0.0):.2f}',
                f"{median:.2f}" if median is not None else "",
                f'{zi.get("total_delay", 0.0) / 60:.2f}',
                f'{zi.get("avg_stops", 0.0):.2f}',
                zi.get("congestion", ("", ""))[0],
            ])
        w.writerow([])
        more = cmp.get("more_delayed_zone")
        more_label = zones.get(more, {}).get("label", "—") if more else "— (roughly even)"
        w.writerow(["More delayed road", more_label])
        w.writerow(["Delay difference (s/vehicle)", f'{cmp.get("delay_diff_s", 0.0):.2f}'])
        w.writerow([])

        # ── per-zone detail ──
        for z, zi in zones.items():
            w.writerow([f'{zi.get("label", z)} — detail'])
            w.writerow(["Vehicles analysed", zi.get("vehicles", 0)])
            w.writerow(["Total delay (veh·min)", f'{zi.get("total_delay", 0.0) / 60:.2f}'])
            w.writerow(["Avg delay (s/vehicle)", f'{zi.get("avg_delay", 0.0):.2f}'])
            median = zi.get("median_delay"); p85 = zi.get("p85_delay")
            w.writerow(["Median delay (s/vehicle)", f"{median:.2f}" if median is not None else ""])
            w.writerow(["85th percentile delay (s/vehicle)", f"{p85:.2f}" if p85 is not None else ""])
            w.writerow(["% vehicles meaningfully delayed", f'{zi.get("pct_delayed", 0.0):.1f}'])
            w.writerow(["Avg stopped time (s/vehicle)", f'{zi.get("avg_stopped", 0.0):.2f}'])
            w.writerow(["Avg stalls (stop-start cycles) / vehicle", f'{zi.get("avg_stops", 0.0):.2f}'])
            w.writerow(["Acceleration noise (motion smoothness, 0=smooth)", f'{zi.get("avg_accel_noise", 0.0):.4f}'])
            w.writerow(["Max queue (vehicles)", zi.get("max_queue", 0)])
            qlen = zi.get("queue_length_m")
            w.writerow(["Queue length (m)", f"{qlen:.1f}" if qlen is not None else "no reference line set"])
            w.writerow(["Congestion level", zi.get("congestion", ("—", ""))[0]])
            ff = zi.get("free_flow_time")
            w.writerow(["Learned free-flow crossing time (s)", f"{ff:.2f}" if ff is not None else "not yet established"])
            w.writerow(["PCU volume (PCU/hr)", f'{zi.get("pcu_per_hour", 0.0):.1f}'])
            vc = zi.get("vc_ratio")
            w.writerow(["Capacity (PCU/hr)", zi.get("capacity_pcu_hr") or "unknown"])
            w.writerow(["Volume/Capacity ratio", f"{vc:.2f}" if vc is not None else "n/a (no capacity set)"])
            if vc is not None:
                w.writerow(["Capacity utilisation level", zi.get("vc_level", ("", ""))[0]])
            ped = zi.get("pedestrians", {})
            w.writerow(["Pedestrians/hawkers currently present", ped.get("current", 0)])
            w.writerow(["Pedestrians/hawkers loitering now", ped.get("loitering_now", 0)])
            w.writerow(["Avg pedestrian dwell time (s)", f'{ped.get("avg_dwell_s", 0.0):.1f}'])
            w.writerow(["Vehicle class", "Count", "Avg Delay (s)", "Avg Stopped (s)"])
            for cls in VEHICLE_CLASSES:
                st = zi.get("class_summary", {}).get(cls, {})
                count = st.get("count", 0)
                avg_d = (st.get("delay", 0.0) / count) if count else 0.0
                avg_s = (st.get("stopped", 0.0) / count) if count else 0.0
                w.writerow([cls, count, f"{avg_d:.1f}", f"{avg_s:.1f}"])
            w.writerow([])


class ResultsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(14)

        hdr = QHBoxLayout()
        t = QLabel("Results & Export")
        t.setStyleSheet("font-size:19px; font-weight:700;")
        hdr.addWidget(t)
        hdr.addStretch()

        self.btn_csv = QPushButton("Export Summary CSV")
        self.btn_csv.setObjectName("btn_primary")
        self.btn_csv.setFixedHeight(34); self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._csv)

        self.btn_log = QPushButton("Export Vehicle Log")
        self.btn_log.setObjectName("btn_ghost")
        self.btn_log.setFixedHeight(34); self.btn_log.setEnabled(False)
        self.btn_log.clicked.connect(self._csv_log)

        self.btn_json = QPushButton("Export JSON")
        self.btn_json.setObjectName("btn_ghost")
        self.btn_json.setFixedHeight(34); self.btn_json.setEnabled(False)
        self.btn_json.clicked.connect(self._json)

        self.btn_prev = QPushButton("Preview Sheet")
        self.btn_prev.setObjectName("btn_ghost")
        self.btn_prev.setFixedHeight(34); self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._preview)

        for b in (self.btn_csv, self.btn_log, self.btn_json, self.btn_prev):
            hdr.addWidget(b)
        lay.addLayout(hdr)

        cmp_card = _card()
        cmpl = QVBoxLayout(cmp_card)
        cmpl.setContentsMargins(14, 10, 14, 10)
        cmpl.setSpacing(6)
        cmpl.addWidget(_lbl("Road A vs Road B"))
        self.comparison = ComparisonPanel()
        cmpl.addWidget(self.comparison)
        lay.addWidget(cmp_card)

        self.tabs = QTabWidget()
        self.board_a = ZoneBoard("A")
        self.board_b = ZoneBoard("B")
        self.tabs.addTab(self.board_a, VideoCanvas.ZONE_LABELS["A"])
        self.tabs.addTab(self.board_b, VideoCanvas.ZONE_LABELS["B"])
        lay.addWidget(self.tabs, 1)

        self.empty_lbl = QLabel("No results yet — run an analysis first.")
        self.empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_lbl.setStyleSheet(f"color:{C['td']}; font-size:13px;")
        lay.addWidget(self.empty_lbl)

    def set_results(self, data):
        self._data = data
        self.empty_lbl.hide()
        zones = data.get("zones", {})
        self.board_a.update_data(zones.get("A"))
        self.board_b.update_data(zones.get("B"))
        self.comparison.update_data(data)
        for b in (self.btn_csv, self.btn_log, self.btn_json, self.btn_prev):
            b.setEnabled(True)
        self._append_session_log(data)

    def _append_session_log(self, data):
        cfg = self.window().cfg
        log_cfg = cfg.get_session_log_config()
        if not log_cfg["enabled"]:
            return
        path = log_cfg["path"]
        meta = cfg.get_meta()
        zones = data.get("zones", {})
        cmp   = data.get("comparison", {})

        fieldnames = ["logged_at", "date", "time_from", "time_to",
                      "intersection", "site_name", "notes"]
        for z in ("A", "B"):
            fieldnames += [f"{z}_label", f"{z}_vehicles", f"{z}_avg_delay_s",
                           f"{z}_median_delay_s", f"{z}_p85_delay_s", f"{z}_pct_delayed",
                           f"{z}_pcu_per_hour", f"{z}_vc_ratio", f"{z}_congestion",
                           f"{z}_ped_loitering_now", f"{z}_queue_length_m"]
        fieldnames += ["more_delayed_zone", "delay_diff_s"]

        row = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "date": meta.get("date", ""), "time_from": meta.get("time_from", ""),
            "time_to": meta.get("time_to", ""), "intersection": meta.get("intersection", ""),
            "site_name": cfg.site_name.text(), "notes": log_cfg["notes"],
        }
        for z in ("A", "B"):
            zi = zones.get(z, {})
            row[f"{z}_label"]    = zi.get("label", z)
            row[f"{z}_vehicles"] = zi.get("vehicles", 0)
            row[f"{z}_avg_delay_s"]    = f'{zi.get("avg_delay", 0.0):.2f}'
            row[f"{z}_median_delay_s"] = f'{(zi.get("median_delay") or 0.0):.2f}'
            row[f"{z}_p85_delay_s"]    = f'{(zi.get("p85_delay") or 0.0):.2f}'
            row[f"{z}_pct_delayed"]    = f'{zi.get("pct_delayed", 0.0):.1f}'
            row[f"{z}_pcu_per_hour"]   = f'{zi.get("pcu_per_hour", 0.0):.1f}'
            vc = zi.get("vc_ratio")
            row[f"{z}_vc_ratio"]     = f"{vc:.2f}" if vc is not None else ""
            row[f"{z}_congestion"]   = zi.get("congestion", ("", ""))[0]
            row[f"{z}_ped_loitering_now"] = zi.get("pedestrians", {}).get("loitering_now", 0)
            qlen = zi.get("queue_length_m")
            row[f"{z}_queue_length_m"] = f"{qlen:.1f}" if qlen is not None else ""
        more = cmp.get("more_delayed_zone")
        row["more_delayed_zone"] = zones.get(more, {}).get("label", "") if more else ""
        row["delay_diff_s"] = f'{cmp.get("delay_diff_s", 0.0):.2f}'

        file_exists = os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            QMessageBox.warning(self, "Session log",
                f"Could not write to session log:\n{path}\n\n{e}")

    def _csv(self):
        if not self._data: return
        p, _ = QFileDialog.getSaveFileName(self, "Save Summary CSV",
            "traffic_delay_summary.csv", "CSV Files (*.csv)")
        if not p: return
        cfg  = self.window().cfg
        meta = cfg.get_meta()
        write_summary_csv(p, self._data, meta, cfg.site_name.text())
        QMessageBox.information(self, "Exported", f"Saved to:\n{p}")

    def _csv_log(self):
        if not self._data: return
        p, _ = QFileDialog.getSaveFileName(self, "Save Vehicle Log CSV",
            "traffic_delay_vehicle_log.csv", "CSV Files (*.csv)")
        if not p: return
        per_vehicle = self._data.get("per_vehicle", {})
        zones = self._data.get("zones", {})
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["zone", "track_id", "class", "entry_time_s", "exit_time_s",
                        "time_in_zone_s", "travel_delay_s", "stopped_delay_s", "pace_pct",
                        "stalls_count", "accel_noise"])
            for z, recs in per_vehicle.items():
                zone_label = zones.get(z, {}).get("label", z)
                for rec in recs:
                    w.writerow([
                        zone_label, rec.get("track_id"), rec.get("cls"),
                        f'{rec.get("entry_time", 0):.2f}', f'{rec.get("exit_time", 0):.2f}',
                        f'{rec.get("time_in_zone", 0):.2f}', f'{rec.get("travel_delay", 0):.2f}',
                        f'{rec.get("stopped_delay", 0):.2f}', f'{rec.get("pace_pct", 0):.1f}',
                        rec.get("stops_count", 0), f'{rec.get("accel_noise", 0.0):.4f}',
                    ])
        QMessageBox.information(self, "Exported", f"Saved to:\n{p}")

    def _json(self):
        if not self._data: return
        p, _ = QFileDialog.getSaveFileName(self, "Save JSON",
            "traffic_delay.json", "JSON Files (*.json)")
        if not p: return
        meta = self.window().cfg.get_meta()
        out  = {"metadata": meta, "results": self._data,
                "exported_at": datetime.now().isoformat()}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        QMessageBox.information(self, "Exported", f"Saved to:\n{p}")

    def _preview(self):
        if not self._data: return
        dlg = SheetPreview(self._data, self.window().cfg.get_meta(), self)
        dlg.exec()


class SheetPreview(QDialog):
    def __init__(self, data, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traffic Delay Data Sheet Preview")
        self.setMinimumSize(980, 620)
        self.setStyleSheet(QSS)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(12)

        hdr = QLabel("TRAFFIC DELAY DATA SHEET")
        hdr.setStyleSheet(
            f"font-size:22px; font-weight:700; color:{C['danger']}; letter-spacing:1.5px;")
        lay.addWidget(hdr)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:{C['danger']}; border:none;"); sep.setFixedHeight(2)
        lay.addWidget(sep)

        mr = QHBoxLayout()
        mr.addWidget(QLabel(f"Intersection: {meta.get('intersection', '___________')}"))
        mr.addStretch()
        mr.addWidget(QLabel(
            f"Date: {meta.get('date', '_______')}   "
            f"Time: {meta.get('time_from', '__')} – {meta.get('time_to', '__')}"))
        lay.addLayout(mr)

        zones = data.get("zones", {})
        cmp   = data.get("comparison", {})
        more  = cmp.get("more_delayed_zone")
        if more and zones.get(more):
            other = "B" if more == "A" else "A"
            summary_txt = (
                f"{zones[more]['label']} is experiencing about "
                f"{abs(cmp.get('delay_diff_s', 0.0)):.1f}s more delay per vehicle than "
                f"{zones.get(other, {}).get('label', other)}.")
        else:
            summary_txt = "Both roads are running about the same."
        summary = QLabel(summary_txt)
        summary.setStyleSheet("font-size:13px; font-weight:600;")
        summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(summary)

        row = QHBoxLayout(); row.setSpacing(16)
        for z, zi in zones.items():
            col = QVBoxLayout()
            zname = QLabel(
                f'{zi.get("label", z)}  ·  {zi.get("vehicles", 0)} vehicles  ·  '
                f'{zi.get("avg_delay", 0.0):.1f}s avg delay  ·  {zi.get("congestion", ("—",""))[0]}')
            zname.setStyleSheet(
                f"color:{VideoCanvas.ZONE_COLORS.get(z, C['acc'])}; font-weight:700; font-size:12px;")
            zname.setWordWrap(True)
            col.addWidget(zname)

            tbl = QTableWidget(len(VEHICLE_CLASSES), 3)
            tbl.setHorizontalHeaderLabels(["Count", "Avg Delay (s)", "Avg Stopped (s)"])
            tbl.setVerticalHeaderLabels(VEHICLE_CLASSES)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tbl.verticalHeader().setDefaultSectionSize(26)
            tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            cls_summary = zi.get("class_summary", {})
            for r, cls in enumerate(VEHICLE_CLASSES):
                st = cls_summary.get(cls, {})
                count = st.get("count", 0)
                avg_d = (st.get("delay", 0.0) / count) if count else 0.0
                avg_s = (st.get("stopped", 0.0) / count) if count else 0.0
                vals = [str(count), f"{avg_d:.1f}", f"{avg_s:.1f}"] if count else ["", "", ""]
                for c, val in enumerate(vals):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
                    item.setForeground(QBrush(QColor(C[f"v{r}"])))
                    tbl.setItem(r, c, item)
            col.addWidget(tbl)
            row.addLayout(col)
        lay.addLayout(row, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject); lay.addWidget(btns)


class HelpPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(14)

        t = QLabel("Help & Documentation")
        t.setStyleSheet("font-size:19px; font-weight:700;")
        lay.addWidget(t)

        tabs = QTabWidget()
        def mktab(md):
            te = QTextEdit(); te.setReadOnly(True); te.setMarkdown(md)
            te.setStyleSheet(f"""
                QTextEdit {{
                    background:{C['s1']}; border:1px solid {C['bd']};
                    border-radius:10px; color:{C['tp']};
                    padding:14px; font-size:12px; }}
            """)
            return te

        tabs.addTab(mktab(f"""
## Quick start

1. **Configuration** — fill in survey metadata, pick YOLOv8 or YOLO26 variant,
   browse a `.pt` file for custom weights (e.g. BNVD fine-tuned), and check the
   **Delay Parameters** defaults (they self-adjust per zone — usually no
   need to touch them on a first run).
2. **Analysis** — drag the 6 corner handles of **Zone A** and **Zone B** on
   the video preview to fit the two roads/approaches you want to compare
   (or type exact % positions on the Configuration tab), then hit **▶ Run
   Analysis**. Each zone gets its own live board (stat cards, flow
   timeline, per-class delay table) plus a Road A vs Road B comparison
   panel, all updating in real time.
3. **Results** — export the summary CSV, the full per-vehicle log CSV,
   JSON, or preview the data sheet — any time, including mid-run.

## How delay is measured — no calibration needed

There are no measuring bars to place and no manual speed to configure.
Each zone learns its **own free-flow baseline live**, from the fastest
crossings it actually observes (the bottom ~15% of real crossing times —
the same idea traffic engineers use to estimate free-flow speed
empirically). That baseline needs a handful of crossings before it's
trusted, so expect early readings in a run to under-report delay.

- **Travel delay** = actual time crossing the zone − that zone's own
  learned free-flow crossing time.
- **Stopped delay** = time spent below a stop-speed threshold, scaled to
  that zone's own on-screen size — no real-world units required.
- Because both zones report delay in the same relative terms (seconds
  against each one's own baseline), **Road A and Road B are directly
  comparable** even if they're completely different lengths or shapes.
- A vehicle that vanishes briefly (occlusion, or a tracker ID switch) is
  held as a "ghost" and re-linked to the next nearby detection instead of
  being double-counted as two separate trips.

Congestion level (FREE FLOW → SEVERE) is graded from each zone's running
average delay per vehicle.

## What counts as "slowed down"

Delay isn't just a stopwatch on entry/exit — each vehicle's actual motion
is tracked throughout its crossing (position sampled every frame, speed
estimated over a short rolling window to stay stable against detection
jitter):

- **Stalls (stop-start cycles)** — every time a vehicle drops below the
  stop-speed threshold and later moves again counts as one stall. A
  vehicle that stops once for 10s and one that stops-and-starts five
  times for 2s each can show the same total stopped time, but the
  stall count tells them apart.
- **Motion smoothness / acceleration noise** — how much a vehicle's speed
  fluctuates over its crossing (near zero for steady travel, high for
  stop-and-go, jerky motion) — a classical traffic-engineering measure of
  how disrupted a traffic stream is, independent of how much total time
  was lost.

Both appear per zone (Extended Stats) and per vehicle (vehicle-log CSV).

## Beyond vehicle delay

- **Pedestrians / hawkers** — tracked separately from vehicles, per zone.
  Anyone dwelling past the loiter threshold (Config → Pedestrians &
  Capacity) reads as vendor/hawker-like activity rather than someone
  just passing through. Turn this off there if you don't need it.
- **PCU volume & capacity** — every completed vehicle is converted to a
  Passenger Car Unit and tallied into a PCU/hr throughput figure. Leave
  capacity at 0/unknown to just see raw throughput; set an estimate (from
  lane count/width) to get a Volume/Capacity ratio — a second, independent
  read on whether a road is genuinely over capacity vs. jammed by friction
  (parking, pedestrians, encroachment) despite modest volume.
- **Queue length in metres** — off by default (queue is just a vehicle
  count). Enable a zone's optional reference line in Config and drag its
  2 endpoints onto something of known real length, and queue length gets
  reported in metres too, measured from the actual spread of currently
  queued vehicles' positions.
- **Delay distribution** — median and 85th-percentile delay, plus % of
  vehicles meaningfully delayed, alongside the average — a mean can hide
  a lot when half of traffic sails through and half crawls.
- **Session log** — every completed run appends one row (Config → Session
  Log) to a CSV that accumulates across multiple videos, so running
  several sessions at different times/days builds a comparable dataset
  automatically instead of you stitching separate exports together.

---

## Supported models

**YOLOv8** (BNVD weights available — use for inference now)

| Variant | Speed | mAP (COCO) |
|---------|-------|------------|
| yolov8n | ⚡⚡⚡⚡ | 37.3 |
| yolov8s | ⚡⚡⚡ | 44.9 |
| **yolov8m** | ⚡⚡⚡ | **50.2** |
| yolov8l | ⚡⚡ | 52.9 |
| yolov8x | ⚡ | 53.9 |

**YOLO26** (train your own weights, then browse the .pt below)

| Variant | Speed | Notes |
|---------|-------|-------|
| yolo26n | ⚡⚡⚡⚡ | Fastest, use for edge/CPU |
| yolo26s | ⚡⚡⚡ | Good balance |
| **yolo26m** | ⚡⚡⚡ | **Recommended once trained** |
| yolo26l | ⚡⚡ | Higher accuracy |
| yolo26x | ⚡ | Maximum accuracy |

---

## BNVD custom weights

Download from `github.com/bipin-saha/BNVD` → Checkpoints.
Browse the `.pt` file via **Configuration → Custom weights (.pt)**.
Classes: Rickshaw, CNG, Bus, Car, Truck, Van, Bicycle, Easybike…
        """), "Quick start")

        tabs.addTab(mktab("""
## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Browse video |
| Space | Pause / Resume |
| Ctrl+S | Export CSV |
| F1 | Help |
| Ctrl+Q | Quit |
        """), "Shortcuts")

        tabs.addTab(mktab(f"""
## About

**IntersectionCV v5 — Two-Road Delay Comparison**
Engine: YOLOv8 / YOLO26 (Ultralytics) + ByteTrack
Interface: PyQt6
Site: Saheb Bazar – Alupotti Rd, Rajshahi

Two independently-shaped, 6-point ROI zones are tracked and analysed in
parallel, each learning its own free-flow baseline live — no measuring
bars, no manual speed calibration. Road A and Road B are reported and
compared side by side in real time.

YOLOv8: anchor-free detection, BNVD weights available (mAP=0.848).
YOLO26: NMS-free, MuSGD + ProgLoss + STAL — load your trained .pt via Custom weights.
BNVD fine-tuned weights: mAP=0.848 on Bangladeshi native vehicles.

Classes: {', '.join(VEHICLE_CLASSES)}
        """), "About")

        lay.addWidget(tabs, 1)


# ──────────────────────────────────────────────────────────────────────
# NAV + MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────

class NavBtn(QPushButton):
    def __init__(self, icon, label, parent=None):
        super().__init__(f"  {icon}   {label}", parent)
        self.setObjectName("btn_nav")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(40)

    def set_active(self, yes):
        self.setProperty("active", "true" if yes else "false")
        self.setStyle(self.style())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntersectionCV  ·  Traffic Delay Analysis")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self.setStyleSheet(QSS)
        self._build()
        self._shortcuts()
        self._nav(0)

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background:{C['bg']};")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────
        sb = QWidget()
        sb.setFixedWidth(210)
        sb.setStyleSheet(f"""
            QWidget {{
                background: {C['s0']};
                border-right: 1px solid {C['bd']};
            }}
        """)
        sb_lay = QVBoxLayout(sb)
        sb_lay.setContentsMargins(0,0,0,0)
        sb_lay.setSpacing(0)

        # Logo strip
        logo_strip = QWidget()
        logo_strip.setFixedHeight(62)
        logo_strip.setStyleSheet(f"""
            QWidget {{
                background: {C['bg']};
                border-bottom: 1px solid {C['bd']};
                border-right: none;
            }}
        """)
        ll = QVBoxLayout(logo_strip)
        ll.setContentsMargins(16,0,16,0)
        ll.setSpacing(1)
        ll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name = QLabel("IntersectionCV")
        name.setStyleSheet(
            f"color:{C['tp']}; font-size:15px; font-weight:700; "
            "background:transparent; border:none;")
        tag = QLabel("ROAD A vs ROAD B · BYTETRACK")
        tag.setStyleSheet(
            f"color:{C['acc']}; font-size:9px; font-weight:700; "
            "letter-spacing:2px; background:transparent; border:none;")
        ll.addWidget(name); ll.addWidget(tag)
        sb_lay.addWidget(logo_strip)

        nav_area = QWidget()
        nav_area.setStyleSheet("background:transparent; border:none;")
        na = QVBoxLayout(nav_area)
        na.setContentsMargins(8,12,8,12)
        na.setSpacing(2)

        self.nav_btns = []
        for i, (ic, lb) in enumerate([
            ("▶","Analysis"), ("⚙","Configuration"),
            ("◈","Results"),  ("?","Help")
        ]):
            btn = NavBtn(ic, lb)
            btn.clicked.connect(lambda _, idx=i: self._nav(idx))
            na.addWidget(btn)
            self.nav_btns.append(btn)

        na.addStretch()
        ver = QLabel("v3.0.0")
        ver.setStyleSheet(
            f"color:{C['td']}; font-size:10px; "
            "padding:0 8px 8px; background:transparent; border:none;")
        na.addWidget(ver)
        sb_lay.addWidget(nav_area, 1)
        root.addWidget(sb)

        # ── Page stack ───────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{C['bg']};")
        self.analysis = AnalysisPage()
        self.cfg      = ConfigPage()
        self.results  = ResultsPage()
        self.help     = HelpPage()
        for page in (self.analysis, self.cfg, self.results, self.help):
            self.stack.addWidget(page)
        root.addWidget(self.stack)

        # ROI / calibration two-way sync: canvas drag → Config spinboxes,
        # and Config spinboxes are wired (in ConfigPage) to push back to canvas.
        self.analysis.canvas.layout_changed.connect(self.cfg._pull_layout_from_canvas)
        self.cfg._pull_layout_from_canvas()

        sb2 = QStatusBar()
        self.setStatusBar(sb2)
        sb2.showMessage(
            "IntersectionCV v5  ·  Two-Road Delay Comparison  ·  Saheb Bazar – Alupotti Rd  ·  "
            "YOLOv8 / YOLO26 + ByteTrack  ·  BNVD weights: github.com/bipin-saha/BNVD")

    def _shortcuts(self):
        for key, fn in [
            ("Ctrl+O",  self.analysis._browse),
            ("Space",   self.analysis._toggle_pause),
            ("Ctrl+S",  self.results._csv),
            ("F1",      lambda: self._nav(3)),
            ("Ctrl+Q",  self.close),
        ]:
            a = QAction(self); a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn); self.addAction(a)

    def _nav(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.set_active(i == idx)

    def closeEvent(self, e):
        if self.analysis.worker and self.analysis.running:
            self.analysis.worker.stop()
            self.analysis.worker.wait(2000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("IntersectionCV")
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()