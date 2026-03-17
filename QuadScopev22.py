"""
Navigator V/f Power-Chain Feasibility UI (v22)

Adds:
- Ld + Lq
- Supplier-friendly Kt/Ke basis handling (peak vs rms)
- Two downhole voltage limits:
    * Motor Vphase_rms (L-N)
    * Contact-block Vll_rms (L-L)
- Cable current limit basis defaults to Per-conductor hard limit (1.6 Arms)
- Field-weakening option (approx steady-state, negative Id)
- Right-side QTabWidget:
    * Envelope plots (existing)
    * Sweeps tab (trade assessment plots)
- Control strategy dropdown (Baseline V/f, Mode A, Mode B)

v13 Adds:
- Optional static load blocks: Magnetic Coupler, Tool Parasitics, BHA (TOB reaction + BHA friction)
- CW/CCW direction selector for CCRS gearbox output

v11 Adds:
- Speed/temperature-dependent extra torque model for torque→current mapping:
    * τ_core = C_L · ω_m^0.5
    * τ_visc = τ_userT(ω_m, T) (Couette / transition / turbulence models)
    * τ_extra(ω) = τ_core(ω) + τ_visc(ω)
- Kt(T) scaling option
- UI groupbox for configuring τ_extra(ω,T) + Kt(T)
"""

from __future__ import annotations

import math
import copy
import textwrap
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QPlainTextEdit,
    QWidget,
)

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# -----------------------------
# Report generation (PDF)
# -----------------------------
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)


# -----------------------------
# Unit conversions
# -----------------------------

def ft_lbf_to_nm(x_ftlbf: float) -> float:
    return x_ftlbf * 1.3558179483314004


def nm_to_ft_lbf(x_nm: float) -> float:
    return x_nm / 1.3558179483314004


# Backwards-compatible alias (older code used nm_to_ftlbf)
def nm_to_ftlbf(x_nm: float) -> float:
    return nm_to_ft_lbf(x_nm)


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(w: float) -> float:
    return w * 60.0 / (2.0 * math.pi)


# -----------------------------
# Quadrant plot interactivity (hover tooltip + crosshair)
# -----------------------------

class QuadrantInteractor:
    """Adds hover tooltip + crosshair to a list of Matplotlib axes on a shared canvas.

    Designed for the 4Q torque plots (RPM vs Torque) in the Feasibility tab.

    Usage:
        inter = QuadrantInteractor(canvas, [ax1, ax2])
        inter.refresh()   # call after any re-plot that clears/redraws axes
    """

    def __init__(self, canvas, axes, pixel_tol: int = 12):
        self.canvas = canvas
        self.axes = list(axes)
        self.pixel_tol = int(pixel_tol)

        # list of (ax, kind, artist) where kind ∈ {"line", "scatter"}
        self._items = []
        self._xhair = {}  # ax -> (vline, hline)
        self._ann = {}  # ax -> annotation

        # Create/attach UI elements
        for ax in self.axes:
            self._ensure_overlays(ax)

        # Connect events once
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._cid_leave = self.canvas.mpl_connect("figure_leave_event", self._on_leave)

        self.refresh()

    def _ensure_overlays(self, ax):
        # If axes were cleared, recreate overlays
        if ax not in self._xhair:
            vline = ax.axvline(0.0, lw=0.9, alpha=0.35, zorder=50)
            hline = ax.axhline(0.0, lw=0.9, alpha=0.35, zorder=50)
            vline.set_visible(False)
            hline.set_visible(False)
            self._xhair[ax] = (vline, hline)

        if ax not in self._ann:
            ann = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92),
                arrowprops=dict(arrowstyle="->", alpha=0.55),
                zorder=60,
            )
            ann.set_visible(False)
            self._ann[ax] = ann

    def refresh(self):
        """Re-collect artists and ensure overlays exist (call after any re-plot)."""
        self._items.clear()
        for ax in self.axes:
            # If the axes was cleared, overlays might have been removed.
            # Detect and re-add if necessary.
            if ax not in self._xhair or self._xhair[ax][0] not in ax.lines:
                if ax in self._xhair:
                    del self._xhair[ax]
                if ax in self._ann:
                    del self._ann[ax]
                self._ensure_overlays(ax)

            vline, hline = self._xhair[ax]

            # Lines (skip crosshair overlays)
            for ln in getattr(ax, "lines", []):
                if not ln.get_visible():
                    continue
                if (ln is vline) or (ln is hline):
                    continue
                self._items.append((ax, "line", ln))

            # Scatter points (PathCollection with offsets)
            for col in getattr(ax, "collections", []):
                if not col.get_visible():
                    continue
                if hasattr(col, "get_offsets"):
                    try:
                        off = col.get_offsets()
                        if off is None:
                            continue
                        off = np.asarray(off, dtype=float)
                        if off.size == 0:
                            continue
                        self._items.append((ax, "scatter", col))
                    except Exception:
                        pass

    def _hide_all(self):
        for ax in self.axes:
            vline, hline = self._xhair.get(ax, (None, None))
            ann = self._ann.get(ax, None)
            if vline is not None:
                vline.set_visible(False)
            if hline is not None:
                hline.set_visible(False)
            if ann is not None:
                ann.set_visible(False)
        self.canvas.draw_idle()

    def _hide_axis(self, ax_keep=None):
        for ax in self.axes:
            if ax_keep is not None and ax is ax_keep:
                continue
            vline, hline = self._xhair.get(ax, (None, None))
            ann = self._ann.get(ax, None)
            if vline is not None:
                vline.set_visible(False)
            if hline is not None:
                hline.set_visible(False)
            if ann is not None:
                ann.set_visible(False)

    def _on_leave(self, event):
        self._hide_all()

    def _nearest_on_line(self, ax, ln, ex, ey):
        try:
            x = np.asarray(ln.get_xdata(orig=False), dtype=float)
            y = np.asarray(ln.get_ydata(orig=False), dtype=float)
        except Exception:
            return None
        if x.size == 0:
            return None
        pts = np.column_stack([x, y])
        pix = ax.transData.transform(pts)
        dx = pix[:, 0] - ex
        dy = pix[:, 1] - ey
        d2 = dx * dx + dy * dy
        i = int(np.nanargmin(d2))
        return i, float(d2[i]), float(x[i]), float(y[i])

    def _nearest_on_scatter(self, ax, col, ex, ey):
        try:
            off = np.asarray(col.get_offsets(), dtype=float)
        except Exception:
            return None
        if off.size == 0:
            return None
        pix = ax.transData.transform(off)
        dx = pix[:, 0] - ex
        dy = pix[:, 1] - ey
        d2 = dx * dx + dy * dy
        i = int(np.nanargmin(d2))
        return i, float(d2[i]), float(off[i, 0]), float(off[i, 1])

    def _clean_label(self, label: str, fallback: str) -> str:
        lab = (label or "").strip()
        if (not lab) or (lab == "_nolegend_"):
            return fallback
        return lab

    def _on_move(self, event):
        ax = getattr(event, "inaxes", None)
        if ax not in self.axes:
            self._hide_all()
            return

        # Ensure overlays exist even if axes were cleared since last refresh
        self._ensure_overlays(ax)

        best = None  # (d2, kind, artist, label, x, y)
        for a, kind, artist in self._items:
            if a is not ax:
                continue
            if kind == "line":
                hit = self._nearest_on_line(ax, artist, event.x, event.y)
                if hit is None:
                    continue
                _, d2, xh, yh = hit
                label = self._clean_label(getattr(artist, "get_label", lambda: "")(), "curve")
            else:
                hit = self._nearest_on_scatter(ax, artist, event.x, event.y)
                if hit is None:
                    continue
                _, d2, xh, yh = hit
                label = self._clean_label(getattr(artist, "get_label", lambda: "")(), "point")

            if (best is None) or (d2 < best[0]):
                best = (d2, kind, artist, label, xh, yh)

        if best is None or best[0] > (self.pixel_tol * self.pixel_tol):
            self._hide_all()
            return

        _, _, _, label, xh, yh = best

        # Hide overlays on the *other* axis
        self._hide_axis(ax_keep=ax)

        vline, hline = self._xhair[ax]
        vline.set_xdata([xh, xh])
        hline.set_ydata([yh, yh])
        vline.set_visible(True)
        hline.set_visible(True)

        ann = self._ann[ax]
        ann.xy = (xh, yh)
        ann.set_text(f"{label}\nRPM={xh:.3g}\nTorque={yh:.3g}")
        ann.set_visible(True)

        self.canvas.draw_idle()


# 1 lbf-in = 0.112984829... N·m
LBIN_TO_NM = 0.1129848290276167

# -----------------------------
# MOOG operating-curve datasets (used by the "Moog Curves" tab)
# -----------------------------
# NOTE: Only the Milling motor dataset is currently populated.
#       "Spear Moog Curve" and "Annular Windings Curve" are placeholders that reuse Milling for now.
#       Units as transcribed from the Moog v129 sheet: rpm, lbf-in, Arms, kW.

MOOG_MILLING_KTTR_LBIN_PER_ARMS = 4.7335  # Kt at 25°C (phase Arms) from the sheet

# Continuous table (max speed 11424.2 rpm)
MOOG_MILLING_CONT = [
    # rpm, visc_drag_lbf_in, torque_out_lbf_in, current_arms, power_kw
    (0.0, 0.000, 24.825, 5.53, 0.000),
    (250.0, 0.003, 24.714, 5.53, 0.073),
    (500.0, 0.007, 24.648, 5.52, 0.146),
    (750.0, 0.010, 24.583, 5.51, 0.218),
    (1000.0, 0.014, 24.517, 5.51, 0.290),
    (1250.0, 0.017, 24.449, 5.50, 0.362),
    (1500.0, 0.020, 24.379, 5.49, 0.433),
    (1750.0, 0.024, 24.306, 5.47, 0.503),
    (2000.0, 0.027, 24.230, 5.46, 0.573),
    (2250.0, 0.030, 24.152, 5.45, 0.643),
    (2500.0, 0.034, 24.071, 5.44, 0.712),
    (2750.0, 0.037, 23.988, 5.42, 0.780),
    (3000.0, 0.041, 23.901, 5.41, 0.848),
    (3250.0, 0.044, 23.813, 5.39, 0.916),
    (3500.0, 0.047, 23.722, 5.37, 0.982),
    (3750.0, 0.051, 23.628, 5.36, 1.048),
    (4000.0, 0.054, 23.531, 5.34, 1.114),
    (4250.0, 0.057, 23.433, 5.32, 1.178),
    (4500.0, 0.061, 23.331, 5.30, 1.242),
    (4750.0, 0.064, 23.228, 5.28, 1.305),
    (5000.0, 0.068, 23.121, 5.26, 1.368),
    (5250.0, 0.071, 23.013, 5.24, 1.429),
    (5500.0, 0.074, 22.902, 5.22, 1.490),
    (5750.0, 0.078, 22.788, 5.19, 1.550),
    (6000.0, 0.081, 22.672, 5.17, 1.610),
    (6250.0, 0.085, 22.554, 5.15, 1.668),
    (6500.0, 0.088, 22.433, 5.12, 1.725),
    (6750.0, 0.091, 22.310, 5.10, 1.782),
    (7000.0, 0.095, 22.184, 5.07, 1.837),
    (7250.0, 0.098, 22.056, 5.05, 1.892),
    (7500.0, 0.101, 21.925, 5.02, 1.946),
    (7750.0, 0.105, 21.792, 4.99, 1.998),
    (8000.0, 0.108, 21.656, 4.97, 2.050),
    (8250.0, 0.112, 21.518, 4.94, 2.100),
    (8500.0, 0.115, 21.377, 4.91, 2.150),
    (8750.0, 0.118, 21.234, 4.88, 2.198),
    (9000.0, 0.122, 21.088, 4.85, 2.246),
    (9250.0, 0.125, 20.939, 4.82, 2.292),
    (9500.0, 0.129, 20.788, 4.79, 2.337),
    (9750.0, 0.132, 20.634, 4.76, 2.380),
    (10000.0, 0.135, 20.477, 4.72, 2.423),
    (10250.0, 0.139, 20.318, 4.69, 2.464),
    (10500.0, 0.142, 20.156, 4.66, 2.504),
    (10750.0, 0.145, 19.990, 4.62, 2.543),
    (11000.0, 0.149, 19.822, 4.59, 2.580),
    (11250.0, 0.152, 19.651, 4.55, 2.616),
    (11424.2, 0.155, 19.530, 4.53, 2.640),
]

# Peak table (max speed 12682.2 rpm)
MOOG_MILLING_PEAK = [
    # rpm, visc_drag_lbf_in, torque_out_lbf_in, current_arms, power_kw
    (10162.4, 0.137, 49.863, 11.14, 5.995),
    (10250.0, 0.139, 47.915, 10.70, 5.811),
    (10500.0, 0.142, 42.446, 9.49, 5.273),
    (10750.0, 0.145, 37.103, 8.30, 4.719),
    (11000.0, 0.149, 31.885, 7.14, 4.150),
    (11250.0, 0.152, 26.789, 6.00, 3.566),
    (11500.0, 0.156, 21.813, 4.89, 2.968),
    (11750.0, 0.159, 16.954, 3.81, 2.357),
    (12000.0, 0.162, 12.210, 2.76, 1.734),
    (12250.0, 0.166, 7.578, 1.72, 1.098),
    (12500.0, 0.169, 3.056, 0.72, 0.452),
    (12682.2, 0.172, -0.172, 0.00, -0.026),
]


def _get_moog_dataset(name: str):
    n = (name or "").strip()
    if n in ("Spear Moog Curve", "Annular Windings Curve"):
        # placeholders until those datasets are available
        n = "Milling Moog Curve"
    if n == "Milling Moog Curve":
        return {
            "name": "Milling Moog Curve",
            "kt_lbin_arms": MOOG_MILLING_KTTR_LBIN_PER_ARMS,
            "cont": MOOG_MILLING_CONT,
            "peak": MOOG_MILLING_PEAK,
        }
    return None


# -----------------------------
# Parameter blocks
# -----------------------------

@dataclass
class VfParams:
    """Surface inverter V/f parameters.

    Canonical internal AC basis: **phase RMS voltage** (L-N).

    Voltage limit entry options:
      • AC fundamental limit (Vrms): enter Line-Line Vrms or Phase (L-N) Vrms
      • DC link (Vdc): enter Vdc and choose modulation (SPWM / SVPWM) plus utilization factor

    The solver converts either entry into an equivalent Phase RMS limit for consistent motor math.
    """

    # --- Voltage entry basis (v11) ---
    voltage_entry_basis: str = "DC link (Vdc)"  # or "AC fundamental limit (Vrms)"
    vdc_link_v: float = 1000.0
    modulation: str = "SVPWM"  # "SVPWM" or "SPWM"
    v_util: float = 1.00  # utilization of achievable AC fundamental (e.g., 0.95)

    # --- AC fundamental entry (legacy) ---
    v_limit_value: float = 750.5
    v_limit_type: str = "Line-Line (Vrms)"  # or "Phase (L-N) (Vrms)"

    # V/f shaping
    base_freq_hz: float = 120.0
    base_v_phase_rms: float = 410.4
    v_boost: float = 8.0

    def ac_limits_from_vdc(self) -> tuple[float, float]:
        """Return (Vph_rms_max, Vll_rms_max) achievable from Vdc in linear modulation.

        Assumptions:
          - 2-level inverter
          - fundamental only (no overmodulation / six-step)
          - v_util scales achievable fundamental

        SPWM:  Vph_rms,max = Vdc/(2*sqrt(2))
        SVPWM: Vph_rms,max = Vdc/sqrt(6)
        and Vll_rms = sqrt(3) * Vph_rms.
        """
        vdc = max(0.0, float(self.vdc_link_v))
        util = float(self.v_util)
        if util < 0.0:
            util = 0.0
        if util > 1.0:
            util = 1.0
        mod = (self.modulation or '').strip().upper()
        if mod.startswith('SV'):
            vph_rms = util * vdc / math.sqrt(6.0)
        else:
            vph_rms = util * vdc / (2.0 * math.sqrt(2.0))
        vll_rms = math.sqrt(3.0) * vph_rms
        return vph_rms, vll_rms

    def v_phase_rms_limit(self) -> float:
        basis = (self.voltage_entry_basis or '').strip().upper()
        if basis.startswith('DC'):
            vph_rms, _ = self.ac_limits_from_vdc()
            return vph_rms
        # AC fundamental entry
        if str(self.v_limit_type).startswith('Line-Line'):
            return float(self.v_limit_value) / math.sqrt(3.0)
        return float(self.v_limit_value)

    def v_ll_rms_limit(self) -> float:
        return math.sqrt(3.0) * self.v_phase_rms_limit()

    def v_surface_cmd_phase_rms(self, f_e_hz: float) -> float:
        """V/f voltage command in phase RMS (L-N), clipped by the available limit."""
        f = max(0.0, float(f_e_hz))
        if self.base_freq_hz <= 0.0:
            return 0.0
        if f <= self.base_freq_hz:
            slope = self.base_v_phase_rms / self.base_freq_hz
            v = self.v_boost + slope * f
        else:
            v = self.v_boost + self.base_v_phase_rms
        return min(max(0.0, float(v)), self.v_phase_rms_limit())


@dataclass
class SineFilterParams:
    """Optional output sine filter between surface inverter and heptacable (steady-state, fundamental).

    Model:
      inverter (Vcmd)  →  series (Rf + jωLf)  →  node  →  shunt capacitor network  →  cable  →  motor

    The shunt capacitor draws reactive current, which increases inverter RMS current and adds extra
    voltage drop across the series element. This block is intentionally simple (RMS phasor magnitudes).

    Capacitance entry is interpreted by connection:
      - DELTA: Cf is per delta branch (line-line). Wye-equivalent per-phase C ≈ 3*Cf.
      - WYE  : Cf is per phase-to-neutral. Wye-equivalent per-phase C = Cf.

    Damping resistor Rd can be modeled either:
      - SERIES: Rd in series with C
      - PARALLEL: Rd in parallel with C
    """
    enabled: bool = True
    enforce_inv_current_limit: bool = True  # Option B: limit inverter-side current including filter reactive draw

    # series element (per phase)
    lf_h: float = 3.64e-3  # H
    rf_ohm: float = 0.243  # ohm

    # shunt network
    cf_f: float = 1.0e-6  # F (interpreted by cap_connection)
    cap_connection: str = "WYE"  # "DELTA" or "WYE"
    damping_topology: str = "SERIES"  # "SERIES" or "PARALLEL"
    rd_ohm: float = 0.0  # ohm


class FieldWeakeningParams:
    enabled: bool = False
    id_max_arms: float = 0.0  # max |Id| in phase-RMS amps
    apply_only_above_base: bool = False  # optional toggle (often OFF for your case with hard downhole limit)
    id_grid_points: int = 100  # resolution for Id scanning


@dataclass
class CableParams:
    length_m: float = 7160.0
    r_ohm_per_m: float = 0.024
    l_h_per_m: float = 0.0000002
    temp_factor_r: float = 1.59

    # Cable temperature reference for R(T)
    temp_ref_C: float = 20.0  # datasheet reference temp for R and ampacity
    temp_alpha_per_C: float = 0.00393  # copper-ish dR/R per °C (~0.393%/°C)

    # 5-segment custom temperature model (explicit segment lengths + temps)
    # When enabled, the cable is treated as 5 independent segments with user-specified length and average temperature.
    temp_model_5seg: bool = True
    temp5_seg_len_m: List[float] = field(default_factory=lambda: [1500.0, 1100.0, 900.0, 1800.0, 1860.0])
    temp5_seg_temp_C: List[float] = field(default_factory=lambda: [100.0, 125.0, 160.0, 175.0, 175.0])

    wires_per_phase: int = 1  # 1 or 2

    # current limit
    i_limit_arms: float = 1.6
    i_limit_basis: str = "Per conductor"  # "Per phase" or "Per conductor" (default hard limit)

    # optional: derate I_limit with temperature (approx): I ∝ 1/sqrt(R(T))
    i_limit_derate_with_temp: bool = True

    # inductance reduction factor when using 2 conductors in parallel (geometry-dependent)
    l_parallel_factor: float = 0.85

    def _len_total_5seg(self) -> float:
        Ls = list(getattr(self, "temp5_seg_len_m", []) or [])
        if len(Ls) == 0:
            return 0.0
        n = min(5, len(Ls))
        return float(sum(max(0.0, float(Ls[i])) for i in range(n)))

    def _temp_factor_r_avg_5seg(self) -> float:
        """Length-weighted average R(T) multiplier for the 5-segment model."""
        Ls = list(getattr(self, "temp5_seg_len_m", []) or [])
        Ts = list(getattr(self, "temp5_seg_temp_C", []) or [])
        n = min(5, len(Ls), len(Ts))
        if n <= 0:
            return float(getattr(self, "temp_factor_r", 1.0))
        Ltot = float(sum(max(0.0, float(Ls[i])) for i in range(n)))
        if Ltot <= 0.0:
            return float(getattr(self, "temp_factor_r", 1.0))
        Tref = float(self.temp_ref_C)
        a = float(self.temp_alpha_per_C)
        s = 0.0
        for i in range(n):
            Li = max(0.0, float(Ls[i]))
            Ti = float(Ts[i])
            s += Li * (1.0 + a * (Ti - Tref))
        return s / Ltot

    def effective_r_phase(self) -> float:
        """Effective per-phase cable resistance (ohms), including optional temperature model."""
        # 5-segment custom model: compute R_total directly from segment lengths and temps
        if bool(getattr(self, "temp_model_5seg", False)):
            Ls = list(getattr(self, "temp5_seg_len_m", []) or [])
            Ts = list(getattr(self, "temp5_seg_temp_C", []) or [])
            n = min(5, len(Ls), len(Ts))
            if n > 0:
                Tref = float(self.temp_ref_C)
                a = float(self.temp_alpha_per_C)
                r_total = 0.0
                for i in range(n):
                    Li = max(0.0, float(Ls[i]))
                    Ti = float(Ts[i])
                    r_total += self.r_ohm_per_m * Li * (1.0 + a * (Ti - Tref))
                if r_total > 0.0:
                    return r_total / max(1, int(self.wires_per_phase))

        tf = float(getattr(self, "temp_factor_r", 1.0))

        r = self.r_ohm_per_m * self.length_m * tf
        return r / max(1, int(self.wires_per_phase))

    def effective_l_phase(self) -> float:
        # L is weakly temperature-dependent; we only update L via the effective modeled length
        Llen = float(self.length_m)
        if bool(getattr(self, "temp_model_5seg", False)):
            Ltot = self._len_total_5seg()
            if Ltot > 0.0:
                Llen = Ltot

        l = self.l_h_per_m * Llen
        if int(self.wires_per_phase) == 2:
            return float(self.l_parallel_factor) * l
        return l

    def i_phase_limit(self) -> float:
        I = float(self.i_limit_arms)
        if bool(getattr(self, "i_limit_derate_with_temp", False)):
            tf = None
            if bool(getattr(self, "temp_model_5seg", False)):
                tf = self._temp_factor_r_avg_5seg()
            else:
                tf = float(getattr(self, "temp_factor_r", 1.0))
            if (tf is not None) and (tf > 0.0):
                I = I / math.sqrt(tf)

        if self.i_limit_basis == "Per conductor":
            return I * max(1, int(self.wires_per_phase))
        return I


@dataclass
class MotorParams:
    """
    PMSM steady-state parameters.

    Canonical internal definitions in this tool:
      - Rs, Ld, Lq are **per-phase** (wye-equivalent), in ohm / H
      - Kt used for torque mapping: **Nm/Arms** (phase RMS magnitude)
      - Ke used for EMF readout: **Vll_rms/krpm**

    Vendor / supplier tools often report:
      - Kt in **lb-in/Arms** (sometimes per Apeak)
      - Ke in **Vll_peak/krpm** (line-line peak)

    We support UI bases:
      - Kt: "Nm/Arms", "Nm/Apeak", "lb-in/Arms", "lb-in/Apeak"
      - Ke: "Vll_rms/krpm", "Vll_peak/krpm"

    Defaults below are loaded from your winding datasheet (301PSTT style):
      - R_ll = 59.05 ohm  => R_phase ≈ 29.525 ohm (wye)
      - L_ll = 29.28 mH   => L_phase ≈ 14.64 mH (wye)
      - Kt = 8.548 lb-in/Arms
      - Ke = 82.604 Vll_peak/krpm
    """

    pole_pairs: int = 4

    # Electrical model parameters (per-phase, wye-equivalent)
    rs_ohm: float = 59.05 / 2.0
    ld_h: float = 29.28e-3 / 2.0
    lq_h: float = 29.28e-3 / 2.0

    # Canonical physical parameter
    lambda_wb: float = 0.113819952

    # Canonical torque mapping constant (Nm/Arms)
    kt_nm_per_arms: float = 8.548 * LBIN_TO_NM

    # Canonical EMF readout constant (Vll_rms/krpm)
    ke_vll_rms_per_krpm: float = 82.604 / math.sqrt(2.0)

    # UI basis handling
    kt_basis: str = "lb-in/Arms"  # "Nm/Arms" | "Nm/Apeak" | "lb-in/Arms" | "lb-in/Apeak"
    ke_basis: str = "Vll_peak/krpm"  # "Vll_rms/krpm" | "Vll_peak/krpm"

    # linking
    link_kt_ke: bool = True
    motor_param_mode: str = "Ke"  # "Lambda" | "Kt" | "Ke"

    def recompute_derived(self):
        """Update Kt & Ke from lambda (if linked), or update lambda from selected mode (basis-safe)."""
        p = max(1, int(self.pole_pairs))
        krpm_to_rad_s = (1000.0 * 2.0 * math.pi / 60.0)

        # sinusoidal PMSM, Id=0 mapping
        def kt_from_lambda(lam: float) -> float:
            # T = 1.5*p*lam*Iq_peak, with Iq_peak = sqrt(2)*Iq_rms
            return 1.5 * p * lam * math.sqrt(2.0)

        def ke_ll_rms_per_rad_from_lambda(lam: float) -> float:
            # Vll_rms_emf = sqrt(3)*Vph_rms_emf; Vph_rms_emf = omega_e*lam/sqrt(2)
            return p * lam * math.sqrt(3.0) / math.sqrt(2.0)

        def ke_per_krpm_from_lambda(lam: float) -> float:
            return ke_ll_rms_per_rad_from_lambda(lam) * krpm_to_rad_s

        def lambda_from_kt_arms(kt_arms: float) -> float:
            return kt_arms / (1.5 * p * math.sqrt(2.0))

        def lambda_from_ke_vll_rms_krpm(ke_krpm_rms: float) -> float:
            ke_per_rad = ke_krpm_rms / krpm_to_rad_s
            return ke_per_rad * math.sqrt(2.0) / (p * math.sqrt(3.0))

        if not self.link_kt_ke:
            # Keep Ke readout coherent with lambda, but don't force Kt.
            self.ke_vll_rms_per_krpm = float(ke_per_krpm_from_lambda(self.lambda_wb))
            return

        if self.motor_param_mode == "Kt":
            self.lambda_wb = float(lambda_from_kt_arms(float(self.kt_nm_per_arms)))
        elif self.motor_param_mode == "Ke":
            self.lambda_wb = float(lambda_from_ke_vll_rms_krpm(float(self.ke_vll_rms_per_krpm)))
        else:
            # Lambda: keep lambda_wb as entered
            pass

        # Update canonical constants from lambda
        self.kt_nm_per_arms = float(kt_from_lambda(self.lambda_wb))
        self.ke_vll_rms_per_krpm = float(ke_per_krpm_from_lambda(self.lambda_wb))

    def ke_ll_rms_per_rad(self) -> float:
        p = max(1, int(self.pole_pairs))
        return p * self.lambda_wb * math.sqrt(3.0) / math.sqrt(2.0)

    def kt_display(self) -> float:
        """Return Kt in the selected UI basis."""
        if self.kt_basis == "Nm/Apeak":
            return self.kt_nm_per_arms / math.sqrt(2.0)
        if self.kt_basis == "lb-in/Arms":
            return self.kt_nm_per_arms / LBIN_TO_NM
        if self.kt_basis == "lb-in/Apeak":
            return (self.kt_nm_per_arms / math.sqrt(2.0)) / LBIN_TO_NM
        return self.kt_nm_per_arms

    def ke_display(self) -> float:
        """Return Ke in the selected UI basis."""
        if self.ke_basis == "Vll_peak/krpm":
            return self.ke_vll_rms_per_krpm * math.sqrt(2.0)
        return self.ke_vll_rms_per_krpm

    def set_kt_from_display(self, val: float):
        """Update canonical Kt (Nm/Arms) from UI value (basis-aware)."""
        v = float(val)
        if self.kt_basis == "Nm/Apeak":
            v *= math.sqrt(2.0)
        elif self.kt_basis == "lb-in/Arms":
            v *= LBIN_TO_NM
        elif self.kt_basis == "lb-in/Apeak":
            v *= (LBIN_TO_NM * math.sqrt(2.0))
        self.kt_nm_per_arms = float(v)

    def set_ke_from_display(self, val: float):
        """Update canonical Ke (Vll_rms/krpm) from UI value (basis-aware)."""
        v = float(val)
        if self.ke_basis == "Vll_peak/krpm":
            v /= math.sqrt(2.0)
        self.ke_vll_rms_per_krpm = float(v)


@dataclass
class GearboxParams:
    stage1: float = 15.0
    stage2: float = 15.0
    stage3: float = 15.0

    eff1: float = 0.85
    eff2: float = 0.85
    eff3: float = 0.85
    eta_misc: float = 1.00

    override_total_eta: bool = True
    eta_total_override: float = 0.40

    backdrivable: bool = True  # if False, output cannot backdrive input; regen/backdrive disabled

    def ratio(self) -> float:
        return float(self.stage1) * float(self.stage2) * float(self.stage3)

    def eff_total(self) -> float:
        if self.override_total_eta:
            return float(np.clip(self.eta_total_override, 1e-6, 0.999999))
        eta = float(self.eff1) * float(self.eff2) * float(self.eff3) * float(self.eta_misc)
        return float(np.clip(eta, 1e-6, 0.999999))


@dataclass
class LimitsParams:
    # motor insulation / winding
    enforce_downhole_vphase_limit: bool = False
    downhole_v_phase_rms_limit: float = 70.0  # hard limit (phase-to-neutral)

    # contact block creepage / connector
    enforce_downhole_vll_limit: bool = True
    downhole_vll_rms_limit: float = 284.0  # optional (line-to-line)


@dataclass
class TargetParams:
    out_rpm: float = 1.0
    out_torque_ftlbf: float = 1000.0
    # If enabled, treat the UI torque target as the required continuous output torque
    # for feasibility checks (i.e., override the load-stack drive torque when larger).
    torque_override_continuous: bool = True


@dataclass
class MagneticCouplerParams:
    """Static magnetic coupler model between motor shaft and gearbox input.

    Piecewise characteristic (in torque magnitude):
      - below T_break: no useful transmission
      - above T_break: linear with slope
      - capped at T_slip: slips (torque limiter)
    """
    enabled: bool = True
    t_break_nm: float = 0.15
    t_slip_nm: float = 1.5
    slope: float = 1.0


@dataclass
class RotatingLossParams:
    """Generic rotating loss model: Coulomb + viscous + quadratic (optional).

    Interpreted as a torque that opposes rotation at the shaft where it is applied.
    """
    enabled: bool = True
    tc_nm: float = 67.790  # Coulomb (Nm)
    b_nm_per_rad_s: float = 0.0  # viscous (Nm/(rad/s))
    c_nm_per_rad_s2: float = 0.0  # quadratic (Nm/(rad/s)^2)


@dataclass
class BHABlockParams:
    """External load model at the CCRS output (lower BHA).

    Assumptions:
      - Bit rotates CW (independent of CCRS output speed).
      - The formation/mud-motor reaction torque on the lower BHA is CCW.
      - BHA friction torque always opposes CCRS output rotation direction.

    NOTE: Drilling TOB magnitude is specified separately (drilling_tob_ftlbf). TargetParams.out_torque_ftlbf is reserved for stuck/stall torque requirement.
    """
    enabled: bool = True
    drilling_tob_ftlbf: float = 150.0  # drilling reactive TOB magnitude (bit CW -> TOB CCW)
    # BHA friction model
    fric_tc_nm: float = 27.12
    fric_b_nm_per_rad_s: float = 0.0
    fric_c_nm_per_rad_s2: float = 0.0


@dataclass
class ExtraTorqueParams:
    """Extra (speed/temperature-dependent) torque that must be overcome by the motor.

    Implements the requested mapping:

        Iq_req = (tau_load + tau_extra(omega_m, T)) / Kt(T)

    where
        tau_extra(omega) = tau_core(omega) + tau_visc(omega, T)

        tau_core(omega) = C_L * |omega|^0.5
        tau_visc(omega, T) is a user-tunable Couette/transition/turbulence model.

    Notes
    -----
    * omega_m is the *mechanical* motor speed in rad/s.
    * All torques are positive magnitudes that oppose motion.
    """

    # overall toggles
    extra_enabled: bool = True
    kt_temp_enabled: bool = True

    # operating temperature (used by Kt(T) and tau_visc)
    temp_C: float = 175.0
    temp_ref_C: float = 25.0

    # winding copper temperature for Rs(T) (winding is typically hotter than ambient)
    #   T_winding = temp_C + winding_rise_C
    rs_temp_enabled: bool = True
    winding_rise_C: float = 10.0  # °C above ambient
    rs_temp_coeff_per_C: float = 0.00393  # fraction/°C (copper ≈ 0.393%/°C)

    # Kt(T) = Kt_ref * (1 + kt_temp_coeff_per_C*(T - Tref))
    kt_temp_coeff_per_C: float = -0.0004  # SmCo fraction per °C https://www.haydonkerkpittman.com/-/media/ametekhaydonkerk/downloads/white-papers/temperature_effects_on_dc_motor_performance_1.pdf?la=en

    # --- core / iron-like torque term ---
    core_enabled: bool = True
    core_cL: float = 0.0  # Nm / (rad/s)^0.5
    core_exp: float = 0.5  # fixed at 0.5 per requested model (left editable for sensitivity)

    # --- viscous / windage torque term ---
    visc_enabled: bool = True
    visc_model: str = "Piecewise (Couette→Transition→Turbulent)"

    # Couette (laminar): tau = k_couette * |omega|
    visc_k_couette: float = 0.0  # Nm / (rad/s)

    # Transition: tau = k_transition * |omega|^n
    visc_k_transition: float = 0.0  # Nm / (rad/s)^n
    visc_n_transition: float = 1.5

    # Turbulent: tau = k_turb * omega^2
    visc_k_turb: float = 0.0  # Nm / (rad/s)^2

    # speed breakpoints for piecewise model (in motor rpm)
    visc_rpm_1: float = 500.0
    visc_rpm_2: float = 2000.0

    # viscous temperature scaling factor S(T)
    visc_temp_scaling: str = "None"  # None | Linear | Exponential
    visc_lin_coeff_per_C: float = 0.0  # fraction per °C (Linear)
    visc_beta_per_C: float = 0.0  # 1/°C (Exponential: exp(-beta*ΔT))

    # smooth blending around rpm1/rpm2 to avoid sharp kinks
    smooth_transitions: bool = True
    smooth_frac: float = 0.15  # blend half-width as a fraction of rpm breakpoint

    # --- Backwards-compatible aliases (UI expects visc_rpm1/visc_rpm2) ---
    @property
    def visc_rpm1(self) -> float:
        return float(self.visc_rpm_1)

    @visc_rpm1.setter
    def visc_rpm1(self, v: float) -> None:
        self.visc_rpm_1 = float(v)

    @property
    def visc_rpm2(self) -> float:
        return float(self.visc_rpm_2)

    @visc_rpm2.setter
    def visc_rpm2(self, v: float) -> None:
        self.visc_rpm_2 = float(v)


@dataclass
class SystemParams:
    # IMPORTANT: use default_factory for dataclass fields that hold other
    # dataclass instances (avoids "mutable default" runtime error).
    vf: VfParams = field(default_factory=VfParams)

    sine_filter: SineFilterParams = field(default_factory=SineFilterParams)

    # Control strategy selector (affects how surface voltage is used + FW behavior)
    #   - 'VF'     : scheduled V/f command (baseline)
    #   - 'MODE_A' : use full inverter voltage ceiling (Vmax), enforce Id=0 (no FW)
    #   - 'MODE_B' : use full inverter voltage ceiling (Vmax), allow FW (Id scan)
    control_strategy: str = "MODE_A"
    fw: FieldWeakeningParams = field(default_factory=FieldWeakeningParams)
    cable: CableParams = field(default_factory=CableParams)
    motor: MotorParams = field(default_factory=MotorParams)
    gearbox: GearboxParams = field(default_factory=GearboxParams)
    limits: LimitsParams = field(default_factory=LimitsParams)
    extra: ExtraTorqueParams = field(default_factory=ExtraTorqueParams)
    target: TargetParams = field(default_factory=TargetParams)
    # Output rotation direction (CCRS gearbox output)
    #   - 'CW'  : clockwise (positive)
    #   - 'CCW' : counter-clockwise (negative)
    out_dir: str = "CW"

    # Operating case toggles
    stuck_mode: bool = False  # if True, force output RPM=0 and apply stall torque requirement
    braking_path_available: bool = True  # if False, regen/braking quadrant is infeasible

    brake_power_limit_enabled: bool = False  # cap allowable braking (regen) power at surface
    brake_power_kw_max: float = 50.0  # kW, only applies when brake_power_limit_enabled

    # Regen constraint over long cable (when braking_path_available is ON)
    # If enabled, maximum electrical braking torque is also limited by: back-EMF must exceed
    # (surface clamp voltage + cable drop). This approximates inverter/brake-resistor clamping at surface.
    regen_cable_limit_enabled: bool = True
    regen_surface_clamp_frac: float = 1.00  # 0..1 fraction of inverter Vrms(ph) limit used as clamp

    # Optional static load blocks
    mag_coupler: MagneticCouplerParams = field(default_factory=MagneticCouplerParams)
    parasitic: RotatingLossParams = field(default_factory=RotatingLossParams)
    bha: BHABlockParams = field(default_factory=BHABlockParams)


@dataclass
class SolveResult:
    feasible: bool
    reasons: List[str]
    notes: List[str]

    gear_ratio: float
    gear_eff: float

    motor_rpm: float
    motor_torque_nm: float

    # extra torque model + Kt(T)
    temp_C: float
    winding_temp_C: float
    rs_eff_ohm: float
    kt_eff_nm_per_arms: float
    tau_core_nm: float
    tau_visc_nm: float
    tau_extra_nm: float
    motor_torque_total_nm: float

    iq_req_base_rms: float

    iq_req_rms: float
    iq_max_rms: float

    id_used_rms: float
    i_mag_used_rms: float
    i_limit_phase_mag: float

    f_e_hz: float

    v_surface_cmd: float
    v_surface_limit: float

    v_motor_req: float
    v_cable_drop: float

    v_node_req: float
    v_filter_drop: float
    v_inverter_req: float
    i_inverter_rms: float
    i_filter_cap_rms: float
    v_downhole_phase_limit: Optional[float]
    v_downhole_ll_limit: Optional[float]

    p_cable_loss_w: float

    # hints
    kt_required_min: float
    ke_required_max_vll_krpm: float

    # --- Static load block reporting (output shaft) ---
    out_dir: str
    # SIGNED output RPM for clarity in UI/plots: CW=+ / CCW=-.
    # (Electrical model still uses magnitude internally.)
    out_rpm_cmd: float
    out_drive_torque_req_ftlbf: float
    out_brake_torque_req_ftlbf: float
    brake_suppressed_ftlbf: float

    tob_reaction_ftlbf: float
    bha_friction_ftlbf: float
    parasitic_ftlbf: float

    # Regen reporting (only meaningful when out_brake_torque_req_ftlbf > 0 and stuck_mode is OFF)
    regen_required: bool
    regen_cap_ftlbf: float
    regen_clamp_phase_v: float

    mag_slipping: bool


# -----------------------------
# Core steady-state model
# -----------------------------

class SystemModel:
    def __init__(self, p: SystemParams):
        self.p = p
        self.p.motor.recompute_derived()
        # bookkeeping for brake/assist reporting at the last evaluated point
        self.last_brake_demand_nm = 0.0
        self.last_brake_suppressed_nm = 0.0

    # ---------- Direction + static load block helpers ----------
    def out_dir_sign(self) -> float:
        return 1.0 if str(self.p.out_dir).upper().startswith("CW") else -1.0

    def _motion_sign(self, omega_out: float) -> float:
        """Sign to use for opposing torques when omega is ~0 (stuck)."""
        if abs(omega_out) < 1e-12:
            return self.out_dir_sign()
        return 1.0 if omega_out > 0 else -1.0

    def _rot_loss_torque_nm(self, omega: float, loss: RotatingLossParams) -> float:
        """Torque applied ON the shaft by losses (opposes motion)."""
        if not loss.enabled:
            return 0.0
        s = self._motion_sign(omega)
        return -(loss.tc_nm * s + loss.b_nm_per_rad_s * omega + loss.c_nm_per_rad_s2 * omega * abs(omega))

    def _bha_external_torques_nm(self, omega_out: float) -> Tuple[float, float]:
        """Return (tau_tob, tau_bha_fric) applied on CCRS output shaft.

        Bit rotates CW, so TOB reaction on lower BHA is CCW (negative).
        BHA friction always opposes CCRS output rotation direction.
        """
        if not self.p.bha.enabled:
            return 0.0, 0.0
        # Drilling TOB is only applied in continuous rotation / drilling mode.
        if bool(getattr(self.p, "stuck_mode", False)):
            tau_tob = 0.0
        else:
            tau_tob = -ft_lbf_to_nm(abs(self.p.bha.drilling_tob_ftlbf))  # CCW

        # BHA friction
        s = self._motion_sign(omega_out)
        fr = self.p.bha
        tau_fric = -(fr.fric_tc_nm * s + fr.fric_b_nm_per_rad_s * omega_out + fr.fric_c_nm_per_rad_s2 * omega_out * abs(
            omega_out))
        return tau_tob, tau_fric

    def _required_output_drive_torque_nm(self, omega_out: float) -> Tuple[float, float, float, float, float, bool]:
        """Compute required CCRS output torque *along direction of motion*.

        Returns:
          (t_drive, t_brake, tau_tob, tau_bha_fric, tau_parasitic)
        where:
          - t_drive >= 0  : torque CCRS must produce in direction of rotation
          - t_brake >= 0  : torque that would need to be absorbed (regen/brake) in direction of rotation
        """
        # external torques on output shaft
        tau_tob, tau_bha = self._bha_external_torques_nm(omega_out)
        tau_par = self._rot_loss_torque_nm(omega_out, self.p.parasitic)

        # Stuck/stall resisting torque requirement (applied only when stuck_mode is enabled).
        tau_stall = 0.0
        if bool(getattr(self.p, "stuck_mode", False)):
            s0 = self._motion_sign(omega_out)
            tau_stall = -ft_lbf_to_nm(abs(self.p.target.out_torque_ftlbf)) * s0

        # net external torque acting on shaft
        tau_ext = tau_stall + tau_tob + tau_bha + tau_par

        # torque CCRS must apply to balance
        # torque CCRS must apply to balance
        tau_ccrs = -tau_ext

        s = self._motion_sign(omega_out)
        t_drive = max(0.0, tau_ccrs * s)
        t_brake = max(0.0, -tau_ccrs * s)

        # record raw brake demand (opposes motion) for reporting
        self.last_brake_demand_nm = float(t_brake)
        self.last_brake_suppressed_nm = 0.0

        backdrivable = bool(getattr(self.p.gearbox, "backdrivable", True))
        backdrive_blocked = False

        # If gearbox is non-backdrivable/self-locking, assisting loads cannot transmit
        # power back to the motor/surface. We suppress the brake/regen torque demand and
        # assume it is dissipated internally (gear friction/self-locking).
        if (not backdrivable) and (t_brake > 1e-9):
            backdrive_blocked = True
            self.last_brake_suppressed_nm = float(t_brake)
            t_brake = 0.0

        return t_drive, t_brake, tau_tob, tau_bha, tau_par, backdrive_blocked

    def _mag_inverse_required_motor_torque_nm(self, t_gb_in: float) -> Tuple[bool, float, bool]:
        """Given required gearbox input torque magnitude (Nm), compute required motor-side torque magnitude (Nm).

        Returns: (ok, t_motor_req, slipping_flag)
        """
        mc = self.p.mag_coupler
        if not mc.enabled:
            return True, abs(t_gb_in), False
        t = abs(t_gb_in)
        if t > mc.t_slip_nm + 1e-12:
            return False, float("inf"), True
        # Inverse of: Tout = slope*(Tin - Tbreak)  (after breakaway)
        t_motor_req = mc.t_break_nm + (t / max(1e-12, mc.slope))
        return True, t_motor_req, False

    def _mag_forward_transmitted_to_gb_nm(self, t_motor_useful: float) -> Tuple[float, bool]:
        """Forward map: given useful motor torque magnitude, return transmitted gearbox input torque magnitude."""
        mc = self.p.mag_coupler
        t = abs(t_motor_useful)
        if not mc.enabled:
            return t, False
        if t <= mc.t_break_nm:
            return 0.0, False
        t_lin = mc.slope * (t - mc.t_break_nm)
        if t_lin >= mc.t_slip_nm:
            return mc.t_slip_nm, True
        return t_lin, False

    # ---------- Regen capability helpers ----------
    def regen_cap_output_torque_ftlbf(self, out_rpm_mag: float, cable: CableParams) -> float:
        """Approx max *electrical* braking torque magnitude available at the CCRS output.

        When cable-aware regen is enabled, this is limited by a simple generator condition:
            E_phase(omega) >= V_clamp_phase + |I|*|Z_cable(omega_e)|
        where E_phase is the no-load back-EMF, and V_clamp_phase approximates the
        surface inverter / brake-resistor clamp voltage.

        This is intentionally conservative and steady-state (RMS phasors).
        """
        p = self.p

        if out_rpm_mag <= 0.0:
            return 0.0

        if not bool(getattr(p, "braking_path_available", True)):
            return 0.0
        if not bool(getattr(p.gearbox, "backdrivable", True)):
            return 0.0

        # If the user disables the cable-aware constraint, we treat regen cap as symmetric with motoring
        # and let other checks (power limit, current limit, etc.) handle feasibility.
        if not bool(getattr(p, "regen_cable_limit_enabled", True)):
            # Fall back to a very large cap; callers should still clip to motoring cap / current limits.
            return float("inf")

        gp = p.gearbox
        mp = p.motor
        G = gp.ratio()
        eta = gp.eff_total()

        motor_rpm = float(out_rpm_mag) * float(G)
        # No-load back EMF (phase RMS)
        e_ll = float(mp.ke_vll_rms_per_krpm) * (motor_rpm / 1000.0)
        e_phase = e_ll / math.sqrt(3.0)

        # Surface clamp voltage for regen (phase RMS)
        v_lim = float(p.vf.v_phase_rms_limit())
        clamp_frac = float(getattr(p, "regen_surface_clamp_frac", 1.0))
        clamp_frac = min(max(clamp_frac, 0.0), 1.0)
        v_clamp = v_lim * clamp_frac

        if e_phase <= v_clamp + 1e-12:
            return 0.0

        # Cable impedance magnitude at electrical frequency
        pole_pairs = max(1, int(mp.pole_pairs))
        f_e = pole_pairs * (motor_rpm / 60.0)
        omega_e = 2.0 * math.pi * float(f_e)
        # Include sine-filter *series* impedance in the conservative regen cable limit (shunt cap ignored here).
        sf = getattr(p, "sine_filter", None)
        R = float(cable.effective_r_phase()) + (
            float(getattr(sf, "rf_ohm", 0.0)) if (sf is not None and bool(getattr(sf, "enabled", False))) else 0.0)
        L = float(cable.effective_l_phase()) + (
            float(getattr(sf, "lf_h", 0.0)) if (sf is not None and bool(getattr(sf, "enabled", False))) else 0.0)
        Z = math.sqrt(R * R + (omega_e * L) * (omega_e * L))
        Z = max(1e-12, Z)

        # Current that can be pushed uphole given E_phase and clamp
        i_cap = (e_phase - v_clamp) / Z

        # Apply cable / conductor current limit
        i_cap = max(0.0, min(float(i_cap), float(cable.i_phase_limit())))

        # Convert to motor torque via Kt(T); treat as torque-producing magnitude
        kt_eff = self.kt_effective_nm_per_arms()
        t_motor = float(kt_eff) * float(i_cap)

        # magnetic coupler (forward)
        t_gb_in_cap, _ = self._mag_forward_transmitted_to_gb_nm(t_motor)

        # gearbox to output
        t_out = t_gb_in_cap * float(G) * float(eta)

        # Optional surface braking power limit (acts as a hyperbola with speed)
        if bool(getattr(p, "brake_power_limit_enabled", False)):
            p_max_kw = float(getattr(p, "brake_power_kw_max", 0.0))
            omega_out = rpm_to_rad_s(float(out_rpm_mag))
            if omega_out > 1e-12 and p_max_kw > 0.0:
                t_pmax = (p_max_kw * 1000.0) / omega_out
                t_out = min(float(t_out), float(t_pmax))

        return float(nm_to_ft_lbf(t_out))

    # ---------- Extra torque + Kt(T) helpers ----------
    def kt_effective_nm_per_arms(self) -> float:
        """Return Kt(T) in canonical units (Nm/Arms).

        If temperature scaling is disabled, returns the canonical motor Kt.
        """
        mp = self.p.motor
        ex = self.p.extra
        kt0 = float(mp.kt_nm_per_arms)
        if not bool(getattr(ex, "kt_temp_enabled", False)):
            return max(1e-12, kt0)
        dT = float(ex.temp_C) - float(ex.temp_ref_C)
        kt = kt0 * (1.0 + float(ex.kt_temp_coeff_per_C) * dT)
        return max(1e-12, float(kt))

    def winding_temp_C(self) -> float:
        """Return winding (copper) temperature used for Rs(T).

        We treat extra.temp_C as the ambient / bulk motor temperature setting in the UI.
        If Rs(T) scaling is enabled, we apply a winding hot-spot rise:
            T_winding = temp_C + winding_rise_C
        """
        ex = self.p.extra
        t_amb = float(getattr(ex, "temp_C", 25.0))
        if not bool(getattr(ex, "rs_temp_enabled", False)):
            return t_amb
        return t_amb + float(getattr(ex, "winding_rise_C", 0.0))

    def rs_effective_ohm(self) -> float:
        """Return effective stator resistance (per-phase) for the dq model.

        If Rs(T) scaling is disabled, returns motor.rs_ohm as entered.
        If enabled, scales with a linear temperature coefficient:
            Rs(Tw) = Rs_ref * (1 + alpha_R * (Tw - T_ref))
        where Tw is the winding temperature (see winding_temp_C()).
        """
        mp = self.p.motor
        ex = self.p.extra
        rs0 = float(mp.rs_ohm)
        if not bool(getattr(ex, "rs_temp_enabled", False)):
            return max(1e-12, rs0)

        Tw = float(self.winding_temp_C())
        Tref = float(getattr(ex, "temp_ref_C", 25.0))
        alpha = float(getattr(ex, "rs_temp_coeff_per_C", 0.0))
        rs = rs0 * (1.0 + alpha * (Tw - Tref))
        return max(1e-12, float(rs))

    def _visc_temp_scale(self) -> float:
        ex = self.p.extra
        if not bool(getattr(ex, "visc_enabled", False)):
            return 1.0
        dT = float(ex.temp_C) - float(ex.temp_ref_C)
        mode = str(getattr(ex, "visc_temp_scaling", "None"))
        if mode.lower().startswith("linear"):
            return max(0.0, 1.0 + float(ex.visc_lin_coeff_per_C) * dT)
        if mode.lower().startswith("exponential"):
            beta = float(ex.visc_beta_per_C)
            # Typical viscosity decreases with temperature: positive beta reduces torque at higher T.
            return float(math.exp(-beta * dT))
        return 1.0

    def tau_core_nm(self, omega_m_rad_s: float) -> float:
        ex = self.p.extra
        if not bool(getattr(ex, "extra_enabled", False)):
            return 0.0
        if not bool(getattr(ex, "core_enabled", True)):
            return 0.0
        cL = float(ex.core_cL)
        if cL == 0.0:
            return 0.0
        exp = float(getattr(ex, "core_exp", 0.5))
        return max(0.0, cL * (abs(float(omega_m_rad_s)) ** max(0.0, exp)))

    def tau_visc_nm(self, omega_m_rad_s: float) -> float:
        ex = self.p.extra
        if not bool(getattr(ex, "extra_enabled", False)):
            return 0.0
        if not bool(getattr(ex, "visc_enabled", False)):
            return 0.0

        w = abs(float(omega_m_rad_s))
        scale = self._visc_temp_scale()
        model = str(getattr(ex, "visc_model", "Off"))

        # single-regime models
        if model.lower().startswith("off"):
            return 0.0
        if model.lower().startswith("couette"):
            return max(0.0, float(ex.visc_k_couette) * w * scale)
        if model.lower().startswith("turb"):
            return max(0.0, float(ex.visc_k_turb) * (w * w) * scale)
        if model.lower().startswith("transition"):
            n = max(0.0, float(ex.visc_n_transition))
            return max(0.0, float(ex.visc_k_transition) * (w ** n) * scale)

        # piecewise (Couette -> Transition -> Turbulent)
        rpm = w * 60.0 / (2.0 * math.pi)
        r1 = max(1e-9, float(ex.visc_rpm_1))
        r2 = max(r1 + 1e-9, float(ex.visc_rpm_2))
        w1 = rpm_to_rad_s(r1)
        w2 = rpm_to_rad_s(r2)

        tau1 = float(ex.visc_k_couette) * w
        n = max(0.0, float(ex.visc_n_transition))
        tau2 = float(ex.visc_k_transition) * (w ** n)
        tau3 = float(ex.visc_k_turb) * (w * w)

        if w <= w1:
            return max(0.0, tau1 * scale)
        if w >= w2:
            return max(0.0, tau3 * scale)

        # middle region: transition torque law
        tau_mid = tau2

        if bool(getattr(ex, "smooth_transitions", True)):
            frac = float(getattr(ex, "smooth_frac", 0.15))
            frac = float(np.clip(frac, 0.0, 0.49))
            # blend near w1
            w1a, w1b = w1 * (1.0 - frac), w1 * (1.0 + frac)
            if w <= w1b:
                s = (w - w1a) / max(1e-12, (w1b - w1a))
                s = float(np.clip(s, 0.0, 1.0))
                s = s * s * (3.0 - 2.0 * s)  # smoothstep
                tau_mid = (1.0 - s) * tau1 + s * tau2
            # blend near w2
            w2a, w2b = w2 * (1.0 - frac), w2 * (1.0 + frac)
            if w >= w2a:
                s = (w - w2a) / max(1e-12, (w2b - w2a))
                s = float(np.clip(s, 0.0, 1.0))
                s = s * s * (3.0 - 2.0 * s)
                tau_mid = (1.0 - s) * tau2 + s * tau3

        return max(0.0, float(tau_mid) * scale)

    def tau_extra_nm(self, omega_m_rad_s: float) -> Tuple[float, float, float]:
        """Return (tau_extra, tau_core, tau_visc) in Nm."""
        tc = self.tau_core_nm(omega_m_rad_s)
        tv = self.tau_visc_nm(omega_m_rad_s)
        return (tc + tv, tc, tv)

    def electrical_freq_hz(self, motor_rpm: float) -> float:
        mp = self.p.motor
        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(mp.pole_pairs)) * omega_m
        return omega_e / (2.0 * math.pi)

    def motor_voltage_required_phase_rms(self, motor_rpm: float, id_rms: float, iq_rms: float) -> float:
        """
        dq magnitude approximation, returned as phase RMS.
        Uses peak dq equations then converts to RMS.
        """
        mp = self.p.motor
        rs = self.rs_effective_ohm()

        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(mp.pole_pairs)) * omega_m

        id_peak = id_rms * math.sqrt(2.0)
        iq_peak = iq_rms * math.sqrt(2.0)

        # dq peak voltages
        # Vd = Rs*Id - omega_e*Lq*Iq
        # Vq = Rs*Iq + omega_e*(lambda + Ld*Id)
        vd_peak = rs * id_peak - omega_e * mp.lq_h * iq_peak
        vq_peak = rs * iq_peak + omega_e * (mp.lambda_wb + mp.ld_h * id_peak)

        v_peak = math.sqrt(vd_peak * vd_peak + vq_peak * vq_peak)
        return v_peak / math.sqrt(2.0)

    def cable_drop_phase_rms(self, motor_rpm: float, i_mag_rms: float, cable: CableParams) -> float:
        mp = self.p.motor

        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(mp.pole_pairs)) * omega_m

        R = cable.effective_r_phase()
        L = cable.effective_l_phase()
        X = omega_e * L

        i_peak = i_mag_rms * math.sqrt(2.0)
        vdrop_peak = i_peak * math.sqrt(R * R + X * X)
        return vdrop_peak / math.sqrt(2.0)

    # -----------------------------
    # Optional inverter output sine filter (steady-state fundamental)
    # -----------------------------
    def _sf(self) -> SineFilterParams:
        # Backward compatible if older settings are loaded without this block
        return getattr(self.p, "sine_filter", SineFilterParams())

    def sine_filter_c_wye_per_phase_f(self) -> float:
        sf = self._sf()
        c = float(getattr(sf, "cf_f", 0.0))
        if (not bool(getattr(sf, "enabled", False))) or c <= 0.0:
            return 0.0
        conn = str(getattr(sf, "cap_connection", "DELTA")).upper()
        if "DELTA" in conn:
            # delta → wye equivalent per-phase capacitance ≈ 3*C_delta
            return 3.0 * c
        return c

    def sine_filter_series_impedance_mag(self, motor_rpm: float) -> float:
        sf = self._sf()
        if not bool(getattr(sf, "enabled", False)):
            return 0.0
        R = max(0.0, float(getattr(sf, "rf_ohm", 0.0)))
        L = max(0.0, float(getattr(sf, "lf_h", 0.0)))
        if R <= 0.0 and L <= 0.0:
            return 0.0

        mp = self.p.motor
        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(getattr(mp, "pole_pairs", 1))) * omega_m
        X = omega_e * L
        return math.sqrt(R * R + X * X)

    def sine_filter_shunt_gb_si(self, motor_rpm: float) -> Tuple[float, float]:
        """Return shunt admittance components (g, b) in Siemens for wye-equivalent per-phase model."""
        sf = self._sf()
        if not bool(getattr(sf, "enabled", False)):
            return 0.0, 0.0

        C = self.sine_filter_c_wye_per_phase_f()
        if C <= 0.0:
            return 0.0, 0.0

        mp = self.p.motor
        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(getattr(mp, "pole_pairs", 1))) * omega_m
        if omega_e <= 1e-12:
            return 0.0, 0.0

        Rd = max(0.0, float(getattr(sf, "rd_ohm", 0.0)))
        topo = str(getattr(sf, "damping_topology", "SERIES")).upper()

        if "PAR" in topo:
            g = 0.0 if Rd <= 1e-12 else 1.0 / Rd
            b = omega_e * C
            return g, b

        # SERIES: Rd + (1/jωC)
        x = 1.0 / (omega_e * C)  # |Xc|
        den = (Rd * Rd + x * x)
        if den <= 1e-24:
            return 0.0, 0.0
        g = Rd / den
        b = x / den
        return g, b

    def sine_filter_shunt_current_components(self, motor_rpm: float, v_node_phase_rms: float) -> Tuple[
        float, float, float]:
        """Return (I_inphase, I_quadrature, I_mag) of shunt branch current in Arms at the node voltage."""
        g, b = self.sine_filter_shunt_gb_si(motor_rpm)
        v = max(0.0, float(v_node_phase_rms))
        i_in = g * v
        i_q = b * v
        i_mag = math.sqrt(i_in * i_in + i_q * i_q)
        return float(i_in), float(i_q), float(i_mag)

    def inverter_current_mag_rms(self, motor_rpm: float, v_node_phase_rms: float, i_load_phase_rms: float) -> float:
        """Approx inverter phase current magnitude at the inverter output."""
        sf = self._sf()
        if not bool(getattr(sf, "enabled", False)):
            return float(abs(i_load_phase_rms))

        i_in, i_q, _ = self.sine_filter_shunt_current_components(motor_rpm, v_node_phase_rms)
        i_load = float(abs(i_load_phase_rms))
        return math.sqrt((i_load + i_in) * (i_load + i_in) + i_q * i_q)

    def inverter_voltage_required_phase_rms(self, motor_rpm: float, v_node_phase_rms: float,
                                            i_load_phase_rms: float) -> float:
        """Approx required inverter phase RMS voltage (L-N) to sustain v_node at the cable/filter node."""
        sf = self._sf()
        if not bool(getattr(sf, "enabled", False)):
            return float(v_node_phase_rms)

        z_s = self.sine_filter_series_impedance_mag(motor_rpm)
        if z_s <= 0.0:
            return float(v_node_phase_rms)

        i_inv = self.inverter_current_mag_rms(motor_rpm, v_node_phase_rms, i_load_phase_rms)
        return float(v_node_phase_rms) + float(i_inv) * float(z_s)

    def cable_loss_w(self, i_mag_rms: float, cable: CableParams) -> float:
        R = cable.effective_r_phase()
        return 3.0 * (i_mag_rms ** 2) * R

    def cable_copper_loss_w(self, i_mag_rms: float, cable: CableParams) -> float:
        # Alias kept for compatibility with older UI versions
        return self.cable_loss_w(i_mag_rms, cable)

    @staticmethod
    def _bisect_root(fn, lo: float, hi: float, iters: int = 60) -> float:
        """Return the largest x in [lo, hi] such that fn(x) <= 0 (monotonic assumption)."""
        flo = fn(lo)
        fhi = fn(hi)
        if flo > 0:
            return lo
        if fhi <= 0:
            return hi
        a, b = lo, hi
        for _ in range(iters):
            m = 0.5 * (a + b)
            fm = fn(m)
            if fm <= 0:
                a = m
            else:
                b = m
        return a

    def _effective_downhole_phase_limit(self) -> Optional[float]:
        """
        Combine downhole limits into an equivalent phase RMS limit at the motor terminals:
          - Motor Vphase_rms limit
          - Contact block Vll_rms limit -> Vphase_rms = Vll_rms/sqrt(3)
        Use the most restrictive enabled limit.
        """
        lims = self.p.limits
        vals = []
        if lims.enforce_downhole_vphase_limit:
            vals.append(float(lims.downhole_v_phase_rms_limit))
        if lims.enforce_downhole_vll_limit:
            vals.append(float(lims.downhole_vll_rms_limit) / math.sqrt(3.0))
        if not vals:
            return None
        return min(vals)

    def _v_cmd_phase(self, motor_rpm: float) -> Tuple[float, float, float]:
        p = self.p
        f_e_hz = self.electrical_freq_hz(motor_rpm)

        v_surface_limit = p.vf.v_phase_rms_limit()

        # Strategy: how we interpret the available surface voltage.
        # - VF: scheduled V/f command (may be below Vmax at low speed)
        # - Mode A/B: treat Vmax as the usable ceiling (SVPWM/current-regulated style)
        mode = getattr(p, "control_strategy", "VF")
        if mode == "VF":
            v_cmd_phase = p.vf.v_surface_cmd_phase_rms(f_e_hz)
        else:
            v_cmd_phase = v_surface_limit  # use full available headroom

        return float(f_e_hz), float(v_cmd_phase), float(v_surface_limit)

    def _max_iq_no_fw(self, motor_rpm: float, cable: CableParams) -> Tuple[
        float, float, float, float, float, Optional[float]]:
        """Id=0 case: find max Iq (which equals Imag) subject to voltage and downhole constraints."""
        p = self.p
        i_lim = cable.i_phase_limit()
        f_e, v_cmd, v_surface_limit = self._v_cmd_phase(motor_rpm)

        v_dh_phase = self._effective_downhole_phase_limit()

        def f_volt(iq: float) -> float:
            v_motor = self.motor_voltage_required_phase_rms(motor_rpm, 0.0, iq)
            v_drop = self.cable_drop_phase_rms(motor_rpm, iq, cable)
            v_node = float(v_motor) + float(v_drop)
            v_inv = self.inverter_voltage_required_phase_rms(motor_rpm, v_node, float(abs(iq)))
            return float(v_inv) - float(v_cmd)

        iq_volt = self._bisect_root(f_volt, 0.0, i_lim)

        if v_dh_phase is not None:
            def f_dh(iq: float) -> float:
                return self.motor_voltage_required_phase_rms(motor_rpm, 0.0, iq) - v_dh_phase

            iq_dh = self._bisect_root(f_dh, 0.0, i_lim)
        else:
            iq_dh = i_lim
        # If a sine filter is enabled, inverter current can exceed load current due to shunt capacitor draw.
        # Option B enforces an inverter-side current magnitude limit using the same limit basis as i_lim (conservative).
        iq_inv = float('inf')
        sf = self.p.sine_filter
        if sf.enabled and getattr(sf, 'enforce_inv_current_limit', True):
            def f_iinv(iq: float) -> float:
                i_load = abs(float(iq))
                v_motor = self.motor_voltage_required_phase_rms(motor_rpm, id_rms=0.0, iq_rms=i_load)
                v_drop = self.cable_drop_phase_rms(motor_rpm, i_load, cable)
                v_node = v_motor + v_drop  # RMS phase at filter output / cable input
                i_inv = self.inverter_current_mag_rms(motor_rpm, v_node, i_load)
                return float(i_inv) - float(i_lim)

            iq_inv = self._bisect_root(f_iinv, 0.0, float(i_lim))
        else:
            iq_inv = float(i_lim)

        iq_max = min(i_lim, iq_volt, iq_dh, iq_inv)
        # with Id=0: i_mag = iq
        return float(iq_max), 0.0, float(iq_max), float(f_e), float(v_cmd), float(v_surface_limit), v_dh_phase

    def max_iq_given_limits(self, motor_rpm: float, cable: CableParams) -> Tuple[
        float, float, float, float, float, float, Optional[float]]:
        """
        Return:
          iq_max_rms, id_best_rms, i_mag_best_rms, f_e, v_cmd, v_surface_limit, v_dh_phase
        """
        p = self.p
        fw = p.fw
        # Effective FW behavior may be overridden by control strategy.
        #   - MODE_A forces Id=0 (no FW)
        #   - MODE_B forces FW enabled (Id scan)
        mode = getattr(p, "control_strategy", "VF")
        fw_enabled_eff = fw.enabled
        if mode == "MODE_A":
            fw_enabled_eff = False
        elif mode == "MODE_B":
            fw_enabled_eff = True

        iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase = self._max_iq_no_fw(motor_rpm, cable)

        if (not fw_enabled_eff) or fw.id_max_arms <= 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        if fw.apply_only_above_base and f_e <= p.vf.base_freq_hz + 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        i_lim = cable.i_phase_limit()
        id_max = min(float(abs(fw.id_max_arms)), float(i_lim))
        if id_max <= 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        # scan Id in [0, -id_max]
        n_id = max(9, int(fw.id_grid_points))
        id_candidates = np.linspace(0.0, -id_max, n_id)

        best_iq = 0.0
        best_id = 0.0
        best_im = 0.0

        for id_rms in id_candidates:
            iq_hi = math.sqrt(max(0.0, i_lim * i_lim - id_rms * id_rms))
            if iq_hi <= 1e-9:
                continue

            def f_volt(iq: float) -> float:
                i_mag = math.sqrt(id_rms * id_rms + iq * iq)
                v_motor = self.motor_voltage_required_phase_rms(motor_rpm, id_rms, iq)
                v_drop = self.cable_drop_phase_rms(motor_rpm, i_mag, cable)
                v_node = float(v_motor) + float(v_drop)
                v_inv = self.inverter_voltage_required_phase_rms(motor_rpm, v_node, float(i_mag))
                return float(v_inv) - float(v_cmd)

            iq_volt = self._bisect_root(f_volt, 0.0, iq_hi)

            if v_dh_phase is not None:
                def f_dh(iq: float) -> float:
                    return self.motor_voltage_required_phase_rms(motor_rpm, id_rms, iq) - v_dh_phase

                iq_dh = self._bisect_root(f_dh, 0.0, iq_hi)
            else:
                iq_dh = iq_hi

            # Option B: also enforce inverter-side current limit when a sine filter is enabled.
            iq_inv = iq_hi
            sf = self.p.sine_filter
            if sf.enabled and getattr(sf, 'enforce_inv_current_limit', True):
                def f_iinv(iq: float) -> float:
                    i_mag = math.sqrt(id_rms * id_rms + iq * iq)
                    v_motor = self.motor_voltage_required_phase_rms(motor_rpm, id_rms, iq)
                    v_drop = self.cable_drop_phase_rms(motor_rpm, i_mag, cable)
                    v_node = float(v_motor) + float(v_drop)
                    i_inv = self.inverter_current_mag_rms(motor_rpm, v_node, float(i_mag))
                    return float(i_inv) - float(i_lim)

                iq_inv = self._bisect_root(f_iinv, 0.0, iq_hi)

            iq_max_id = min(iq_hi, iq_volt, iq_dh, iq_inv)

            if iq_max_id > best_iq:
                best_iq = iq_max_id
                best_id = float(id_rms)
                best_im = math.sqrt(best_id * best_id + best_iq * best_iq)

        # If FW didn't improve, keep no-FW solution
        if best_iq <= iq0 + 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        return float(best_iq), float(best_id), float(best_im), float(f_e), float(v_cmd), float(
            v_surface_limit), v_dh_phase

    def max_iq_given_limits_fast(self, motor_rpm: float, cable: CableParams) -> Tuple[
        float, float, float, float, float, float, Optional[float]]:
        """A faster (approx) variant used for sweeps.

        Instead of scanning a grid of Id values, it evaluates only:
          - Id = 0 (no field weakening)
          - Id = -Id_max (max weakening), if enabled

        This is usually close enough for trade-off plots and is much faster.
        """
        p = self.p
        fw = p.fw
        fw_enabled_eff = fw.enabled  # effective FW enable

        # baseline no-FW
        iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase = self._max_iq_no_fw(motor_rpm, cable)

        # if FW isn't active at this speed, return baseline
        if (not fw_enabled_eff) or fw.id_max_arms <= 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase
        if fw.apply_only_above_base and f_e <= p.vf.base_freq_hz + 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        i_lim = cable.i_phase_limit()
        id_max = min(float(abs(fw.id_max_arms)), float(i_lim))
        if id_max <= 1e-9:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        id_try = -id_max
        iq_hi = math.sqrt(max(0.0, i_lim * i_lim - id_try * id_try))
        if iq_hi <= 1e-12:
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        def violates(iq: float) -> bool:
            i_mag = math.sqrt(id_try * id_try + iq * iq)
            if i_mag > i_lim + 1e-12:
                return True
            v_motor = self.motor_voltage_required_phase_rms(motor_rpm, id_try, iq)
            v_drop = self.cable_drop_phase_rms(motor_rpm, i_mag, cable)
            v_node = float(v_motor) + float(v_drop)
            sf = self.p.sine_filter
            if sf.enabled and getattr(sf, 'enforce_inv_current_limit', True):
                i_inv = self.inverter_current_mag_rms(motor_rpm, v_node, float(i_mag))
                if float(i_inv) > float(i_lim) + 1e-12:
                    return True

            v_inv = self.inverter_voltage_required_phase_rms(motor_rpm, v_node, float(i_mag))
            if float(v_inv) > float(v_cmd) + 1e-12:
                return True
            if v_dh_phase is not None and v_motor > v_dh_phase + 1e-12:
                return True
            return False

        # if even iq=0 violates (rare), give up
        if violates(0.0):
            return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

        # if max iq is feasible, take it
        if not violates(iq_hi):
            iq_best = iq_hi
        else:
            lo, hi = 0.0, iq_hi
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if violates(mid):
                    hi = mid
                else:
                    lo = mid
            iq_best = lo

        if iq_best > iq0 + 1e-9:
            im_best = math.sqrt(id_try * id_try + iq_best * iq_best)
            return float(iq_best), float(id_try), float(im_best), float(f_e), float(v_cmd), float(
                v_surface_limit), v_dh_phase

        return iq0, id0, im0, f_e, v_cmd, v_surface_limit, v_dh_phase

    def find_feasible_id_for_iq(self, motor_rpm: float, iq_req: float, cable: CableParams) -> Tuple[
        bool, float, float, float, float, float, Optional[float]]:
        """
        Check if required Iq can be achieved with some Id (possibly 0) under all constraints.

        Returns:
          feasible, id_used_rms, i_mag_used_rms, f_e, v_cmd, v_surface_limit, v_dh_phase
        """
        p = self.p
        fw = p.fw
        i_lim = cable.i_phase_limit()
        f_e, v_cmd, v_surface_limit = self._v_cmd_phase(motor_rpm)
        v_dh_phase = self._effective_downhole_phase_limit()

        # candidate Id range
        if (not fw.enabled) or fw.id_max_arms <= 1e-9 or (fw.apply_only_above_base and f_e <= p.vf.base_freq_hz + 1e-9):
            id_candidates = [0.0]
        else:
            id_max = min(float(abs(fw.id_max_arms)), float(i_lim))
            n_id = max(11, int(fw.id_grid_points))
            id_candidates = np.linspace(0.0, -id_max, n_id)

        best = None  # minimize i_mag for the first feasible solution
        for id_rms in id_candidates:
            i_mag = math.sqrt(id_rms * id_rms + iq_req * iq_req)
            if i_mag > i_lim + 1e-12:
                continue

            v_motor = self.motor_voltage_required_phase_rms(motor_rpm, id_rms, iq_req)
            v_drop = self.cable_drop_phase_rms(motor_rpm, i_mag, cable)
            v_node = float(v_motor) + float(v_drop)
            sf = self.p.sine_filter
            if sf.enabled and getattr(sf, 'enforce_inv_current_limit', True):
                i_inv = self.inverter_current_mag_rms(motor_rpm, v_node, float(i_mag))
                if float(i_inv) > float(i_lim) + 1e-12:
                    continue

            v_inv = self.inverter_voltage_required_phase_rms(motor_rpm, v_node, float(i_mag))

            if float(v_inv) > float(v_cmd) + 1e-12:
                continue

            if v_dh_phase is not None and v_motor > v_dh_phase + 1e-12:
                continue

            # feasible
            if best is None or i_mag < best[1]:
                best = (float(id_rms), float(i_mag))

        if best is None:
            return False, 0.0, float(i_lim), f_e, v_cmd, v_surface_limit, v_dh_phase

        return True, best[0], best[1], f_e, v_cmd, v_surface_limit, v_dh_phase

    def solve_target(self, cable_override: Optional[CableParams] = None) -> SolveResult:
        p = self.p
        gp = p.gearbox
        mp = p.motor

        cable = cable_override if cable_override is not None else p.cable

        G = gp.ratio()
        eta = gp.eff_total()

        # Commanded CCRS output speed:
        #   - UI stores a magnitude (>=0)
        #   - direction is selected separately (CW/CCW)
        out_rpm_cmd_mag = 0.0 if bool(getattr(p, "stuck_mode", False)) else float(max(0.0, p.target.out_rpm))
        out_rpm_cmd_signed = out_rpm_cmd_mag * self.out_dir_sign()
        omega_out = rpm_to_rad_s(out_rpm_cmd_mag) * self.out_dir_sign()

        # Required output drive torque is computed from enabled static load blocks.
        # NOTE: TargetParams.out_torque_ftlbf is the stuck/stall torque requirement (used only when stuck_mode is enabled).
        #       Drilling TOB magnitude is separately defined in BHABlockParams.drilling_tob_ftlbf.
        t_drive_out_nm, t_brake_out_nm, tau_tob_nm, tau_bha_nm, tau_par_nm, backdrive_blocked = self._required_output_drive_torque_nm(
            omega_out)

        # Optional: UI torque target override for continuous mode.
        # Historically, out_torque_ftlbf was used only for stuck/stall. In Navigator CCRS feasibility
        # we often want to judge the design against an explicit UI torque requirement at the UI RPM.
        # When enabled, we enforce the larger of (load-stack motoring requirement, UI torque target).
        _ui_override_on = bool(getattr(getattr(p, 'target', None), 'torque_override_continuous', True))
        _ui_tq_nm = ft_lbf_to_nm(abs(float(getattr(p.target, 'out_torque_ftlbf', 0.0))))
        _ui_override_msgs: List[str] = []
        if (not bool(getattr(p, 'stuck_mode', False))) and _ui_override_on and (_ui_tq_nm > 1e-9):
            _t_drive_stack_nm = float(t_drive_out_nm)
            _t_brake_stack_nm = float(t_brake_out_nm)
            # Treat UI torque target as a resisting (motoring) requirement.
            t_drive_out_nm = max(float(t_drive_out_nm), float(_ui_tq_nm))
            t_brake_out_nm = 0.0
            if abs(t_drive_out_nm - _t_drive_stack_nm) > 1e-9 or abs(_t_brake_stack_nm) > 1e-9:
                _ui_override_msgs.append(
                    f"UI torque override: using {nm_to_ft_lbf(t_drive_out_nm):.0f} ft-lbf @ output (stack was {nm_to_ft_lbf(_t_drive_stack_nm):.0f} ft-lbf)."
                )

        # Determine whether the requested operating point needs motoring torque or electrical braking (regen).
        #  - Motoring: output torque required is in the same direction as commanded rotation.
        #  - Regen: output torque required opposes commanded rotation (electrical braking / TOB-assist case).
        regen_required = (t_brake_out_nm > 1e-9) and (not bool(getattr(p, "stuck_mode", False))) and (
                out_rpm_cmd_mag > 1e-12)
        t_out_for_elec_nm = t_brake_out_nm if regen_required else t_drive_out_nm

        # Convert the required output torque (drive or brake) into gearbox input torque magnitude
        t_gb_in_nm = t_out_for_elec_nm / max(1e-12, (G * eta))

        # Optional magnetic coupler (inverse mapping from gearbox input torque → motor torque)
        ok_cpl, t_motor_load_nm_mag, mag_slip = self._mag_inverse_required_motor_torque_nm(t_gb_in_nm)

        motor_rpm = out_rpm_cmd_mag * G  # electrical model uses magnitude
        motor_torque_nm = t_motor_load_nm_mag  # motor-side load torque magnitude (Nm)

        # ---- Requested mapping: Iq_req = (tau_load + tau_extra(omega, T)) / Kt(T)
        kt_eff = self.kt_effective_nm_per_arms()
        omega_m = rpm_to_rad_s(motor_rpm)
        tau_extra, tau_core, tau_visc = self.tau_extra_nm(omega_m)
        motor_torque_total = motor_torque_nm + tau_extra

        iq_req_base = motor_torque_nm / max(1e-9, kt_eff)
        iq_req = motor_torque_total / max(1e-9, kt_eff)
        # capability (max iq)
        iq_max, id_best_at_max, im_best_at_max, f_e, v_cmd, v_surface_limit, v_dh_phase = self.max_iq_given_limits(
            motor_rpm, cable)

        feasible_elec, id_used, im_used, _, _, _, _ = self.find_feasible_id_for_iq(motor_rpm, iq_req, cable)

        # decide which operating point to report for voltage budget
        if feasible_elec:
            id_rep = id_used
            iq_rep = iq_req
            im_rep = im_used
        else:
            id_rep = id_best_at_max
            iq_rep = iq_max
            im_rep = im_best_at_max

        # In regen (electrical braking), the inverter's V/f scheduled command is not the governing clamp;
        # regen feasibility is handled separately. For reporting, we use Id=0 and negative Iq.
        if regen_required:
            feasible_elec = True  # may still be rejected by explicit regen checks below
            id_rep = 0.0
            iq_rep = -abs(iq_req)
            im_rep = abs(iq_req)
        v_motor_req = self.motor_voltage_required_phase_rms(motor_rpm, id_rep, iq_rep)
        v_drop = self.cable_drop_phase_rms(motor_rpm, im_rep, cable)
        # Voltage at the cable/filter node (after sine filter series element, before cable)
        v_node_req = float(v_motor_req) + float(v_drop)

        # Optional sine filter adds inverter-side current and voltage demand
        i_inverter_rms = self.inverter_current_mag_rms(motor_rpm, v_node_req, float(im_rep))
        _, _, i_filter_cap_rms = self.sine_filter_shunt_current_components(motor_rpm, v_node_req)

        v_inverter_req = self.inverter_voltage_required_phase_rms(motor_rpm, v_node_req, float(im_rep))
        v_filter_drop = max(0.0, float(v_inverter_req) - float(v_node_req))
        fails: List[str] = []
        notes: List[str] = []
        if '_ui_override_msgs' in locals() and _ui_override_msgs:
            notes.extend(_ui_override_msgs)
        if backdrive_blocked:
            notes.append(
                "Gearbox set non-backdrivable (self-locking): assisting loads cannot backdrive the motor; electrical regen/braking is not available.")
            if getattr(self, "last_brake_suppressed_nm", 0.0) > 1e-9:
                notes.append(
                    f"Assist torque suppressed (assumed dissipated in gearbox): {nm_to_ft_lbf(self.last_brake_suppressed_nm):.0f} ft-lbf.")

        # Regen: optional cable-aware constraint (back-EMF must exceed surface clamp + cable drop)
        regen_cap_ftlbf = 0.0
        regen_clamp_phase_v = 0.0
        if regen_required:
            regen_cap_ftlbf = float(self.regen_cap_output_torque_ftlbf(out_rpm_cmd_mag, cable))
            clamp_frac = float(getattr(p, 'regen_surface_clamp_frac', 1.0))
            regen_clamp_phase_v = float(p.vf.v_phase_rms_limit()) * max(0.0, clamp_frac)

        if not ok_cpl:
            fails.append(
                f"Magnetic coupler slip limit exceeded: T_gb_in={t_gb_in_nm:.2f} Nm > T_slip={p.mag_coupler.t_slip_nm:.2f} Nm"
            )

        # If braking would be required (regen quadrant), check whether we can absorb power.
        if t_brake_out_nm > 1e-9:
            braking_ok = bool(getattr(p, "braking_path_available", True)) and bool(
                getattr(p.gearbox, "backdrivable", True))
            if not braking_ok:
                if not bool(getattr(p, "braking_path_available", True)):
                    fails.append(
                        f"REGEN REQUIRED but braking path is OFF: need to absorb {nm_to_ft_lbf(t_brake_out_nm):.0f} ft-lbf at this speed"
                    )
                else:
                    fails.append(
                        f"REGEN REQUIRED but gearbox is set non-backdrivable: need to absorb {nm_to_ft_lbf(t_brake_out_nm):.0f} ft-lbf at this speed"
                    )
            else:
                # Cable-aware regen cap (based on motor back-EMF vs surface clamp + cable impedance)
                if regen_required:
                    if bool(getattr(p, 'regen_cable_limit_enabled', True)):
                        t_req_ftlbf = nm_to_ft_lbf(t_brake_out_nm)
                        if regen_cap_ftlbf + 1e-9 < t_req_ftlbf:
                            fails.append(
                                f'Regen limited by cable+surface clamp: Tcap≈{regen_cap_ftlbf:.0f} ft-lbf < Treq={t_req_ftlbf:.0f} ft-lbf'
                            )
                        else:
                            notes.append(
                                f'Regen cable-aware cap: Tcap≈{regen_cap_ftlbf:.0f} ft-lbf (clamp≈{regen_clamp_phase_v:.0f} Vrms/ph)')
                    else:
                        notes.append('Regen cable-aware cap: DISABLED (using symmetric torque limits).')

                if bool(getattr(p, "brake_power_limit_enabled", False)) and abs(omega_out) > 1e-9:
                    p_req_kw = (t_brake_out_nm * abs(omega_out)) / 1000.0
                    p_max_kw = float(getattr(p, "brake_power_kw_max", 0.0))
                    if p_req_kw > p_max_kw + 1e-9:
                        fails.append(
                            f"Surface braking power limit exceeded: Preq={p_req_kw:.2f} kW > Pmax={p_max_kw:.2f} kW"
                        )
                    else:
                        notes.append(
                            f"Regen quadrant: T_brake={nm_to_ft_lbf(t_brake_out_nm):.0f} ft-lbf, P={p_req_kw:.2f} kW (≤ {p_max_kw:.2f} kW)"
                        )
                else:
                    notes.append(
                        f"Regen quadrant: T_brake={nm_to_ft_lbf(t_brake_out_nm):.0f} ft-lbf (braking path available)"
                    )

        i_lim = cable.i_phase_limit()

        if iq_req > iq_max + 1e-12:
            fails.append(f"Iq_max={iq_max:.3f} Arms < Iq_req={iq_req:.3f} Arms")

        # Explain which constraint is likely binding
        # (we do direct checks at the required operating point with Id=0 for readability)
        if not regen_required:
            v_node_no_fw = self.motor_voltage_required_phase_rms(motor_rpm, 0.0, iq_req) + self.cable_drop_phase_rms(
                motor_rpm, iq_req, cable)
            v_need_no_fw = self.inverter_voltage_required_phase_rms(motor_rpm, float(v_node_no_fw), float(abs(iq_req)))
            if float(v_need_no_fw) > float(v_cmd) + 1e-12:
                fails.append(
                    f"Surface V/f command insufficient (Id=0): Vneed={v_need_no_fw:.1f} Vrms(ph) > Vcmd={v_cmd:.1f} Vrms(ph)")
        else:
            notes.append(
                "Regen: V/f scheduled command is not used as the clamp; regen capability is evaluated separately.")

        # Downhole limits reporting
        vll_lim = p.limits.downhole_vll_rms_limit if p.limits.enforce_downhole_vll_limit else None
        vph_lim = p.limits.downhole_v_phase_rms_limit if p.limits.enforce_downhole_vphase_limit else None
        v_dh_eff = self._effective_downhole_phase_limit()
        if v_dh_eff is not None:
            vm_req_id0 = self.motor_voltage_required_phase_rms(motor_rpm, 0.0, iq_req)
            if vm_req_id0 > v_dh_eff + 1e-12:
                fails.append(
                    f"Downhole motor-side voltage limit exceeded (Id=0): Vmotor={vm_req_id0:.1f} Vrms(ph) > {v_dh_eff:.1f} Vrms(ph)")

        # Also show if pure current magnitude is exceeded (only possible with FW enabled)
        if im_used > i_lim + 1e-12:
            fails.append(
                f"Cable phase-current magnitude exceeded: |I|={im_used:.3f} Arms > Ilim_phase={i_lim:.3f} Arms")

        p_loss = self.cable_loss_w(im_rep, cable)

        # hints
        # (best-case: assumes voltage is not limiting; includes tau_extra if enabled)
        kt_min = motor_torque_total / max(1e-9, i_lim)

        v_motor_avail = max(0.0, v_cmd - self.cable_drop_phase_rms(motor_rpm, im_rep, cable))
        omega_m = rpm_to_rad_s(motor_rpm)
        omega_e = max(1, int(mp.pole_pairs)) * omega_m
        if omega_e > 1e-9:
            lambda_max = v_motor_avail * math.sqrt(2.0) / omega_e
            p_pairs = max(1, int(mp.pole_pairs))
            ke_ll_rms_per_rad = p_pairs * lambda_max * math.sqrt(3.0) / math.sqrt(2.0)
            ke_vll_krpm_max = ke_ll_rms_per_rad * (1000.0 * 2.0 * math.pi / 60.0)
        else:
            ke_vll_krpm_max = float("inf")

        return SolveResult(
            feasible=bool(feasible_elec) and (len(fails) == 0),
            reasons=fails,
            notes=notes,
            gear_ratio=G,
            gear_eff=eta,
            motor_rpm=motor_rpm,
            motor_torque_nm=motor_torque_nm,
            temp_C=float(p.extra.temp_C),
            winding_temp_C=float(self.winding_temp_C()),
            rs_eff_ohm=float(self.rs_effective_ohm()),
            kt_eff_nm_per_arms=float(kt_eff),
            tau_core_nm=float(tau_core),
            tau_visc_nm=float(tau_visc),
            tau_extra_nm=float(tau_extra),
            motor_torque_total_nm=float(motor_torque_total),
            iq_req_base_rms=float(iq_req_base),
            iq_req_rms=iq_req,
            iq_max_rms=iq_max,
            id_used_rms=id_rep,
            i_mag_used_rms=im_rep,
            i_limit_phase_mag=i_lim,
            f_e_hz=f_e,
            v_surface_cmd=(regen_clamp_phase_v if (
                    regen_required and bool(getattr(p, 'regen_cable_limit_enabled', True))) else v_cmd),
            v_surface_limit=v_surface_limit,
            v_motor_req=v_motor_req,
            v_cable_drop=v_drop,
            v_node_req=v_node_req,
            v_filter_drop=v_filter_drop,
            v_inverter_req=v_inverter_req,
            i_inverter_rms=i_inverter_rms,
            i_filter_cap_rms=i_filter_cap_rms,
            v_downhole_phase_limit=vph_lim,
            v_downhole_ll_limit=vll_lim,
            p_cable_loss_w=p_loss,
            kt_required_min=kt_min,
            ke_required_max_vll_krpm=ke_vll_krpm_max,
            out_dir=str(p.out_dir),
            out_rpm_cmd=float(out_rpm_cmd_signed),
            out_drive_torque_req_ftlbf=float(nm_to_ft_lbf(t_drive_out_nm)),
            out_brake_torque_req_ftlbf=float(nm_to_ft_lbf(t_brake_out_nm)),
            brake_suppressed_ftlbf=float(nm_to_ft_lbf(getattr(self, 'last_brake_suppressed_nm', 0.0))),
            tob_reaction_ftlbf=float(nm_to_ft_lbf(tau_tob_nm)),
            bha_friction_ftlbf=float(nm_to_ft_lbf(tau_bha_nm)),
            parasitic_ftlbf=float(nm_to_ft_lbf(tau_par_nm)),
            regen_required=bool(regen_required),
            regen_cap_ftlbf=float(regen_cap_ftlbf),
            regen_clamp_phase_v=float(regen_clamp_phase_v),
            mag_slipping=bool(mag_slip),

        )

    def solve_point(
            self,
            out_rpm: float,
            out_torque_ftlbf: float,
            cable_override: Optional[CableParams] = None,
    ) -> SolveResult:
        """Solve a single operating point without changing the UI state permanently."""
        p = self.p
        old = copy.deepcopy(p.target)
        try:
            p.target.out_rpm = float(out_rpm)
            p.target.out_torque_ftlbf = float(out_torque_ftlbf)
            return self.solve_target(cable_override=cable_override)
        finally:
            p.target = old

    def compute_envelope(
            self,
            out_rpm_max: float = 1.2,
            n: int = 180,
            cable_override: Optional[CableParams] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          out_rpm, t_out_max_ftlbf, iq_max_arr, cable_loss_arr  (loss computed at the current magnitude used at max torque)
        """
        p = self.p
        gp = p.gearbox
        mp = p.motor
        cable = cable_override if cable_override is not None else p.cable

        G = gp.ratio()
        eta = gp.eff_total()

        rpms = np.linspace(0.01, max(0.02, out_rpm_max), n)
        t_out_max_ftlbf = np.zeros_like(rpms)
        iq_max_arr = np.zeros_like(rpms)
        loss_arr = np.zeros_like(rpms)

        for k, out_rpm in enumerate(rpms):
            motor_rpm = float(out_rpm * G)
            iq_max, id_best, im_best, _, _, _, _ = self.max_iq_given_limits(motor_rpm, cable)
            iq_max_arr[k] = iq_max

            kt_eff = self.kt_effective_nm_per_arms()
            omega_m = rpm_to_rad_s(motor_rpm)
            tau_extra, _, _ = self.tau_extra_nm(omega_m)

            # available *useful* motor torque after internal/core/viscous torque is overcome
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))

            # optional magnetic coupler limits how much of that motor torque reaches gearbox input
            t_gb_in_cap, mag_slip = self._mag_forward_transmitted_to_gb_nm(t_motor_useful)

            # gearbox delivers to output
            t_out_raw_nm = t_gb_in_cap * G * eta

            # subtract internal tool parasitics at the output shaft (if enabled)
            omega_out = rpm_to_rad_s(float(out_rpm)) * self.out_dir_sign()
            tau_par = self._rot_loss_torque_nm(omega_out, p.parasitic)
            t_out_cap_nm = max(0.0, t_out_raw_nm - abs(tau_par))

            t_out_max_ftlbf[k] = nm_to_ft_lbf(t_out_cap_nm)

            loss_arr[k] = self.cable_loss_w(im_best, cable)

        return rpms, t_out_max_ftlbf, iq_max_arr, loss_arr


# -----------------------------
# UI helpers
# -----------------------------

def make_dspin(minv: float, maxv: float, step: float, val: float, decimals: int = 3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(minv, maxv)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    w.setKeyboardTracking(False)
    return w


def make_ispin(minv: int, maxv: int, step: int, val: int) -> QSpinBox:
    w = QSpinBox()
    w.setRange(minv, maxv)
    w.setSingleStep(step)
    w.setValue(val)
    w.setKeyboardTracking(False)
    return w


def slider_for_spin(spin) -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setTracking(True)
    s.setFocusPolicy(Qt.NoFocus)
    s.setTickPosition(QSlider.NoTicks)
    s.setMinimumWidth(170)

    if isinstance(spin, QDoubleSpinBox):
        lo = float(spin.minimum())
        hi = float(spin.maximum())
        step = float(spin.singleStep()) or 0.001
        # If (range/step) becomes huge (e.g., micro-step spinboxes), the slider's
        # integer range can explode and make Qt sluggish. Cap the slider resolution
        # and use a derived slider-step while keeping the spinbox step intact.
        imax_raw = int(round((hi - lo) / step))
        imax_raw = max(1, imax_raw)
        max_slider_steps = 50000
        if imax_raw > max_slider_steps:
            imax = max_slider_steps
            step_s = (hi - lo) / float(imax)
            step_s = max(step_s, 1e-12)
        else:
            imax = imax_raw
            step_s = step

        s.setRange(0, imax)
        s.setSingleStep(1)
        s.setPageStep(max(1, imax // 20))

        def _spin_to_slider(_=None):
            idx = int(round((float(spin.value()) - lo) / step_s))
            idx = min(max(0, idx), imax)
            s.blockSignals(True)
            s.setValue(idx)
            s.blockSignals(False)

        def _slider_to_spin(idx: int):
            val = lo + float(idx) * step_s
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            try:
                spin.valueChanged.emit(float(spin.value()))
            except TypeError:
                spin.valueChanged.emit(spin.value())

        spin.valueChanged.connect(_spin_to_slider)
        s.valueChanged.connect(_slider_to_spin)
        _spin_to_slider()

    else:
        lo = int(spin.minimum())
        hi = int(spin.maximum())
        s.setRange(lo, hi)
        s.setSingleStep(int(max(1, spin.singleStep())))
        s.setPageStep(max(1, (hi - lo) // 20) if hi > lo else 1)

        def _spin_to_slider(val: int):
            s.blockSignals(True)
            s.setValue(int(val))
            s.blockSignals(False)

        def _slider_to_spin(val: int):
            spin.blockSignals(True)
            spin.setValue(int(val))
            spin.blockSignals(False)
            try:
                spin.valueChanged.emit(int(spin.value()))
            except TypeError:
                spin.valueChanged.emit(spin.value())

        spin.valueChanged.connect(_spin_to_slider)
        s.valueChanged.connect(_slider_to_spin)
        _spin_to_slider(spin.value())

    return s


def add_slider_row(grid: QGridLayout, row: int, label: str, spin, tooltip: Optional[str] = None):
    lab = QLabel(label)
    if tooltip:
        lab.setToolTip(tooltip)
        spin.setToolTip(tooltip)
    grid.addWidget(lab, row, 0)
    sl = slider_for_spin(spin)
    grid.addWidget(sl, row, 1)
    grid.addWidget(spin, row, 2)
    return sl


class MplPane(QWidget):
    """Envelope plots pane.

    Layout update (v15.x):
      - Left: large 4-quadrant torque envelope (main plot)
      - Right: three diagnostics stacked vertically (Voltage / Current / Loss)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Larger canvas so the 4Q envelope is readable and dominant.
        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)

        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[2.45, 1.0],
            wspace=0.18,
        )

        # Main 4Q plot on the left
        self.ax_env = self.fig.add_subplot(gs[0, 0])

        # Three stacked plots on the right
        gs_right = gs[0, 1].subgridspec(nrows=3, ncols=1, hspace=0.32)
        self.ax_v = self.fig.add_subplot(gs_right[0, 0])
        self.ax_i = self.fig.add_subplot(gs_right[1, 0])
        self.ax_loss = self.fig.add_subplot(gs_right[2, 0])

        # Manual margins (avoid tight_layout warnings with nested gridspec)
        self.fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.07)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)


class SweepPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fig = Figure(figsize=(11, 7.5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax_t = self.fig.add_subplot(2, 2, 1)
        self.ax_emf = self.fig.add_subplot(2, 2, 2)
        self.ax_ke = self.fig.add_subplot(2, 2, 3)
        self.ax_eta = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=2.0)


class QuadSweepPane(QWidget):
    """A generic 2x2 matplotlib pane for additional sweep tabs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fig = Figure(figsize=(11, 7.5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=2.0)


class BandPane(QWidget):
    """Band plots pane.

    v15 layout:
      - Left: two large band plots stacked (important)
      - Right: three utilization/diagnostic plots stacked vertically

    We avoid tight_layout() warnings by using explicit GridSpec spacing.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Larger canvas – band plots are meant to be readable.
        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)

        # Nested grids: left column (2 rows), right column (3 rows)
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[2.35, 1.0],
            wspace=0.18,
        )
        gs_left = gs[0].subgridspec(nrows=2, ncols=1, hspace=0.18)
        gs_right = gs[1].subgridspec(nrows=3, ncols=1, hspace=0.30)

        self.ax1 = self.fig.add_subplot(gs_left[0, 0])
        self.ax2 = self.fig.add_subplot(gs_left[1, 0])

        self.ax3 = self.fig.add_subplot(gs_right[0, 0])
        self.ax4 = self.fig.add_subplot(gs_right[1, 0])
        self.ax5 = self.fig.add_subplot(gs_right[2, 0])

        # Manual margins (avoid tight_layout warnings with nested gridspec)
        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)


class TablesPane(QWidget):
    """Decision tables pane.

    Provides compact, copy-friendly tables for quick wiring/limit trade decisions.
    Tables are rendered with matplotlib (same styling approach as other in-app tables).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)
        gs = self.fig.add_gridspec(nrows=2, ncols=2, hspace=0.20, wspace=0.14)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])

        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.06)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)


class MotorOpsPane(QWidget):
    """Motor operating behavior pane.

    Shows what the motor is doing versus output RPM for 1-wire/phase and 2-wires/phase:
    - Currents (Iq and |I|)
    - Motor terminal voltage (VLL,rms required)
    - Back-EMF (VLL,rms)
    - Electrical frequency
    Plus compact snapshot tables for both wiring cases.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)

        # Left: 2x2 plots. Right: two tables (1-wire, 2-wire).
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 1.0], wspace=0.16)

        gs_left = gs[0].subgridspec(nrows=3, ncols=2, hspace=0.30, wspace=0.22)
        gs_right = gs[1].subgridspec(nrows=2, ncols=1, hspace=0.32)

        self.ax_i = self.fig.add_subplot(gs_left[0, 0])
        self.ax_v = self.fig.add_subplot(gs_left[0, 1])
        self.ax_emf = self.fig.add_subplot(gs_left[1, 0])
        self.ax_f = self.fig.add_subplot(gs_left[1, 1])

        self.ax_tm = self.fig.add_subplot(gs_left[2, 0])
        self.ax_out = self.fig.add_subplot(gs_left[2, 1])
        self.ax_out_p = self.ax_out.twinx()

        self.ax_tbl_1w = self.fig.add_subplot(gs_right[0, 0])
        self.ax_tbl_2w = self.fig.add_subplot(gs_right[1, 0])

        # Manual margins (avoid tight_layout warnings with nested gridspec)
        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    @staticmethod
    def _make_table(ax, title: str, columns: List[str], rows: List[List[str]],
                    col_widths: Optional[List[float]] = None,
                    fontsize: int = 8):
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=11, pad=6, fontweight="600")

        header_bg = "#111827"
        header_fg = "white"
        zebra = ["#f8fafc", "#eef2f7"]

        tbl = ax.table(cellText=rows, colLabels=columns, loc="center",
                       cellLoc="center", colLoc="center", colWidths=col_widths)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)

        # Style header and zebra rows
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor(header_bg)
                cell.get_text().set_color(header_fg)
                cell.get_text().set_fontweight("600")
            else:
                cell.set_facecolor(zebra[(r - 1) % 2])

        # Scale row height a bit for readability
        tbl.scale(1.0, 1.22)

    @staticmethod
    def _interp(x: np.ndarray, y: np.ndarray, xq: float) -> float:
        if len(x) == 0:
            return float("nan")
        if xq <= float(x[0]):
            return float(y[0])
        if xq >= float(x[-1]):
            return float(y[-1])
        return float(np.interp(float(xq), x.astype(float), y.astype(float)))

    def render(self,
               m1: Dict[str, np.ndarray],
               m2: Dict[str, np.ndarray],
               res_1w: Optional[SolveResult],
               res_2w: Optional[SolveResult],
               params: SystemParams):

        # Clear plots
        for ax in (self.ax_i, self.ax_v, self.ax_emf, self.ax_f, self.ax_tm, self.ax_out, self.ax_out_p, self.ax_tbl_1w,
                   self.ax_tbl_2w):
            ax.cla()

        # Unpack
        x1 = m1.get("out_rpm", np.array([]))
        x2 = m2.get("out_rpm", np.array([]))

        # --- Currents ---
        self.ax_i.plot(x1, m1.get("iq_max", np.zeros_like(x1)), label="Iq max (1w)")
        self.ax_i.plot(x1, m1.get("i_mag", np.zeros_like(x1)), label="|I| (1w)")
        self.ax_i.plot(x2, m2.get("iq_max", np.zeros_like(x2)), label="Iq max (2w)")
        self.ax_i.plot(x2, m2.get("i_mag", np.zeros_like(x2)), label="|I| (2w)")

        # Current limit lines
        if "i_limit" in m1:
            self.ax_i.plot(x1, m1["i_limit"], linestyle="--", label="I limit (1w)")
        if "i_limit" in m2:
            self.ax_i.plot(x2, m2["i_limit"], linestyle="--", label="I limit (2w)")

        # Mark UI target points (requested operating point, not max-torque envelope)
        if res_1w is not None:
            self.ax_i.scatter([abs(float(res_1w.out_rpm_cmd))], [float(res_1w.i_mag_used_rms)], marker="o", s=35,
                              label="UI |I| (1w)")
        if res_2w is not None:
            self.ax_i.scatter([abs(float(res_2w.out_rpm_cmd))], [float(res_2w.i_mag_used_rms)], marker="o", s=35,
                              label="UI |I| (2w)")

        self.ax_i.set_title("Motor currents vs output RPM")
        self.ax_i.set_xlabel("Output RPM (abs)")
        self.ax_i.set_ylabel("Arms")
        self.ax_i.grid(True, alpha=0.25)
        self.ax_i.legend(loc="best", fontsize=8)

        # --- Motor terminal VLL,rms required (envelope max-torque point) ---
        self.ax_v.plot(x1, m1.get("vll_motor", np.zeros_like(x1)), label="VLL req (1w)")
        self.ax_v.plot(x2, m2.get("vll_motor", np.zeros_like(x2)), label="VLL req (2w)")

        # Downhole VLL limit (if enabled)
        try:
            if getattr(params.limits, "enforce_downhole_vll", False):
                vll_lim = float(getattr(params.limits, "downhole_vll_rms_limit", 0.0))
                if vll_lim > 0:
                    self.ax_v.axhline(vll_lim, linestyle="--", label="Downhole VLL limit")
        except Exception:
            pass

        if res_1w is not None:
            self.ax_v.scatter([abs(float(res_1w.out_rpm_cmd))], [math.sqrt(3.0) * float(res_1w.v_motor_req)],
                              marker="o", s=35, label="UI VLL (1w)")
        if res_2w is not None:
            self.ax_v.scatter([abs(float(res_2w.out_rpm_cmd))], [math.sqrt(3.0) * float(res_2w.v_motor_req)],
                              marker="o", s=35, label="UI VLL (2w)")

        self.ax_v.set_title("Motor terminal voltage vs output RPM")
        self.ax_v.set_xlabel("Output RPM (abs)")
        self.ax_v.set_ylabel("VLL,rms (V)")
        self.ax_v.grid(True, alpha=0.25)
        self.ax_v.legend(loc="best", fontsize=8)

        # --- Back-EMF (no-load) ---
        self.ax_emf.plot(x1, m1.get("emf_ll", np.zeros_like(x1)), label="Back-EMF VLL (1w)")
        self.ax_emf.plot(x2, m2.get("emf_ll", np.zeros_like(x2)), label="Back-EMF VLL (2w)")

        if res_1w is not None:
            ke = float(params.motor.ke_vll_rms_per_krpm)
            emf_ui = ke * (abs(float(res_1w.motor_rpm)) / 1000.0)
            self.ax_emf.scatter([abs(float(res_1w.out_rpm_cmd))], [emf_ui], marker="o", s=35, label="UI EMF (1w)")
        if res_2w is not None:
            ke = float(params.motor.ke_vll_rms_per_krpm)
            emf_ui = ke * (abs(float(res_2w.motor_rpm)) / 1000.0)
            self.ax_emf.scatter([abs(float(res_2w.out_rpm_cmd))], [emf_ui], marker="o", s=35, label="UI EMF (2w)")

        self.ax_emf.set_title("Back-EMF vs output RPM (derived from Ke)")
        self.ax_emf.set_xlabel("Output RPM (abs)")
        self.ax_emf.set_ylabel("VLL,rms (V)")
        self.ax_emf.grid(True, alpha=0.25)
        self.ax_emf.legend(loc="best", fontsize=8)

        # --- Electrical frequency ---
        self.ax_f.plot(x1, m1.get("f_e", np.zeros_like(x1)), label="f_e (1w)")
        self.ax_f.plot(x2, m2.get("f_e", np.zeros_like(x2)), label="f_e (2w)")
        if res_1w is not None:
            self.ax_f.scatter([abs(float(res_1w.out_rpm_cmd))], [float(res_1w.f_e_hz)], marker="o", s=35,
                              label="UI f_e (1w)")
        if res_2w is not None:
            self.ax_f.scatter([abs(float(res_2w.out_rpm_cmd))], [float(res_2w.f_e_hz)], marker="o", s=35,
                              label="UI f_e (2w)")

        self.ax_f.set_title("Electrical frequency vs output RPM")
        self.ax_f.set_xlabel("Output RPM (abs)")
        self.ax_f.set_ylabel("Hz")
        self.ax_f.grid(True, alpha=0.25)
        self.ax_f.legend(loc="best", fontsize=8)

        # --- Motor torque vs output RPM (envelope max-torque point) ---
        self.ax_tm.plot(x1, m1.get("t_motor_em_nm", np.zeros_like(x1)), label="T_em (1w)")
        self.ax_tm.plot(x1, m1.get("t_motor_use_nm", np.zeros_like(x1)), label="T_useful (1w)")
        self.ax_tm.plot(x2, m2.get("t_motor_em_nm", np.zeros_like(x2)), linestyle="--", label="T_em (2w)")
        self.ax_tm.plot(x2, m2.get("t_motor_use_nm", np.zeros_like(x2)), linestyle="--", label="T_useful (2w)")

        if res_1w is not None:
            self.ax_tm.scatter([abs(float(res_1w.out_rpm_cmd))], [float(res_1w.motor_torque_total_nm)], marker="o",
                               s=35, label="UI T_motor (1w)")
        if res_2w is not None:
            self.ax_tm.scatter([abs(float(res_2w.out_rpm_cmd))], [float(res_2w.motor_torque_total_nm)], marker="o",
                               s=35, label="UI T_motor (2w)")

        self.ax_tm.set_title("Motor torque vs output RPM")
        self.ax_tm.set_xlabel("Output RPM (abs)")
        self.ax_tm.set_ylabel("N·m")
        self.ax_tm.grid(True, alpha=0.25)
        self.ax_tm.legend(loc="best", fontsize=8)

        # --- Output torque and power vs output RPM (capability at max-torque point) ---
        self.ax_out.plot(x1, m1.get("t_out_ftlbf", np.zeros_like(x1)), label="Tout cap (1w)")
        self.ax_out.plot(x2, m2.get("t_out_ftlbf", np.zeros_like(x2)), linestyle="--", label="Tout cap (2w)")

        # UI required output torque marker (drive torque, magnitude)
        if res_1w is not None:
            self.ax_out.scatter([abs(float(res_1w.out_rpm_cmd))], [abs(float(res_1w.out_drive_torque_req_ftlbf))],
                                marker="o", s=35, label="UI Tout req (1w)")
        if res_2w is not None:
            self.ax_out.scatter([abs(float(res_2w.out_rpm_cmd))], [abs(float(res_2w.out_drive_torque_req_ftlbf))],
                                marker="o", s=35, label="UI Tout req (2w)")

        # Power on the twin axis
        self.ax_out_p.plot(x1, m1.get("p_out_w", np.zeros_like(x1)) / 1000.0, linestyle=":", label="Pout cap (1w)")
        self.ax_out_p.plot(x2, m2.get("p_out_w", np.zeros_like(x2)) / 1000.0, linestyle=":", label="Pout cap (2w)")

        if res_1w is not None:
            p_kw = (ft_lbf_to_nm(abs(float(res_1w.out_drive_torque_req_ftlbf))) * rpm_to_rad_s(
                abs(float(res_1w.out_rpm_cmd)))) / 1000.0
            self.ax_out_p.scatter([abs(float(res_1w.out_rpm_cmd))], [p_kw], marker="o", s=35, label="UI Pout req (1w)")
        if res_2w is not None:
            p_kw = (ft_lbf_to_nm(abs(float(res_2w.out_drive_torque_req_ftlbf))) * rpm_to_rad_s(
                abs(float(res_2w.out_rpm_cmd)))) / 1000.0
            self.ax_out_p.scatter([abs(float(res_2w.out_rpm_cmd))], [p_kw], marker="o", s=35, label="UI Pout req (2w)")

        self.ax_out.set_title("Output torque & power vs output RPM (capability)")
        self.ax_out.set_xlabel("Output RPM (abs)")
        self.ax_out.set_ylabel("ft-lbf")
        self.ax_out.grid(True, alpha=0.25)

        self.ax_out_p.set_ylabel("kW")

        # Two legends: left for torque, right for power
        self.ax_out.legend(loc="upper left", fontsize=8)
        self.ax_out_p.legend(loc="upper right", fontsize=8)

        # --- Snapshot tables ---
        def build_table(metrics: Dict[str, np.ndarray], res: Optional[SolveResult], title: str):
            out_rpm = metrics.get("out_rpm", np.array([]))
            if len(out_rpm) == 0:
                return ["(no data)"], []

            max_rpm = float(out_rpm[-1])
            pts = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
            pts = [p for p in pts if p <= max_rpm + 1e-9]

            if res is not None:
                pts.append(abs(float(res.out_rpm_cmd)))
            # unique + sorted
            pts = sorted({round(float(p), 6) for p in pts})

            cols = ["Out RPM", "Motor RPM", "f_e (Hz)", "Iq (A)", "|I| (A)", "VLL (V)", "Tmot_use (N·m)",
                    "Tout (ft-lbf)", "Pout (kW)", "Pcu (W)"]
            rows = []
            for p in pts:
                mrpm = self._interp(out_rpm, metrics.get("motor_rpm", out_rpm * 0.0), p)
                fe = self._interp(out_rpm, metrics.get("f_e", out_rpm * 0.0), p)
                iq = self._interp(out_rpm, metrics.get("iq_max", out_rpm * 0.0), p)
                im = self._interp(out_rpm, metrics.get("i_mag", out_rpm * 0.0), p)
                vll = self._interp(out_rpm, metrics.get("vll_motor", out_rpm * 0.0), p)
                emf = self._interp(out_rpm, metrics.get("emf_ll", out_rpm * 0.0), p)
                pcu = self._interp(out_rpm, metrics.get("p_cu", out_rpm * 0.0), p)
                tmu = self._interp(out_rpm, metrics.get("t_motor_use_nm", out_rpm * 0.0), p)
                tout = self._interp(out_rpm, metrics.get("t_out_ftlbf", out_rpm * 0.0), p)
                pout_kw = self._interp(out_rpm, metrics.get("p_out_w", out_rpm * 0.0), p) / 1000.0

                rows.append([
                    f"{p:.2f}",
                    f"{mrpm:.0f}",
                    f"{fe:.1f}",
                    f"{iq:.2f}",
                    f"{im:.2f}",
                    f"{vll:.1f}",
                    f"{tmu:.2f}",
                    f"{tout:.0f}",
                    f"{pout_kw:.2f}",
                    f"{pcu:.0f}",
                ])

            return cols, rows

        cols1, rows1 = build_table(m1, res_1w, "1-wire")
        cols2, rows2 = build_table(m2, res_2w, "2-wire")

        self._make_table(self.ax_tbl_1w, "Motor snapshot (1-wire / phase)", cols1, rows1,
                         col_widths=[0.09, 0.10, 0.09, 0.08, 0.08, 0.09, 0.11, 0.10, 0.09, 0.08],
                         fontsize=7)
        self._make_table(self.ax_tbl_2w, "Motor snapshot (2-wires / phase)", cols2, rows2,
                         col_widths=[0.09, 0.10, 0.09, 0.08, 0.08, 0.09, 0.11, 0.10, 0.09, 0.08],
                         fontsize=7)

        self.canvas.draw_idle()


class MoogCurvesPane(QWidget):
    """Compare the current τ_extra implementation against MOOG operating-curve tables.

    This pane is intentionally narrow-scope: it plots the vendor (MOOG) curves and overlays what the
    current τ_extra(ω,T) block would predict when back-calculating torque from the MOOG current column.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl = QLabel(
            "Select an Extra Torque preset to view its MOOG curve (Milling is populated; Spear/Annular are placeholders).")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#444;")
        layout.addWidget(self.lbl)

        self.fig = Figure(figsize=(15.0, 8.5), dpi=100)
        gs = self.fig.add_gridspec(nrows=2, ncols=2, wspace=0.28, hspace=0.34)

        self.ax_torque = self.fig.add_subplot(gs[0, 0])
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.ax_visc = self.fig.add_subplot(gs[1, 0])
        self.ax_i = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(120)
        self.txt.setStyleSheet("font-family:Consolas,Menlo,monospace; font-size:11px;")
        layout.addWidget(self.txt)

    def _clear(self, msg: str = ""):
        for ax in (self.ax_torque, self.ax_loss, self.ax_visc, self.ax_i):
            ax.clear()
        self.txt.setPlainText(msg or "")
        self.canvas.draw_idle()

    def render(self, model: "SystemModel", params: "SystemParams", preset_name: str):
        ds = _get_moog_dataset(preset_name)
        if not ds:
            self.lbl.setText(
                "No MOOG dataset selected. Choose 'Milling Moog Curve' (or a placeholder preset) in Extra Torque.")
            self._clear(
                "MOOG dataset not available for this preset.\n\nTip: Select 'Milling Moog Curve' in Extra Torque.")
            return

        self.lbl.setText(f"MOOG dataset: {ds['name']}  |  Preset selected: {preset_name}")

        RPM_TO_RAD_S = 2.0 * math.pi / 60.0
        NM_TO_LBIN = 1.0 / LBIN_TO_NM

        kt_moog_lbin_arms = float(ds["kt_lbin_arms"])
        kt_model_nm_arms = float(model.kt_effective_nm_per_arms())
        kt_model_lbin_arms = kt_model_nm_arms * NM_TO_LBIN

        # Build arrays (continuous + peak)
        cont = np.asarray(ds["cont"], dtype=float)
        peak = np.asarray(ds["peak"], dtype=float)

        def unpack(a):
            rpm = a[:, 0]
            visc_lbf = a[:, 1]
            tq_lbf = a[:, 2]
            i_arms = a[:, 3]
            return rpm, visc_lbf, tq_lbf, i_arms

        rpm_c, visc_c, tq_c, i_c = unpack(cont)
        rpm_p, visc_p, tq_p, i_p = unpack(peak)

        # Model τ_extra evaluated at the MOOG speeds (using current UI parameters)
        def eval_tau(rpm_arr):
            tau_extra_nm = np.zeros_like(rpm_arr, dtype=float)
            tau_core_nm = np.zeros_like(rpm_arr, dtype=float)
            tau_visc_nm = np.zeros_like(rpm_arr, dtype=float)
            for k, rpm in enumerate(rpm_arr):
                omega = float(rpm) * RPM_TO_RAD_S
                te, tc, tv = model.tau_extra_nm(omega)
                tau_extra_nm[k] = float(te)
                tau_core_nm[k] = float(tc)
                tau_visc_nm[k] = float(tv)
            return tau_extra_nm, tau_core_nm, tau_visc_nm

        tau_extra_c_nm, tau_core_c_nm, tau_visc_c_nm = eval_tau(rpm_c)
        tau_extra_p_nm, tau_core_p_nm, tau_visc_p_nm = eval_tau(rpm_p)

        tau_extra_c_lbf = tau_extra_c_nm * NM_TO_LBIN
        tau_extra_p_lbf = tau_extra_p_nm * NM_TO_LBIN
        tau_visc_c_lbf = tau_visc_c_nm * NM_TO_LBIN
        tau_visc_p_lbf = tau_visc_p_nm * NM_TO_LBIN

        # MOOG-inferred total loss torque: τ_loss = Kt*I - τ_out
        loss_c_lbf = kt_moog_lbin_arms * i_c - tq_c
        loss_p_lbf = kt_moog_lbin_arms * i_p - tq_p

        # Reconstructed torque-out using current τ_extra and MOOG Kt: τ̂_out = Kt*I - τ_extra
        tq_hat_c_lbf = kt_moog_lbin_arms * i_c - tau_extra_c_lbf
        tq_hat_p_lbf = kt_moog_lbin_arms * i_p - tau_extra_p_lbf

        # ---------- Plot: Torque-out ----------
        ax = self.ax_torque
        ax.clear()
        ax.set_title("Torque-out vs Speed (MOOG vs τ_extra back-calc)")
        ax.set_xlabel("Speed (rpm)")
        ax.set_ylabel("Torque-out (lbf-in)")
        ax.plot(rpm_c, tq_c, marker="o", linestyle="None", label="MOOG Continuous")
        ax.plot(rpm_p, tq_p, marker="o", linestyle="None", label="MOOG Peak")
        ax.plot(rpm_c, tq_hat_c_lbf, linestyle="--", label="Back-calc using τ_extra (cont)")
        ax.plot(rpm_p, tq_hat_p_lbf, linestyle="--", label="Back-calc using τ_extra (peak)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # ---------- Plot: Loss torque ----------
        ax = self.ax_loss
        ax.clear()
        ax.set_title("Loss torque (MOOG inferred vs τ_extra model)")
        ax.set_xlabel("Speed (rpm)")
        ax.set_ylabel("Loss torque (lbf-in)")
        ax.plot(rpm_c, loss_c_lbf, marker="o", linestyle="None", label="MOOG inferred (cont)")
        ax.plot(rpm_p, loss_p_lbf, marker="o", linestyle="None", label="MOOG inferred (peak)")
        ax.plot(rpm_c, tau_extra_c_lbf, linestyle="-", label="τ_extra model (cont)")
        ax.plot(rpm_p, tau_extra_p_lbf, linestyle="-", label="τ_extra model (peak)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # ---------- Plot: Viscous drag ----------
        ax = self.ax_visc
        ax.clear()
        ax.set_title("Viscous drag (MOOG column vs τ_visc model)")
        ax.set_xlabel("Speed (rpm)")
        ax.set_ylabel("Viscous drag (lbf-in)")
        ax.plot(rpm_c, visc_c, marker="o", linestyle="None", label="MOOG viscous (cont)")
        ax.plot(rpm_p, visc_p, marker="o", linestyle="None", label="MOOG viscous (peak)")
        ax.plot(rpm_c, tau_visc_c_lbf, linestyle="-", label="τ_visc model (cont)")
        ax.plot(rpm_p, tau_visc_p_lbf, linestyle="-", label="τ_visc model (peak)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # ---------- Plot: Current ----------
        ax = self.ax_i
        ax.clear()
        ax.set_title("Phase current (MOOG)")
        ax.set_xlabel("Speed (rpm)")
        ax.set_ylabel("Current (Arms)")
        ax.plot(rpm_c, i_c, marker="o", linestyle="None", label="MOOG Continuous")
        ax.plot(rpm_p, i_p, marker="o", linestyle="None", label="MOOG Peak")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # ---------- Summary ----------
        def rmse(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if len(a) == 0:
                return float("nan")
            return float(np.sqrt(np.mean((a - b) ** 2)))

        s = []
        s.append("Back-calc uses MOOG Kt (KtTR) and your current τ_extra model.")
        s.append(f"  Kt_moog = {kt_moog_lbin_arms:.4g} lbf-in/Arms")
        s.append(
            f"  Kt_model(current) = {kt_model_lbin_arms:.4g} lbf-in/Arms  (Kt(T)={'ON' if params.extra.kt_temp_enabled else 'OFF'}, T={params.extra.temp_C:.0f}°C)")
        s.append("")
        s.append("Continuous set errors (lbf-in):")
        s.append(f"  RMSE[loss]  = {rmse(loss_c_lbf, tau_extra_c_lbf):.3g}")
        s.append(f"  RMSE[visc]  = {rmse(visc_c, tau_visc_c_lbf):.3g}")
        s.append(f"  RMSE[tq_out]= {rmse(tq_c, tq_hat_c_lbf):.3g}")
        s.append("")
        s.append("Peak set errors (lbf-in):")
        s.append(f"  RMSE[loss]  = {rmse(loss_p_lbf, tau_extra_p_lbf):.3g}")
        s.append(f"  RMSE[visc]  = {rmse(visc_p, tau_visc_p_lbf):.3g}")
        s.append(f"  RMSE[tq_out]= {rmse(tq_p, tq_hat_p_lbf):.3g}")
        self.txt.setPlainText("\n".join(s))

        self.canvas.draw_idle()


class StonehousePresetPane(QWidget):
    """Reference pane: Stonehouse-based load presets.

    This pane is meant as a *read-only reference*:
      - It DOES NOT change the user's UI entries.
      - It re-solves the model using preset load scenarios (TOB, BHA friction, GB parasitic)
        and shows the resulting feasibility/markers against the torque envelope.

    Presets implemented (ft-lbf):
      - Low (mild lateral)   : TOB=175,  Tc,BHA=50,  Tc,GB=55
      - Nominal (continuous) : TOB=216,  Tc,BHA=158, Tc,GB=83    (Stonehouse Scenario A)
      - High (peak corr.)    : TOB=216,  Tc,BHA=315, Tc,GB=250   (Stonehouse Scenario B)
      - Stall (non-rotating) : 945 ft-lbf (Stonehouse Scenario C, handled via stuck_mode)
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)

        # Similar readability layout to BandPane: two large 4Q plots + 3 info panes.
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[2.35, 1.0],
            wspace=0.18,
        )
        gs_left = gs[0].subgridspec(nrows=2, ncols=1, hspace=0.18)
        gs_right = gs[1].subgridspec(nrows=3, ncols=1, hspace=0.30)

        self.ax1 = self.fig.add_subplot(gs_left[0, 0])
        self.ax2 = self.fig.add_subplot(gs_left[1, 0])

        self.ax3 = self.fig.add_subplot(gs_right[0, 0])  # preset table
        self.ax4 = self.fig.add_subplot(gs_right[1, 0])  # feasibility summary
        self.ax5 = self.fig.add_subplot(gs_right[2, 0])  # stacked bar / visualization

        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07)

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._interactor: Optional[QuadrantInteractor] = None

    def enable_interactivity(self):
        """Enable hover tooltip + crosshair on the two 4Q plots (ax1/ax2).

        Call this after every re-plot because the axes are cleared and re-drawn.
        """
        try:
            if self._interactor is None:
                self._interactor = QuadrantInteractor(self.canvas, [self.ax1, self.ax2], pixel_tol=12)
            else:
                self._interactor.refresh()
        except Exception:
            pass


class ArchitecturePane(QWidget):
    """Live block-diagram view of the current simulation architecture.

    Shows which blocks are included/excluded and highlights key active features.
    This is meant as a one-stop "what is the simulator actually modeling right now" view.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.fig = Figure(figsize=(15.2, 8.8), dpi=100)
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.4, 1.0], wspace=0.05)

        self.ax_diag = self.fig.add_subplot(gs[0, 0])
        self.ax_feat = self.fig.add_subplot(gs[0, 1])

        # Backwards-compatible aliases: some plotting code uses older names.
        # Keep these as direct references to the current axes.
        self.ax_main = self.ax_diag
        self.ax_flags = self.ax_feat

        for ax in (self.ax_diag, self.ax_feat):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.06)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)


class DashboardPane(QWidget):
    """Highly-detailed, blockwise dashboard for the current target point.

    Goal: present the same diagnostics as the left status box, but organized by
    the architecture blocks (surface → filter → cable → contact → motor → gearbox → loads).

    This pane is display-only; it does not change any parameters.
    """

    def __init__(self):
        super().__init__()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.hdr = QLabel('—')
        self.hdr.setWordWrap(True)
        self.hdr.setStyleSheet('font-size: 13px; padding: 6px;')
        outer.addWidget(self.hdr)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameStyle(QFrame.NoFrame)

        inner = QWidget()
        self.scroll.setWidget(inner)
        self.vbox = QVBoxLayout(inner)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(10)

        outer.addWidget(self.scroll)

        self._val = {}  # key -> QLabel (value cell)

        def add_block(title: str, rows):
            gb = QGroupBox(title)
            gl = QGridLayout(gb)
            gl.setContentsMargins(10, 10, 10, 10)
            gl.setHorizontalSpacing(12)
            gl.setVerticalSpacing(6)

            for i, (k, label) in enumerate(rows):
                l = QLabel(label)
                l.setStyleSheet('color:#111827;')
                v = QLabel('—')
                v.setTextInteractionFlags(Qt.TextSelectableByMouse)
                v.setWordWrap(True)
                v.setFont(QFont('Courier New', 9))
                gl.addWidget(l, i, 0)
                gl.addWidget(v, i, 1)
                self._val[k] = v

            gl.setColumnStretch(0, 0)
            gl.setColumnStretch(1, 1)
            self.vbox.addWidget(gb)

        add_block('Command + Mode', [
            ('pass_fail', 'Status'),
            ('cmd', 'Command'),
            ('quad', 'Quadrant / Pout'),
            ('braking', 'Braking / Regen'),
            ('backdrive', 'Gearbox backdrivable'),
        ])

        add_block('Mechanical stack (output shaft)', [
            ('tout_req', 'Required @ output'),
            ('tout_terms', 'TOB / BHA / parasitic'),
            ('gear', 'Gear ratio / efficiency'),
        ])

        add_block('Motor operating point', [
            ('motor_rpm', 'Motor RPM / f_e'),
            ('tmot', 'Motor torque (pre-extra)'),
            ('textra', 'τ_core / τ_visc / τ_extra'),
            ('kt', 'Kt_eff(T)'),
        ])

        add_block('Currents', [
            ('iq', 'Iq (phase RMS)'),
            ('imag', '|I| used / limit'),
            ('fw', 'Field-weakening'),
        ])

        add_block('Voltages (phase RMS)', [
            ('vbudget', 'Vcmd / Vreq(inv) / Vnode / Vmotor / Vdrop(cable)'),
            ('vlimits', 'Limits (surface / downhole phase / downhole VLL)'),
        ])

        add_block('Losses + filter', [
            ('cable_loss', 'Cable copper loss'),
            ('filter', 'Sine-filter loading'),
        ])

        add_block('Headroom + limiter', [
            ('i_margin', 'Current headroom'),
            ('v_margin', 'Voltage headroom'),
            ('limiter', 'Limiting constraint'),
        ])

        # Raw diagnostics at the end (copy/paste friendly)
        gb_raw = QGroupBox('Raw diagnostics')
        raw_l = QVBoxLayout(gb_raw)
        raw_l.setContentsMargins(10, 10, 10, 10)

        btn_row = QHBoxLayout()
        self.btn_copy = QPushButton('Copy')
        self.btn_copy.clicked.connect(self._copy_raw)
        btn_row.addWidget(self.btn_copy)
        btn_row.addStretch(1)
        raw_l.addLayout(btn_row)

        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setFont(QFont('Courier New', 9))
        self.raw.setLineWrapMode(QPlainTextEdit.NoWrap)
        raw_l.addWidget(self.raw)

        self.vbox.addWidget(gb_raw)
        self.vbox.addStretch(1)

    def _copy_raw(self):
        try:
            QApplication.clipboard().setText(self.raw.toPlainText())
        except Exception:
            pass

    def _set(self, k: str, s: str):
        w = self._val.get(k)
        if w is not None:
            w.setText(s)

    def update_from_status(self, full_text: str, p, res_sel, res_1, res_2, dominant=None):
        """Update dashboard fields from the latest solve results."""
        import math

        try:
            sel_wires = getattr(p.cable, 'wires_per_phase', 1)
        except Exception:
            sel_wires = 1
        res = res_1 if sel_wires == 1 else res_2

        # Header
        pass_fail = '✅ PASS' if getattr(res, 'feasible', False) else '❌ FAIL'
        self.hdr.setText(f"{pass_fail} — Detailed dashboard (selected: {sel_wires}-wire/phase)")

        # Derived
        RPM_TO_RADPS = 2.0 * math.pi / 60.0
        FTLBF_TO_NM = 1.3558179483314004

        sgn = 1.0 if str(getattr(p, 'out_dir', 'CW')).upper().startswith('CW') else -1.0
        out_rpm = float(getattr(res, 'out_rpm_cmd', 0.0))
        omega_out = out_rpm * RPM_TO_RADPS
        tau_out_req_ftlbf_signed = sgn * (float(getattr(res, 'out_drive_torque_req_ftlbf', 0.0)) - float(
            getattr(res, 'out_brake_torque_req_ftlbf', 0.0)))
        p_out_w = (tau_out_req_ftlbf_signed * FTLBF_TO_NM) * omega_out

        quad = 'STATIC (ω≈0)' if abs(out_rpm) < 1e-9 else ('MOTORING (Pout>0)' if p_out_w > 0 else 'BRAKING (Pout<0)')

        # Command
        self._set('pass_fail',
                  'PASS — target point feasible' if getattr(res, 'feasible', False) else 'FAIL — constraint violation')
        self._set('cmd', f"{getattr(p, 'out_dir', '—')} {out_rpm:.3f} rpm  (CW=+, CCW=-)")
        self._set('quad', f"{quad}  |  P_out≈{p_out_w / 1000.0:.3f} kW")

        braking_path = bool(getattr(p, 'braking_path_available', True))
        regen_limit_on = bool(getattr(p, 'regen_cable_limit_enabled', True))
        clamp_frac = float(getattr(p, 'regen_surface_clamp_frac', 1.0))
        backdrivable = bool(getattr(getattr(p, 'gearbox', p), 'backdrivable', True))

        self._set('braking',
                  f"braking_path={'YES' if braking_path else 'NO'} | cable_limit={'ON' if regen_limit_on else 'OFF'} (clamp={clamp_frac:.2f})")
        self._set('backdrive', 'YES' if backdrivable else 'NO')

        # Mechanical
        self._set('tout_req',
                  f"drive={float(getattr(res, 'out_drive_torque_req_ftlbf', 0.0)):.0f} ft-lbf, brake={float(getattr(res, 'out_brake_torque_req_ftlbf', 0.0)):.0f} ft-lbf")
        self._set('tout_terms',
                  f"τ_TOB={float(getattr(res, 'tob_reaction_ftlbf', 0.0)):.0f}, τ_BHA_fric={float(getattr(res, 'bha_friction_ftlbf', 0.0)):.0f}, τ_parasitic={float(getattr(res, 'parasitic_ftlbf', 0.0)):.0f} ft-lbf")
        self._set('gear',
                  f"G={float(getattr(res, 'gear_ratio', 0.0)):.1f}, η={float(getattr(res, 'gear_eff', 0.0)):.3f}")

        # Motor
        motor_rpm = float(getattr(res, 'motor_rpm', 0.0))
        pole_pairs = getattr(getattr(p, 'motor', p), 'pole_pairs', None)
        f_e = None
        try:
            if pole_pairs is not None:
                f_e = float(pole_pairs) * motor_rpm / 60.0
        except Exception:
            f_e = None
        fe_s = f"{f_e:.2f} Hz" if f_e is not None else '—'
        self._set('motor_rpm', f"{motor_rpm:.0f} rpm  |  f_e≈{fe_s}")
        self._set('tmot', f"{float(getattr(res, 'motor_torque_nm', 0.0)):.3f} N·m" + (
            "  [MAG SLIP RISK]" if bool(getattr(res, 'mag_slipping', False)) else ""))
        self._set('textra',
                  f"τ_core={float(getattr(res, 'tau_core_nm', 0.0)):.3f}, τ_visc={float(getattr(res, 'tau_visc_nm', 0.0)):.3f}, τ_extra={float(getattr(res, 'tau_extra_nm', 0.0)):.3f} N·m")
        self._set('kt',
                  f"{float(getattr(res, 'kt_eff_nm_per_arms', 0.0)):.6f} N·m/Arms  @T={float(getattr(res, 'temp_C', 0.0)):.1f} °C")

        # Currents
        self._set('iq',
                  f"Iq_base={float(getattr(res, 'iq_req_base_rms', 0.0)):.3f}, Iq_total={float(getattr(res, 'iq_req_rms', 0.0)):.3f}, Iq_max={float(getattr(res, 'iq_max_rms', 0.0)):.3f}")
        self._set('imag',
                  f"|I|_used={float(getattr(res, 'i_mag_used_rms', 0.0)):.3f} Arms, |I|_limit={float(getattr(res, 'i_limit_phase_mag', 0.0)):.3f} Arms  (basis: {getattr(getattr(p, 'cable', p), 'i_limit_basis', '—')})")
        self._set('fw',
                  f"{'ON' if bool(getattr(getattr(p, 'fw', p), 'enabled', False)) else 'OFF'}  (Id_used={float(getattr(res, 'id_used_rms', 0.0)):.3f} Arms)")

        # Voltages
        ll_surface_cmd = float(getattr(res, 'v_surface_cmd', 0.0)) * math.sqrt(3.0)
        ll_inverter_req = float(getattr(res, 'v_inverter_req', 0.0)) * math.sqrt(3.0)
        ll_motor_req = float(getattr(res, 'v_motor_req', 0.0)) * math.sqrt(3.0)

        self._set('vbudget',
                  f"Vcmd={float(getattr(res, 'v_surface_cmd', 0.0)):.1f} (≈{ll_surface_cmd:.1f} Vll), Vreq(inv)={float(getattr(res, 'v_inverter_req', 0.0)):.1f} (≈{ll_inverter_req:.1f} Vll), Vnode={float(getattr(res, 'v_node_req', 0.0)):.1f}, Vmotor={float(getattr(res, 'v_motor_req', 0.0)):.1f} (≈{ll_motor_req:.1f} Vll), Vdrop(cable)={float(getattr(res, 'v_cable_drop', 0.0)):.1f}")

        v_surface_lim = getattr(res, 'v_surface_limit', None)
        v_dh_ph_lim = getattr(res, 'v_downhole_phase_limit', None)
        v_dh_ll_lim = getattr(res, 'v_downhole_ll_limit', None)
        lim_parts = []
        if v_surface_lim is not None:
            lim_parts.append(f"surface={float(v_surface_lim):.1f} Vrms")
        if v_dh_ph_lim is not None:
            lim_parts.append(f"dh_phase={float(v_dh_ph_lim):.1f} Vrms")
        if v_dh_ll_lim is not None:
            lim_parts.append(f"dh_VLL(contact)={float(v_dh_ll_lim):.1f} Vrms")
        self._set('vlimits', ' | '.join(lim_parts) if lim_parts else '—')

        # Losses
        self._set('cable_loss', f"{float(getattr(res, 'p_cable_loss_w', 0.0)):.1f} W")
        if (float(getattr(res, 'i_filter_cap_rms', 0.0)) > 1e-9) or (float(getattr(res, 'v_filter_drop', 0.0)) > 1e-9):
            self._set('filter',
                      f"I_load={float(getattr(res, 'i_mag_used_rms', 0.0)):.3f} Arms, I_inv≈{float(getattr(res, 'i_inverter_rms', 0.0)):.3f} Arms, I_shunt≈{float(getattr(res, 'i_filter_cap_rms', 0.0)):.3f} Arms, Vdrop≈{float(getattr(res, 'v_filter_drop', 0.0)):.2f} Vrms")
        else:
            self._set('filter', '—')

        # Headroom
        I_used = getattr(res, 'i_mag_used_rms', None)
        I_lim = getattr(res, 'i_limit_phase_mag', None)
        I_margin = (float(I_lim) - float(I_used)) if (I_used is not None and I_lim is not None) else None

        Vsurf_cmd = getattr(res, 'v_surface_cmd', None)
        Vsurf_lim = getattr(res, 'v_surface_limit', None)
        Vsurf_margin_cmd = (float(Vsurf_lim) - float(Vsurf_cmd)) if (
                    Vsurf_cmd is not None and Vsurf_lim is not None) else None

        Vdh_ll_margin = None
        if v_dh_ll_lim is not None:
            Vdh_ll_margin = float(v_dh_ll_lim) - ll_motor_req

        self._set('i_margin', f"{I_margin:.2f} A" if I_margin is not None else '—')
        self._set('v_margin', f"surface(cmd)={Vsurf_margin_cmd:.2f} V, dh_VLL={Vdh_ll_margin:.2f} V" if (
                    Vsurf_margin_cmd is not None and Vdh_ll_margin is not None) else '—')

        if dominant is not None:
            lab, m, used, lim = dominant
            try:
                util = (100.0 * float(used) / float(lim)) if (lim is not None and float(lim) > 0) else None
            except Exception:
                util = None
            util_s = f"{util:.1f}%" if util is not None else '—'
            self._set('limiter', f"{lab} (util={util_s}, headroom={m:.2f})")
        else:
            self._set('limiter', '—')

        # Raw
        try:
            self.raw.setPlainText(full_text)
        except Exception:
            pass


class DirectionRiskPane(QWidget):
    """Direction Risk tab.

    Shows the CCW risk as two decision-grade visuals side-by-side:
      (1) Robustness histogram of the CCW margin m = (Tc - TOB), with an inset TOB-vs-Tc sign-flip map.
      (2) Scenario summary bars (UI + Low/Nominal/High) showing m directly, with the UI what-if band.

    This layout is intentionally minimal: it is meant to answer, in one view,
    whether CCW can be treated as a default continuous-rotation direction.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(17.5, 6.6), dpi=100)
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.45, 1.0],
            wspace=0.22,
        )

        self.ax_hist = self.fig.add_subplot(gs[0, 0])
        self.ax_bars = self.fig.add_subplot(gs[0, 1])

        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.10)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)


class AnimationPane(QWidget):
    """Animation tab: continuous RPM control, CW vs CCW.

    Purpose
    -------
    This pane is not a high-fidelity transient tool model. It is a decision-support
    visual that uses the *static* torque/voltage limits and the same load sign
    conventions as the solver to illustrate what a speed loop is up against in:

      - CW continuous rotation (TOB typically resists), versus
      - CCW continuous rotation (TOB can assist ⇒ braking/regeneration may be required).

    The animation integrates a simple output-shaft dynamic:
        J * dω/dt = τ_ccrs + τ_ext
    with τ_ccrs limited by the computed motoring envelope and (if enabled)
    the regen/braking cap.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._model: Optional[SystemModel] = None
        self._params: Optional[SystemParams] = None
        self._env_cache = None  # (env_rpm_1, env_tq_1, env_rpm_2, env_tq_2)

        self._sim = None  # dict with CW/CCW traces
        self._k = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # --- Controls row
        ctrl = QGroupBox("Continuous RPM control (illustrative transient)")
        gl = QGridLayout(ctrl)
        gl.setContentsMargins(10, 10, 10, 10)
        gl.setHorizontalSpacing(12)
        gl.setVerticalSpacing(6)

        self.lbl_ctx = QLabel("—")
        self.lbl_ctx.setWordWrap(True)
        self.lbl_ctx.setStyleSheet("color:#111827;")
        gl.addWidget(self.lbl_ctx, 0, 0, 1, 6)

        def _spin(minv, maxv, step, val, suf=""):
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            w.setValue(val)
            if suf:
                w.setSuffix(suf)
            return w

        self.sim_target_rpm = _spin(0.01, 5.0, 0.05, 1.0, " rpm")
        self.sim_duration = _spin(1.0, 60.0, 1.0, 10.0, " s")
        self.sim_dt = _spin(0.005, 0.200, 0.005, 0.020, " s")
        self.sim_tau = _spin(0.05, 5.0, 0.05, 0.35, " s")
        self.sim_J = _spin(0.01, 500.0, 0.50, 5.0, " kg·m²")

        gl.addWidget(QLabel("Target RPM"), 1, 0)
        gl.addWidget(self.sim_target_rpm, 1, 1)
        gl.addWidget(QLabel("Duration"), 1, 2)
        gl.addWidget(self.sim_duration, 1, 3)
        gl.addWidget(QLabel("dt"), 1, 4)
        gl.addWidget(self.sim_dt, 1, 5)

        gl.addWidget(QLabel("Speed loop τ"), 2, 0)
        gl.addWidget(self.sim_tau, 2, 1)
        gl.addWidget(QLabel("Output inertia J"), 2, 2)
        gl.addWidget(self.sim_J, 2, 3)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Compute")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_reset = QPushButton("Reset")
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        gl.addLayout(btn_row, 2, 4, 1, 2)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        gl.addWidget(self.slider, 3, 0, 1, 6)

        outer.addWidget(ctrl)

        # --- Plots
        self.fig = Figure(figsize=(17.5, 7.2), dpi=100)
        gs = self.fig.add_gridspec(2, 2, wspace=0.25, hspace=0.26)
        self.ax_rpm_cw = self.fig.add_subplot(gs[0, 0])
        self.ax_rpm_ccw = self.fig.add_subplot(gs[0, 1])
        self.ax_tau_cw = self.fig.add_subplot(gs[1, 0])
        self.ax_tau_ccw = self.fig.add_subplot(gs[1, 1])
        self.fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.10)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas)

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(40)  # ~25 fps
        self.timer.timeout.connect(self._tick)

        # Wire up controls
        self.btn_run.clicked.connect(self.compute)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self._on_slider)

        self._init_plot()

    def sync_from_model(self, model: SystemModel, params: SystemParams, env_cache=None):
        """Attach latest model/params (called by the main window)."""
        self._model = model
        self._params = params
        self._env_cache = env_cache

        # Only auto-sync the target RPM when the pane has no cached run.
        # (Avoid clobbering the user's animation-specific value.)
        try:
            if (self._sim is None) and (not self.timer.isActive()) and (not self.sim_target_rpm.hasFocus()):
                self.sim_target_rpm.blockSignals(True)
                self.sim_target_rpm.setValue(max(0.01, float(abs(params.target.out_rpm))))
                self.sim_target_rpm.blockSignals(False)
        except Exception:
            pass

        # Context label
        try:
            wires = int(getattr(params.cable, 'wires_per_phase', 1))
        except Exception:
            wires = 1
        bp = bool(getattr(params, 'braking_path_available', True))
        rlim = bool(getattr(params, 'regen_cable_limit_enabled', True))
        bd = bool(getattr(getattr(params, 'gearbox', params), 'backdrivable', True))
        tob = float(getattr(getattr(params, 'bha', params), 'drilling_tob_ftlbf', 0.0))
        self.lbl_ctx.setText(
            f"Selected: {wires}-wire/phase | braking_path={'YES' if bp else 'NO'} | regen_limit={'ON' if rlim else 'OFF'} | "
            f"gearbox_backdrivable={'YES' if bd else 'NO'} | TOB={tob:.0f} ft-lbf (TOB torque sign is CCW)"
        )

    # ---------------- plotting ----------------
    def _init_plot(self):
        for ax in (self.ax_rpm_cw, self.ax_rpm_ccw, self.ax_tau_cw, self.ax_tau_ccw):
            ax.cla()
            ax.grid(True, alpha=0.25)

        self.ax_rpm_cw.set_title("CW: speed tracking")
        self.ax_rpm_ccw.set_title("CCW: speed tracking")
        self.ax_tau_cw.set_title("CW: output torque (signed)")
        self.ax_tau_ccw.set_title("CCW: output torque (signed)")

        self.ax_rpm_cw.set_ylabel("RPM")
        self.ax_rpm_ccw.set_ylabel("RPM")
        self.ax_tau_cw.set_ylabel("ft-lbf")
        self.ax_tau_ccw.set_ylabel("ft-lbf")
        self.ax_tau_cw.set_xlabel("time (s)")
        self.ax_tau_ccw.set_xlabel("time (s)")

        # Placeholders
        (self.l_rpm_cw,) = self.ax_rpm_cw.plot([], [], lw=2, label="actual")
        (self.l_cmd_cw,) = self.ax_rpm_cw.plot([], [], lw=1.5, ls="--", label="cmd")
        (self.p_rpm_cw,) = self.ax_rpm_cw.plot([], [], marker="o", ms=6)

        (self.l_rpm_ccw,) = self.ax_rpm_ccw.plot([], [], lw=2, label="actual")
        (self.l_cmd_ccw,) = self.ax_rpm_ccw.plot([], [], lw=1.5, ls="--", label="cmd")
        (self.p_rpm_ccw,) = self.ax_rpm_ccw.plot([], [], marker="o", ms=6)

        (self.l_tau_cw,) = self.ax_tau_cw.plot([], [], lw=2, label="τ_ccrs")
        (self.l_ext_cw,) = self.ax_tau_cw.plot([], [], lw=1.2, ls=":", label="τ_ext")
        (self.l_cap_m_cw,) = self.ax_tau_cw.plot([], [], lw=1.2, ls="--", label="mot cap")
        (self.l_cap_b_cw,) = self.ax_tau_cw.plot([], [], lw=1.2, ls="--", label="brake cap")
        (self.p_tau_cw,) = self.ax_tau_cw.plot([], [], marker="o", ms=6)

        (self.l_tau_ccw,) = self.ax_tau_ccw.plot([], [], lw=2, label="τ_ccrs")
        (self.l_ext_ccw,) = self.ax_tau_ccw.plot([], [], lw=1.2, ls=":", label="τ_ext")
        (self.l_cap_m_ccw,) = self.ax_tau_ccw.plot([], [], lw=1.2, ls="--", label="mot cap")
        (self.l_cap_b_ccw,) = self.ax_tau_ccw.plot([], [], lw=1.2, ls="--", label="brake cap")
        (self.p_tau_ccw,) = self.ax_tau_ccw.plot([], [], marker="o", ms=6)

        for ax in (self.ax_rpm_cw, self.ax_rpm_ccw, self.ax_tau_cw, self.ax_tau_ccw):
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        self.txt_cw = self.ax_rpm_cw.text(0.02, 0.92, "—", transform=self.ax_rpm_cw.transAxes, fontsize=9,
                                          bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))
        self.txt_ccw = self.ax_rpm_ccw.text(0.02, 0.92, "—", transform=self.ax_rpm_ccw.transAxes, fontsize=9,
                                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

        self.canvas.draw_idle()

    # ---------------- simulation ----------------
    def compute(self):
        """Compute traces (no playback)."""
        self.pause()
        if self._params is None:
            return

        # Snapshot params for a consistent run
        p = copy.deepcopy(self._params)
        p.stuck_mode = False  # this pane is for continuous rotation

        # Use the selected wiring (1w/2w) for caps
        try:
            wires = int(getattr(p.cable, 'wires_per_phase', 1))
        except Exception:
            wires = 1
        cab = CableParams(**vars(p.cable))
        cab.wires_per_phase = int(max(1, min(2, wires)))

        model = SystemModel(p)

        tgt_rpm = float(self.sim_target_rpm.value())
        t_end = float(self.sim_duration.value())
        dt = float(self.sim_dt.value())
        tau = float(self.sim_tau.value())
        J = float(self.sim_J.value())

        # Motoring cap curve from the cached envelope (if provided)
        env_rpm = None
        env_tq = None
        if self._env_cache is not None:
            try:
                env_rpm_1, env_tq_1, env_rpm_2, env_tq_2 = self._env_cache
                if cab.wires_per_phase == 1:
                    env_rpm, env_tq = np.array(env_rpm_1, dtype=float), np.array(env_tq_1, dtype=float)
                else:
                    env_rpm, env_tq = np.array(env_rpm_2, dtype=float), np.array(env_tq_2, dtype=float)
            except Exception:
                env_rpm, env_tq = None, None

        if env_rpm is None or env_tq is None:
            out_rpm_max = max(1.0, float(tgt_rpm) * 1.5)
            env_rpm, env_tq, _, _ = model.compute_envelope(out_rpm_max=out_rpm_max, n=200, cable_override=cab)

        env_rpm = np.array(env_rpm, dtype=float)
        env_tq = np.array(env_tq, dtype=float)

        def mot_cap_ftlbf(rpm_mag: float) -> float:
            r = float(max(env_rpm[0], min(env_rpm[-1], abs(rpm_mag))))
            return float(np.interp(r, env_rpm, env_tq))

        def brake_cap_ftlbf(rpm_mag: float) -> float:
            try:
                cap = float(model.regen_cap_output_torque_ftlbf(float(abs(rpm_mag)), cable_override=cab))
            except Exception:
                cap = 0.0
            if (not bool(getattr(p, 'braking_path_available', True))):
                cap = 0.0
            if (not bool(getattr(getattr(p, 'gearbox', p), 'backdrivable', True))):
                cap = 0.0
            # Keep brake cap within the motoring envelope for plotting
            return float(min(mot_cap_ftlbf(rpm_mag), max(0.0, cap)))

        def simulate(sgn: float) -> Dict[str, np.ndarray]:
            n = int(max(2, math.ceil(t_end / max(1e-6, dt))) + 1)
            t = np.linspace(0.0, t_end, n)

            omega_cmd = sgn * rpm_to_rad_s(tgt_rpm)
            omega = 0.0

            rpm = np.zeros(n)
            tau_ccrs = np.zeros(n)
            tau_ext = np.zeros(n)
            cap_m = np.zeros(n)
            cap_b = np.zeros(n)
            sat = np.zeros(n, dtype=bool)
            assist = np.zeros(n, dtype=bool)

            for i in range(n):
                omega_eff = omega if abs(omega) > 1e-6 else omega_cmd
                rpm_mag = abs(rad_s_to_rpm(omega_eff))

                # External torque stack (sign conventions match solver)
                ttob, tbha = model._bha_external_torques_nm(omega_eff)
                tpar = model._rot_loss_torque_nm(omega_eff, p.parasitic)
                text = float(ttob + tbha + tpar)

                # Simple speed-loop: request a first-order response in ω
                omega_dot_des = (omega_cmd - omega) / max(1e-6, tau)
                tau_cmd = float(J * omega_dot_des - text)

                # Caps
                cm = ft_lbf_to_nm(mot_cap_ftlbf(rpm_mag))
                cb = ft_lbf_to_nm(brake_cap_ftlbf(rpm_mag))
                cap_m[i] = nm_to_ft_lbf(cm) * sgn
                cap_b[i] = -nm_to_ft_lbf(cb) * sgn

                # Motoring vs braking depends on power sign
                is_mot = (tau_cmd * omega_eff) >= 0.0
                cap = cm if is_mot else cb
                if cap < 1e-9:
                    tau_use = 0.0
                else:
                    tau_use = math.copysign(min(abs(tau_cmd), cap), tau_cmd)

                # Integrate
                omega_dot = (tau_use + text) / max(1e-9, J)
                omega = omega + omega_dot * dt

                rpm[i] = rad_s_to_rpm(omega)
                tau_ccrs[i] = nm_to_ft_lbf(tau_use)
                tau_ext[i] = nm_to_ft_lbf(text)
                sat[i] = bool(abs(tau_use) >= (cap - 1e-6)) if cap > 1e-9 else True
                assist[i] = bool((text * omega_eff) > 0.0)

            return {
                "t": t,
                "rpm": rpm,
                "tau": tau_ccrs,
                "text": tau_ext,
                "cap_m": cap_m,
                "cap_b": cap_b,
                "sat": sat,
                "assist": assist,
                "cmd_rpm": np.full_like(t, sgn * tgt_rpm, dtype=float),
            }

        self._sim = {
            "cw": simulate(+1.0),
            "ccw": simulate(-1.0),
        }
        self._k = 0

        # Update plots (static curves)
        self._render_traces()

        # Slider
        try:
            nmax = int(len(self._sim["cw"]["t"]) - 1)
            self.slider.blockSignals(True)
            self.slider.setEnabled(True)
            self.slider.setRange(0, nmax)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
        except Exception:
            pass

        self._update_markers(0)

    def _render_traces(self):
        if not self._sim:
            return
        cw = self._sim["cw"]
        ccw = self._sim["ccw"]

        self.l_rpm_cw.set_data(cw["t"], cw["rpm"])
        self.l_cmd_cw.set_data(cw["t"], cw["cmd_rpm"])
        self.l_rpm_ccw.set_data(ccw["t"], ccw["rpm"])
        self.l_cmd_ccw.set_data(ccw["t"], ccw["cmd_rpm"])

        self.l_tau_cw.set_data(cw["t"], cw["tau"])
        self.l_ext_cw.set_data(cw["t"], cw["text"])
        self.l_cap_m_cw.set_data(cw["t"], cw["cap_m"])
        self.l_cap_b_cw.set_data(cw["t"], cw["cap_b"])

        self.l_tau_ccw.set_data(ccw["t"], ccw["tau"])
        self.l_ext_ccw.set_data(ccw["t"], ccw["text"])
        self.l_cap_m_ccw.set_data(ccw["t"], ccw["cap_m"])
        self.l_cap_b_ccw.set_data(ccw["t"], ccw["cap_b"])

        # Axes limits
        for ax, tr in ((self.ax_rpm_cw, cw), (self.ax_rpm_ccw, ccw)):
            y = tr["rpm"]
            m = float(max(0.2, np.max(np.abs(y)) * 1.15))
            ax.set_xlim(0.0, float(tr["t"][-1]))
            ax.set_ylim(-m, +m)

        for ax, tr in ((self.ax_tau_cw, cw), (self.ax_tau_ccw, ccw)):
            y = np.concatenate([tr["tau"], tr["text"], tr["cap_m"], tr["cap_b"]])
            m = float(max(50.0, np.max(np.abs(y)) * 1.20))
            ax.set_xlim(0.0, float(tr["t"][-1]))
            ax.set_ylim(-m, +m)
            ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.35)

        self.canvas.draw_idle()

    def play(self):
        if not self._sim:
            self.compute()
        if not self._sim:
            return
        self.timer.start()

    def pause(self):
        try:
            self.timer.stop()
        except Exception:
            pass

    def reset(self):
        self.pause()
        self._k = 0
        if self._sim:
            try:
                self.slider.blockSignals(True)
                self.slider.setValue(0)
                self.slider.blockSignals(False)
            except Exception:
                pass
            self._update_markers(0)

    def _on_slider(self, v: int):
        self.pause()
        self._k = int(v)
        self._update_markers(self._k)

    def _tick(self):
        if not self._sim:
            return
        nmax = len(self._sim["cw"]["t"]) - 1
        self._k = int(min(nmax, self._k + 1))
        try:
            self.slider.blockSignals(True)
            self.slider.setValue(self._k)
            self.slider.blockSignals(False)
        except Exception:
            pass
        self._update_markers(self._k)
        if self._k >= nmax:
            self.timer.stop()

    def _update_markers(self, k: int):
        if not self._sim:
            return
        k = int(max(0, k))
        cw = self._sim["cw"]
        ccw = self._sim["ccw"]
        k = min(k, len(cw["t"]) - 1, len(ccw["t"]) - 1)

        t = float(cw["t"][k])

        self.p_rpm_cw.set_data([t], [float(cw["rpm"][k])])
        self.p_rpm_ccw.set_data([t], [float(ccw["rpm"][k])])

        self.p_tau_cw.set_data([t], [float(cw["tau"][k])])
        self.p_tau_ccw.set_data([t], [float(ccw["tau"][k])])

        sat_cw = "SAT" if bool(cw["sat"][k]) else "OK"
        sat_ccw = "SAT" if bool(ccw["sat"][k]) else "OK"
        as_cw = "ASSIST" if bool(cw["assist"][k]) else "RESIST"
        as_ccw = "ASSIST" if bool(ccw["assist"][k]) else "RESIST"

        pwr_cw = float(cw["tau"][k] * rpm_to_rad_s(float(cw["rpm"][k])))
        pwr_ccw = float(ccw["tau"][k] * rpm_to_rad_s(float(ccw["rpm"][k])))
        quad_cw = "MOTOR" if pwr_cw >= 0.0 else "BRAKE"
        quad_ccw = "MOTOR" if pwr_ccw >= 0.0 else "BRAKE"

        self.txt_cw.set_text(f"t={t:.2f}s  |  {as_cw}  |  {quad_cw}  |  {sat_cw}")
        self.txt_ccw.set_text(f"t={t:.2f}s  |  {as_ccw}  |  {quad_ccw}  |  {sat_ccw}")

        self.canvas.draw_idle()


class NavigatorVfWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuadScope v22.0")
        # self.resize(1650, 900)
        self.resize(3000, 1700)

        self.params = SystemParams()
        self.model = SystemModel(self.params)

        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: scrollable controls ----
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.left = QWidget()
        self.left_scroll.setWidget(self.left)
        self.left_layout = QVBoxLayout(self.left)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)

        self._build_controls()

        # ---- Right: tabs ----
        self.tabs = QTabWidget()

        self.mpl_presets = StonehousePresetPane()
        self.tabs.addTab(self.mpl_presets, "Feasibility")
        self.preset_pane = self.mpl_presets

        self.mpl_env = MplPane()
        self.tabs.addTab(self.mpl_env, "Envelope")

        self.mpl_motor = MotorOpsPane()
        self.tabs.addTab(self.mpl_motor, "Motor")
        self.motor_pane = self.mpl_motor

        self.mpl_moog = MoogCurvesPane()
        self.tabs.addTab(self.mpl_moog, "Moog Curves")
        self.moog_pane = self.mpl_moog

        self.mpl_tables = TablesPane()
        self.tabs.addTab(self.mpl_tables, "Tables")
        self.tables_pane = self.mpl_tables

        self.mpl_arch = ArchitecturePane()
        self.tabs.addTab(self.mpl_arch, "Architecture")
        self.arch_pane = self.mpl_arch

        self.dashboard = DashboardPane()
        self.tabs.addTab(self.dashboard, "Dashboard")
        self.dashboard_pane = self.dashboard

        self.mpl_dirrisk = DirectionRiskPane()
        self.tabs.addTab(self.mpl_dirrisk, "Direction Risk")
        self.dirrisk_pane = self.mpl_dirrisk

        self.mpl_anim = AnimationPane()
        self.tabs.addTab(self.mpl_anim, "Animation")
        self.anim_pane = self.mpl_anim

        # Nested sweep tabs (trade assessment)
        self.sweep_tabs = QTabWidget()
        self.sweep_ratio = SweepPane()
        self.sweep_speed = QuadSweepPane()
        self.sweep_motor = QuadSweepPane()
        self.sweep_fw = QuadSweepPane()
        self.sweep_cable = QuadSweepPane()
        self.sweep_inverter = QuadSweepPane()
        self.sweep_poles = QuadSweepPane()
        self.sweep_limits = QuadSweepPane()
        self.sweep_power = QuadSweepPane()
        self.sweep_load = QuadSweepPane()

        self.sweep_tabs.addTab(self.sweep_ratio, "Ratio Trade")
        self.sweep_tabs.addTab(self.sweep_speed, "Speed Voltage")
        self.sweep_tabs.addTab(self.sweep_motor, "Motor Design")
        self.sweep_tabs.addTab(self.sweep_fw, "Field Weakening")
        self.sweep_tabs.addTab(self.sweep_cable, "Cable Sensitivity")
        self.sweep_tabs.addTab(self.sweep_inverter, "Inverter")
        self.sweep_tabs.addTab(self.sweep_poles, "Motor Poles")
        self.sweep_tabs.addTab(self.sweep_limits, "Voltage Limits")
        self.sweep_tabs.addTab(self.sweep_power, "Power Assessment")
        self.sweep_tabs.addTab(self.sweep_load, "Loads & Braking")

        self.tabs.addTab(self.sweep_tabs, "Sweeps")

        self.mpl_band = BandPane()
        self.tabs.addTab(self.mpl_band, "Band Plots")

        # Backward-compatible alias used by plotting helpers
        self.band_pane = self.mpl_band

        splitter.addWidget(self.left_scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 1090])

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        # Guard to prevent Auto-update storms during bulk UI updates (Reset)
        self._bulk_updating = False
        # Snapshot initial UI values so Reset restores widgets (not just params)
        self._capture_ui_defaults()

        self._connect_signals()
        self._apply_sweeps_tab_enabled()
        self.update_all()

    # ---------- UI build ----------
    def _build_controls(self):
        title = QLabel("QuadScope: Navigator CCRS PowerTrain Static Solver")
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        title.setFont(f)
        self.left_layout.addWidget(title)

        # Status row
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("✅ Ready")
        self.lbl_status.setStyleSheet("color:#065f46; font-weight:600;")
        self.chk_auto = QCheckBox("Auto")
        self.chk_auto.setChecked(True)

        self.chk_sweeps = QCheckBox("Sweeps")
        self.chk_sweeps.setChecked(False)
        self.btn_reset = QPushButton("Reset")
        self.btn_recompute = QPushButton("Rerun")
        self.btn_report = QPushButton("Report")

        status_row.addWidget(self.lbl_status)
        status_row.addStretch(1)
        status_row.addWidget(self.chk_auto)
        status_row.addWidget(self.chk_sweeps)
        status_row.addWidget(self.btn_reset)
        status_row.addWidget(self.btn_recompute)
        status_row.addWidget(self.btn_report)
        self.left_layout.addLayout(status_row)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.status.setStyleSheet("padding: 10px;")
        self.left_layout.addWidget(self.status)

        # Control strategy
        gb_strategy = QGroupBox("CONTROL STRATEGY")
        Gs = QGridLayout(gb_strategy)

        self.combo_strategy = QComboBox()
        # Store internal keys via itemData so labels can evolve without breaking logic.
        self.combo_strategy.addItem("Baseline: V/f (scheduled volts, open-loop)", "VF")
        self.combo_strategy.addItem("Mode A: Vmax ceiling (Id = 0, no field-weakening)", "MODE_A")
        self.combo_strategy.addItem("Mode B: Vmax + Field-Weakening (Id scan)", "MODE_B")
        idx = self.combo_strategy.findData(self.params.control_strategy)
        self.combo_strategy.setCurrentIndex(idx if idx >= 0 else 0)

        self.lbl_strategy_note = QLabel("")
        self.lbl_strategy_note.setWordWrap(True)

        Gs.addWidget(QLabel("Strategy"), 0, 0)
        Gs.addWidget(self.combo_strategy, 0, 1)
        Gs.addWidget(self.lbl_strategy_note, 1, 0, 1, 2)

        self.left_layout.addWidget(gb_strategy)

        # Target
        gb_target = QGroupBox("TARGET OUTPUT")
        G = QGridLayout(gb_target)

        self.t_out_rpm = make_dspin(0.0, 5.0, 0.05, self.params.target.out_rpm, 3)
        self.t_out_tq = make_dspin(0.0, 3000.0, 25.0, self.params.target.out_torque_ftlbf, 1)
        self.env_out_rpm_max = make_dspin(0.1, 5.0, 0.1, 1.5, 2)

        add_slider_row(G, 0, "Output RPM", self.t_out_rpm)
        add_slider_row(G, 1, "UI torque target (ft-lbf)", self.t_out_tq)
        add_slider_row(G, 2, "Envelope plot max RPM", self.env_out_rpm_max)

        self.chk_tq_override = QCheckBox("Use UI torque target for continuous feasibility (override load stack)")
        self.chk_tq_override.setChecked(bool(getattr(self.params.target, "torque_override_continuous", True)))
        G.addWidget(self.chk_tq_override, 3, 0, 1, 3)

        self.left_layout.addWidget(gb_target)

        # Static load blocks (v13)
        gb_load = QGroupBox("BHA EXTERNAL LOADS")
        Gl = QGridLayout(gb_load)

        self.combo_out_dir = QComboBox()
        self.combo_out_dir.addItem("CW", "CW")
        self.combo_out_dir.addItem("CCW", "CCW")
        # set current from params
        idx_dir = self.combo_out_dir.findData(self.params.out_dir)
        self.combo_out_dir.setCurrentIndex(idx_dir if idx_dir >= 0 else 0)

        self.chk_stuck = QCheckBox("Stuck / stall mode (force output RPM = 0)")
        self.chk_stuck.setChecked(bool(getattr(self.params, "stuck_mode", False)))

        self.chk_brakepath = QCheckBox("Braking path available (4-quadrant / brake resistor)")
        self.chk_brakepath.setChecked(bool(getattr(self.params, "braking_path_available", True)))

        self.chk_brake_pwr = QCheckBox("Limit braking power at surface (regen sink)")
        self.chk_brake_pwr.setChecked(bool(getattr(self.params, "brake_power_limit_enabled", False)))
        self.brake_pwr_kw = make_dspin(0.0, 500.0, 1.0, float(getattr(self.params, "brake_power_kw_max", 50.0)), 1)
        # cable-aware regen constraint (surface clamp + cable drop vs motor back-EMF)
        self.chk_regen_cable = QCheckBox("Cable-aware regen limit (back-EMF must exceed surface clamp + cable drop)")
        self.chk_regen_cable.setChecked(bool(getattr(self.params, "regen_cable_limit_enabled", True)))
        self.regen_clamp_frac = make_dspin(0.0, 1.0, 0.05, float(getattr(self.params, "regen_surface_clamp_frac", 1.0)),
                                           2)
        self.regen_clamp_frac.setEnabled(self.chk_regen_cable.isChecked())
        self.brake_pwr_kw.setEnabled(self.chk_brake_pwr.isChecked())

        self.chk_bha = QCheckBox("Enable BHA block (TOB reaction + BHA friction)")
        self.chk_bha.setChecked(bool(self.params.bha.enabled))
        self.bha_tob = make_dspin(0.0, 2000.0, 10.0, float(getattr(self.params.bha, "drilling_tob_ftlbf", 150.0)), 1)
        self.bha_tc = make_dspin(0.0, 5000.0, 10.0, nm_to_ft_lbf(self.params.bha.fric_tc_nm), 1)
        self.bha_b = make_dspin(0.0, 2000.0, 1.0,
                                nm_to_ft_lbf(self.params.bha.fric_b_nm_per_rad_s * (2.0 * math.pi / 60.0)), 3)

        self.chk_par = QCheckBox("Enable gearbox parasitic losses (bearings, seals, etc.)")
        self.chk_par.setChecked(bool(self.params.parasitic.enabled))
        self.par_tc = make_dspin(0.0, 2000.0, 5.0, nm_to_ft_lbf(self.params.parasitic.tc_nm), 1)
        self.par_b = make_dspin(0.0, 500.0, 0.5,
                                nm_to_ft_lbf(self.params.parasitic.b_nm_per_rad_s * (2.0 * math.pi / 60.0)), 3)

        self.chk_mag = QCheckBox("Enable magnetic coupler (between motor and gearbox)")
        self.chk_mag.setChecked(bool(self.params.mag_coupler.enabled))
        self.mag_tbreak = make_dspin(0.0, 5.0, 0.01, float(self.params.mag_coupler.t_break_nm), 3)
        self.mag_tslip = make_dspin(0.0, 5.0, 0.01, float(self.params.mag_coupler.t_slip_nm), 3)
        self.mag_slope = make_dspin(0.1, 5.0, 0.01, float(self.params.mag_coupler.slope), 3)

        r = 0
        Gl.addWidget(QLabel("CCRS output direction"), r, 0)
        Gl.addWidget(self.combo_out_dir, r, 1);
        r += 1

        Gl.addWidget(self.chk_stuck, r, 0, 1, 2);
        r += 1
        Gl.addWidget(self.chk_brakepath, r, 0, 1, 2);
        r += 1

        Gl.addWidget(self.chk_brake_pwr, r, 0, 1, 2);
        r += 1
        add_slider_row(Gl, r, "Max surface braking power (kW)", self.brake_pwr_kw);
        r += 1
        Gl.addWidget(self.chk_regen_cable, r, 0, 1, 2);
        r += 1
        Gl.addWidget(QLabel("Regen clamp fraction (× inverter V-limit)"), r, 0)
        Gl.addWidget(self.regen_clamp_frac, r, 1);
        r += 1

        Gl.addWidget(self.chk_bha, r, 0, 1, 2);
        r += 1
        add_slider_row(Gl, r, "Drilling TOB magnitude (ft-lbf) [bit CW → TOB CCW]", self.bha_tob);
        r += 1
        add_slider_row(Gl, r, "BHA friction Tc (ft-lbf)", self.bha_tc);
        r += 1
        add_slider_row(Gl, r, "BHA viscous B (ft-lbf per RPM)", self.bha_b);
        r += 1
        self.left_layout.addWidget(gb_load)

        # V/f + inverter + field weakening
        gb_vf = QGroupBox("SURFACE INVERTER")
        vb = QVBoxLayout(gb_vf)
        vb.setContentsMargins(8, 8, 8, 8)
        vb.setSpacing(6)

        # Voltage entry basis (v11): DC link (recommended) or AC fundamental limit
        basis_grid = QGridLayout()
        basis_grid.setContentsMargins(0, 0, 0, 0)
        basis_grid.setHorizontalSpacing(8)
        lbl_basis = QLabel("Voltage entry basis")
        self.v_entry_basis = QComboBox()
        self.v_entry_basis.addItems(["DC link (Vdc)", "AC fundamental limit (Vrms)"])
        self.v_entry_basis.setCurrentText(self.params.vf.voltage_entry_basis)
        basis_grid.addWidget(lbl_basis, 0, 0)
        basis_grid.addWidget(self.v_entry_basis, 0, 1, 1, 2)
        vb.addLayout(basis_grid)

        # Derived AC limits (always displayed; solver uses phase RMS internally)
        self.lbl_vlimits = QLabel("")
        self.lbl_vlimits.setWordWrap(True)
        self.lbl_vlimits.setStyleSheet("color:#444;")
        vb.addWidget(self.lbl_vlimits)

        # --- AC fundamental entry (legacy) ---
        self.w_ac_vlim = QWidget()
        V1 = QGridLayout(self.w_ac_vlim)
        V1.setContentsMargins(0, 0, 0, 0)
        V1.setHorizontalSpacing(8)
        V1.setVerticalSpacing(4)

        self.v_limit_type = QComboBox()
        self.v_limit_type.addItems(["Line-Line (Vrms)", "Phase (L-N) (Vrms)"])
        self.v_limit_type.setCurrentText(self.params.vf.v_limit_type)
        self.v_limit_value = make_dspin(10.0, 2000.0, 5.0, self.params.vf.v_limit_value, 1)

        V1.addWidget(QLabel("AC V limit type"), 0, 0)
        V1.addWidget(self.v_limit_type, 0, 1, 1, 2)
        add_slider_row(V1, 1, "AC V limit value (Vrms)", self.v_limit_value,
                       "AC fundamental limit. If type is Line-Line, tool converts to Phase RMS internally.")
        vb.addWidget(self.w_ac_vlim)

        # --- DC link entry (recommended) ---
        self.w_dc_vlim = QWidget()
        V2 = QGridLayout(self.w_dc_vlim)
        V2.setContentsMargins(0, 0, 0, 0)
        V2.setHorizontalSpacing(8)
        V2.setVerticalSpacing(4)

        self.vdc_link = make_dspin(0.0, 2000.0, 5.0, self.params.vf.vdc_link_v, 0)
        add_slider_row(V2, 0, "DC link voltage Vdc (V)", self.vdc_link,
                       "DC bus feeding the inverter. Tool converts Vdc to AC fundamental based on modulation (linear region).")

        self.modulation = QComboBox()
        self.modulation.addItems(["SVPWM", "SPWM"])
        self.modulation.setCurrentText(self.params.vf.modulation)
        V2.addWidget(QLabel("Modulation"), 1, 0)
        V2.addWidget(self.modulation, 1, 1, 1, 2)

        self.v_util = make_dspin(0.50, 1.00, 0.01, self.params.vf.v_util, 2)
        add_slider_row(V2, 2, "Voltage utilization", self.v_util,
                       "0.95 is a practical default to account for deadtime / device drops. 1.00 = ideal linear modulation limit.")
        vb.addWidget(self.w_dc_vlim)

        # --- Common V/f settings ---
        w_common = QWidget()
        V = QGridLayout(w_common)
        V.setContentsMargins(0, 0, 0, 0)
        V.setHorizontalSpacing(8)
        V.setVerticalSpacing(4)

        self.base_f = make_dspin(10.0, 2000.0, 10.0, self.params.vf.base_freq_hz, 0)
        self.base_v = make_dspin(1.0, 2000.0, 2.0, self.params.vf.base_v_phase_rms, 1)
        self.v_boost = make_dspin(0.0, 80.0, 0.5, self.params.vf.v_boost, 1)

        add_slider_row(V, 0, "Base freq (Hz)", self.base_f)
        add_slider_row(V, 1, "Base Vphase (Vrms)", self.base_v)
        add_slider_row(V, 2, "Vboost (Vrms)", self.v_boost)

        self.lbl_slope = QLabel("")
        self.lbl_slope.setStyleSheet("color:#444;")
        V.addWidget(self.lbl_slope, 3, 0, 1, 3)

        self.chk_fw = QCheckBox("Enable field weakening (approx, negative Id)")
        self.chk_fw.setChecked(self.params.fw.enabled)
        V.addWidget(self.chk_fw, 4, 0, 1, 3)

        self.chk_fw_base_only = QCheckBox("Apply only above base freq")
        self.chk_fw_base_only.setChecked(self.params.fw.apply_only_above_base)
        V.addWidget(self.chk_fw_base_only, 5, 0, 1, 3)

        self.fw_idmax = make_dspin(0.0, 50.0, 0.05, self.params.fw.id_max_arms, 2)
        add_slider_row(V, 6, "Max |Id| (Arms, phase RMS)", self.fw_idmax,
                       "Field weakening current budget (negative Id)")

        note = QLabel(
            "Above base freq, Vcmd is constant. FW reduces motor EMF requirement via Id<0 (steady-state approx).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#444;")
        V.addWidget(note, 7, 0, 1, 3)

        vb.addWidget(w_common)

        self.left_layout.addWidget(gb_vf)

        # Sine filter (optional block between inverter and heptacable)
        gb_sf = QGroupBox("SINE FILTER (INVERTER OUTPUT)")
        gl_sf = QGridLayout(gb_sf)

        self.sf_enable = QCheckBox("Enable sine filter (Lf + Cf)")
        self.sf_enable.setChecked(bool(self.params.sine_filter.enabled))
        gl_sf.addWidget(self.sf_enable, 0, 0, 1, 2)

        gl_sf.addWidget(QLabel("Lf (mH/phase)"), 1, 0)
        self.sf_lf_mH = make_dspin(0.0, 500.0, 0.1, float(self.params.sine_filter.lf_h) * 1e3, 2)
        gl_sf.addWidget(self.sf_lf_mH, 1, 1)

        gl_sf.addWidget(QLabel("Rf (ohm/phase)"), 2, 0)
        self.sf_rf_ohm = make_dspin(0.0, 50.0, 0.01, float(self.params.sine_filter.rf_ohm), 3)
        gl_sf.addWidget(self.sf_rf_ohm, 2, 1)

        gl_sf.addWidget(QLabel("Cf (uF)"), 3, 0)
        self.sf_cf_uF = make_dspin(0.0, 500.0, 0.1, float(self.params.sine_filter.cf_f) * 1e6, 2)
        gl_sf.addWidget(self.sf_cf_uF, 3, 1)

        gl_sf.addWidget(QLabel("Cap connection"), 4, 0)
        self.sf_cap_conn = QComboBox()
        self.sf_cap_conn.addItems(["DELTA", "WYE"])
        self.sf_cap_conn.setCurrentText(str(self.params.sine_filter.cap_connection).upper())
        gl_sf.addWidget(self.sf_cap_conn, 4, 1)

        gl_sf.addWidget(QLabel("Damping topology"), 5, 0)
        self.sf_damp_topo = QComboBox()
        self.sf_damp_topo.addItems(["SERIES", "PARALLEL"])
        self.sf_damp_topo.setCurrentText(str(self.params.sine_filter.damping_topology).upper())
        gl_sf.addWidget(self.sf_damp_topo, 5, 1)

        gl_sf.addWidget(QLabel("Rd (ohm)"), 6, 0)
        self.sf_rd_ohm = make_dspin(0.0, 5000.0, 1.0, float(self.params.sine_filter.rd_ohm), 1)
        gl_sf.addWidget(self.sf_rd_ohm, 6, 1)

        note = QLabel("Note: Cf is interpreted by connection (DELTA: per branch, WYE: per phase).\n"
                      "Steady-state fundamental approximation; adds I_shunt and Vdrop across Lf/Rf.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        gl_sf.addWidget(note, 7, 0, 1, 2)

        self.left_layout.addWidget(gb_sf)

        # Cable
        gb_cable = QGroupBox("HEPTACABLE (PER PHASE MODEL)")
        C = QGridLayout(gb_cable)

        self.c_len = make_dspin(1.0, 10000.0, 1.0, self.params.cable.length_m, 0)
        self.c_rpm = make_dspin(0.01, 0.50, 0.01, self.params.cable.r_ohm_per_m, 3)
        self.c_lpm = make_dspin(0.0, 1e-2, 1e-7, self.params.cable.l_h_per_m, 9)
        self.c_temp = make_dspin(0.5, 4.0, 0.05, self.params.cable.temp_factor_r, 2)

        self.c_wires = QComboBox()
        self.c_wires.addItems(["1 wire/phase", "2 wires/phase"])
        self.c_wires.setCurrentIndex(0 if self.params.cable.wires_per_phase == 1 else 1)

        self.c_ilim_basis = QComboBox()
        self.c_ilim_basis.addItems(["Per phase", "Per conductor"])
        self.c_ilim_basis.setCurrentText(self.params.cable.i_limit_basis)

        self.c_ilim = make_dspin(0.1, 50.0, 0.015, self.params.cable.i_limit_arms, 2)
        self.c_lpar = make_dspin(0.2, 1.0, 0.01, self.params.cable.l_parallel_factor, 2)
        # Band Plots: optional 'shorter cable' overlay (L - ΔL)
        self.c_len_band = make_dspin(0.0, 20000.0, 100.0, 1000.0, 0)

        add_slider_row(C, 0, "Length (m)", self.c_len)
        add_slider_row(C, 1, "R (ohm/m)", self.c_rpm)
        add_slider_row(C, 2, "L (H/m)", self.c_lpm)
        add_slider_row(C, 3, "Temp factor on R", self.c_temp)

        C.addWidget(QLabel("Conductor count"), 4, 0)
        C.addWidget(self.c_wires, 4, 1, 1, 2)
        C.addWidget(QLabel("Current limit basis"), 5, 0)
        C.addWidget(self.c_ilim_basis, 5, 1, 1, 2)
        add_slider_row(C, 6, "I limit (Arms)", self.c_ilim, "Hard limit basis: per conductor or per phase")
        add_slider_row(C, 7, "2-wire L factor", self.c_lpar, "Inductance multiplier when using 2 wires/phase")
        # --- Cable temperature model (optional) ---
        self.c_temp_ref = make_dspin(-50.0, 80.0, 1.0, float(getattr(self.params.cable, "temp_ref_C", 20.0)), 1)
        self.c_temp_alpha = make_dspin(0.0, 0.02, 0.00010,
                                       float(getattr(self.params.cable, "temp_alpha_per_C", 0.00393)), 5)

        # 2) 5-segment custom profile (explicit Li + Ti per segment)
        self.c_temp_5seg = QCheckBox("5-seg custom temp profile (length + temp per segment)")
        self.c_temp_5seg.setChecked(bool(getattr(self.params.cable, "temp_model_5seg", True)))

        self.c_temp_derate = QCheckBox("Derate I-limit with temp (approx: I ∝ 1/√R(T))")
        self.c_temp_derate.setChecked(bool(getattr(self.params.cable, "i_limit_derate_with_temp", True)))

        # Defaults for 5-seg profile (used only if no saved profile exists)
        T_def = [100.0, 125.0, 160.0, 175.0, 175.0]
        L_def = [float(self.params.cable.length_m) / 5.0] * 5

        _L5 = list(getattr(self.params.cable, "temp5_seg_len_m", L_def) or L_def)
        _T5 = list(getattr(self.params.cable, "temp5_seg_temp_C", T_def) or T_def)
        if len(_L5) < 5:
            _L5 = _L5 + L_def[len(_L5):]
        if len(_T5) < 5:
            _T5 = _T5 + T_def[len(_T5):]
        _L5 = _L5[:5]
        _T5 = _T5[:5]

        self.c_5seg_L = []
        self.c_5seg_T = []
        for i in range(5):
            self.c_5seg_L.append(make_dspin(0.0, 20000.0, 50.0, float(_L5[i]), 0))
            self.c_5seg_T.append(make_dspin(-50.0, 250.0, 1.0, float(_T5[i]), 1))

        self.lbl_c_5seg_total = QLabel("—")

        # Place below the existing cable controls
        row0 = 15
        C.addWidget(QLabel("— Temperature model —"), row0, 0, 1, 3)

        # Shared R(T) reference / alpha
        add_slider_row(C, row0 + 1, "Reference temp (°C)", self.c_temp_ref)
        add_slider_row(C, row0 + 2, "Alpha dR/R per °C", self.c_temp_alpha)

        # 5-seg custom profile
        C.addWidget(self.c_temp_5seg, row0 + 3, 0, 1, 3)
        C.addWidget(self.c_temp_derate, row0 + 4, 0, 1, 3)

        C.addWidget(QLabel("Seg"), row0 + 5, 0)
        C.addWidget(QLabel("Length (m)"), row0 + 5, 1)
        C.addWidget(QLabel("Temp (°C)"), row0 + 5, 2)
        for i in range(5):
            C.addWidget(QLabel(str(i + 1)), row0 + 6 + i, 0)
            C.addWidget(self.c_5seg_L[i], row0 + 6 + i, 1)
            C.addWidget(self.c_5seg_T[i], row0 + 6 + i, 2)

        C.addWidget(QLabel("5-seg total length"), row0 + 11, 0)
        C.addWidget(self.lbl_c_5seg_total, row0 + 11, 1, 1, 2)

        # Ensure enable/disable state is correct at startup; live recompute is handled by _connect_signals().
        self._update_cable_temp_ui_enabled()

        self.lbl_cable_R = QLabel("—")
        self.lbl_cable_L = QLabel("—")
        self.lbl_cable_Iph = QLabel("—")
        C.addWidget(QLabel("Effective R_phase"), 8, 0)
        C.addWidget(self.lbl_cable_R, 8, 1, 1, 2)
        C.addWidget(QLabel("Effective L_phase"), 9, 0)
        C.addWidget(self.lbl_cable_L, 9, 1, 1, 2)
        C.addWidget(QLabel("Effective I_phase limit"), 10, 0)
        C.addWidget(self.lbl_cable_Iph, 10, 1, 1, 2)

        add_slider_row(C, 11, "Band: -ΔL (m)", self.c_len_band,
                       "Band Plots: overlay envelope for cable length (L - ΔL)")

        self.left_layout.addWidget(gb_cable)

        # Motor
        gb_motor = QGroupBox("PMSM (DESIGN-ORIENTED PARAMS)")
        M = QGridLayout(gb_motor)

        self.m_preset = QComboBox()
        self.m_preset.addItems(
            ["Spear motor", "Milling motor", "Annular Motor - John", "Annular Motor - John2", "Annular Motor - John3", "User defined"])
        self.m_preset.setCurrentText("Spear motor")

        self.m_pp = make_ispin(1, 10, 1, self.params.motor.pole_pairs)
        self.m_rs = make_dspin(0.001, 50.0, 0.05, self.params.motor.rs_ohm, 3)
        self.m_ld = make_dspin(0.0, 0.5, 0.0005, self.params.motor.ld_h, 6)
        self.m_lq = make_dspin(0.0, 0.5, 0.0005, self.params.motor.lq_h, 6)

        self.m_link = QCheckBox("Link Kt ↔ Ke (sinusoidal PMSM, basis-aware)")
        self.m_link.setChecked(self.params.motor.link_kt_ke)

        self.m_mode = QComboBox()
        self.m_mode.addItems(["Lambda", "Kt", "Ke"])
        self.m_mode.setCurrentText(self.params.motor.motor_param_mode)

        self.m_kt_basis = QComboBox()
        self.m_kt_basis.addItems(["Nm/Arms", "Nm/Apeak", "lb-in/Arms", "lb-in/Apeak"])
        self.m_kt_basis.setCurrentText(self.params.motor.kt_basis)

        self.m_ke_basis = QComboBox()
        self.m_ke_basis.addItems(["Vll_rms/krpm", "Vll_peak/krpm"])
        self.m_ke_basis.setCurrentText(self.params.motor.ke_basis)

        self.m_lambda = make_dspin(0.0005, 2.0, 0.002, self.params.motor.lambda_wb, 4)
        self.m_kt = make_dspin(0.0005, 100.0, 0.0001, self.params.motor.kt_display(), 4)
        self.m_ke = make_dspin(0.01, 500.0, 0.01, max(0.1, self.params.motor.ke_display()), 2)

        # rows
        M.addWidget(QLabel("Preset"), 0, 0)
        M.addWidget(self.m_preset, 0, 1, 1, 2)

        M.addWidget(QLabel("Pole pairs"), 1, 0)
        M.addWidget(slider_for_spin(self.m_pp), 1, 1)
        M.addWidget(self.m_pp, 1, 2)

        add_slider_row(M, 2, "Rs (ohm)", self.m_rs)
        add_slider_row(M, 3, "Ld (H)", self.m_ld)
        add_slider_row(M, 4, "Lq (H)", self.m_lq)

        M.addWidget(self.m_link, 5, 0, 1, 3)

        M.addWidget(QLabel("Design input mode"), 6, 0)
        M.addWidget(self.m_mode, 6, 1, 1, 2)

        M.addWidget(QLabel("Kt basis"), 7, 0)
        M.addWidget(self.m_kt_basis, 7, 1, 1, 2)
        M.addWidget(QLabel("Ke basis"), 8, 0)
        M.addWidget(self.m_ke_basis, 8, 1, 1, 2)

        add_slider_row(M, 9, "Lambda (Wb)", self.m_lambda, "Flux linkage used for back-EMF")
        add_slider_row(M, 10, "Kt (selected basis)", self.m_kt, "Torque constant input (basis-aware)")
        add_slider_row(M, 11, "Ke (selected basis)", self.m_ke, "Back-EMF constant input (basis-aware)")

        self.lbl_kt_ke_rule = QLabel("—")
        self.lbl_kt_ke_rule.setWordWrap(True)
        self.lbl_kt_ke_rule.setStyleSheet("color:#444;")
        M.addWidget(self.lbl_kt_ke_rule, 12, 0, 1, 3)

        self.left_layout.addWidget(gb_motor)

        # Extra torque + Kt(T)
        gb_extra = QGroupBox("EXTRA TORQUE + Kt(T)")
        X = QGridLayout(gb_extra)

        self.x_preset = QComboBox()
        self.x_preset.addItems([
            "Milling Moog Curve",
            "Spear Moog Curve",
            "Annular Windings Curve",
            "User Defined",
        ])
        # "User Defined" = restore built-in defaults (ExtraTorqueParams defaults)
        self.x_preset.setCurrentText("User Defined")

        self.x_enable_extra = QCheckBox("Use τ_extra(ω,T) in torque→current mapping")
        self.x_enable_extra.setChecked(self.params.extra.extra_enabled)

        self.x_enable_ktT = QCheckBox("Enable Kt(T) scaling")
        self.x_enable_ktT.setChecked(self.params.extra.kt_temp_enabled)

        self.x_enable_rsT = QCheckBox("Enable Rs(T) scaling (T_winding = T + ΔT)")
        self.x_enable_rsT.setChecked(bool(getattr(self.params.extra, "rs_temp_enabled", False)))
        self.x_wind_dT = make_dspin(0.0, 60.0, 0.5, float(getattr(self.params.extra, "winding_rise_C", 10.0)), 1)
        # UI shows %/°C; params store fraction/°C
        self.x_rs_tc = make_dspin(0.0, 2.0, 0.01,
                                  100.0 * float(getattr(self.params.extra, "rs_temp_coeff_per_C", 0.00393)), 3)

        self.x_temp = make_dspin(-20.0, 250.0, 1.0, self.params.extra.temp_C, 1)
        self.x_tref = make_dspin(-20.0, 250.0, 1.0, self.params.extra.temp_ref_C, 1)
        # store as %/°C in UI, fraction/°C in params
        self.x_kt_tc = make_dspin(-2.0, 2.0, 0.01, 100.0 * self.params.extra.kt_temp_coeff_per_C, 3)

        self.x_core_en = QCheckBox("Core: τ_core = C_L · ω^0.5")
        self.x_core_en.setChecked(self.params.extra.core_enabled)
        # Core-loss coefficient values are typically small; allow micro-resolution.
        self.x_core_cl = make_dspin(0.0, 50.0, 1e-5, self.params.extra.core_cL, 6)
        self.x_core_exp = make_dspin(0.0, 2.0, 0.01, self.params.extra.core_exp, 2)

        self.x_visc_en = QCheckBox("Viscous: τ_visc(ω,T) (Couette/transition/turbulence)")
        self.x_visc_en.setChecked(self.params.extra.visc_enabled)

        self.x_visc_model = QComboBox()
        self.x_visc_model.addItems([
            "Couette (τ=k·ω)",
            "Transition (τ=k·ω^n)",
            "Turbulent (τ=k·ω²)",
            "Piecewise (Couette→Transition→Turbulent)",
        ])
        self.x_visc_model.setCurrentText(self.params.extra.visc_model)

        # Viscous coefficients can be extremely small for oil-filled motors.
        # Keep wide ranges for manual sweeps, but allow micro-resolution.
        self.x_k_c = make_dspin(0.0, 100.0, 1e-6, self.params.extra.visc_k_couette, 8)
        self.x_k_tr = make_dspin(0.0, 100.0, 1e-6, self.params.extra.visc_k_transition, 8)
        self.x_n_tr = make_dspin(0.5, 3.0, 0.05, self.params.extra.visc_n_transition, 2)
        self.x_k_tb = make_dspin(0.0, 1.0, 1e-8, self.params.extra.visc_k_turb, 10)
        self.x_rpm1 = make_dspin(0.0, 20000.0, 50.0, self.params.extra.visc_rpm1, 0)
        self.x_rpm2 = make_dspin(0.0, 20000.0, 50.0, self.params.extra.visc_rpm2, 0)

        self.x_visc_T = QComboBox()
        self.x_visc_T.addItems(["None", "Linear", "Exponential"])
        self.x_visc_T.setCurrentText(self.params.extra.visc_temp_scaling)
        self.x_visc_a = make_dspin(-5.0, 5.0, 0.01, 100.0 * self.params.extra.visc_lin_coeff_per_C, 3)  # %/°C
        self.x_visc_beta = make_dspin(0.0, 1.0, 0.001, self.params.extra.visc_beta_per_C, 4)  # 1/°C

        self.x_smooth = QCheckBox("Smooth transitions")
        self.x_smooth.setChecked(self.params.extra.smooth_transitions)
        self.x_smooth_f = make_dspin(0.0, 0.5, 0.01, self.params.extra.smooth_frac, 2)

        self.lbl_extra_preview = QLabel("—")
        self.lbl_extra_preview.setWordWrap(True)
        self.lbl_extra_preview.setStyleSheet("color:#444;")

        r = 0
        X.addWidget(QLabel("Preset"), r, 0)
        X.addWidget(self.x_preset, r, 1, 1, 2)
        r += 1

        X.addWidget(self.x_enable_extra, r, 0, 1, 3);
        r += 1
        X.addWidget(self.x_enable_ktT, r, 0, 1, 3);
        r += 1
        X.addWidget(self.x_enable_rsT, r, 0, 1, 3);
        r += 1

        add_slider_row(X, r, "T (°C)", self.x_temp);
        r += 1
        add_slider_row(X, r, "T_ref (°C)", self.x_tref);
        r += 1
        add_slider_row(X, r, "Kt temp coeff (%/°C)", self.x_kt_tc, "Kt(T)=Kt_ref·(1+α·(T-T_ref))");
        r += 1
        add_slider_row(X, r, "ΔT_winding (°C)", self.x_wind_dT, "Winding temperature used for Rs(T): T_w = T + ΔT");
        r += 1
        add_slider_row(X, r, "Rs temp coeff (%/°C)", self.x_rs_tc, "Copper ≈ 0.393%/°C");
        r += 1

        X.addWidget(self.x_core_en, r, 0, 1, 3);
        r += 1
        add_slider_row(X, r, "C_L (Nm/(rad/s)^0.5)", self.x_core_cl);
        r += 1
        add_slider_row(X, r, "Core exponent", self.x_core_exp, "Default 0.5 per requested model");
        r += 1

        X.addWidget(self.x_visc_en, r, 0, 1, 3);
        r += 1
        X.addWidget(QLabel("Viscous model"), r, 0);
        X.addWidget(self.x_visc_model, r, 1, 1, 2);
        r += 1
        add_slider_row(X, r, "k_couette (Nm/(rad/s))", self.x_k_c);
        r += 1
        add_slider_row(X, r, "k_transition (Nm/(rad/s)^n)", self.x_k_tr);
        r += 1
        add_slider_row(X, r, "n_transition", self.x_n_tr);
        r += 1
        add_slider_row(X, r, "k_turb (Nm/(rad/s)^2)", self.x_k_tb);
        r += 1
        add_slider_row(X, r, "RPM_1 (Couette→Trans)", self.x_rpm1);
        r += 1
        add_slider_row(X, r, "RPM_2 (Trans→Turb)", self.x_rpm2);
        r += 1

        X.addWidget(QLabel("Viscous temp scaling"), r, 0);
        X.addWidget(self.x_visc_T, r, 1, 1, 2);
        r += 1
        add_slider_row(X, r, "a (%/°C) [Linear]", self.x_visc_a);
        r += 1
        add_slider_row(X, r, "β (1/°C) [Exp]", self.x_visc_beta);
        r += 1

        X.addWidget(self.x_smooth, r, 0, 1, 2);
        X.addWidget(self.x_smooth_f, r, 2);
        r += 1
        X.addWidget(self.lbl_extra_preview, r, 0, 1, 3);
        r += 1

        self.left_layout.addWidget(gb_extra)

        # Magnetic coupler (between motor and gearbox)
        gb_mag = QGroupBox("MAGNETIC COUPLER (MOTOR ↔ GEARBOX)")
        Gm = QGridLayout(gb_mag)

        rr = 0
        Gm.addWidget(self.chk_mag, rr, 0, 1, 3);
        rr += 1
        add_slider_row(Gm, rr, "T_break (Nm)", self.mag_tbreak,
                       "Low-slip drag torque (eddy-current / shear) that always opposes relative motion.");
        rr += 1
        add_slider_row(Gm, rr, "T_slip (Nm)", self.mag_tslip,
                       "Max transmitted torque before the coupler slips (torque clamp).");
        rr += 1
        add_slider_row(Gm, rr, "Slope", self.mag_slope,
                       "Shape factor for slip curve (higher = stiffer transition).");
        rr += 1

        note_mag = QLabel("Note: This block sits between PMSM and gearbox. It is not an external BHA load.")
        note_mag.setWordWrap(True)
        note_mag.setStyleSheet("color:#444;")
        Gm.addWidget(note_mag, rr, 0, 1, 3);
        rr += 1

        self.left_layout.addWidget(gb_mag)

        # Gearbox
        gb_gear = QGroupBox("3-STAGE CYCLOIDAL GEARBOX")
        Gg = QGridLayout(gb_gear)

        self.g_preset = QComboBox()
        self.g_preset.addItems(["Spear gearbox", "Milling gearbox", "User defined"])
        self.g_preset.setCurrentText("Spear gearbox")

        Gg.addWidget(QLabel("Preset"), 0, 0)
        Gg.addWidget(self.g_preset, 0, 1, 1, 2)

        self.g1 = make_dspin(1.0, 400.0, 1.0, self.params.gearbox.stage1, 1)
        self.g2 = make_dspin(1.0, 400.0, 1.0, self.params.gearbox.stage2, 1)
        self.g3 = make_dspin(1.0, 400.0, 1.0, self.params.gearbox.stage3, 1)

        self.e1 = make_dspin(0.2, 0.99, 0.01, self.params.gearbox.eff1, 2)
        self.e2 = make_dspin(0.2, 0.99, 0.01, self.params.gearbox.eff2, 2)
        self.e3 = make_dspin(0.2, 0.99, 0.01, self.params.gearbox.eff3, 2)
        self.eta_misc = make_dspin(0.5, 1.0, 0.01, self.params.gearbox.eta_misc, 2)

        self.chk_eta_override = QCheckBox("Override total efficiency")
        self.chk_eta_override.setChecked(self.params.gearbox.override_total_eta)
        self.eta_override = make_dspin(0.05, 0.99, 0.01, self.params.gearbox.eta_total_override, 2)

        # Band plots: gearbox efficiency uncertainty band (±, absolute per-unit)
        # Example: 0.10 => η ∈ [η_nom-0.10, η_nom+0.10] (clipped to (0,1))
        self.eta_band_pu = make_dspin(0.0, 0.50, 0.01, 0.10, 3)

        add_slider_row(Gg, 1, "Stage 1 ratio", self.g1)
        add_slider_row(Gg, 2, "Stage 2 ratio", self.g2)
        add_slider_row(Gg, 3, "Stage 3 ratio", self.g3)

        add_slider_row(Gg, 4, "Stage 1 eff", self.e1)
        add_slider_row(Gg, 5, "Stage 2 eff", self.e2)
        add_slider_row(Gg, 6, "Stage 3 eff", self.e3)
        add_slider_row(Gg, 7, "Misc η multiplier", self.eta_misc)

        Gg.addWidget(self.chk_eta_override, 8, 0, 1, 2)
        Gg.addWidget(self.eta_override, 8, 2)

        self.lbl_G = QLabel("—")
        self.lbl_eta = QLabel("—")
        Gg.addWidget(QLabel("Total ratio"), 9, 0)
        Gg.addWidget(self.lbl_G, 9, 1, 1, 2)
        Gg.addWidget(QLabel("Total efficiency"), 10, 0)
        Gg.addWidget(self.lbl_eta, 10, 1, 1, 2)

        add_slider_row(Gg, 11, "η band ± (abs pu)", self.eta_band_pu)

        self.chk_backdrive = QCheckBox("Gearbox backdrivable (allow backdrive/regen)")
        self.chk_backdrive.setChecked(bool(getattr(self.params.gearbox, "backdrivable", True)))
        Gg.addWidget(self.chk_backdrive, 12, 0, 1, 3)

        # Gearbox internal parasitic losses (bearings, seals, oil shear, etc.)
        Gg.addWidget(self.chk_par, 13, 0, 1, 3)
        add_slider_row(Gg, 14, "Parasitic Tc (ft-lbf)", self.par_tc,
                       "Lumped gearbox internal loss torque resisting output rotation.");
        add_slider_row(Gg, 15, "Parasitic B (ft-lbf per RPM)", self.par_b)

        self.left_layout.addWidget(gb_gear)

        # Downhole limits
        gb_lim = QGroupBox("DOWNHOLE VOLTAGE LIMITS (HARD/OPTIONAL)")
        Lm = QGridLayout(gb_lim)

        self.chk_dh_vph = QCheckBox("Enforce motor Vphase RMS limit (L-N)")
        self.chk_dh_vph.setChecked(self.params.limits.enforce_downhole_vphase_limit)
        self.v_dh_vph = make_dspin(1.0, 500.0, 1.0, self.params.limits.downhole_v_phase_rms_limit, 1)

        self.chk_dh_vll = QCheckBox("Enforce contact-block Vll RMS limit (L-L)")
        self.chk_dh_vll.setChecked(self.params.limits.enforce_downhole_vll_limit)
        self.v_dh_vll = make_dspin(10.0, 1000.0, 5.0, self.params.limits.downhole_vll_rms_limit, 1)

        # Band plots: contact-block Vll uncertainty band (± Vrms)
        self.vll_band_vrms = make_dspin(0.0, 200.0, 1.0, 20.0, 1)

        Lm.addWidget(self.chk_dh_vph, 0, 0, 1, 3)
        add_slider_row(Lm, 1, "Downhole motor Vphase limit (Vrms)", self.v_dh_vph)
        Lm.addWidget(self.chk_dh_vll, 2, 0, 1, 3)
        add_slider_row(Lm, 3, "Downhole contact-block Vll limit (Vrms)", self.v_dh_vll)

        add_slider_row(Lm, 4, "Contact Vll band ± (Vrms)", self.vll_band_vrms)

        self.left_layout.addWidget(gb_lim)

        # Sweep settings (for trade studies)
        gb_sw = QGroupBox("SWEEP SETTINGS (TRADE-OFFS)")
        S = QGridLayout(gb_sw)

        self.sw_ratio_min = make_dspin(100.0, 15000.0, 50.0, 500.0, 0)
        self.sw_ratio_max = make_dspin(200.0, 20000.0, 50.0, 4500.0, 0)
        self.sw_points = make_ispin(25, 401, 1, 161)

        self.sw_ke_min = make_dspin(1.0, 500.0, 1.0, 10.0, 1)
        self.sw_ke_max = make_dspin(1.0, 500.0, 1.0, 150.0, 1)

        self.sw_len_min = make_dspin(100.0, 50000.0, 100.0, 1000.0, 0)
        self.sw_len_max = make_dspin(200.0, 50000.0, 100.0, 15000.0, 0)

        vref = self.params.vf.v_ll_rms_limit()  # reference for sweep defaults (effective Vll,rms)

        # ----- Inverter sweeps -----
        self.sw_inv_vlim_min = make_dspin(10.0, 2000.0, 5.0, max(10.0, vref * 0.5), 1)
        self.sw_inv_vlim_max = make_dspin(10.0, 2000.0, 5.0, max(20.0, vref * 1.1), 1)

        self.sw_basef_min = make_dspin(10.0, 5000.0, 10.0, max(10.0, self.params.vf.base_freq_hz * 0.5), 0)
        self.sw_basef_max = make_dspin(10.0, 5000.0, 10.0, max(20.0, self.params.vf.base_freq_hz * 1.5), 0)

        self.sw_basev_min = make_dspin(1.0, 2000.0, 2.0, max(1.0, self.params.vf.base_v_phase_rms * 0.5), 1)
        self.sw_basev_max = make_dspin(1.0, 2000.0, 2.0, max(2.0, self.params.vf.base_v_phase_rms * 1.5), 1)

        self.sw_vboost_min = make_dspin(0.0, 200.0, 0.5, max(0.0, self.params.vf.v_boost * 0.0), 1)
        self.sw_vboost_max = make_dspin(0.0, 200.0, 0.5, max(1.0, self.params.vf.v_boost * 2.0), 1)

        # ----- Motor pole-pairs sweep -----
        self.sw_pp_min = make_ispin(1, 24, 1, max(1, int(self.params.motor.pole_pairs)))
        self.sw_pp_max = make_ispin(1, 24, 1, max(2, int(self.params.motor.pole_pairs) * 2))

        self.sw_pp_hold = QComboBox()
        self.sw_pp_hold.addItems(["Hold lambda (Wb)", "Hold Ke (Vll_rms/krpm)", "Hold Kt (Nm/Arms)"])
        self.sw_pp_hold.setCurrentIndex(0)

        # ----- Downhole voltage-limit sweeps -----
        self.sw_dh_vph_min = make_dspin(1.0, 800.0, 1.0, max(1.0, self.params.limits.downhole_v_phase_rms_limit * 0.5),
                                        1)
        self.sw_dh_vph_max = make_dspin(1.0, 800.0, 1.0, max(2.0, self.params.limits.downhole_v_phase_rms_limit * 1.5),
                                        1)

        self.sw_dh_vll_min = make_dspin(10.0, 2000.0, 5.0, max(10.0, self.params.limits.downhole_vll_rms_limit * 0.5),
                                        1)
        self.sw_dh_vll_max = make_dspin(10.0, 2000.0, 5.0, max(20.0, self.params.limits.downhole_vll_rms_limit * 1.5),
                                        1)

        # layout rows
        add_slider_row(S, 8, 'Inverter Vlimit min (same units)', self.sw_inv_vlim_min)
        add_slider_row(S, 9, 'Inverter Vlimit max (same units)', self.sw_inv_vlim_max)
        add_slider_row(S, 10, 'Base freq sweep min (Hz)', self.sw_basef_min)
        add_slider_row(S, 11, 'Base freq sweep max (Hz)', self.sw_basef_max)
        add_slider_row(S, 12, 'Base Vphase sweep min (Vrms)', self.sw_basev_min)
        add_slider_row(S, 13, 'Base Vphase sweep max (Vrms)', self.sw_basev_max)
        add_slider_row(S, 14, 'Vboost sweep min (Vrms)', self.sw_vboost_min)
        add_slider_row(S, 15, 'Vboost sweep max (Vrms)', self.sw_vboost_max)

        S.addWidget(QLabel('Pole pairs sweep min/max'), 16, 0)
        S.addWidget(self.sw_pp_min, 16, 1)
        S.addWidget(self.sw_pp_max, 16, 2)

        S.addWidget(QLabel('Pole sweep hold-mode'), 17, 0)
        S.addWidget(self.sw_pp_hold, 17, 1, 1, 2)

        add_slider_row(S, 18, 'Downhole Vphase sweep min (Vrms)', self.sw_dh_vph_min)
        add_slider_row(S, 19, 'Downhole Vphase sweep max (Vrms)', self.sw_dh_vph_max)
        add_slider_row(S, 20, 'Downhole Vll sweep min (Vrms)', self.sw_dh_vll_min)
        add_slider_row(S, 21, 'Downhole Vll sweep max (Vrms)', self.sw_dh_vll_max)
        self.chk_fast_sweeps = QCheckBox("Fast sweeps (Id=0 and -Idmax only)")
        self.chk_fast_sweeps.setChecked(True)

        add_slider_row(S, 0, "Ratio min", self.sw_ratio_min)
        add_slider_row(S, 1, "Ratio max", self.sw_ratio_max)
        S.addWidget(QLabel("Sweep points"), 2, 0)
        S.addWidget(slider_for_spin(self.sw_points), 2, 1)
        S.addWidget(self.sw_points, 2, 2)

        add_slider_row(S, 3, "Ke sweep min (Vll_rms/krpm)", self.sw_ke_min)
        add_slider_row(S, 4, "Ke sweep max (Vll_rms/krpm)", self.sw_ke_max)

        add_slider_row(S, 5, "Cable length sweep min (m)", self.sw_len_min)
        add_slider_row(S, 6, "Cable length sweep max (m)", self.sw_len_max)

        S.addWidget(self.chk_fast_sweeps, 23, 0, 1, 3)

        note_sw = QLabel("Tip: use fast sweeps for interactive trade studies; switch off for full Id grid sweeps.")
        note_sw.setWordWrap(True)
        note_sw.setStyleSheet("color:#444;")
        S.addWidget(note_sw, 24, 0, 1, 3)

        self.left_layout.addWidget(gb_sw)

        self.left_layout.addStretch(1)

        # -------- Save plots (exports all tabs + sweeps) --------
        self.btn_save_plots = QPushButton("Save plots")
        self.btn_save_plots.setToolTip(
            "Save PNGs of all plots from all tabs (including Sweeps), and also saves each individual subplot as its own image.\n"
            "A new unique output folder is created every time you press this button."
        )
        self.btn_save_plots.setStyleSheet("padding: 9px; font-weight:700;")
        self.left_layout.addWidget(self.btn_save_plots)

        footer = QLabel("GEMS 106068737 | ADARSH GOUDA | Version 22.0")
        footer.setAlignment(Qt.AlignHCenter)
        footer.setStyleSheet("color: #4169E1; font-weight:600; padding-top:8px;")
        self.left_layout.addWidget(footer)

    def _connect_signals(self):
        widgets = [
            self.t_out_rpm, self.t_out_tq, self.env_out_rpm_max,
            self.chk_tq_override,
            self.v_limit_type, self.v_limit_value, self.v_entry_basis, self.vdc_link, self.modulation, self.v_util,
            self.base_f, self.base_v, self.v_boost,
            self.combo_strategy,
            self.chk_fw, self.chk_fw_base_only, self.fw_idmax,
            self.sf_enable, self.sf_lf_mH, self.sf_rf_ohm, self.sf_cf_uF, self.sf_cap_conn, self.sf_damp_topo,
            self.sf_rd_ohm,
            self.c_len, self.c_rpm, self.c_lpm, self.c_temp, self.c_wires, self.c_ilim_basis, self.c_ilim, self.c_lpar,
            self.c_temp_ref, self.c_temp_alpha,
            self.c_len_band,
            self.m_pp, self.m_rs, self.m_ld, self.m_lq, self.m_link, self.m_mode, self.m_kt_basis, self.m_ke_basis,
            self.m_lambda, self.m_kt, self.m_ke,
            # extra torque + Kt(T)
            self.x_enable_extra, self.x_enable_ktT, self.x_enable_rsT, self.x_temp, self.x_tref, self.x_kt_tc,
            self.x_wind_dT, self.x_rs_tc,
            self.x_core_en, self.x_core_cl, self.x_core_exp,
            self.x_visc_en, self.x_visc_model,
            self.x_k_c, self.x_k_tr, self.x_n_tr, self.x_k_tb,
            self.x_rpm1, self.x_rpm2,
            self.x_visc_T, self.x_visc_a, self.x_visc_beta,
            self.x_smooth, self.x_smooth_f,
            self.g1, self.g2, self.g3, self.e1, self.e2, self.e3, self.eta_misc, self.chk_eta_override,
            self.eta_override, self.eta_band_pu, self.chk_backdrive, self.chk_backdrive,
            self.chk_dh_vph, self.v_dh_vph, self.chk_dh_vll, self.v_dh_vll, self.vll_band_vrms,
            self.sw_ratio_min, self.sw_ratio_max, self.sw_points,
            self.sw_ke_min, self.sw_ke_max, self.sw_len_min, self.sw_len_max,
            self.sw_inv_vlim_min, self.sw_inv_vlim_max, self.sw_basef_min, self.sw_basef_max, self.sw_basev_min,
            self.sw_basev_max, self.sw_vboost_min, self.sw_vboost_max,
            self.sw_pp_min, self.sw_pp_max, self.sw_pp_hold,
            self.sw_dh_vph_min, self.sw_dh_vph_max, self.sw_dh_vll_min, self.sw_dh_vll_max,
            self.chk_fast_sweeps,
            self.chk_sweeps,
            self.combo_out_dir, self.chk_stuck, self.chk_brakepath, self.chk_brake_pwr, self.brake_pwr_kw,
            self.chk_regen_cable, self.regen_clamp_frac, self.chk_bha, self.bha_tob, self.bha_tc, self.bha_b,
            self.chk_par, self.par_tc, self.par_b, self.chk_mag, self.mag_tbreak, self.mag_tslip, self.mag_slope,
        ]

        # Extra cable temperature widgets (5-seg custom profile + I-limit derating)
        if hasattr(self, "c_temp_5seg"):
            widgets.append(self.c_temp_5seg)
        if hasattr(self, "c_temp_derate"):
            widgets.append(self.c_temp_derate)
        if hasattr(self, "c_5seg_L"):
            widgets.extend(list(self.c_5seg_L))
        if hasattr(self, "c_5seg_T"):
            widgets.extend(list(self.c_5seg_T))

        def hook(w):
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.valueChanged.connect(self._on_any_change)
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._on_any_change)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._on_any_change)

        for w in widgets:
            hook(w)

        self.btn_recompute.clicked.connect(self.update_all)
        self.btn_reset.clicked.connect(self._reset_defaults)
        self.btn_report.clicked.connect(self._generate_pdf_report)
        self.btn_save_plots.clicked.connect(self._save_all_plots)
        self.chk_sweeps.stateChanged.connect(self._on_sweeps_toggle)
        # Presets (Motor/Gearbox/Extra torque)
        if hasattr(self, "m_preset"):
            self.m_preset.currentIndexChanged.connect(self._on_motor_preset_changed)
        if hasattr(self, "g_preset"):
            self.g_preset.currentIndexChanged.connect(self._on_gearbox_preset_changed)
        if hasattr(self, "x_preset"):
            self.x_preset.currentIndexChanged.connect(self._on_extra_torque_preset_changed)

        self.tabs.currentChanged.connect(self._on_main_tab_changed)

    def _on_motor_preset_changed(self):
        if getattr(self, "_bulk_updating", False):
            return
        name = self.m_preset.currentText() if hasattr(self, "m_preset") else ""
        self._apply_motor_preset(name)

    def _on_gearbox_preset_changed(self):
        if getattr(self, "_bulk_updating", False):
            return
        name = self.g_preset.currentText() if hasattr(self, "g_preset") else ""
        self._apply_gearbox_preset(name)

    def _on_extra_torque_preset_changed(self):
        if getattr(self, "_bulk_updating", False):
            return
        name = self.x_preset.currentText() if hasattr(self, "x_preset") else ""
        self._apply_extra_torque_preset(name)

    def _apply_extra_torque_preset(self, name: str):
        """Apply τ_extra(ω,T) + Kt(T)/Rs(T) configuration presets.

        Presets requested:
          - Milling Moog Curve
          - Spear Moog Curve (placeholder = milling for now)
          - Annular Windings Curve (placeholder = milling for now)
          - User Defined (restore built-in defaults)
        """
        n = (name or "").strip()
        if not n:
            return

        # UI guard
        need = ("x_enable_extra", "x_enable_ktT", "x_enable_rsT", "x_temp", "x_tref", "x_kt_tc",
                "x_core_en", "x_core_cl", "x_core_exp", "x_visc_en", "x_visc_model", "x_k_c", "x_k_tr",
                "x_n_tr", "x_k_tb", "x_rpm1", "x_rpm2", "x_visc_T", "x_visc_a", "x_visc_beta", "x_smooth", "x_smooth_f",
                "x_wind_dT", "x_rs_tc")
        if not all(hasattr(self, a) for a in need):
            return

        self._bulk_updating = True
        try:
            if n.lower().startswith("user"):
                d = ExtraTorqueParams()  # defaults = current tool defaults
                self.x_enable_extra.setChecked(bool(d.extra_enabled))
                self.x_enable_ktT.setChecked(bool(d.kt_temp_enabled))
                self.x_enable_rsT.setChecked(bool(getattr(d, "rs_temp_enabled", False)))

                self.x_temp.setValue(float(d.temp_C))
                self.x_tref.setValue(float(d.temp_ref_C))
                self.x_kt_tc.setValue(100.0 * float(d.kt_temp_coeff_per_C))

                self.x_wind_dT.setValue(float(getattr(d, "winding_rise_C", 10.0)))
                self.x_rs_tc.setValue(100.0 * float(getattr(d, "rs_temp_coeff_per_C", 0.00393)))

                self.x_core_en.setChecked(bool(d.core_enabled))
                self.x_core_cl.setValue(float(d.core_cL))
                self.x_core_exp.setValue(float(getattr(d, "core_exp", 0.5)))

                self.x_visc_en.setChecked(bool(d.visc_enabled))
                self.x_visc_model.setCurrentText(
                    str(getattr(d, "visc_model", "Piecewise (Couette→Transition→Turbulent)")))
                self.x_k_c.setValue(float(getattr(d, "visc_k_couette", 0.0)))
                self.x_k_tr.setValue(float(getattr(d, "visc_k_transition", 0.0)))
                self.x_n_tr.setValue(float(getattr(d, "visc_n_transition", 1.5)))
                self.x_k_tb.setValue(float(getattr(d, "visc_k_turb", 0.0)))
                self.x_rpm1.setValue(float(getattr(d, "visc_rpm1", 500.0)))
                self.x_rpm2.setValue(float(getattr(d, "visc_rpm2", 2000.0)))

                self.x_visc_T.setCurrentText(str(getattr(d, "visc_temp_scaling", "None")))
                self.x_visc_a.setValue(100.0 * float(getattr(d, "visc_lin_coeff_per_C", 0.0)))
                self.x_visc_beta.setValue(float(getattr(d, "visc_beta_per_C", 0.0)))

                self.x_smooth.setChecked(bool(getattr(d, "smooth_transitions", True)))
                self.x_smooth_f.setValue(float(getattr(d, "smooth_frac", 0.15)))
            else:
                # For now, Spear/Annular presets are placeholders that reuse the Milling Moog curve.
                self._apply_milling_motor_extra_torque()

                # Keep the Moog curve interpretation simple/transparent:
                # - curve is at 175°C oil ambient (per sheet)
                # - do NOT bake Kt(T) scaling into the Moog preset (avoids double counting confusion)
                self.x_temp.setValue(175.0)
                self.x_tref.setValue(25.0)
                self.x_enable_ktT.setChecked(False)
                self.x_visc_T.setCurrentText("None")
        finally:
            self._bulk_updating = False

        if self.chk_auto.isChecked():
            self.update_all()

    def _apply_motor_preset(self, name: str):
        """Apply named PMSM presets by updating the UI fields.

        Mapping (per user clarification):
          - Spear motor: defaults already in code (winding data sheet)
          - Milling motor: parameters-at-25C table (lower Kt/Ke)
        """
        n = (name or "").strip()
        if not n:
            return

        def lambda_from_ke_vll_peak_per_krpm(ke_vll_peak: float, pole_pairs: int) -> float:
            # Match MotorParams.recompute_derived() internal convention
            krpm_to_rad_s = (1000.0 * 2.0 * math.pi / 60.0)
            ke_rms = float(ke_vll_peak) / math.sqrt(2.0)
            ke_per_rad = ke_rms / krpm_to_rad_s
            p = max(1, int(pole_pairs))
            return ke_per_rad * math.sqrt(2.0) / (p * math.sqrt(3.0))

        # Preset tables
        presets = {
            "Spear motor": {
                "pole_pairs": 4,
                "rs_ohm": 59.05 / 2.0,
                "ld_h": 29.28e-3 / 2.0,
                "lq_h": 29.28e-3 / 2.0,
                "kt_lbin_arms": 8.548,
                "ke_vll_peak_krpm": 82.604,
            },
            "Milling motor": {
                "pole_pairs": 4,
                "rs_ohm": 3.10 / 2.0,
                "ld_h": 2.10e-3,
                "lq_h": 2.21e-3,
                "kt_lbin_arms": 4.73,
                "ke_vll_peak_krpm": 45.74,
            },
            "Annular Motor - John": {
                "pole_pairs": 4,  # 8 poles @ 3000 rpm → 200 Hz fundamental
                "rs_ohm": 7.334 / 2.0,  # line-line (wye) → per-phase
                "ld_h": 1.692e-3,  # D-axis inductance (H)
                "lq_h": 1.983e-3,  # Q-axis inductance (H)
                "kt_lbin_arms": 6.237,  # torque constant (lb-in/Arms)
                "ke_vll_peak_krpm": 83.82,  # back-EMF (Vll_peak/krpm)
            },
            "Annular Motor - John2": {
                "pole_pairs": 4,  # 8 poles @ 3000 rpm → 200 Hz fundamental
                "rs_ohm": 9.114 / 2.0,  # line-line (wye) → per-phase
                "ld_h": 2.045e-3,  # D-axis inductance (H)
                "lq_h": 2.467e-3,  # Q-axis inductance (H)
                "kt_lbin_arms": 6.852,  # torque constant (lb-in/Arms)
                "ke_vll_peak_krpm": 93.9,  # back-EMF (Vll_peak/krpm) (fundamental)
            },
            "Annular Motor - John3": {
                "pole_pairs": 4,  # assumed same annular-motor pole-pair count as John / John2
                "rs_ohm": 3.52 / 2.0,  # line-line (wye) → per-phase = 1.76 ohm
                "ld_h": 1.044e-3,  # D-axis inductance (H)
                "lq_h": 1.192e-3,  # Q-axis inductance (H)
                "kt_lbin_arms": 3.725,  # torque constant (lb-in/Arms)
                "ke_vll_peak_krpm": 51.37,  # back-EMF (Vll_peak/krpm), using screenshot fundamental value
            },
            "User defined": {
                # "User defined" means: restore the tool's built-in defaults (same basis as startup).
                "pole_pairs": 4,
                "rs_ohm": 59.05 / 2.0,
                "ld_h": 29.28e-3 / 2.0,
                "lq_h": 29.28e-3 / 2.0,
                "kt_lbin_arms": 8.548,
                "ke_vll_peak_krpm": 82.604,
            },
        }

        if n not in presets:
            return
        d = presets[n]

        self._bulk_updating = True
        try:
            # Basis + linkage (datasheet-friendly)
            self.m_link.setChecked(True)
            self.m_mode.setCurrentText("Ke")
            self.m_kt_basis.setCurrentText("lb-in/Arms")
            self.m_ke_basis.setCurrentText("Vll_peak/krpm")

            # Electrical params (per-phase, wye-equivalent)
            self.m_pp.setValue(int(d["pole_pairs"]))
            self.m_rs.setValue(float(d["rs_ohm"]))
            self.m_ld.setValue(float(d["ld_h"]))
            self.m_lq.setValue(float(d["lq_h"]))

            # Constants
            self.m_ke.setValue(float(d["ke_vll_peak_krpm"]))
            self.m_kt.setValue(float(d["kt_lbin_arms"]))

            lam = lambda_from_ke_vll_peak_per_krpm(float(d["ke_vll_peak_krpm"]), int(d["pole_pairs"]))
            self.m_lambda.setValue(float(lam))

            # If the milling motor preset is chosen, also seed the Extra-Torque block
            # from the Moog thermal operating curve / v129 sheet (CLDF + viscous drag).
            if n == "Milling motor":
                if hasattr(self, "x_preset"):
                    self.x_preset.setCurrentText("Milling Moog Curve")
                self._apply_extra_torque_preset("Milling Moog Curve")
        finally:
            self._bulk_updating = False

        if self.chk_auto.isChecked():
            self.update_all()

    def _apply_milling_motor_extra_torque(self) -> None:
        """Seed τ_extra(ω,T) from the Moog milling motor operating-curve sheet.

        Source (from the shared sheet screenshot):
          - CLDF = 0.0049 lbf-in/(rpm)^0.5  (core loss torque)
          - Viscous drag ≈ 0.155 lbf-in at 11424.2 rpm (continuous table)

        Mapping used by this UI:
          τ_core = C_L · |ω|^exp
          τ_visc = k_couette · |ω|

        where ω is motor mechanical speed in rad/s.
        """
        if not all(hasattr(self, a) for a in ("x_core_cl", "x_core_exp", "x_visc_model", "x_k_c")):
            return

        LBF_IN_TO_NM = 0.1129848290276167
        RPM_TO_RAD_S = 2.0 * math.pi / 60.0

        # --- core-loss coefficient (convert from lbf-in/(rpm^0.5) to Nm/(rad/s)^0.5) ---
        cldf_lbf_in_per_rpm_pow = 0.0049
        exp = 0.5
        cL_nm = cldf_lbf_in_per_rpm_pow * LBF_IN_TO_NM / (RPM_TO_RAD_S ** exp)

        # --- viscous Couette coefficient (convert from lbf-in vs rpm to Nm/(rad/s)) ---
        # Use a simple through-origin slope based on the max-speed point.
        visc_lbf_in = 0.155
        rpm_ref = 11424.2
        omega_ref = rpm_ref * RPM_TO_RAD_S
        k_couette = (visc_lbf_in * LBF_IN_TO_NM) / max(1e-12, omega_ref)

        # Apply to the Extra torque block.
        self.x_enable_extra.setChecked(True)
        self.x_core_en.setChecked(True)
        self.x_core_exp.setValue(exp)
        self.x_core_cl.setValue(float(cL_nm))

        self.x_visc_en.setChecked(True)
        self.x_visc_model.setCurrentText("Couette (τ=k·ω)")
        self.x_k_c.setValue(float(k_couette))
        self.x_k_tr.setValue(0.0)
        self.x_k_tb.setValue(0.0)

    def _apply_gearbox_preset(self, name: str):
        """Apply named gearbox presets by updating the UI fields."""
        n = (name or "").strip()
        if not n:
            return

        presets = {
            "Spear gearbox": {
                "stage1": 15.0,
                "stage2": 15.0,
                "stage3": 15.0,
                "override_eta": True,
                "eta_total": 0.40,
            },
            "Milling gearbox": {
                "stage1": 217.3,
                "stage2": 15.0,
                "stage3": 1.0,
                "override_eta": True,
                "eta_total": 0.70,
            },
            "User defined": {
                # Restore the tool's built-in defaults for gearbox fields.
                "stage1": 15.0,
                "stage2": 15.0,
                "stage3": 15.0,
                "override_eta": True,
                "eta_total": 0.40,
            },
        }

        if n not in presets:
            return
        d = presets[n]

        self._bulk_updating = True
        try:
            self.g1.setValue(float(d["stage1"]))
            self.g2.setValue(float(d["stage2"]))
            self.g3.setValue(float(d["stage3"]))

            # Keep stage eff knobs at their defaults; total η is what drives feasibility.
            self.chk_eta_override.setChecked(bool(d["override_eta"]))
            self.eta_override.setValue(float(d["eta_total"]))
        finally:
            self._bulk_updating = False

        if self.chk_auto.isChecked():
            self.update_all()

    def _update_strategy_ui_enabled(self):
        """Enable/disable UI controls that are irrelevant under selected strategy."""
        if not hasattr(self, "combo_strategy"):
            return

        mode = self.combo_strategy.currentData() or "VF"

        # Preserve user's FW checkbox preference when in VF, so we can restore it.
        if not hasattr(self, "_fw_user_pref"):
            self._fw_user_pref = bool(self.chk_fw.isChecked())

        if mode == "VF":
            was_disabled = not self.chk_fw.isEnabled()
            self.chk_fw.setEnabled(True)
            if was_disabled:
                self.chk_fw.setChecked(bool(self._fw_user_pref))
            else:
                self._fw_user_pref = bool(self.chk_fw.isChecked())

            self.chk_fw_base_only.setEnabled(bool(self.chk_fw.isChecked()))
            self.fw_idmax.setEnabled(bool(self.chk_fw.isChecked()))

            for w in (self.base_f, self.base_v, self.v_boost):
                w.setEnabled(True)

            self.lbl_strategy_note.setText(
                "Baseline scheduled V/f command: surface voltage follows V = Vboost + slope·f (capped by SVPWM voltage limit). "
                "Field-weakening follows the FW checkbox."
            )
        elif mode == "MODE_A":
            if self.chk_fw.isEnabled():
                self._fw_user_pref = bool(self.chk_fw.isChecked())
            self.chk_fw.setChecked(False)
            self.chk_fw.setEnabled(False)
            self.chk_fw_base_only.setEnabled(False)
            self.fw_idmax.setEnabled(False)

            for w in (self.base_f, self.base_v, self.v_boost):
                w.setEnabled(False)

            self.lbl_strategy_note.setText(
                "Mode A uses the full inverter voltage ceiling (Vmax) as headroom (SVPWM/current-regulated style) and enforces Id = 0 (no FW). "
                "V/f schedule fields are ignored."
            )
        else:  # MODE_B
            if self.chk_fw.isEnabled():
                self._fw_user_pref = bool(self.chk_fw.isChecked())
            self.chk_fw.setChecked(True)
            self.chk_fw.setEnabled(False)
            self.chk_fw_base_only.setEnabled(True)
            self.fw_idmax.setEnabled(True)

            for w in (self.base_f, self.base_v, self.v_boost):
                w.setEnabled(False)

            self.lbl_strategy_note.setText(
                "Mode B uses the full inverter voltage ceiling (Vmax) as headroom (SVPWM/current-regulated style) and forces FW ON (Id scan). "
                "V/f schedule fields are ignored."
            )

    def _update_cable_temp_ui_enabled(self) -> None:
        """Enable/disable cable temperature model widgets and keep derived values (like 5-seg total length) consistent."""
        five_on = bool(getattr(self, "c_temp_5seg", None) and self.c_temp_5seg.isChecked())

        # Reference/alpha are used by the 5-seg R(T) model
        for w in (getattr(self, "c_temp_ref", None), getattr(self, "c_temp_alpha", None)):
            if w is not None:
                w.setEnabled(True)

        # Uniform multiplier only used when 5-seg is OFF
        if hasattr(self, "c_temp"):
            self.c_temp.setEnabled(not five_on)

        # Derating can be applied either from 5-seg (true R(T)) or from the uniform multiplier
        if hasattr(self, "c_temp_derate"):
            self.c_temp_derate.setEnabled(True)

        # 5-seg widgets
        if hasattr(self, "c_temp_5seg"):
            self.c_temp_5seg.setEnabled(True)

        if hasattr(self, "c_5seg_L") and hasattr(self, "c_5seg_T"):
            for w in list(self.c_5seg_L) + list(self.c_5seg_T):
                w.setEnabled(five_on)

        # If 5-seg is active, total cable length is derived from sum(Li)
        if five_on and hasattr(self, "c_5seg_L") and hasattr(self, "c_len"):
            Ltot = float(sum(float(w.value()) for w in self.c_5seg_L))
            if hasattr(self, "lbl_c_5seg_total"):
                self.lbl_c_5seg_total.setText(f"{Ltot:.0f} m")
            try:
                self.c_len.blockSignals(True)
                self.c_len.setEnabled(False)
                if Ltot > 0.0:
                    self.c_len.setValue(Ltot)
            finally:
                self.c_len.blockSignals(False)
        else:
            if hasattr(self, "lbl_c_5seg_total"):
                self.lbl_c_5seg_total.setText("—")
            if hasattr(self, "c_len"):
                self.c_len.setEnabled(True)

    def _apply_voltage_basis_visibility(self) -> None:
        """Show/hide the AC vs DC voltage-entry blocks based on dropdown."""
        basis = (self.v_entry_basis.currentText() if hasattr(self, 'v_entry_basis') else '').strip().upper()
        is_dc = basis.startswith('DC')
        if hasattr(self, 'w_dc_vlim'):
            self.w_dc_vlim.setVisible(is_dc)
        if hasattr(self, 'w_ac_vlim'):
            self.w_ac_vlim.setVisible(not is_dc)

    def _enforce_vf_schedule_against_voltage_limit(self, p: "SystemParams") -> None:
        """Clamp V/f schedule knobs so the commanded AC fundamental never exceeds the active voltage limit.

        We treat:
          - v_phase_rms_limit() as the hard inverter/cable limit (either user-entered AC, or derived from Vdc)
          - v_boost as an offset at very low frequency
          - base_v_phase_rms as the *additional* RMS voltage at base_freq (so Vcmd(base)=v_boost + base_v_phase_rms)

        Therefore we enforce:
          0 <= v_boost <= Vlim
          0 <= base_v_phase_rms <= (Vlim - v_boost)
        """
        vlim = max(0.0, float(p.vf.v_phase_rms_limit()))

        vb_req = float(p.vf.v_boost)
        vb = min(max(vb_req, 0.0), vlim)

        base_max = max(0.0, vlim - vb)
        bv_req = float(p.vf.base_v_phase_rms)
        bv = min(max(bv_req, 0.0), base_max)

        # If we had to clamp, update both params and UI (without feedback loops).
        def _set_spin(spin: QDoubleSpinBox, val: float) -> None:
            if abs(spin.value() - val) > 1e-9:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)

        if abs(vb - vb_req) > 1e-12:
            p.vf.v_boost = vb
            _set_spin(self.v_boost, vb)

        if abs(bv - bv_req) > 1e-12:
            p.vf.base_v_phase_rms = bv
            _set_spin(self.base_v, bv)

    def _on_any_change(self, *_):
        if self._bulk_updating:
            return

        # Strategy may override/lock some controls (FW, V/f schedule knobs)
        self._bulk_updating = True
        try:
            self._update_strategy_ui_enabled()
            self._update_cable_temp_ui_enabled()
        finally:
            self._bulk_updating = False

        if self.chk_auto.isChecked():
            self.update_all()

    def _reset_defaults(self):
        # Reset MUST restore UI widgets (otherwise UI overwrites params on next update_all)
        self.params = SystemParams()
        self._restore_ui_defaults()
        self._update_strategy_ui_enabled()
        self.update_all()

    def _sync_params_from_ui(self):
        p = self.params

        if hasattr(self, "combo_strategy"):
            p.control_strategy = self.combo_strategy.currentData() or "VF"

        # target
        p.target.out_rpm = float(self.t_out_rpm.value())
        p.target.out_torque_ftlbf = float(self.t_out_tq.value())
        p.target.torque_override_continuous = bool(self.chk_tq_override.isChecked())

        # stuck/stall + regen availability
        if hasattr(self, "chk_stuck"):
            p.stuck_mode = bool(self.chk_stuck.isChecked())
            # Force RPM=0 in stuck mode (UI + model)
            if p.stuck_mode:
                self.t_out_rpm.blockSignals(True)
                self.t_out_rpm.setValue(0.0)
                self.t_out_rpm.blockSignals(False)
                self.t_out_rpm.setEnabled(False)
            else:
                self.t_out_rpm.setEnabled(True)

        if hasattr(self, "chk_brakepath"):
            p.braking_path_available = bool(self.chk_brakepath.isChecked())
        if hasattr(self, "chk_brake_pwr"):
            p.brake_power_limit_enabled = bool(self.chk_brake_pwr.isChecked())
        if hasattr(self, "brake_pwr_kw"):
            p.brake_power_kw_max = float(self.brake_pwr_kw.value())

        # cable-aware regen constraint (regen current requires back-EMF headroom over clamp + cable drop)
        if hasattr(self, "chk_regen_cable"):
            p.regen_cable_limit_enabled = bool(self.chk_regen_cable.isChecked())
            p.regen_surface_clamp_frac = float(self.regen_clamp_frac.value())
            self.regen_clamp_frac.setEnabled(self.chk_regen_cable.isChecked())

        # direction + static load blocks
        if hasattr(self, "combo_out_dir"):
            p.out_dir = str(self.combo_out_dir.currentData() or self.combo_out_dir.currentText() or "CW")

        if hasattr(self, "chk_bha"):
            p.bha.enabled = bool(self.chk_bha.isChecked())
            if hasattr(self, "bha_tob"):
                p.bha.drilling_tob_ftlbf = float(self.bha_tob.value())
            p.bha.fric_tc_nm = ft_lbf_to_nm(float(self.bha_tc.value()))
            # B entered as ft-lbf per RPM -> convert to Nm per (rad/s)
            p.bha.fric_b_nm_per_rad_s = ft_lbf_to_nm(float(self.bha_b.value())) / (2.0 * math.pi / 60.0)
            p.bha.fric_c_nm_per_rad_s2 = 0.0

        if hasattr(self, "chk_par"):
            p.parasitic.enabled = bool(self.chk_par.isChecked())
            p.parasitic.tc_nm = ft_lbf_to_nm(float(self.par_tc.value()))
            p.parasitic.b_nm_per_rad_s = ft_lbf_to_nm(float(self.par_b.value())) / (2.0 * math.pi / 60.0)
            p.parasitic.c_nm_per_rad_s2 = 0.0

        if hasattr(self, "chk_mag"):
            p.mag_coupler.enabled = bool(self.chk_mag.isChecked())
            p.mag_coupler.t_break_nm = float(self.mag_tbreak.value())
            p.mag_coupler.t_slip_nm = float(self.mag_tslip.value())
            p.mag_coupler.slope = float(self.mag_slope.value())

        # vf
        p.vf.v_limit_type = self.v_limit_type.currentText()
        p.vf.voltage_entry_basis = self.v_entry_basis.currentText()
        p.vf.vdc_link_v = float(self.vdc_link.value())
        p.vf.modulation = self.modulation.currentText()
        p.vf.v_util = float(self.v_util.value())
        p.vf.v_limit_value = float(self.v_limit_value.value())
        p.vf.base_freq_hz = float(self.base_f.value())
        p.vf.base_v_phase_rms = float(self.base_v.value())
        p.vf.v_boost = float(self.v_boost.value())

        # fw
        p.fw.enabled = bool(self.chk_fw.isChecked())
        p.fw.apply_only_above_base = bool(self.chk_fw_base_only.isChecked())
        p.fw.id_max_arms = float(self.fw_idmax.value())

        # sine filter
        p.sine_filter.enabled = bool(self.sf_enable.isChecked())
        p.sine_filter.lf_h = float(self.sf_lf_mH.value()) * 1e-3
        p.sine_filter.rf_ohm = float(self.sf_rf_ohm.value())
        p.sine_filter.cf_f = float(self.sf_cf_uF.value()) * 1e-6
        p.sine_filter.cap_connection = self.sf_cap_conn.currentText()
        p.sine_filter.damping_topology = self.sf_damp_topo.currentText()
        p.sine_filter.rd_ohm = float(self.sf_rd_ohm.value())

        # cable
        p.cable.length_m = float(self.c_len.value())
        p.cable.r_ohm_per_m = float(self.c_rpm.value())
        p.cable.l_h_per_m = float(self.c_lpm.value())
        p.cable.temp_factor_r = float(self.c_temp.value())
        p.cable.temp_ref_C = float(self.c_temp_ref.value())
        p.cable.temp_alpha_per_C = float(self.c_temp_alpha.value())
        # 5-seg custom temperature profile inputs
        if hasattr(self, "c_temp_5seg"):
            p.cable.temp_model_5seg = bool(self.c_temp_5seg.isChecked())
        if hasattr(self, "c_temp_derate"):
            p.cable.i_limit_derate_with_temp = bool(self.c_temp_derate.isChecked())
        if hasattr(self, "c_5seg_L"):
            p.cable.temp5_seg_len_m = [float(w.value()) for w in self.c_5seg_L]
        if hasattr(self, "c_5seg_T"):
            p.cable.temp5_seg_temp_C = [float(w.value()) for w in self.c_5seg_T]

        # Enforce model exclusivity and derive overall length from segments when 5-seg is active
        if bool(getattr(p.cable, "temp_model_5seg", False)):
            # segmented model removed
            Ltot = float(sum(max(0.0, float(x)) for x in getattr(p.cable, "temp5_seg_len_m", [])[:5]))
            if Ltot > 0.0:
                p.cable.length_m = Ltot
        p.cable.wires_per_phase = 1 if self.c_wires.currentIndex() == 0 else 2
        p.cable.i_limit_basis = self.c_ilim_basis.currentText()
        p.cable.i_limit_arms = float(self.c_ilim.value())
        p.cable.l_parallel_factor = float(self.c_lpar.value())

        # motor
        p.motor.pole_pairs = int(self.m_pp.value())
        p.motor.rs_ohm = float(self.m_rs.value())
        p.motor.ld_h = float(self.m_ld.value())
        p.motor.lq_h = float(self.m_lq.value())
        p.motor.link_kt_ke = bool(self.m_link.isChecked())
        p.motor.motor_param_mode = self.m_mode.currentText()

        p.motor.kt_basis = self.m_kt_basis.currentText()
        p.motor.ke_basis = self.m_ke_basis.currentText()

        p.motor.lambda_wb = float(self.m_lambda.value())
        p.motor.set_kt_from_display(float(self.m_kt.value()))
        p.motor.set_ke_from_display(float(self.m_ke.value()))

        p.motor.recompute_derived()

        # extra torque + Kt(T)
        p.extra.extra_enabled = bool(self.x_enable_extra.isChecked())
        p.extra.kt_temp_enabled = bool(self.x_enable_ktT.isChecked())
        p.extra.temp_C = float(self.x_temp.value())
        p.extra.temp_ref_C = float(self.x_tref.value())
        p.extra.kt_temp_coeff_per_C = float(self.x_kt_tc.value()) / 100.0  # %/C -> fraction/C
        p.extra.rs_temp_enabled = bool(self.x_enable_rsT.isChecked())
        p.extra.winding_rise_C = float(self.x_wind_dT.value())
        p.extra.rs_temp_coeff_per_C = float(self.x_rs_tc.value()) / 100.0  # %/C -> fraction/C

        p.extra.core_enabled = bool(self.x_core_en.isChecked())
        p.extra.core_cL = float(self.x_core_cl.value())
        p.extra.core_exp = float(self.x_core_exp.value())

        p.extra.visc_enabled = bool(self.x_visc_en.isChecked())
        p.extra.visc_model = self.x_visc_model.currentText()
        p.extra.visc_k_couette = float(self.x_k_c.value())
        p.extra.visc_k_transition = float(self.x_k_tr.value())
        p.extra.visc_n_transition = float(self.x_n_tr.value())
        p.extra.visc_k_turb = float(self.x_k_tb.value())
        p.extra.visc_rpm1 = float(self.x_rpm1.value())
        p.extra.visc_rpm2 = float(self.x_rpm2.value())

        p.extra.visc_temp_scaling = self.x_visc_T.currentText()
        p.extra.visc_lin_coeff_per_C = float(self.x_visc_a.value()) / 100.0  # %/C -> fraction/C
        p.extra.visc_beta_per_C = float(self.x_visc_beta.value())

        p.extra.smooth_transitions = bool(self.x_smooth.isChecked())
        p.extra.smooth_frac = float(self.x_smooth_f.value())

        # gearbox
        p.gearbox.stage1 = float(self.g1.value())
        p.gearbox.stage2 = float(self.g2.value())
        p.gearbox.stage3 = float(self.g3.value())
        p.gearbox.eff1 = float(self.e1.value())
        p.gearbox.eff2 = float(self.e2.value())
        p.gearbox.eff3 = float(self.e3.value())
        p.gearbox.eta_misc = float(self.eta_misc.value())
        p.gearbox.override_total_eta = bool(self.chk_eta_override.isChecked())
        p.gearbox.eta_total_override = float(self.eta_override.value())
        p.gearbox.backdrivable = bool(self.chk_backdrive.isChecked())

        # limits
        p.limits.enforce_downhole_vphase_limit = bool(self.chk_dh_vph.isChecked())
        p.limits.downhole_v_phase_rms_limit = float(self.v_dh_vph.value())
        p.limits.enforce_downhole_vll_limit = bool(self.chk_dh_vll.isChecked())
        p.limits.downhole_vll_rms_limit = float(self.v_dh_vll.value())

        # keep FW Id max sane vs current limit (soft clamp)
        i_lim = p.cable.i_phase_limit()
        if p.fw.id_max_arms > i_lim:
            p.fw.id_max_arms = i_lim
            self.fw_idmax.blockSignals(True)
            self.fw_idmax.setValue(i_lim)
            self.fw_idmax.blockSignals(False)

        # refresh display Kt/Ke (basis-aware)
        self.m_kt.blockSignals(True)
        self.m_ke.blockSignals(True)
        self.m_kt.setValue(p.motor.kt_display())
        self.m_ke.setValue(max(0.0, p.motor.ke_display()))
        self.m_kt.blockSignals(False)
        self.m_ke.blockSignals(False)

    def update_all(self):
        try:
            # Ensure strategy note + enablement are correct even before any user interaction.
            self._update_strategy_ui_enabled()
            self._update_cable_temp_ui_enabled()
            # Ensure voltage-entry dependent widgets are in the right state at startup.
            self._apply_voltage_basis_visibility()
            self._refresh_extra_ui_enabled()
            if hasattr(self, "brake_pwr_kw") and hasattr(self, "chk_brake_pwr"):
                self.brake_pwr_kw.setEnabled(self.chk_brake_pwr.isChecked())
            self._sync_params_from_ui()
        except Exception as e:
            self.lbl_status.setText("❌")
            self.lbl_status.setStyleSheet("color:#991b1b; font-weight:600;")
            self.status.setText(f"Parameter error: {e}")
            return

        # Clamp V/f schedule knobs against the active voltage limit before building the model.
        self._enforce_vf_schedule_against_voltage_limit(self.params)

        self.model = SystemModel(self.params)

        # derived labels
        slope = self.params.vf.base_v_phase_rms / max(1e-9, self.params.vf.base_freq_hz)
        self.lbl_slope.setText(f"{slope:.3f} V/Hz (phase RMS)")

        # Voltage-basis visibility + derived AC limits readout
        self._apply_voltage_basis_visibility()
        vph_lim = self.params.vf.v_phase_rms_limit()
        vll_lim = self.params.vf.v_ll_rms_limit()
        if self.params.vf.voltage_entry_basis.strip().upper().startswith('DC'):
            self.lbl_vlimits.setText(
                f"Derived AC limit from Vdc:  VLL,rms,max = {vll_lim:.1f}  |  Vϕ,rms,max = {vph_lim:.1f}   "
                f"(Vdc={self.params.vf.vdc_link_v:.0f} V, {self.params.vf.modulation}, util={self.params.vf.v_util:.2f})"
            )
        else:
            self.lbl_vlimits.setText(
                f"Derived AC limit:  VLL,rms,max = {vll_lim:.1f}  |  Vϕ,rms,max = {vph_lim:.1f}   (from AC Vlimit entry)"
            )

        self.lbl_cable_R.setText(f"{self.params.cable.effective_r_phase():.5f} ohm")
        self.lbl_cable_L.setText(f"{self.params.cable.effective_l_phase():.7f} H")
        self.lbl_cable_Iph.setText(f"{self.params.cable.i_phase_limit():.3f} Arms (phase magnitude)")

        self.lbl_G.setText(f"{self.params.gearbox.ratio():.1f}:1")
        self.lbl_eta.setText(f"{self.params.gearbox.eff_total():.3f}")

        ke_per_rad = self.params.motor.ke_ll_rms_per_rad()
        kt_expected = math.sqrt(3.0) * ke_per_rad
        kt_expected_lbin = kt_expected / LBIN_TO_NM
        self.lbl_kt_ke_rule.setText(
            f"Canonical relationship (tool): Kt(Nm/Arms) ≈ √3 · Ke_ll_rms_per_rad.  "
            f"Ke_ll_rms_per_rad={ke_per_rad:.4f} => Kt_expected≈{kt_expected:.4f} Nm/Arms "
            f"({kt_expected_lbin:.2f} lb-in/Arms)."
        )

        # solve target (selected wiring)
        res_sel = self.model.solve_target()

        # extra torque preview (at target)
        try:
            self.lbl_extra_preview.setText(
                f"At target: T_amb={res_sel.temp_C:.1f}°C,  T_w={res_sel.winding_temp_C:.1f}°C,  Rs_eff={res_sel.rs_eff_ohm:.3f} Ω,  "
                f"Kt_eff={res_sel.kt_eff_nm_per_arms:.4f} Nm/Arms.  "
                f"τ_core={res_sel.tau_core_nm:.3f} Nm,  τ_visc={res_sel.tau_visc_nm:.3f} Nm,  "
                f"τ_extra={res_sel.tau_extra_nm:.3f} Nm.  "
                f"Iq_base={res_sel.iq_req_base_rms:.3f} Arms,  Iq_total={res_sel.iq_req_rms:.3f} Arms."
            )
        except Exception:
            self.lbl_extra_preview.setText("At target: —")

        # compare 1-wire vs 2-wire
        cab1 = CableParams(**vars(self.params.cable));
        cab1.wires_per_phase = 1
        cab2 = CableParams(**vars(self.params.cable));
        cab2.wires_per_phase = 2
        res_1 = self.model.solve_target(cable_override=cab1)
        res_2 = self.model.solve_target(cable_override=cab2)

        out_rpm_max = float(self.env_out_rpm_max.value())
        env_rpm_1, env_tq_1, env_iq_1, env_loss_1 = self.model.compute_envelope(out_rpm_max=out_rpm_max, n=200,
                                                                                cable_override=cab1)
        env_rpm_2, env_tq_2, env_iq_2, env_loss_2 = self.model.compute_envelope(out_rpm_max=out_rpm_max, n=200,
                                                                                cable_override=cab2)

        # Cache a light envelope for other panes (e.g., Animation)
        self._env_cache = (env_rpm_1, env_tq_1, env_rpm_2, env_tq_2)

        self._update_status(res_sel, res_1, res_2)
        self._plot_envelope(res_sel, (env_rpm_1, env_tq_1, env_iq_1, env_loss_1),
                            (env_rpm_2, env_tq_2, env_iq_2, env_loss_2))

        # Motor metrics along envelope for utilization + Motor tab
        # We compute values at the max-torque operating point for each RPM.
        def _env_motor_metrics(env_rpm: np.ndarray, cable: CableParams) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            out["out_rpm"] = np.array(env_rpm, dtype=float)
            out["motor_rpm"] = np.zeros_like(env_rpm, dtype=float)
            out["iq_max"] = np.zeros_like(env_rpm, dtype=float)
            out["id_best"] = np.zeros_like(env_rpm, dtype=float)
            out["i_mag"] = np.zeros_like(env_rpm, dtype=float)
            out["f_e"] = np.zeros_like(env_rpm, dtype=float)
            out["vll_motor"] = np.zeros_like(env_rpm, dtype=float)
            out["emf_ll"] = np.zeros_like(env_rpm, dtype=float)
            out["p_cu"] = np.zeros_like(env_rpm, dtype=float)
            out["i_limit"] = np.full_like(env_rpm, float(cable.i_phase_limit()), dtype=float)

            # Torque/power reporting
            out["t_motor_em_nm"] = np.zeros_like(env_rpm, dtype=float)  # Kt*Iq
            out["t_motor_use_nm"] = np.zeros_like(env_rpm, dtype=float)  # after τ_extra
            out["t_gb_in_nm"] = np.zeros_like(env_rpm, dtype=float)  # after mag coupler cap
            out["t_out_ftlbf"] = np.zeros_like(env_rpm, dtype=float)  # after gearbox + parasitics
            out["p_out_w"] = np.zeros_like(env_rpm, dtype=float)

            G_local = float(self.params.gearbox.ratio())
            eta_gb = float(self.params.gearbox.eff_total())
            rs_eff = float(self.model.rs_effective_ohm())
            ke_vll_per_krpm = float(self.params.motor.ke_vll_rms_per_krpm)
            kt_eff = float(self.model.kt_effective_nm_per_arms())

            for i, out_rpm in enumerate(env_rpm):
                out_rpm_mag = float(abs(float(out_rpm)))
                motor_rpm = out_rpm_mag * G_local

                iq_max, id_best, im_best, f_e, _, _, _ = self.model.max_iq_given_limits(motor_rpm, cable)

                vph_motor = self.model.motor_voltage_required_phase_rms(motor_rpm, float(id_best), float(iq_max))
                vll_motor = math.sqrt(3.0) * float(vph_motor)

                emf_ll = ke_vll_per_krpm * (motor_rpm / 1000.0)
                p_cu = 3.0 * (float(im_best) ** 2) * rs_eff

                # Torque chain (consistent with compute_envelope())
                omega_m = rpm_to_rad_s(motor_rpm)
                tau_extra, _, _ = self.model.tau_extra_nm(omega_m)

                t_em = kt_eff * float(iq_max)
                t_use = max(0.0, float(t_em) - float(tau_extra))

                t_gb_in_cap, _ = self.model._mag_forward_transmitted_to_gb_nm(float(t_use))

                t_out_raw_nm = float(t_gb_in_cap) * G_local * eta_gb
                omega_out = rpm_to_rad_s(out_rpm_mag) * self.model.out_dir_sign()
                tau_par = self.model._rot_loss_torque_nm(omega_out, self.params.parasitic)
                t_out_cap_nm = max(0.0, float(t_out_raw_nm) - abs(float(tau_par)))

                p_out_w = float(t_out_cap_nm) * float(abs(rpm_to_rad_s(out_rpm_mag)))

                out["motor_rpm"][i] = motor_rpm
                out["iq_max"][i] = float(iq_max)
                out["id_best"][i] = float(id_best)
                out["i_mag"][i] = float(im_best)
                out["f_e"][i] = float(f_e)
                out["vll_motor"][i] = float(vll_motor)
                out["emf_ll"][i] = float(emf_ll)
                out["p_cu"][i] = float(p_cu)

                out["t_motor_em_nm"][i] = float(t_em)
                out["t_motor_use_nm"][i] = float(t_use)
                out["t_gb_in_nm"][i] = float(t_gb_in_cap)
                out["t_out_ftlbf"][i] = float(nm_to_ft_lbf(t_out_cap_nm))
                out["p_out_w"][i] = float(p_out_w)

            return out

        mtr_1 = _env_motor_metrics(env_rpm_1, cab1)
        mtr_2 = _env_motor_metrics(env_rpm_2, cab2)

        vll_1 = mtr_1["vll_motor"]
        vll_2 = mtr_2["vll_motor"]

        # Cache for on-demand Motor tab rendering
        self._last_motor_metrics = (mtr_1, mtr_2, res_1, res_2)
        self._plot_band_torque(env_rpm_1, env_tq_1, env_rpm_2, env_tq_2, vll_1, vll_2)
        # Motor ops tab (only render when visible)
        try:
            if hasattr(self, "motor_pane") and (self.tabs.currentWidget() == self.motor_pane):
                self._plot_motor_ops()
        except Exception:
            pass

        # Moog curve comparison tab (only render when visible)
        try:
            if hasattr(self, "moog_pane") and (self.tabs.currentWidget() == self.moog_pane):
                self._plot_moog_curves()
        except Exception:
            pass

        # Decision tables (only render when tab is visible)
        try:
            if hasattr(self, "tables_pane") and (self.tabs.currentWidget() == self.tables_pane):
                self._plot_tables()
        except Exception:
            pass

        # Stonehouse reference presets (only compute when that tab is visible)
        try:
            if hasattr(self, "preset_pane") and (self.tabs.currentWidget() == self.preset_pane):
                self._plot_stonehouse_presets()
                self._sync_quadrant_yaxis()
                try:
                    self.preset_pane.enable_interactivity()
                except Exception:
                    pass

        except Exception:
            # Never block normal updates if the reference pane errors
            pass

        # Direction risk (only render when tab is visible)
        try:
            if hasattr(self, "dirrisk_pane") and (self.tabs.currentWidget() == self.dirrisk_pane):
                self._plot_direction_risk(res_sel)
        except Exception:
            pass

        # Animation tab (keep it in sync when visible)
        try:
            if hasattr(self, "anim_pane") and (self.tabs.currentWidget() == self.anim_pane):
                self.anim_pane.sync_from_model(self.model, self.params, env_cache=getattr(self, "_env_cache", None))
        except Exception:
            pass

        if self.chk_sweeps.isChecked():
            self._plot_sweeps()

        # Architecture diagram (always updated; lightweight)
        # Sync y-axis limits across all 4Q torque plots (Envelope + Bands + Presets)
        try:
            self._sync_quadrant_yaxis()
        except Exception:
            pass

        self._plot_architecture()

        self.lbl_status.setText("✅ Up-to-date")
        self.lbl_status.setStyleSheet("color:#065f46; font-weight:600;")

    def _on_sweeps_toggle(self, *_):
        self._apply_sweeps_tab_enabled()
        # If user just enabled sweeps, refresh them once (especially if Auto update is ON)
        if self.chk_sweeps.isChecked() and self.chk_auto.isChecked() and (not self._bulk_updating):
            self.update_all()

    def _plot_motor_ops(self):
        """Render the Motor tab using cached envelope metrics (fast)."""
        if not hasattr(self, "motor_pane"):
            return
        if not hasattr(self, "_last_motor_metrics"):
            return
        try:
            m1, m2, res_1, res_2 = self._last_motor_metrics
            self.motor_pane.render(m1, m2, res_1, res_2, self.params)
        except Exception:
            pass

    def _plot_moog_curves(self):
        """Render the Moog Curves tab (vendor curves vs current τ_extra model)."""
        if not hasattr(self, "moog_pane"):
            return
        preset = self.x_preset.currentText() if hasattr(self, "x_preset") else "Milling Moog Curve"
        try:
            self.moog_pane.render(self.model, self.params, preset)
        except Exception:
            pass

    def _on_main_tab_changed(self, idx: int):
        """Update lightweight reference panes when user switches tabs."""
        try:
            w = self.tabs.widget(int(idx))
        except Exception:
            return

        # Render Stonehouse reference pane on-demand when it becomes visible.
        if hasattr(self, "preset_pane") and (w == self.preset_pane):
            try:
                self._plot_stonehouse_presets()
                self._sync_quadrant_yaxis()
                try:
                    self.preset_pane.enable_interactivity()
                except Exception:
                    pass
            except Exception:
                pass

        # Render decision tables pane on-demand.
        if hasattr(self, "tables_pane") and (w == self.tables_pane):
            try:
                self._plot_tables()
            except Exception:
                pass

        # Render Motor ops pane on-demand.
        if hasattr(self, "motor_pane") and (w == self.motor_pane):
            try:
                self._plot_motor_ops()
            except Exception:
                pass

        # Render Moog curve comparison pane on-demand.
        if hasattr(self, "moog_pane") and (w == self.moog_pane):
            try:
                self._plot_moog_curves()
            except Exception:
                pass

        # Keep Animation pane context fresh when it becomes visible.
        if hasattr(self, "anim_pane") and (w == self.anim_pane):
            try:
                self.anim_pane.sync_from_model(self.model, self.params, env_cache=getattr(self, "_env_cache", None))
            except Exception:
                pass

    def _apply_sweeps_tab_enabled(self):
        """Enable/disable Sweeps tab to avoid slow auto updates when disabled."""
        enabled = bool(self.chk_sweeps.isChecked())
        idx = self.tabs.indexOf(self.sweep_tabs)
        if idx >= 0:
            self.tabs.setTabEnabled(idx, enabled)
            if (not enabled) and (self.tabs.currentIndex() == idx):
                self.tabs.setCurrentIndex(0)

    def _state_widgets_for_defaults(self):
        """All widgets whose values should be restored by Reset."""
        ws = [
            # top toggles
            self.chk_auto, self.chk_sweeps,
            # target
            self.t_out_rpm, self.t_out_tq, self.env_out_rpm_max, self.chk_tq_override,
            self.chk_tq_override,
            # vf/fw
            self.v_limit_type, self.v_limit_value, self.v_entry_basis, self.vdc_link, self.modulation, self.v_util,
            self.base_f, self.base_v, self.v_boost,
            self.v_entry_basis,
            self.vdc_link,
            self.modulation,
            self.v_util,
            self.chk_fw, self.chk_fw_base_only, self.fw_idmax,
            # cable            self.sf_enable, self.sf_lf_mH, self.sf_rf_ohm, self.sf_cf_uF, self.sf_cap_conn, self.sf_damp_topo, self.sf_rd_ohm,
            # cable
            self.c_len, self.c_rpm, self.c_lpm, self.c_temp, self.c_wires, self.c_ilim_basis, self.c_ilim, self.c_lpar,
            self.c_temp_ref, self.c_temp_alpha, self.c_temp_5seg, self.c_temp_derate, *getattr(self, "c_5seg_L", []),
            *getattr(self, "c_5seg_T", []),
            self.c_len_band,
            # motor
            getattr(self, "m_preset", None),
            self.m_pp, self.m_rs, self.m_ld, self.m_lq, self.m_link, self.m_mode, self.m_kt_basis, self.m_ke_basis,
            self.m_lambda, self.m_kt, self.m_ke,
            # extra torque + Kt(T)
            self.x_enable_extra, self.x_enable_ktT, self.x_enable_rsT, self.x_temp, self.x_tref, self.x_kt_tc,
            self.x_wind_dT, self.x_rs_tc,
            self.x_core_en, self.x_core_cl, self.x_core_exp,
            self.x_visc_en, self.x_visc_model, self.x_k_c, self.x_k_tr, self.x_n_tr, self.x_k_tb,
            self.x_rpm1, self.x_rpm2, self.x_visc_T, self.x_visc_a, self.x_visc_beta,
            self.x_smooth, self.x_smooth_f,
            # gearbox
            getattr(self, "g_preset", None),
            self.g1, self.g2, self.g3, self.e1, self.e2, self.e3, self.eta_misc, self.chk_eta_override,
            self.eta_override, self.eta_band_pu,
            # limits
            self.chk_dh_vph, self.v_dh_vph, self.chk_dh_vll, self.v_dh_vll, self.vll_band_vrms,
            # sweeps controls
            self.sw_ratio_min, self.sw_ratio_max, self.sw_points,
            self.sw_ke_min, self.sw_ke_max, self.sw_len_min, self.sw_len_max,
            self.sw_inv_vlim_min, self.sw_inv_vlim_max, self.sw_basef_min, self.sw_basef_max, self.sw_basev_min,
            self.sw_basev_max, self.sw_vboost_min, self.sw_vboost_max,
            self.sw_pp_min, self.sw_pp_max, self.sw_pp_hold,
            self.sw_dh_vph_min, self.sw_dh_vph_max, self.sw_dh_vll_min, self.sw_dh_vll_max,
            self.chk_fast_sweeps,
            self.combo_strategy,
        ]
        return [w for w in ws if w is not None]

    def _capture_ui_defaults(self):
        """Snapshot initial widget states so Reset restores *actual* defaults."""
        self._ui_defaults = []
        for w in self._state_widgets_for_defaults():
            if isinstance(w, QDoubleSpinBox):
                self._ui_defaults.append((w, float(w.value())))
            elif isinstance(w, QSpinBox):
                self._ui_defaults.append((w, int(w.value())))
            elif isinstance(w, QComboBox):
                self._ui_defaults.append((w, str(w.currentText())))
            elif isinstance(w, QCheckBox):
                self._ui_defaults.append((w, bool(w.isChecked())))

    def _restore_ui_defaults(self):
        """Restore widgets to the snapshot defaults."""
        if not hasattr(self, "_ui_defaults"):
            return
        self._bulk_updating = True
        try:
            for w, val in self._ui_defaults:
                if isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QComboBox):
                    # Prefer setCurrentText (stable across list ordering)
                    w.setCurrentText(str(val))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
        finally:
            self._bulk_updating = False
        self._apply_sweeps_tab_enabled()

    def _refresh_extra_ui_enabled(self) -> None:
        """Enable/disable extra-torque UI elements based on the selected options."""
        extra_on = bool(self.x_enable_extra.isChecked())
        ktT_on = bool(self.x_enable_ktT.isChecked())

        # Kt(T)
        self.x_kt_tc.setEnabled(ktT_on)

        # Rs(T)
        rsT_on = bool(self.x_enable_rsT.isChecked())
        self.x_wind_dT.setEnabled(rsT_on)
        self.x_rs_tc.setEnabled(rsT_on)

        # Everything else is gated by extra torque enable
        for w in [
            self.x_core_en, self.x_core_cl, self.x_core_exp,
            self.x_visc_en, self.x_visc_model,
            self.x_k_c, self.x_k_tr, self.x_n_tr, self.x_k_tb,
            self.x_rpm1, self.x_rpm2,
            self.x_visc_T, self.x_visc_a, self.x_visc_beta,
            self.x_smooth, self.x_smooth_f,
        ]:
            w.setEnabled(extra_on)

        core_on = extra_on and bool(self.x_core_en.isChecked())
        for w in [self.x_core_cl, self.x_core_exp]:
            w.setEnabled(core_on)

        visc_on = extra_on and bool(self.x_visc_en.isChecked())
        for w in [
            self.x_visc_model, self.x_k_c, self.x_k_tr, self.x_n_tr, self.x_k_tb,
            self.x_rpm1, self.x_rpm2,
            self.x_visc_T, self.x_visc_a, self.x_visc_beta,
            self.x_smooth, self.x_smooth_f,
        ]:
            w.setEnabled(visc_on)

        # Per-model parameter relevance
        model = self.x_visc_model.currentText()
        if visc_on:
            is_couette = model.startswith("Couette")
            is_trans = model.startswith("Transition")
            is_turb = model.startswith("Turbulent")
            is_piece = model.startswith("Piecewise")

            self.x_k_c.setEnabled(is_couette or is_piece)
            self.x_k_tr.setEnabled(is_trans or is_piece)
            self.x_n_tr.setEnabled(is_trans or is_piece)
            self.x_k_tb.setEnabled(is_turb or is_piece)
            self.x_rpm1.setEnabled(is_piece)
            self.x_rpm2.setEnabled(is_piece)
            self.x_smooth.setEnabled(is_piece)
            self.x_smooth_f.setEnabled(is_piece and bool(self.x_smooth.isChecked()))

            # viscous temperature scaling
            tmode = self.x_visc_T.currentText()
            self.x_visc_a.setEnabled(tmode.startswith("Linear"))
            self.x_visc_beta.setEnabled(tmode.startswith("Exponential"))
        else:
            self.x_smooth_f.setEnabled(False)

    def _update_status(self, res_sel: SolveResult, res_1: SolveResult, res_2: SolveResult):
        import math

        p = self.params
        sel_wires = getattr(p.cable, "wires_per_phase", 1)
        res = res_1 if sel_wires == 1 else res_2

        # -----------------------------
        # Helpers (safe formatting + safe get)
        # -----------------------------
        FTLBF_TO_NM = 1.3558179483314004
        RPM_TO_RADPS = 2.0 * math.pi / 60.0

        def _sg(obj, name, default=None):
            try:
                return getattr(obj, name)
            except Exception:
                return default

        def _fmt(x, nd=2, unit=""):
            if x is None:
                return "—"
            try:
                if isinstance(x, (float, int)) and (math.isnan(x) or math.isinf(x)):
                    return "—"
            except Exception:
                pass
            try:
                return f"{x:.{nd}f}{unit}"
            except Exception:
                return f"{x}{unit}"

        def _pct_used(used, limit):
            if used is None or limit is None:
                return None
            try:
                if limit <= 0:
                    return None
                return 100.0 * float(used) / float(limit)
            except Exception:
                return None

        def _min_key(pairs):
            """pairs = [(label, margin_value, used, limit)]"""
            best = None
            for it in pairs:
                label, margin, used, limit = it
                if margin is None:
                    continue
                if best is None or margin < best[1]:
                    best = it
            return best

        # -----------------------------
        # Basic derived values
        # -----------------------------
        ll_surface_cmd = res.v_surface_cmd * math.sqrt(3.0)
        ll_inverter_req = res.v_inverter_req * math.sqrt(3.0)
        ll_motor_req = res.v_motor_req * math.sqrt(3.0)

        out_rpm = float(res.out_rpm_cmd)
        omega_out = out_rpm * RPM_TO_RADPS
        motor_rpm = float(res.motor_rpm)
        omega_m = motor_rpm * RPM_TO_RADPS

        s = 1.0 if str(p.out_dir).upper().startswith("CW") else -1.0  # sign convention for output direction

        # Signed "required" output torque (best-effort from split drive/brake requirements)
        # (drive req is positive magnitude, brake req is positive magnitude)
        tau_out_req_ftlbf_signed = s * (float(res.out_drive_torque_req_ftlbf) - float(res.out_brake_torque_req_ftlbf))
        tau_out_req_nm_signed = tau_out_req_ftlbf_signed * FTLBF_TO_NM
        p_out_w = tau_out_req_nm_signed * omega_out

        # TOB assist vs resist relative to commanded direction (TOB is negative/CCW if bit rotates CW)
        tob_mode = "—"
        try:
            tob_mode = "RESIST" if (res.tob_reaction_ftlbf * s) < 0 else (
                "ASSIST" if abs(res.tob_reaction_ftlbf) > 1e-9 else "—"
            )
        except Exception:
            pass

        # Quadrant / power-flow (simple)
        if abs(out_rpm) < 1e-9:
            quad = "STATIC (ω≈0)"
        else:
            quad = "MOTORING (P_out>0)" if p_out_w > 0 else "BRAKING (P_out<0)"

        # Electrical frequency estimates (best-effort)
        pole_pairs = _sg(_sg(p, "motor", p), "pole_pairs", None)
        if pole_pairs is None:
            pole_pairs = _sg(_sg(p, "motor", p), "p", None)
        f_e = None
        try:
            if pole_pairs is not None:
                f_e = float(pole_pairs) * motor_rpm / 60.0
        except Exception:
            f_e = None

        # -----------------------------
        # Margins + dominant limiter
        # -----------------------------
        I_used = _sg(res, "i_mag_used_rms", None)
        I_lim = _sg(res, "i_limit_phase_mag", None)
        I_margin = (I_lim - I_used) if (I_used is not None and I_lim is not None) else None

        Vsurf_cmd = _sg(res, "v_surface_cmd", None)
        Vsurf_lim = _sg(res, "v_surface_limit", None)
        Vsurf_margin_cmd = (Vsurf_lim - Vsurf_cmd) if (Vsurf_cmd is not None and Vsurf_lim is not None) else None

        Vinv_req = _sg(res, "v_inverter_req", None)
        Vsurf_margin_req = (Vsurf_lim - Vinv_req) if (Vinv_req is not None and Vsurf_lim is not None) else None

        Vdh_ph_lim = _sg(res, "v_downhole_phase_limit", None)
        Vdh_ll_lim = _sg(res, "v_downhole_ll_limit", None)

        Vmotor_req = _sg(res, "v_motor_req", None)
        Vdh_ph_margin = (Vdh_ph_lim - Vmotor_req) if (Vdh_ph_lim is not None and Vmotor_req is not None) else None
        Vdh_ll_margin = (Vdh_ll_lim - ll_motor_req) if (Vdh_ll_lim is not None and ll_motor_req is not None) else None

        # Compose candidate constraints (use whichever surface margin is tighter)
        constraints = []
        constraints.append(("Current |I|", I_margin, I_used, I_lim))
        # surface: track both cmd and inverter-req; take the worse for dominance decision
        if Vsurf_margin_cmd is not None:
            constraints.append(("Surface Vph (cmd)", Vsurf_margin_cmd, Vsurf_cmd, Vsurf_lim))
        if Vsurf_margin_req is not None:
            constraints.append(("Surface Vph (inv-req)", Vsurf_margin_req, Vinv_req, Vsurf_lim))
        if Vdh_ph_margin is not None:
            constraints.append(("Downhole Vph (motor)", Vdh_ph_margin, Vmotor_req, Vdh_ph_lim))
        if Vdh_ll_margin is not None:
            constraints.append(("Downhole VLL (contact)", Vdh_ll_margin, ll_motor_req, Vdh_ll_lim))

        dominant = _min_key(constraints)

        def _constraint_line(label, margin, used, limit, unit):
            pct = _pct_used(used, limit)
            pct_s = f"{pct:.1f}%" if pct is not None else "—"
            m = _fmt(margin, 2, unit)
            u = _fmt(used, 2, unit)
            l = _fmt(limit, 2, unit)
            flag = " ⛔" if (margin is not None and margin < 0) else ""
            return f"• {label}: used={u} / lim={l}  (util={pct_s}, headroom={m}){flag}"

        # -----------------------------
        # Optional EMF headroom proof (best-effort)
        # -----------------------------
        # Try to compute no-load EMF from available motor Ke information
        # We prefer a precomputed SolveResult field if present; else compute from params if we can.
        e_ll_rms_est = _sg(res, "e_ll_emf_rms", None)
        if e_ll_rms_est is None:
            ke_ll_rms_per_krpm = _sg(_sg(p, "motor", p), "ke_ll_rms_per_krpm", None)
            if ke_ll_rms_per_krpm is None:
                ke_ll_rms_per_krpm = _sg(_sg(p, "motor", p), "ke_vll_rms_per_krpm", None)
            if ke_ll_rms_per_krpm is None:
                # Sometimes stored as peak; if so, convert to RMS.
                ke_ll_peak_per_krpm = _sg(_sg(p, "motor", p), "ke_ll_peak_per_krpm", None)
                if ke_ll_peak_per_krpm is None:
                    ke_ll_peak_per_krpm = _sg(_sg(p, "motor", p), "ke_vll_peak_per_krpm", None)
                if ke_ll_peak_per_krpm is not None:
                    try:
                        ke_ll_rms_per_krpm = float(ke_ll_peak_per_krpm) / math.sqrt(2.0)
                    except Exception:
                        ke_ll_rms_per_krpm = None
            if ke_ll_rms_per_krpm is not None:
                try:
                    e_ll_rms_est = float(ke_ll_rms_per_krpm) * (motor_rpm / 1000.0)
                except Exception:
                    e_ll_rms_est = None

        emf_headroom_ll = None
        if (e_ll_rms_est is not None) and (Vdh_ll_lim is not None):
            try:
                emf_headroom_ll = float(Vdh_ll_lim) - float(e_ll_rms_est)
            except Exception:
                emf_headroom_ll = None

        # -----------------------------
        # Regen feasibility (numerical) – only best-effort
        # -----------------------------
        braking_path = bool(getattr(p, "braking_path_available", True))
        regen_limit_on = bool(getattr(p, "regen_cable_limit_enabled", True))
        clamp_frac = float(getattr(p, "regen_surface_clamp_frac", 1.0))
        backdrivable = bool(getattr(_sg(p, "gearbox", p), "backdrivable", True))

        # Clamp voltage (phase RMS) best-effort:
        # If clamp is modeled as a fraction of available surface phase RMS, use that.
        v_clamp_ph = None
        if Vsurf_lim is not None:
            v_clamp_ph = clamp_frac * float(Vsurf_lim)

        # Phase EMF estimate (from ll estimate) best-effort: Vph = Vll / sqrt(3)
        e_ph_rms_est = None
        if e_ll_rms_est is not None:
            try:
                e_ph_rms_est = float(e_ll_rms_est) / math.sqrt(3.0)
            except Exception:
                e_ph_rms_est = None

        # Cable series impedance magnitude at operating fe (best-effort from params)
        z_series_mag = None
        try:
            R_phase = _sg(_sg(p, "cable", p), "r_phase_ohm", None)
            L_phase = _sg(_sg(p, "cable", p), "l_phase_h", None)
            if R_phase is None:
                R_phase = _sg(_sg(p, "cable", p), "R_phase", None)
            if L_phase is None:
                L_phase = _sg(_sg(p, "cable", p), "L_phase", None)
            if (R_phase is not None) and (f_e is not None):
                x = 0.0
                if L_phase is not None:
                    x = 2.0 * math.pi * float(f_e) * float(L_phase)
                z_series_mag = math.sqrt(float(R_phase) ** 2 + float(x) ** 2)
        except Exception:
            z_series_mag = None

        i_regen_max_est = None
        if braking_path and backdrivable and regen_limit_on and (e_ph_rms_est is not None) and (
                v_clamp_ph is not None) and (z_series_mag is not None):
            try:
                num = float(e_ph_rms_est) - float(v_clamp_ph)
                i_regen_max_est = max(0.0, num / float(z_series_mag)) if z_series_mag > 0 else 0.0
                if I_lim is not None:
                    i_regen_max_est = min(i_regen_max_est, float(I_lim))
            except Exception:
                i_regen_max_est = None

        # -----------------------------
        # Build status text
        # -----------------------------
        lines = []
        lines.append(
            "✅ PASS — UI target feasible" if res.feasible else "❌ FAIL — UI target infeasible (binding constraint hit)")
        lines.append("")
        lines.append(f"Command: {p.out_dir} {res.out_rpm_cmd:.3f} rpm  (CW=+, CCW=-)")
        lines.append(f"Quadrant: {quad}  |  P_out≈{_fmt(p_out_w / 1000.0, 3, ' kW')}")
        lines.append(f"Braking path available: {'YES' if braking_path else 'NO'}")
        lines.append(f"Regen cable-limit: {'ON' if regen_limit_on else 'OFF'} (clamp frac {clamp_frac:.2f})")
        lines.append(f"Gearbox backdrivable: {'YES' if backdrivable else 'NO'}")
        if bool(getattr(p, "brake_power_limit_enabled", False)):
            lines.append(f"Surface brake power limit: {float(getattr(p, 'brake_power_kw_max', 0.0)):.1f} kW")

        if bool(getattr(p, "stuck_mode", False)):
            lines.append(
                f"Mode: STUCK/STALL (RPM forced 0). Required output torque: {p.target.out_torque_ftlbf:.0f} ft-lbf")
        else:
            lines.append(
                f"Mode: CONTINUOUS ROTATION. UI torque target: {p.target.out_torque_ftlbf:.0f} ft-lbf (override={'ON' if bool(getattr(p.target, 'torque_override_continuous', True)) else 'OFF'})")
            if p.bha.enabled:
                lines.append(
                    f"Drilling TOB (bit CW → TOB CCW): {abs(getattr(p.bha, 'drilling_tob_ftlbf', 0.0)):.0f} ft-lbf")

        lines.append(
            f"Required @ output: drive={res.out_drive_torque_req_ftlbf:.0f} ft-lbf, "
            f"brake={res.out_brake_torque_req_ftlbf:.0f} ft-lbf  |  TOB is {tob_mode}ing"
        )
        lines.append(
            f"  Components on output shaft: τ_TOB={res.tob_reaction_ftlbf:.0f}, τ_BHA_fric={res.bha_friction_ftlbf:.0f}, "
            f"τ_parasitic={res.parasitic_ftlbf:.0f} ft-lbf"
        )

        lines.append(
            f"Gear: G={res.gear_ratio:.1f}, η={res.gear_eff:.3f} (override={'ON' if p.gearbox.override_total_eta else 'OFF'})")
        lines.append(
            f"Motor: {res.motor_rpm:.0f} rpm  (extra={'ON' if p.extra.extra_enabled else 'OFF'}, Kt(T)={'ON' if p.extra.kt_temp_enabled else 'OFF'})")
        if f_e is not None:
            lines.append(f"Electrical frequency: f_e≈{_fmt(f_e, 2, ' Hz')}  (pole-pairs={pole_pairs})")

        lines.append(f"  Motor load torque (pre-extra, includes coupler if enabled): {res.motor_torque_nm:.3f} N·m" + (
            "  [MAG SLIP RISK]" if res.mag_slipping else ""))
        lines.append(
            f"  τ_core={res.tau_core_nm:.3f}, τ_visc={res.tau_visc_nm:.3f}, τ_extra={res.tau_extra_nm:.3f} N·m")
        lines.append(f"  Total torque for Iq mapping: {res.motor_torque_total_nm:.3f} N·m")
        lines.append(f"  Kt_eff(T): {res.kt_eff_nm_per_arms:.6f} N·m/Arms  at T={res.temp_C:.1f} °C")

        lines.append(
            f"Iq (phase RMS): Iq_base={res.iq_req_base_rms:.3f}, Iq_total={res.iq_req_rms:.3f}, Iq_max={res.iq_max_rms:.3f}")
        lines.append(
            f"Current magnitude: |I|_used={res.i_mag_used_rms:.3f} Arms, |I|_limit={res.i_limit_phase_mag:.3f} Arms  (basis: {p.cable.i_limit_basis})")
        lines.append(f"FW: {'ON' if p.fw.enabled else 'OFF'} (Id_used={res.id_used_rms:.3f} Arms)")

        # Voltage budget
        lines.append("")
        lines.append(
            f"Voltage budget (phase RMS): Vcmd(surface)={res.v_surface_cmd:.1f} (≈{ll_surface_cmd:.1f} Vll), "
            f"Vreq(inv)={res.v_inverter_req:.1f} (≈{ll_inverter_req:.1f} Vll), "
            f"Vdrop(filter)={res.v_filter_drop:.1f}, Vnode={res.v_node_req:.1f}, "
            f"Vmotor_req={res.v_motor_req:.1f} (≈{ll_motor_req:.1f} Vll), Vdrop(cable)={res.v_cable_drop:.1f}"
        )
        lines.append(f"Inverter phase limit: {res.v_surface_limit:.1f} Vrms")
        if res.v_downhole_phase_limit is not None:
            lines.append(f"Downhole motor Vphase limit: {res.v_downhole_phase_limit:.1f} Vrms")
        if res.v_downhole_ll_limit is not None:
            lines.append(f"Downhole contact-block Vll limit: {res.v_downhole_ll_limit:.1f} Vrms")

        # Sine filter loading
        lines.append(f"Cable copper loss @ reported point: {res.p_cable_loss_w:.1f} W")
        if (res.i_filter_cap_rms > 1e-9) or (res.v_filter_drop > 1e-9):
            lines.append(
                f"Sine-filter loading (phase RMS): I_load={res.i_mag_used_rms:.3f} Arms, "
                f"I_inv≈{res.i_inverter_rms:.3f} Arms, I_shunt≈{res.i_filter_cap_rms:.3f} Arms, "
                f"Vdrop(filter)≈{res.v_filter_drop:.2f} Vrms"
            )

        # Constraint margins + dominant limiter
        lines.append("")
        lines.append("Constraint margins (headroom):")
        lines.append(_constraint_line("Current |I|", I_margin, I_used, I_lim, " A"))
        # surface: show both cmd + inv-req if available
        if (Vsurf_margin_cmd is not None) and (Vsurf_cmd is not None) and (Vsurf_lim is not None):
            lines.append(_constraint_line("Surface Vph (cmd)", Vsurf_margin_cmd, Vsurf_cmd, Vsurf_lim, " V"))
        if (Vsurf_margin_req is not None) and (Vinv_req is not None) and (Vsurf_lim is not None):
            lines.append(_constraint_line("Surface Vph (inv-req)", Vsurf_margin_req, Vinv_req, Vsurf_lim, " V"))
        if (Vdh_ph_margin is not None) and (Vdh_ph_lim is not None) and (Vmotor_req is not None):
            lines.append(_constraint_line("Downhole Vph (motor)", Vdh_ph_margin, Vmotor_req, Vdh_ph_lim, " V"))
        if (Vdh_ll_margin is not None) and (Vdh_ll_lim is not None):
            lines.append(_constraint_line("Downhole VLL (contact)", Vdh_ll_margin, ll_motor_req, Vdh_ll_lim, " V"))

        if dominant is not None:
            dom_label, dom_margin, dom_used, dom_lim = dominant
            dom_pct = _pct_used(dom_used, dom_lim)
            dom_pct_s = f"{dom_pct:.1f}%" if dom_pct is not None else "—"
            lines.append(f"LIMITING constraint: {dom_label}  (util={dom_pct_s}, headroom={_fmt(dom_margin, 2, '')})")

        # EMF headroom (review-friendly proof)
        if e_ll_rms_est is not None:
            lines.append("")
            lines.append("EMF / speed ceiling check (best-effort):")
            lines.append(f"• Estimated no-load EMF: E_ll,rms≈{_fmt(e_ll_rms_est, 1, ' V')}")
            if Vdh_ll_lim is not None:
                lines.append(f"• Contact-block headroom vs EMF: Vll_limit−E_ll≈{_fmt(emf_headroom_ll, 1, ' V')}")

        # Regen numerical feasibility (best-effort)
        if braking_path or backdrivable:
            lines.append("")
            lines.append("Regen feasibility (best-effort):")
            lines.append(
                f"• Preconditions: backdrivable={'YES' if backdrivable else 'NO'}, braking_path={'YES' if braking_path else 'NO'}, cable_limit={'ON' if regen_limit_on else 'OFF'}")
            if (e_ph_rms_est is not None) and (v_clamp_ph is not None):
                lines.append(
                    f"• EMF vs clamp (phase RMS): E_ph≈{_fmt(e_ph_rms_est, 2, ' V')}, V_clamp≈{_fmt(v_clamp_ph, 2, ' V')}")
            if z_series_mag is not None and f_e is not None:
                lines.append(f"• Series |Z| at f_e≈{_fmt(f_e, 2, ' Hz')}: |Z_series|≈{_fmt(z_series_mag, 3, ' Ω')}")
            if i_regen_max_est is not None:
                lines.append(f"• I_regen,max≈{_fmt(i_regen_max_est, 3, ' Arms')} (cable-limited)")

        # Design hints (existing)
        lines.append("")
        lines.append("Motor design hints for this target (given cable I limit):")
        lines.append(
            f"• Minimum Kt required (best case, incl τ_extra): {res.kt_required_min:.4f} Nm/Arms "
            f"({res.kt_required_min / LBIN_TO_NM:.2f} lb-in/Arms)"
        )
        if math.isfinite(res.ke_required_max_vll_krpm):
            lines.append(
                f"• Approx max Ke w.r.t motor-side V margin: {res.ke_required_max_vll_krpm:.1f} Vll_rms/krpm "
                f"({res.ke_required_max_vll_krpm * math.sqrt(2.0):.1f} Vll_peak/krpm)"
            )

        # 1-wire vs 2-wire comparison (+ binding regime)
        def _limiting_label(r: SolveResult):
            ll_m = r.v_motor_req * math.sqrt(3.0)
            cands = []
            # current
            if (r.i_limit_phase_mag is not None) and (r.i_mag_used_rms is not None):
                cands.append(("Current", r.i_limit_phase_mag - r.i_mag_used_rms))
            # surface cmd
            if (r.v_surface_limit is not None) and (r.v_surface_cmd is not None):
                cands.append(("Surface V(cmd)", r.v_surface_limit - r.v_surface_cmd))
            # surface inv req
            if (r.v_surface_limit is not None) and (r.v_inverter_req is not None):
                cands.append(("Surface V(inv)", r.v_surface_limit - r.v_inverter_req))
            # downhole
            if r.v_downhole_phase_limit is not None:
                cands.append(("Downhole Vph", r.v_downhole_phase_limit - r.v_motor_req))
            if r.v_downhole_ll_limit is not None:
                cands.append(("Downhole VLL", r.v_downhole_ll_limit - ll_m))
            if not cands:
                return "—"
            lab, m = min(cands, key=lambda t: t[1])
            return f"{lab} ({_fmt(m, 2, '')})"

        lines.append("")
        lines.append("1-wire vs 2-wire @ target (same basis):")
        lines.append(
            f"• 1-wire: Iq_max={res_1.iq_max_rms:.3f}, |I|_used={res_1.i_mag_used_rms:.3f}, "
            f"Vdrop={res_1.v_cable_drop:.2f}, Ploss={res_1.p_cable_loss_w:.1f} W, LIMITING={_limiting_label(res_1)}"
        )
        lines.append(
            f"• 2-wire: Iq_max={res_2.iq_max_rms:.3f}, |I|_used={res_2.i_mag_used_rms:.3f}, "
            f"Vdrop={res_2.v_cable_drop:.2f}, Ploss={res_2.p_cable_loss_w:.1f} W, LIMITING={_limiting_label(res_2)}"
        )

        # Notes / blocking reasons (existing)
        if getattr(res, "notes", None):
            lines.append("")
            lines.append("Notes:")
            for n in res.notes:
                lines.append(f"• {n}")

        if (not res.feasible) and res.reasons:
            lines.append("")
            lines.append("Blocking reasons:")
            for r in res.reasons:
                lines.append(f"• {r}")

        # Quick auto-guidance (based on dominant limiter)
        if dominant is not None:
            dom_label = dominant[0].lower()
            lines.append("")
            lines.append("Suggested levers (based on limiting constraint):")
            if "downhole vll" in dom_label or "contact" in dom_label:
                lines.append(
                    "• Raise contact-block VLL (creepage/insulation), OR reduce Ke, OR reduce ratio G, OR enable field-weakening.")
            elif "downhole vph" in dom_label:
                lines.append("• Raise downhole phase limit, OR reduce required Vmotor via Ke/G/FW adjustments.")
            elif "current" in dom_label:
                lines.append(
                    "• Increase wires/phase (ampacity), OR increase Kt, OR reduce required torque (losses/BHA load), OR improve gearbox efficiency.")
            elif "surface v" in dom_label:
                lines.append(
                    "• Increase Vdc/utilization, OR reduce filter/cable drops, OR reduce required motor voltage (Ke/G/FW).")
            else:
                lines.append(
                    "• Review voltage/current headroom, coupler slip margin, and brake power path assumptions.")

        full_text = "\n".join(lines)

        # Update Dashboard tab (full diagnostics)
        if hasattr(self, "dashboard_pane"):
            try:
                self.dashboard_pane.update_from_status(full_text, p, res_sel, res_1, res_2, dominant)
            except Exception:
                pass

        # Left-side status: keep it short (dashboard has the full dump)
        try:
            short = []
            short.append(
                "✅ PASS — UI target feasible" if res.feasible else "❌ FAIL — UI target infeasible (binding constraint hit)")
            short.append(
                f"Cmd: {p.out_dir} {res.out_rpm_cmd:.3f} rpm   |   {quad}   |   P_out≈{_fmt(p_out_w / 1000.0, 3, ' kW')}")
            if dominant is not None:
                dom_label, dom_margin, dom_used, dom_lim = dominant
                dom_pct = _pct_used(dom_used, dom_lim)
                dom_pct_s = f"{dom_pct:.1f}%" if dom_pct is not None else "—"
                short.append(f"Limiter: {dom_label}  (util={dom_pct_s}, headroom={_fmt(dom_margin, 2, '')})")
            if I_margin is not None:
                short.append(f"|I| headroom: {_fmt(I_margin, 2, ' A')}")
            if Vdh_ll_margin is not None:
                short.append(f"Contact VLL headroom: {_fmt(Vdh_ll_margin, 1, ' V')}")
            short.append("See Dashboard tab for full details.")
            self.status.setText("\n".join(short))
        except Exception:
            self.status.setText(full_text)

    def _plot_envelope(self, res: SolveResult, env1, env2):
        (rpm1, tq1, iq1, loss1) = env1
        (rpm2, tq2, iq2, loss2) = env2

        # Always show full +/- RPM axis (CW=+ , CCW=-), independent of selected command direction.
        rpm1a = np.asarray(rpm1, dtype=float)
        rpm2a = np.asarray(rpm2, dtype=float)
        tq1a = np.asarray(tq1, dtype=float)
        tq2a = np.asarray(tq2, dtype=float)

        ax1 = self.mpl_env.ax_env
        ax2 = self.mpl_env.ax_v
        ax3 = self.mpl_env.ax_i
        ax4 = self.mpl_env.ax_loss

        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        braking_ok = bool(getattr(self.params, "braking_path_available", True)) and bool(
            getattr(self.params.gearbox, "backdrivable", True))

        # --- Torque envelope with signed quadrants ---
        # Build *continuous* signed curves over [-RPMmax, +RPMmax] to avoid missing quadrants.
        rpm1_pos = np.asarray(rpm1a, dtype=float)
        rpm2_pos = np.asarray(rpm2a, dtype=float)
        tq1_abs = np.asarray(tq1a, dtype=float)
        tq2_abs = np.asarray(tq2a, dtype=float)

        # Symmetric RPM vectors over [-max..+max] with an explicit break at 0.
        # The inserted NaN prevents Matplotlib from drawing a near-vertical "seam" line
        # (which can appear slightly tilted due to finite grid resolution around 0 RPM).
        rpm1_sym = np.concatenate((-rpm1_pos[::-1], [np.nan], rpm1_pos))
        rpm2_sym = np.concatenate((-rpm2_pos[::-1], [np.nan], rpm2_pos))

        tq1_sym = np.concatenate((tq1_abs[::-1], [np.nan], tq1_abs))
        tq2_sym = np.concatenate((tq2_abs[::-1], [np.nan], tq2_abs))

        # Motoring cap: torque sign follows speed sign (Quadrant I and III)
        s1 = np.sign(rpm1_sym);
        s1[s1 == 0.0] = 1.0
        s2 = np.sign(rpm2_sym);
        s2[s2 == 0.0] = 1.0
        tq1_mot = tq1_sym * s1
        tq2_mot = tq2_sym * s2

        ax1.plot(rpm1_sym, tq1_mot, linewidth=2, label='1 wire/phase (motoring)')
        ax1.plot(rpm2_sym, tq2_mot, linewidth=2, linestyle='--', label='2 wires/phase (motoring)')

        # Regen/brake capability (Quadrant II and IV): torque sign opposes speed sign.
        # If cable-aware regen is enabled, compute a per-speed regen torque magnitude cap.
        if braking_ok:
            m_reg = SystemModel(self.params)
            import copy as _copy
            cab1 = _copy.deepcopy(self.params.cable)
            cab2 = _copy.deepcopy(self.params.cable)
            cab1.wires_per_phase = 1
            cab2.wires_per_phase = 2

            regen1_abs = np.array([
                min(float(tq), float(m_reg.regen_cap_output_torque_ftlbf(float(r), cab1)))
                for r, tq in zip(rpm1_pos, tq1_abs)
            ], dtype=float)
            regen2_abs = np.array([
                min(float(tq), float(m_reg.regen_cap_output_torque_ftlbf(float(r), cab2)))
                for r, tq in zip(rpm2_pos, tq2_abs)
            ], dtype=float)

            regen1_sym = np.concatenate((regen1_abs[::-1], [np.nan], regen1_abs))
            regen2_sym = np.concatenate((regen2_abs[::-1], [np.nan], regen2_abs))

            tq1_regen = -regen1_sym * s1
            tq2_regen = -regen2_sym * s2

            ax1.plot(rpm1_sym, tq1_regen, linewidth=1.5, linestyle=':', label='1 wire/phase (regen cap)')
            ax1.plot(rpm2_sym, tq2_regen, linewidth=1.5, linestyle=':', label='2 wires/phase (regen cap)')

            # Optional surface braking power limit: |T_brake| <= Pmax/ω
            if bool(getattr(self.params, 'brake_power_limit_enabled', False)):
                pmax_kw = float(getattr(self.params, 'brake_power_kw_max', 0.0))
                if pmax_kw > 0.0:
                    rp = np.linspace(0.05, max(0.05, float(self.env_out_rpm_max.value()) if hasattr(self,
                                                                                                    'env_out_rpm_max') else 1.0),
                                     500)
                    omega = rp * 2.0 * math.pi / 60.0
                    tlim_nm = (pmax_kw * 1000.0) / np.maximum(omega, 1e-12)
                    tlim_ftlb = nm_to_ft_lbf(tlim_nm)
                    # Braking quadrants: (+RPM, -T) and (-RPM, +T)
                    ax1.plot(rp, -tlim_ftlb, linewidth=1.2, linestyle='-.', label='Surface brake power limit')
                    ax1.plot(-rp, tlim_ftlb, linewidth=1.2, linestyle='-.', label='Surface brake power limit')

        else:
            why = []
            if not bool(getattr(self.params, 'braking_path_available', True)):
                why.append('Braking path OFF')
            if not bool(getattr(self.params.gearbox, 'backdrivable', True)):
                why.append('Gearbox non-backdrivable')
            msg = 'REGEN QUADRANTS NOT AVAILABLE (' + ', '.join(why) + ')' if why else 'REGEN QUADRANTS NOT AVAILABLE'
            ax1.text(0.02, 0.06, msg, transform=ax1.transAxes, fontsize=8)

        # Target operating point (signed on shaft): show T_CCRS along with its drive/brake decomposition
        rpm_tgt = float(res.out_rpm_cmd)
        s_tgt = (1.0 if rpm_tgt > 1e-12 else (
            -1.0 if rpm_tgt < -1e-12 else (1.0 if str(self.params.out_dir).upper().startswith("CW") else -1.0)))

        t_drive = float(res.out_drive_torque_req_ftlbf)  # >=0, along direction of motion
        t_brake = float(res.out_brake_torque_req_ftlbf)  # >=0, opposite direction of motion (regen/brake)

        tau_drive_signed = t_drive * s_tgt
        tau_brake_signed = -t_brake * s_tgt
        tau_ccrs_signed = tau_drive_signed + tau_brake_signed
        tau_load_signed = -tau_ccrs_signed  # net external torque acting on shaft

        # Main "required" point: torque CCRS must apply
        ax1.scatter([rpm_tgt], [tau_ccrs_signed], s=70)
        ax1.annotate("T_CCRS", (rpm_tgt, tau_ccrs_signed), textcoords="offset points", xytext=(8, 8), fontsize=8)

        # Decomposition (plotted only if non-trivial)
        if abs(t_drive) > 1e-6:
            ax1.scatter([rpm_tgt], [tau_drive_signed], s=55, marker="o")
            ax1.annotate("T_drive", (rpm_tgt, tau_drive_signed), textcoords="offset points", xytext=(8, -14),
                         fontsize=8)
        if abs(t_brake) > 1e-6:
            ax1.scatter([rpm_tgt], [tau_brake_signed], s=55, marker="x")
            ax1.annotate("T_brake", (rpm_tgt, tau_brake_signed), textcoords="offset points", xytext=(8, -14),
                         fontsize=8)

        # Net external torque point (load on shaft)
        ax1.scatter([rpm_tgt], [tau_load_signed], s=55, marker="s", facecolors="none", edgecolors="k")
        ax1.annotate("T_load", (rpm_tgt, tau_load_signed), textcoords="offset points", xytext=(8, 8), fontsize=8)

        # Required torque vs speed curve (from static load blocks), plotted across +/- speed
        if bool(self.params.bha.enabled) or bool(self.params.parasitic.enabled) or bool(
                getattr(self.params, "stuck_mode", False)):
            m_req = SystemModel(self.params)
            rpm_max_env = float(max(np.max(np.abs(rpm2a)) if rpm2a.size else 0.0,
                                    np.max(np.abs(rpm1a)) if rpm1a.size else 0.0,
                                    float(self.env_out_rpm_max.value()) if hasattr(self, "env_out_rpm_max") else 0.0,
                                    abs(rpm_tgt),
                                    1.0))
            rpms_req = np.linspace(-rpm_max_env, rpm_max_env, 401)
            req_curve = []
            for r in rpms_req:
                omega = rpm_to_rad_s(float(r))
                t_drive_nm, t_brake_nm, *_ = m_req._required_output_drive_torque_nm(omega)
                s = m_req._motion_sign(omega)
                tau_ccrs_nm = (t_drive_nm - t_brake_nm) * s
                req_curve.append(nm_to_ft_lbf(tau_ccrs_nm))
            req_curve = np.asarray(req_curve, dtype=float)
            ax1.plot(rpms_req, req_curve, linewidth=1.8, label="T_CCRS required (static blocks)")
            ax1.plot(rpms_req, -req_curve, linewidth=1.2, linestyle="--", label="Net external torque on shaft (load)")

        # Axis scaling
        rpm_max_plot = float(self.env_out_rpm_max.value()) if hasattr(self, "env_out_rpm_max") else float(
            max(np.max(rpm2a), 1.0))
        ax1.set_xlim(-rpm_max_plot, rpm_max_plot)

        # y-limit should consider: motoring envelopes + the target required CCRS torque.
        ymax = float(max(np.max(np.abs(tq1a)) if tq1a.size else 0.0,
                         np.max(np.abs(tq2a)) if tq2a.size else 0.0,
                         abs(tau_ccrs_signed),
                         1.0) * 1.12)
        ax1.set_ylim(-ymax, ymax)

        # Shade regen half-planes for clarity (still shown even if braking is OFF).
        # Shade REGEN quadrants (Q2 and Q4): torque opposes speed
        ax1.fill_between([0.0, rpm_max_plot], [0.0, 0.0], [-ymax, -ymax], alpha=0.08)
        ax1.fill_between([-rpm_max_plot, 0.0], [0.0, 0.0], [ymax, ymax], alpha=0.08)
        ax1.axhline(0.0, linewidth=1.0)
        ax1.axvline(0.0, linewidth=1.0)

        ax1.set_title("Output Torque Envelope (signed, CW=+, CCW=-)")
        ax1.set_xlabel("Output RPM")
        # ax1.set_ylim(-1500,1500)
        ax1.set_ylabel("Output Torque (ft-lbf)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        # Context annotation (TOB vs Stall)
        if bool(getattr(self.params, "stuck_mode", False)):
            ann = f"STALL requirement = {abs(self.params.target.out_torque_ftlbf):.0f} ft-lbf"
        else:
            if bool(self.params.bha.enabled):
                ann = f"Drilling TOB (bit CW → TOB CCW) = {abs(getattr(self.params.bha, 'drilling_tob_ftlbf', 0.0)):.0f} ft-lbf"
            else:
                ann = "BHA block OFF"
        ax1.text(0.02, 0.02, ann, transform=ax1.transAxes, fontsize=8)

        if float(getattr(res, "brake_suppressed_ftlbf", 0.0)) > 1e-6:
            ax1.text(
                0.02, 0.10,
                f"Assist torque suppressed by self-locking gearbox: {float(res.brake_suppressed_ftlbf):.0f} ft-lbf (no electrical regen)",
                transform=ax1.transAxes, fontsize=8
            )
        elif float(res.out_brake_torque_req_ftlbf) > 1e-6:
            if braking_ok:
                ax1.text(
                    0.02, 0.10,
                    f"BRAKING/REGEN required at target: {float(res.out_brake_torque_req_ftlbf):.0f} ft-lbf",
                    transform=ax1.transAxes, fontsize=8
                )
            else:
                ax1.text(
                    0.02, 0.10,
                    f"BRAKING required at target (regen not available): {float(res.out_brake_torque_req_ftlbf):.0f} ft-lbf",
                    transform=ax1.transAxes, fontsize=8
                )

        # --- Voltage budget @ reported point ---
        v_motor = res.v_motor_req
        v_drop = res.v_cable_drop
        v_cmd = res.v_surface_cmd
        if bool(getattr(res, "regen_required", False)):
            ax2.set_title("Regen Voltage Condition @ Target (phase RMS)")
            ax2.bar([0], [v_cmd], label="Surface clamp (Vclamp)")
            ax2.bar([0], [v_drop], bottom=[v_cmd], label="Cable drop")
            # back-EMF estimate (phase RMS)
            e_phase = (self.params.motor.ke_vll_rms_per_krpm * (res.motor_rpm / 1000.0)) / math.sqrt(3.0)
            ax2.axhline(e_phase, linestyle="--", linewidth=2, label="Motor back-EMF (phase)")
        else:
            ax2.set_title("Voltage Budget @ Target (phase RMS)")
            ax2.bar([0], [v_motor], label="Motor required")
            ax2.bar([0], [v_drop], bottom=[v_motor], label="Cable drop")
            ax2.axhline(v_cmd, linestyle="--", linewidth=2, label="Surface command (Vcmd)")
        ax2.axhline(res.v_surface_limit, linestyle=":", linewidth=2, label="Inverter phase limit")
        if res.v_downhole_phase_limit is not None:
            ax2.axhline(res.v_downhole_phase_limit, linestyle="-.", linewidth=2, label="Downhole motor Vphase limit")
        if res.v_downhole_ll_limit is not None:
            ax2.axhline(res.v_downhole_ll_limit / math.sqrt(3.0), linestyle=(0, (3, 2, 1, 2)), linewidth=2,
                        label="Downhole contact Vll limit (as phase)")
        ax2.set_xticks([0])
        ax2.set_xticklabels([f"{self.params.cable.wires_per_phase}-wire"])
        ax2.set_ylabel("Vrms (phase)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        # --- Iq envelope + requirement ---
        ax3.plot(rpm1a, iq1, linewidth=2, label="Iq_max 1-wire (+RPM)")
        ax3.plot(-rpm1a, iq1, linewidth=2, label="Iq_max 1-wire (-RPM)")
        ax3.plot(rpm2a, iq2, linewidth=2, linestyle="--", label="Iq_max 2-wire (+RPM)")
        ax3.plot(-rpm2a, iq2, linewidth=2, linestyle="--", label="Iq_max 2-wire (-RPM)")
        ax3.axhline(res.iq_req_rms, linestyle=":", linewidth=2, label="Iq_req @ target")
        ax3.axhline(self.params.cable.i_phase_limit(), linestyle="-.", linewidth=2, label="|I| limit (phase)")
        ax3.set_title("Torque-Producing Current Margin")
        ax3.set_xlabel("Output RPM")
        ax3.set_ylabel("Arms (Iq RMS)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")

        # --- Cable loss vs RPM (mirror for symmetry) ---
        loss1a = np.asarray(loss1, dtype=float)
        loss2a = np.asarray(loss2, dtype=float)
        ax4.plot(rpm1a, loss1a, linewidth=2, label="Cable loss @ max torque (1-wire)")
        ax4.plot(-rpm1a, loss1a, linewidth=2, label="Cable loss @ max torque (1-wire)")
        ax4.plot(rpm2a, loss2a, linewidth=2, linestyle="--", label="Cable loss @ max torque (2-wire)")
        ax4.plot(-rpm2a, loss2a, linewidth=2, linestyle="--", label="Cable loss @ max torque (2-wire)")
        ax4.set_title("Cable Copper Loss vs Output RPM")
        ax4.set_xlabel("Output RPM")
        ax4.set_ylabel("Loss (W)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")

        self.mpl_env.fig.tight_layout(pad=2.0)
        self.mpl_env.canvas.draw_idle()

    def _efficiency_torque_scalers(self):
        """Return (low, high) torque scalers for the efficiency uncertainty band.

        The nominal torque envelopes are computed using the nominal total gearbox efficiency η_nom.
        The UI provides an absolute efficiency uncertainty band ±Δη (per-unit), so we form:
            η_lo = clip(η_nom - Δη),  η_hi = clip(η_nom + Δη)
            scalers = (η_lo/η_nom, η_hi/η_nom)
        """
        eta_nom = float(np.clip(self.params.gearbox.eff_total(), 1e-6, 0.999999))
        # Δη from UI (abs per-unit)
        try:
            d_eta = float(self.eta_band_pu.value())
        except Exception:
            d_eta = 0.0
        eta_lo = float(np.clip(eta_nom - d_eta, 1e-6, 0.999999))
        eta_hi = float(np.clip(eta_nom + d_eta, 1e-6, 0.999999))
        return (eta_lo / eta_nom, eta_hi / eta_nom)

    def _effective_downhole_vll_limit(self) -> Optional[float]:
        """Return the effective downhole VLL RMS limit (line-line) considering enabled limits.

        If only a motor Vphase (L-N) limit is enabled, convert to VLL via √3.
        If both are enabled, take the most restrictive.
        Returns None if no downhole voltage limit is enabled.
        """
        lims = self.params.limits
        vals = []
        if lims.enforce_downhole_vll_limit:
            vals.append(float(lims.downhole_vll_rms_limit))
        if lims.enforce_downhole_vphase_limit:
            vals.append(float(lims.downhole_v_phase_rms_limit) * math.sqrt(3.0))
        if not vals:
            return None
        return min(vals)

    def _downhole_vll_band(self, rpm_mag: np.ndarray, cable_override: Optional[CableParams] = None) -> Optional[dict]:
        """Return a torque band (lo/hi) due to downhole VLL limit uncertainty (±ΔV).

        The band is computed by re-solving the envelope twice with VLL = (VLL_nom - ΔV) and (VLL_nom + ΔV),
        then interpolating onto rpm_mag.
        """
        p0 = self.params
        if not bool(p0.limits.enforce_downhole_vll_limit):
            return None
        try:
            dv = float(self.vll_band_vrms.value())
        except Exception:
            dv = 0.0
        if dv <= 0.0:
            return None

        vll_nom = float(p0.limits.downhole_vll_rms_limit)

        def _torque_for_vll(vll_val: float) -> np.ndarray:
            p = copy.deepcopy(p0)
            p.limits.enforce_downhole_vll_limit = True
            p.limits.downhole_vll_rms_limit = float(vll_val)
            m = SystemModel(p)
            rp, tq, _, _ = m.compute_envelope(
                out_rpm_max=float(np.max(rpm_mag)),
                n=int(len(rpm_mag)),
                cable_override=cable_override,
            )
            if len(rp) != len(rpm_mag) or float(np.max(np.abs(rp - rpm_mag))) > 1e-6:
                tq = np.interp(rpm_mag, rp, tq)
            return np.asarray(tq, dtype=float)

        tq_lo = _torque_for_vll(max(1e-6, vll_nom - dv))
        tq_hi = _torque_for_vll(vll_nom + dv)

        return {"lo": np.minimum(tq_lo, tq_hi), "hi": np.maximum(tq_lo, tq_hi)}

    def _cable_length_torque_band(self, rpm_mag: np.ndarray, tq_mag: np.ndarray,
                                  cable_override: Optional[CableParams] = None) -> Optional[dict]:
        """Return a torque band (lo/hi) due to cable length uncertainty.

        Uses a symmetric ±ΔL about the nominal cable length:
          - L_low = max(1 m, L0 - ΔL)
          - L_high = L0 + ΔL

        If ΔL is 0 (or both perturbed lengths collapse back to L0), returns None.
        """
        dL = float(self.c_len_band.value()) if hasattr(self, "c_len_band") else 0.0
        if dL <= 0.0:
            return None

        cab0 = cable_override if cable_override is not None else self.params.cable
        L0 = float(cab0.length_m)

        L_low = max(1.0, L0 - dL)
        L_high = max(1.0, L0 + dL)

        if abs(L_low - L0) < 1e-9 and abs(L_high - L0) < 1e-9:
            return None

        def _torque_at_length(Lm: float) -> np.ndarray:
            cab = copy.deepcopy(cab0)
            cab.length_m = float(Lm)
            rp, tq_adj, _, _ = self.model.compute_envelope(
                out_rpm_max=float(np.max(rpm_mag)),
                n=int(len(rpm_mag)),
                cable_override=cab,
            )
            rp = np.asarray(rp, dtype=float)
            tq_adj = np.asarray(tq_adj, dtype=float)
            if len(rp) != len(rpm_mag) or float(np.max(np.abs(rp - rpm_mag))) > 1e-6:
                tq_adj = np.interp(rpm_mag, rp, tq_adj)
            return tq_adj

        tq_nom = np.asarray(tq_mag, dtype=float)
        curves = [tq_nom]

        if abs(L_low - L0) >= 1e-9:
            try:
                curves.append(_torque_at_length(L_low))
            except Exception:
                pass
        if abs(L_high - L0) >= 1e-9:
            try:
                curves.append(_torque_at_length(L_high))
            except Exception:
                pass

        if len(curves) <= 1:
            return None

        stack = np.vstack([np.asarray(c, dtype=float) for c in curves])
        return {"lo": np.nanmin(stack, axis=0), "hi": np.nanmax(stack, axis=0)}

    def _plot_band_torque(self,
                          env_rpm_1: np.ndarray, env_tq_1: np.ndarray,
                          env_rpm_2: np.ndarray, env_tq_2: np.ndarray,
                          vll_1: np.ndarray, vll_2: np.ndarray):
        """Band plots (v15).

        Left column: two large band plots (1-wire and 2-wire) showing **4-quadrant** torque envelopes:
          - Motoring quadrants (QI, QIII): torque sign matches speed sign
          - Regen/braking quadrants (QII, QIV): torque sign opposes speed sign, limited by regen/braking path

        Right column: three stacked utilization plots.
        """
        p = self.params

        # Clear
        for ax in (self.band_pane.ax1, self.band_pane.ax2, self.band_pane.ax3, self.band_pane.ax4, self.band_pane.ax5):
            ax.cla()

        rpm_axis = float(self.env_out_rpm_max.value())

        # Regen/braking is only meaningful if BOTH are true:
        #   - Surface has a braking/regen path
        #   - Geartrain is backdrivable (i.e., can transmit torque/power back upstream)
        braking_path = bool(getattr(p, 'braking_path_available', True))
        backdrivable = bool(getattr(getattr(p, 'gearbox', None), 'backdrivable', True))
        regen_ok = braking_path and backdrivable

        # Convenience: build cable overrides for 1- and 2-wire/phase
        cable_1 = copy.deepcopy(p.cable)
        cable_1.wires_per_phase = 1
        cable_2 = copy.deepcopy(p.cable)
        cable_2.wires_per_phase = 2

        def _sorted_xy(x: np.ndarray, y: np.ndarray):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            idx = np.argsort(x)
            return x[idx], y[idx]

        def _plot_motoring_and_bands(ax, rpm_mag: np.ndarray, tq_mag: np.ndarray, cable_override: CableParams,
                                     label_prefix: str, show_legend: bool):
            rpm_mag, tq_mag = _sorted_xy(rpm_mag, tq_mag)

            # Shade braking quadrants (QII and QIV) for readability
            try:
                ax.axvspan(0.0, rpm_axis, ymin=0.0, ymax=0.5, facecolor="0.6", alpha=0.06,
                           hatch="///", edgecolor="0.6", linewidth=0.0, zorder=0)
                ax.axvspan(-rpm_axis, 0.0, ymin=0.5, ymax=1.0, facecolor="0.6", alpha=0.06,
                           hatch="///", edgecolor="0.6", linewidth=0.0, zorder=0)
            except Exception:
                pass

            # Efficiency band (±loss model)
            s_eta_lo, s_eta_hi = self._efficiency_torque_scalers()
            tq_eta_lo = tq_mag * s_eta_lo
            tq_eta_hi = tq_mag * s_eta_hi

            # --- Motoring envelope (QI and QIII)
            # QI: +rpm, +tq
            ax.fill_between(rpm_mag, tq_eta_lo, tq_eta_hi, alpha=0.18,
                            label='η band (loss uncertainty)' if show_legend else None)
            ax.plot(rpm_mag, tq_mag, lw=2.2, label=f'{label_prefix} motoring' if show_legend else None)

            # QIII: -rpm, -tq (reverse arrays so x is increasing)
            rpm_neg = -rpm_mag[::-1]
            tq_neg = -tq_mag[::-1]
            ax.fill_between(rpm_neg, -tq_eta_hi[::-1], -tq_eta_lo[::-1], alpha=0.18)
            ax.plot(rpm_neg, tq_neg, lw=2.2)

            # --- Downhole VLL band (creepage limit uncertainty)
            vll_band = self._downhole_vll_band(rpm_mag, cable_override=cable_override)
            if vll_band is not None:
                ax.fill_between(rpm_mag, vll_band['lo'], vll_band['hi'], alpha=0.12,
                                label='VLL band (downhole limit)' if show_legend else None)
                ax.fill_between(rpm_neg, -vll_band['hi'][::-1], -vll_band['lo'][::-1], alpha=0.12)

            # --- Cable-length sensitivity band
            len_band = self._cable_length_torque_band(rpm_mag, tq_mag, cable_override=cable_override)
            if len_band is not None:
                ax.fill_between(rpm_mag, len_band['lo'], len_band['hi'], alpha=0.12,
                                label='Cable length band' if show_legend else None)
                ax.fill_between(rpm_neg, -len_band['hi'][::-1], -len_band['lo'][::-1], alpha=0.12)
                # Draw band edges so it remains visible even when overlapping other fills
                try:
                    lo = np.asarray(len_band['lo'], dtype=float)
                    hi = np.asarray(len_band['hi'], dtype=float)
                    ax.plot(rpm_mag, lo, ls='--', lw=0.9, alpha=0.7)
                    ax.plot(rpm_mag, hi, ls='--', lw=0.9, alpha=0.7)
                    ax.plot(rpm_neg, -lo[::-1], ls='--', lw=0.9, alpha=0.7)
                    ax.plot(rpm_neg, -hi[::-1], ls='--', lw=0.9, alpha=0.7)
                except Exception:
                    pass

            # --- Regen/braking caps (QII and QIV)
            if regen_ok:
                c_use = cable_override if (cable_override is not None) else p.cable

                # Ideal/symmetric regen cap: assumes the electrical chain can absorb power and
                # does NOT impose the long-cable + surface-clamp constraint.
                regen_ideal = np.asarray(tq_mag, dtype=float)

                # Cable-aware regen cap: back-EMF must exceed (surface clamp + cable drop).
                # Force this ON in the plotted capability so the user can *see* whether regen is
                # realistically possible on a long cable.
                try:
                    if bool(getattr(p, 'regen_cable_limit_enabled', True)):
                        m_reg = self.model
                    else:
                        p_reg = copy.deepcopy(p)
                        p_reg.regen_cable_limit_enabled = True
                        m_reg = SystemModel(p_reg)
                    regen_cable = np.array([m_reg.regen_cap_output_torque_ftlbf(float(r), c_use)
                                            for r in rpm_mag], float)
                except Exception:
                    regen_cable = np.zeros_like(rpm_mag, dtype=float)

                regen_cable = np.minimum(regen_cable, tq_mag)

                # QIV: +rpm, -tq
                ax.plot(rpm_mag, -regen_ideal, ls=':', lw=1.2, color='0.55',
                        label=f'{label_prefix} regen cap (ideal)' if show_legend else None)
                ax.plot(rpm_mag, -regen_cable, ls=':', lw=1.8, color='#ff4fbf',
                        label=f'{label_prefix} regen cap (cable-aware)' if show_legend else None)

                # QII: -rpm, +tq
                ax.plot(rpm_neg, regen_ideal[::-1], ls=':', lw=1.2, color='0.55')
                ax.plot(rpm_neg, regen_cable[::-1], ls=':', lw=1.8, color='#ff4fbf')

            # Axes styling
            ax.axhline(0.0, lw=1.0)
            ax.axvline(0.0, lw=1.0)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-rpm_axis, rpm_axis)

        # ---------- Left: two big band plots ----------
        show_leg1 = True
        _plot_motoring_and_bands(self.band_pane.ax1, env_rpm_1, env_tq_1, cable_1, '1 wire/phase', show_leg1)
        self.band_pane.ax1.set_title('Torque band (1 wire/phase, 4Q view)')
        self.band_pane.ax1.set_ylabel('Output Torque (ft-lbf)')
        # y-axis limits are synced across all 4Q plots (see _sync_quadrant_yaxis)

        show_leg2 = True
        _plot_motoring_and_bands(self.band_pane.ax2, env_rpm_2, env_tq_2, cable_2, '2 wires/phase', show_leg2)
        self.band_pane.ax2.set_title('Torque band (2 wires/phase, 4Q view)')
        self.band_pane.ax2.set_xlabel('Output RPM (signed, CW=+, CCW=-)')
        self.band_pane.ax2.set_ylabel('Output Torque (ft-lbf)')
        # y-axis limits are synced across all 4Q plots (see _sync_quadrant_yaxis)

        # Regen annotation if disabled
        if not regen_ok:
            msg = 'Regen/braking quadrants disabled\n(no backdrive OR no braking path)'
            for ax in (self.band_pane.ax1, self.band_pane.ax2):
                ax.text(0.02, 0.98, msg, transform=ax.transAxes, va='top', ha='left', fontsize=9)

        # Target marker (drive torque in selected direction)
        s_dir = self.model.out_dir_sign()
        rpm_tgt = float(p.target.out_rpm) * s_dir
        tq_tgt = float(p.target.out_torque_ftlbf) * s_dir
        for ax in (self.band_pane.ax1, self.band_pane.ax2):
            ax.scatter([rpm_tgt], [tq_tgt], s=55, marker='o')
            ax.annotate('Target', (rpm_tgt, tq_tgt), textcoords='offset points', xytext=(6, 6), fontsize=9)

        # Legends for the 4Q band plots (ax1/ax2)
        for ax in (self.band_pane.ax1, self.band_pane.ax2):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc="best", framealpha=0.85)

        # ---------- Right: utilization / diagnostics ----------
        # Revert to v14-style diagnostics:
        #   (1) Contact-block Vll utilization vs output RPM (computed against configured limit even if not enforced)
        #   (2) Vll headroom vs output RPM using *no-load* back-EMF ELL_rms
        #   (3) Approx voltage-limited Imax vs output RPM using a simple |Z| model

        vll_lim = float(self.params.limits.downhole_vll_rms_limit)

        if vll_lim <= 1e-9:
            for ax_u, title in (
                    (self.band_pane.ax3, "Contact-block Vll Utilization vs Output RPM"),
                    (self.band_pane.ax4, "Vll Headroom vs Output RPM (Contact-block)"),
                    (self.band_pane.ax5, "Approx V-Limited Imax vs Output RPM"),
            ):
                ax_u.set_title(title)
                ax_u.text(0.5, 0.5, "Contact-block Vll limit is 0 / unset", ha="center", va="center",
                          transform=ax_u.transAxes)
                ax_u.set_axis_off()
            self.band_pane.canvas.draw_idle()
            return

        # Use the already-computed VLL along the envelope (motor terminal VLL_rms at the max-torque point)
        rpm1, vll1 = _sorted_xy(env_rpm_1, vll_1)
        rpm2, vll2 = _sorted_xy(env_rpm_2, vll_2)

        # For these utilization diagnostics, direction only changes sign; magnitudes are symmetric.
        # Plot both CW (+) and CCW (-) by mirroring the curves about 0.
        r1m = np.abs(np.asarray(rpm1, dtype=float))
        r2m = np.abs(np.asarray(rpm2, dtype=float))

        # -----------------------------
        # (1) Vll utilization (% of contact-block limit) vs Output RPM
        # -----------------------------
        u1 = 100.0 * (np.asarray(vll1, dtype=float) / vll_lim)
        u2 = 100.0 * (np.asarray(vll2, dtype=float) / vll_lim)

        r1p, u1p = _sorted_xy(r1m, u1)
        r2p, u2p = _sorted_xy(r2m, u2)

        ax3 = self.band_pane.ax3
        ax3.plot(r1p, u1p, linewidth=2, color="C0", label="1 wire/phase")
        ax3.plot(r2p, u2p, linewidth=2, color="C1", label="2 wires/phase")
        ax3.plot(-r1p[::-1], u1p[::-1], linewidth=2, color="C0", label=None)
        ax3.plot(-r2p[::-1], u2p[::-1], linewidth=2, color="C1", label=None)
        ax3.axhline(100.0, linewidth=2, linestyle="--", label=f"Vll limit = {vll_lim:.0f} Vrms (100%)")
        ax3.set_title("Contact-block Vll Utilization vs Output RPM")
        ax3.set_xlabel("Output RPM (CW=+, CCW=-)")
        ax3.set_ylabel("Utilization (%)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")
        ax3.set_xlim(-rpm_axis, rpm_axis)
        ymax_u = max(110.0, float(np.nanmax([np.nanmax(u1p), np.nanmax(u2p)]) * 1.05))
        ax3.set_ylim(0.0, min(200.0, ymax_u))

        # -----------------------------
        # (2) Vll headroom and no-load back-EMF vs Output RPM (Contact-block)
        # -----------------------------
        G = float(self.params.gearbox.ratio())
        mp = self.params.motor

        def _ell_rms_from_out_rpm(out_rpm_arr: np.ndarray) -> np.ndarray:
            motor_rpm = np.asarray(out_rpm_arr, dtype=float) * G
            return (float(mp.ke_vll_rms_per_krpm) * motor_rpm) / 1000.0

        r_ref = np.abs(np.asarray(rpm1, dtype=float))
        r_refp, ell_ref = _sorted_xy(r_ref, _ell_rms_from_out_rpm(r_ref))

        ax4 = self.band_pane.ax4
        ax4.plot(r_refp, ell_ref, linewidth=2, label="ELL_rms (no-load back-EMF)")
        ax4.plot(-r_refp[::-1], ell_ref[::-1], linewidth=2, label=None)
        ax4.axhline(vll_lim, linewidth=2, linestyle="--", label=f"Vll_max = {vll_lim:.0f} Vrms")
        # Fill headroom region where ELL < Vll_max
        try:
            ax4.fill_between(r_refp, ell_ref, vll_lim, where=(ell_ref <= vll_lim),
                             alpha=0.20, label="Voltage headroom")
            # Mirror headroom to CCW side
            try:
                ax4.fill_between(-r_refp[::-1], ell_ref[::-1], vll_lim,
                                 where=(ell_ref[::-1] <= vll_lim), alpha=0.20)
            except Exception:
                pass
        except Exception:
            pass
        ax4.set_title("Vll Headroom vs Output RPM (Contact-block)")
        ax4.set_xlabel("Output RPM (CW=+, CCW=-)")
        ax4.set_ylabel("Vrms (L-L)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")
        ax4.set_xlim(-rpm_axis, rpm_axis)

        # -----------------------------
        # (3) Approx voltage-limited Imax vs Output RPM
        # -----------------------------
        p_pairs = max(1, int(getattr(mp, "pole_pairs", 1)))

        # Voltage-limited Imax approximation:
        #   Imax ≈ (Vll_max - ELL_rms) / (sqrt(3) * |Z_phase|)
        #   |Z_phase| = sqrt(R_total^2 + (ω_e * L_total)^2)
        def _imax_curve(out_rpm_arr: np.ndarray, wires_per_phase: int) -> np.ndarray:
            out_rpm_arr = np.asarray(out_rpm_arr, dtype=float)
            motor_rpm = out_rpm_arr * G
            omega_m = motor_rpm * (2.0 * math.pi / 60.0)
            omega_e = omega_m * float(p_pairs)

            cab = CableParams(**vars(self.params.cable))
            cab.wires_per_phase = int(wires_per_phase)

            R_total = float(self.model.rs_effective_ohm()) + float(cab.effective_r_phase())
            L_total = float(mp.lq_h) + float(cab.effective_l_phase())

            ell = (float(mp.ke_vll_rms_per_krpm) * motor_rpm) / 1000.0
            dvll = np.maximum(0.0, vll_lim - ell)
            Zmag = np.sqrt(R_total * R_total + (omega_e * L_total) * (omega_e * L_total))
            Zmag = np.maximum(1e-12, Zmag)

            return dvll / (math.sqrt(3.0) * Zmag)

        r1 = np.abs(np.asarray(rpm1, dtype=float))
        r2 = np.abs(np.asarray(rpm2, dtype=float))

        imax1 = _imax_curve(r1, 1)
        imax2 = _imax_curve(r2, 2)

        # Also show current caps so it's clear where voltage dominates.
        cab1 = CableParams(**vars(self.params.cable));
        cab1.wires_per_phase = 1
        cab2 = CableParams(**vars(self.params.cable));
        cab2.wires_per_phase = 2
        i_cap1 = float(cab1.i_phase_limit())
        i_cap2 = float(cab2.i_phase_limit())

        x1, y1 = _sorted_xy(r1, imax1)
        x2, y2 = _sorted_xy(r2, imax2)

        ax5 = self.band_pane.ax5
        ax5.plot(x1, y1, linewidth=2, color="C0", label="Imax (Vll headroom) — 1 wire/phase")
        ax5.plot(x2, y2, linewidth=2, color="C1", label="Imax (Vll headroom) — 2 wires/phase")
        ax5.plot(-x1[::-1], y1[::-1], linewidth=2, color="C0", label=None)
        ax5.plot(-x2[::-1], y2[::-1], linewidth=2, color="C1", label=None)
        ax5.axhline(i_cap1, linewidth=1.5, color="C0", linestyle=":",
                    label=f"I limit (1-wire) = {i_cap1:.2f} Arms")
        ax5.axhline(i_cap2, linewidth=1.5, color="C1", linestyle=":",
                    label=f"I limit (2-wire) = {i_cap2:.2f} Arms")

        ax5.set_title("Approx V-Limited Imax vs Output RPM")
        ax5.set_xlabel("Output RPM (CW=+, CCW=-)")
        ax5.set_ylabel("Arms (phase RMS)")
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8, loc="best")
        ax5.set_xlim(-rpm_axis, rpm_axis)

        # Friendly view: keep some headroom, but avoid absurd spikes at very low RPM
        ymax_i = float(np.nanmax([np.nanmax(y1), np.nanmax(y2), i_cap1, i_cap2]) * 1.10)
        if math.isfinite(ymax_i) and ymax_i > 0.0:
            ax5.set_ylim(0.0, min(10.0 * max(i_cap2, 1.0), ymax_i))

        self.band_pane.canvas.draw_idle()

    # -----------------------------
    # Sweep helper grids
    # -----------------------------
    def _plot_stonehouse_presets(self):
        """Render the Stonehouse presets reference pane.

        NOTE: This pane is a *reference* only. It does not modify the user's UI values.
        """
        if (not hasattr(self, "preset_pane")):
            return

        p_base = copy.deepcopy(self.params)
        out_rpm_max = float(self.env_out_rpm_max.value())
        rpm_axis = max(0.2, out_rpm_max)

        # Regen/braking feasibility depends on system-level assumptions
        braking_path = bool(getattr(p_base, 'braking_path_available', True))
        backdrivable = bool(getattr(getattr(p_base, 'gearbox', None), 'backdrivable', True))
        regen_ok = braking_path and backdrivable

        # Prepare 1- and 2-wire cable overrides (per-phase model)
        cab1 = copy.deepcopy(p_base.cable)
        cab1.wires_per_phase = 1
        cab2 = copy.deepcopy(p_base.cable)
        cab2.wires_per_phase = 2

        # ---- Presets (ft-lbf) ----
        presets = [
            ("Low (mild lateral)", 175.0, 50.0, 55.0, False, None),
            ("Nominal (continuous)", 216.0, 158.0, 83.0, False, None),  # Stonehouse Scenario A
            ("High (peak correction)", 216.0, 315.0, 250.0, False, None),  # Stonehouse Scenario B
            ("Stall (mud motor stall)", 0.0, 0.0, 0.0, True, 945.0),  # Stonehouse Scenario C
        ]

        def _make_params(tob_ftlbf: float, bha_tc_ftlbf: float, gb_tc_ftlbf: float,
                         stuck: bool = False, stall_tq_ftlbf: Optional[float] = None) -> SystemParams:
            pc = copy.deepcopy(p_base)
            # Preset reference curves should use the load-stack torque (do not apply UI torque override).
            pc.target.torque_override_continuous = False
            if stuck:
                pc.stuck_mode = True
                if stall_tq_ftlbf is not None:
                    pc.target.out_torque_ftlbf = float(stall_tq_ftlbf)
                # For stall reference, treat rotating losses & BHA loads as inactive
                # (torque requirement is driven by the stuck/stall target).
                if hasattr(pc, "bha"):
                    pc.bha.enabled = False
                    pc.bha.drilling_tob_ftlbf = 0.0
                    pc.bha.fric_tc_nm = 0.0
                    pc.bha.fric_b_nm_per_rad_s = 0.0
                    pc.bha.fric_c_nm_per_rad_s2 = 0.0
                if hasattr(pc, "parasitic"):
                    pc.parasitic.enabled = False
                    pc.parasitic.tc_nm = 0.0
                    pc.parasitic.b_nm_per_rad_s = 0.0
                    pc.parasitic.c_nm_per_rad_s2 = 0.0
            else:
                pc.stuck_mode = False
                if hasattr(pc, "bha"):
                    pc.bha.enabled = True
                    pc.bha.drilling_tob_ftlbf = float(tob_ftlbf)
                    pc.bha.fric_tc_nm = float(ft_lbf_to_nm(bha_tc_ftlbf))
                if hasattr(pc, "parasitic"):
                    pc.parasitic.enabled = True
                    pc.parasitic.tc_nm = float(ft_lbf_to_nm(gb_tc_ftlbf))
            return pc

        def _signed_torque_point(res: SolveResult, model: SystemModel) -> Tuple[float, float]:
            """Return (rpm_signed, tq_signed_ftlbf) for plotting in 4Q space.

            Uses *electrical* torque demand at output:
              - Motoring: torque sign matches rpm sign
              - Regen/braking: torque sign opposes rpm sign
            """
            rpm_s = float(res.out_rpm_cmd)
            if abs(rpm_s) < 1e-12:
                s = float(model.out_dir_sign())
            else:
                s = 1.0 if rpm_s > 0 else -1.0
            if bool(res.regen_required):
                tq = -s * float(res.out_brake_torque_req_ftlbf)
            else:
                tq = s * float(res.out_drive_torque_req_ftlbf)
            return rpm_s, tq

        def _plot_4q_band(ax,
                          rpm_mag: np.ndarray,
                          tq_nom: np.ndarray,
                          tq_lo: np.ndarray,
                          tq_hi: np.ndarray,
                          model_for_regen: SystemModel,
                          cable_for_regen: CableParams,
                          title: str,
                          ylabel: bool = False,
                          xlabel: bool = False,
                          bands: Optional[dict] = None,
                          show_legend: bool = True):
            """4Q torque view with (1) Stonehouse preset band + (2) system variability bands.

            bands keys (optional):
              - 'eta': {'lo':..., 'hi':...}          gearbox efficiency uncertainty (applied as torque scaling)
              - 'vll': {'lo':..., 'hi':...}          downhole VLL (creepage) limit uncertainty band
              - 'len': {'lo':..., 'hi':...}          cable-length uncertainty band
            """
            rpm_mag = np.asarray(rpm_mag, dtype=float)
            tq_nom = np.asarray(tq_nom, dtype=float)
            tq_lo = np.asarray(tq_lo, dtype=float)
            tq_hi = np.asarray(tq_hi, dtype=float)

            # Shade braking quadrants (QII and QIV) for readability
            try:
                ax.axvspan(0.0, rpm_axis, ymin=0.0, ymax=0.5, facecolor="0.6", alpha=0.06,
                           hatch="///", edgecolor="0.6", linewidth=0.0, zorder=0)
                ax.axvspan(-rpm_axis, 0.0, ymin=0.5, ymax=1.0, facecolor="0.6", alpha=0.06,
                           hatch="///", edgecolor="0.6", linewidth=0.0, zorder=0)
            except Exception:
                pass

            # --- Variability bands (same idea as Band Plots) ---
            if isinstance(bands, dict):
                # η band
                eta = bands.get("eta", None)
                if isinstance(eta, dict) and ("lo" in eta) and ("hi" in eta):
                    ax.fill_between(rpm_mag, eta["lo"], eta["hi"], alpha=0.12,
                                    label="η band (loss uncertainty)" if show_legend else None)
                    rpm_neg = -rpm_mag[::-1]
                    ax.fill_between(rpm_neg, -np.asarray(eta["hi"])[::-1], -np.asarray(eta["lo"])[::-1], alpha=0.12)

                # Downhole VLL band
                vllb = bands.get("vll", None)
                if isinstance(vllb, dict) and ("lo" in vllb) and ("hi" in vllb):
                    ax.fill_between(rpm_mag, vllb["lo"], vllb["hi"], alpha=0.10,
                                    label="VLL band (downhole limit)" if show_legend else None)
                    rpm_neg = -rpm_mag[::-1]
                    ax.fill_between(rpm_neg, -np.asarray(vllb["hi"])[::-1], -np.asarray(vllb["lo"])[::-1], alpha=0.10)

                # Cable length band (with dashed edges to stay visible)
                lb = bands.get("len", None)
                if isinstance(lb, dict) and ("lo" in lb) and ("hi" in lb):
                    ax.fill_between(rpm_mag, lb["lo"], lb["hi"], alpha=0.10,
                                    label="Cable length band" if show_legend else None)
                    rpm_neg = -rpm_mag[::-1]
                    ax.fill_between(rpm_neg, -np.asarray(lb["hi"])[::-1], -np.asarray(lb["lo"])[::-1], alpha=0.10)
                    try:
                        lo = np.asarray(lb["lo"], dtype=float)
                        hi = np.asarray(lb["hi"], dtype=float)
                        ax.plot(rpm_mag, lo, ls="--", lw=0.9, alpha=0.7)
                        ax.plot(rpm_mag, hi, ls="--", lw=0.9, alpha=0.7)
                        ax.plot(rpm_neg, -lo[::-1], ls="--", lw=0.9, alpha=0.7)
                        ax.plot(rpm_neg, -hi[::-1], ls="--", lw=0.9, alpha=0.7)
                    except Exception:
                        pass

            # --- Stonehouse preset band (Low↔High) and nominal line ---
            # QI (+,+)
            ax.fill_between(rpm_mag, tq_lo, tq_hi, alpha=0.16, label="Preset band (Low↔High)" if show_legend else None)
            ax.plot(rpm_mag, tq_nom, lw=2.2, label="Nominal preset" if show_legend else None)

            # QIII (-,-)
            rpm_neg = -rpm_mag[::-1]
            ax.fill_between(rpm_neg, -tq_hi[::-1], -tq_lo[::-1], alpha=0.16)
            ax.plot(rpm_neg, -tq_nom[::-1], lw=2.2)

            # Regen caps (QII/QIV) if enabled by assumptions
            if regen_ok:
                # Ideal symmetric cap (no cable/clamp limitation): equals the motoring cap.
                regen_ideal = np.asarray(tq_nom, dtype=float)

                # Cable-aware cap: back-EMF must exceed surface clamp + cable drop.
                try:
                    m_reg = model_for_regen
                    if not bool(getattr(getattr(m_reg, 'p', None), 'regen_cable_limit_enabled', True)):
                        p_reg = copy.deepcopy(getattr(m_reg, 'p'))
                        p_reg.regen_cable_limit_enabled = True
                        m_reg = SystemModel(p_reg)
                    regen_cable = np.array([m_reg.regen_cap_output_torque_ftlbf(float(r), cable_for_regen)
                                            for r in rpm_mag], float)
                except Exception:
                    regen_cable = np.zeros_like(rpm_mag, dtype=float)

                regen_cable = np.minimum(regen_cable, tq_nom)

                # QIV: +rpm, -tq
                ax.plot(rpm_mag, -regen_ideal, ls=':', lw=1.0, color='0.55',
                        label='Regen cap (ideal)' if show_legend else None)
                ax.plot(rpm_mag, -regen_cable, ls=':', lw=1.8, color='#ff4fbf',
                        label='Regen cap (cable-aware)' if show_legend else None)
                # QII: -rpm, +tq
                ax.plot(rpm_neg, regen_ideal[::-1], ls=':', lw=1.0, color='0.55')
                ax.plot(rpm_neg, regen_cable[::-1], ls=':', lw=1.8, color='#ff4fbf')

            ax.axhline(0.0, lw=1.0)
            ax.axvline(0.0, lw=1.0)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-rpm_axis, rpm_axis)
            # y-axis limits are synced across all 4Q plots (see _sync_quadrant_yaxis)
            ax.set_title(title)
            if ylabel:
                ax.set_ylabel("Output Torque (ft-lbf)")
            if xlabel:
                ax.set_xlabel("Output RPM (signed, CW=+, CCW=-)")

        # Clear axes
        for ax in (
        self.preset_pane.ax1, self.preset_pane.ax2, self.preset_pane.ax3, self.preset_pane.ax4, self.preset_pane.ax5):
            ax.cla()

        # Build low/nom/high envelopes (each includes its own GB parasitic torque)
        # Use fewer points than the main envelope to keep the reference pane snappy.
        n_env = 140

        p_low = _make_params(175.0, 50.0, 55.0, stuck=False)
        p_nom = _make_params(216.0, 158.0, 83.0, stuck=False)
        p_hi = _make_params(216.0, 315.0, 250.0, stuck=False)

        m_low = SystemModel(p_low)
        m_nom = SystemModel(p_nom)
        m_hi = SystemModel(p_hi)

        # --- Variability bands (same knobs as Band Plots) ---
        s_eta_lo, s_eta_hi = self._efficiency_torque_scalers()

        def _torque_band_downhole_vll(params_ref: SystemParams, rpm_mag: np.ndarray,
                                      cable_override: CableParams) -> Optional[dict]:
            # Only if the VLL limit is actively enforced AND the band knob is non-zero.
            if not bool(getattr(params_ref.limits, "enforce_downhole_vll_limit", False)):
                return None
            try:
                dv = float(self.vll_band_vrms.value())
            except Exception:
                dv = 0.0
            if dv <= 0.0:
                return None

            vll_nom = float(params_ref.limits.downhole_vll_rms_limit)

            def _env_tq_for_vll(vll_val: float) -> np.ndarray:
                p = copy.deepcopy(params_ref)
                p.limits.enforce_downhole_vll_limit = True
                p.limits.downhole_vll_rms_limit = float(vll_val)
                m = SystemModel(p)
                rp, tq, _, _ = m.compute_envelope(
                    out_rpm_max=float(np.max(rpm_mag)),
                    n=int(len(rpm_mag)),
                    cable_override=cable_override,
                )
                rp = np.asarray(rp, dtype=float)
                tq = np.asarray(tq, dtype=float)
                if len(rp) != len(rpm_mag) or float(np.max(np.abs(rp - rpm_mag))) > 1e-6:
                    tq = np.interp(rpm_mag, rp, tq)
                return tq

            tq_lo = _env_tq_for_vll(max(1e-6, vll_nom - dv))
            tq_hi = _env_tq_for_vll(vll_nom + dv)
            return {"lo": np.minimum(tq_lo, tq_hi), "hi": np.maximum(tq_lo, tq_hi)}

        def _torque_band_cable_len(model_ref: SystemModel, rpm_mag: np.ndarray,
                                   cable_override: CableParams) -> Optional[dict]:
            if not hasattr(self, "c_len_band"):
                return None
            try:
                dL = float(self.c_len_band.value())
            except Exception:
                dL = 0.0
            if dL <= 0.0:
                return None

            cab0 = copy.deepcopy(cable_override)
            L0 = float(cab0.length_m)
            L_low = max(1.0, L0 - dL)
            L_high = max(1.0, L0 + dL)

            if abs(L_low - L0) < 1e-9 and abs(L_high - L0) < 1e-9:
                return None

            def _env_tq_for_len(Lm: float) -> np.ndarray:
                cab = copy.deepcopy(cab0)
                cab.length_m = float(Lm)
                rp, tq, _, _ = model_ref.compute_envelope(
                    out_rpm_max=float(np.max(rpm_mag)),
                    n=int(len(rpm_mag)),
                    cable_override=cab,
                )
                rp = np.asarray(rp, dtype=float)
                tq = np.asarray(tq, dtype=float)
                if len(rp) != len(rpm_mag) or float(np.max(np.abs(rp - rpm_mag))) > 1e-6:
                    tq = np.interp(rpm_mag, rp, tq)
                return tq

            try:
                curves = [_env_tq_for_len(L0)]
            except Exception:
                curves = [np.zeros_like(rpm_mag, dtype=float)]

            try:
                if abs(L_low - L0) >= 1e-9:
                    curves.append(_env_tq_for_len(L_low))
                if abs(L_high - L0) >= 1e-9:
                    curves.append(_env_tq_for_len(L_high))
            except Exception:
                pass

            if len(curves) <= 1:
                return None
            stack = np.vstack([np.asarray(c, dtype=float) for c in curves])
            return {"lo": np.nanmin(stack, axis=0), "hi": np.nanmax(stack, axis=0)}

        rpm_lo_1, tq_lo_1, _, _ = m_low.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab1)
        rpm_no_1, tq_no_1, _, _ = m_nom.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab1)
        rpm_hi_1, tq_hi_1, _, _ = m_hi.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab1)

        rpm_lo_2, tq_lo_2, _, _ = m_low.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab2)
        rpm_no_2, tq_no_2, _, _ = m_nom.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab2)
        rpm_hi_2, tq_hi_2, _, _ = m_hi.compute_envelope(out_rpm_max=out_rpm_max, n=n_env, cable_override=cab2)

        # Build the same variability bands shown in the Band Plots pane (η, VLL, cable length)
        vll_band_1 = _torque_band_downhole_vll(p_nom, np.asarray(rpm_no_1, dtype=float), cab1)
        vll_band_2 = _torque_band_downhole_vll(p_nom, np.asarray(rpm_no_2, dtype=float), cab2)

        len_band_1 = _torque_band_cable_len(m_nom, np.asarray(rpm_no_1, dtype=float), cab1)
        len_band_2 = _torque_band_cable_len(m_nom, np.asarray(rpm_no_2, dtype=float), cab2)

        bands_1 = {"eta": {"lo": np.asarray(tq_no_1, dtype=float) * s_eta_lo,
                           "hi": np.asarray(tq_no_1, dtype=float) * s_eta_hi}}
        if vll_band_1 is not None:
            bands_1["vll"] = vll_band_1
        if len_band_1 is not None:
            bands_1["len"] = len_band_1

        bands_2 = {"eta": {"lo": np.asarray(tq_no_2, dtype=float) * s_eta_lo,
                           "hi": np.asarray(tq_no_2, dtype=float) * s_eta_hi}}
        if vll_band_2 is not None:
            bands_2["vll"] = vll_band_2
        if len_band_2 is not None:
            bands_2["len"] = len_band_2

        # Plot the 4Q band views
        _plot_4q_band(self.preset_pane.ax1, rpm_no_1, tq_no_1, tq_lo_1, tq_hi_1, m_nom, cab1,
                      "Stonehouse presets band (1 wire/phase)", ylabel=True, xlabel=False, bands=bands_1,
                      show_legend=True)
        _plot_4q_band(self.preset_pane.ax2, rpm_no_2, tq_no_2, tq_lo_2, tq_hi_2, m_nom, cab2,
                      "Stonehouse presets band (2 wires/phase)", ylabel=True, xlabel=True, bands=bands_2,
                      show_legend=True)

        # Compute & plot preset operating points (external-only requirement vs net envelope).
        # For consistency with compute_envelope() (which subtracts parasitics),
        # we compute the required point with parasitic disabled, but we judge feasibility with full params.
        markers = ['o', 's', '^', 'X']
        for i, (name, tob, bha_tc, gb_tc, stuck, stall_tq) in enumerate(presets):
            p_full = _make_params(tob, bha_tc, gb_tc, stuck=stuck, stall_tq_ftlbf=stall_tq)
            p_req = copy.deepcopy(p_full)
            if hasattr(p_req, "parasitic"):
                p_req.parasitic.enabled = False
                p_req.parasitic.tc_nm = 0.0

            m_full = SystemModel(p_full)
            m_req = SystemModel(p_req)

            # Requirement (external-only) point
            res_req = m_req.solve_target(cable_override=cab1)
            x_pt, y_pt = _signed_torque_point(res_req, m_req)

            # Feasibility (full) for 1-wire and 2-wire
            res_full_1 = m_full.solve_target(cable_override=cab1)
            res_full_2 = m_full.solve_target(cable_override=cab2)

            ok1 = bool(res_full_1.feasible)
            ok2 = bool(res_full_2.feasible)

            # Marker style based on feasibility in the corresponding wiring plot
            self.preset_pane.ax1.scatter([x_pt], [y_pt], s=55, marker=markers[i % len(markers)],
                                         alpha=0.95)
            self.preset_pane.ax2.scatter([x_pt], [y_pt], s=55, marker=markers[i % len(markers)],
                                         alpha=0.95)

            # Annotate lightly (avoid clutter)
            tag = "✓" if (ok1 and ok2) else ("✓/✗" if (ok1 != ok2) else "✗")
            for ax in (self.preset_pane.ax1, self.preset_pane.ax2):
                ax.annotate(f"{name} ({tag})", (x_pt, y_pt), textcoords="offset points", xytext=(6, 6), fontsize=8)

        # --- UI target reference (user-entered speed + torque) ---
        try:
            s_dir = float(self.model.out_dir_sign())
        except Exception:
            s_dir = 1.0

        try:
            ui_rpm = float(self.params.target.out_rpm) * s_dir
            ui_tq = float(self.params.target.out_torque_ftlbf) * s_dir
        except Exception:
            ui_rpm, ui_tq = 0.0, 0.0

        for ax in (self.preset_pane.ax1, self.preset_pane.ax2):
            ax.scatter([ui_rpm], [ui_tq], s=95, marker="*", alpha=0.95, label="UI target")
            # ax.scatter([ui_rpm], [0.0], s=40, marker="o", alpha=0.75, label="UI RPM")
            ax.annotate("UI target", (ui_rpm, ui_tq), textcoords="offset points", xytext=(6, -10), fontsize=8)
            # ax.annotate("UI rpm", (ui_rpm, 0.0), textcoords="offset points", xytext=(6, 6), fontsize=8)

        # Legends
        for ax in (self.preset_pane.ax1, self.preset_pane.ax2):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc="best", framealpha=0.85)

            # -------------------------
            # Right-side tables (improved styling)
            # -------------------------
            import textwrap

            # --- gather UI target and UI load values (for table context) ---
            ui_rpm = float(self.params.target.out_rpm)
            ui_tq = float(self.params.target.out_torque_ftlbf)
            ui_dir = str(getattr(self.params, "out_dir", "CW"))

            def _ui_load_components_ftlbf() -> tuple[float, float, float, float]:
                tob_u = 0.0
                tc_bha_u = 0.0
                tc_gb_u = 0.0
                try:
                    if hasattr(self.params, "bha") and bool(getattr(self.params.bha, "enabled", False)):
                        tob_u = float(getattr(self.params.bha, "drilling_tob_ftlbf", 0.0))
                        tc_bha_u = float(nm_to_ft_lbf(getattr(self.params.bha, "fric_tc_nm", 0.0)))
                except Exception:
                    pass
                try:
                    if hasattr(self.params, "parasitic") and bool(getattr(self.params.parasitic, "enabled", False)):
                        tc_gb_u = float(nm_to_ft_lbf(getattr(self.params.parasitic, "tc_nm", 0.0)))
                except Exception:
                    pass
                tot_u = tob_u + tc_bha_u + tc_gb_u
                return tob_u, tc_bha_u, tc_gb_u, tot_u

            ui_tob, ui_tc_bha, ui_tc_gb, ui_total_load = _ui_load_components_ftlbf()

            # ---------- matplotlib table helpers ----------
            def _wrap(s: str, width: int) -> str:
                s = "" if s is None else str(s)
                if s.strip() == "":
                    return "—"
                return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))

            def _make_table(ax, title: str, subtitle: str, col_labels: list[str], rows: list[list[str]],
                            col_widths: list[float], header_bg: str = "#f3f4f6",
                            row_bg1: str = "white", row_bg2: str = "#fafafa",
                            fontsize: int = 9, header_fontsize: int = 9, left_align_cols: tuple[int, ...] = (0,),
                            special_row_bg=None,
                            cell_bg_overrides=None):
                ax.set_axis_off()

                # Title + subtitle
                ax.text(0.01, 0.98, title, va="top", ha="left", fontsize=11, fontweight="bold")
                if subtitle:
                    ax.text(0.01, 0.91, subtitle, va="top", ha="left", fontsize=9.2, color="0.25")

                # Table bbox leaves room for title/subtitle
                tbl = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    cellLoc="center",
                    colLoc="center",
                    loc="upper left",
                    bbox=[0.01, 0.03, 0.98, 0.82],
                )
                tbl.auto_set_font_size(False)

                # Apply widths
                ncols = len(col_labels)
                for c in range(ncols):
                    for r in range(len(rows) + 1):  # +1 header
                        cell = tbl[(r, c)]
                        cell.set_width(col_widths[c])

                # Style header + alternating rows
                for (r, c), cell in tbl.get_celld().items():
                    cell.set_edgecolor("0.15")
                    cell.set_linewidth(0.8)

                    if r == 0:
                        cell.set_facecolor(header_bg)
                        cell.get_text().set_fontweight("bold")
                        cell.get_text().set_fontsize(header_fontsize)
                    else:
                        base_bg = row_bg1 if ((r - 1) % 2 == 0) else row_bg2
                        if special_row_bg and ((r - 1) in special_row_bg):
                            base_bg = special_row_bg[(r - 1)]
                        cell.set_facecolor(base_bg)
                        cell.get_text().set_fontsize(fontsize)

                    if c in left_align_cols:
                        cell.get_text().set_ha("left")

                    # Optional per-cell override (e.g., feasibility coloring)
                    if cell_bg_overrides and ((r, c) in cell_bg_overrides):
                        cell.set_facecolor(cell_bg_overrides[(r, c)])

                return tbl

            # (ax3) Presets table + UI load row
            ax3 = self.preset_pane.ax3

            preset_rows = []
            for (name, tob, bha_tc, gb_tc, stuck, stall_tq) in presets:
                if stuck and (stall_tq is not None):
                    total = float(stall_tq)
                    preset_rows.append([name, "0", "0", "0", f"{total:.0f}"])
                else:
                    total = float(tob) + float(bha_tc) + float(gb_tc)
                    preset_rows.append([name, f"{tob:.0f}", f"{bha_tc:.0f}", f"{gb_tc:.0f}", f"{total:.0f}"])

            # Append UI values row for quick comparison
            preset_rows.append([
                "UI (current)",
                f"{ui_tob:.0f}",
                f"{ui_tc_bha:.0f}",
                f"{ui_tc_gb:.0f}",
                f"{ui_total_load:.0f}",
            ])

            subtitle3 = f"UI target: {ui_rpm:.3f} RPM, {ui_tq:.0f} ft-lbf, out_dir={ui_dir}"
            col_labels3 = ["Preset", "T_TOB", "Tc_BHA", "Tc_GB", "Total"]
            col_widths3 = [0.46, 0.13, 0.13, 0.13, 0.15]
            special3 = {len(preset_rows) - 1: "#e0f2fe"}  # UI row
            # Highlight stall row subtly (if present)
            for ridx, r in enumerate(preset_rows):
                if "Stall" in r[0]:
                    special3[ridx] = "#fff7ed"

            _make_table(
                ax3,
                "Stonehouse presets (ft-lbf) — reference overrides",
                subtitle3,
                col_labels3,
                preset_rows,
                col_widths3,
                left_align_cols=(0,),
                special_row_bg=special3,
                fontsize=9,
                header_fontsize=9,
            )

            # (ax4) Feasibility table (color-coded) + UI row
            ax4 = self.preset_pane.ax4

            def _feas_row(label: str, res1: SolveResult, res2: SolveResult) -> tuple[list[str], str]:
                ok1 = bool(res1.feasible)
                ok2 = bool(res2.feasible)
                s1 = "✓" if ok1 else "✗"
                s2 = "✓" if ok2 else "✗"
                reason = "—"
                if not ok1 and getattr(res1, "reasons", None):
                    reason = res1.reasons[0]
                elif not ok2 and getattr(res2, "reasons", None):
                    reason = res2.reasons[0]
                reason = _wrap(reason, width=34)
                return [label, s1, s2, reason], ("ok" if (ok1 and ok2) else "mixed" if (ok1 != ok2) else "bad")

            # UI row
            # ui_r1 = self.model.solve_target(cable_override=cab1)
            # ui_r2 = self.model.solve_target(cable_override=cab2)

            # UI target-point row (matches the star marker on the plots)
            def _point_feasible(
                    model: SystemModel,
                    cable: CableParams,
                    out_rpm_mag: float,
                    out_tq_mag_ftlbf: float
            ) -> Tuple[bool, str]:
                p = model.p
                G = float(p.gearbox.ratio())
                eta = float(p.gearbox.eff_total())

                rpm_m = float(out_rpm_mag) * G
                omega_m = rpm_to_rad_s(rpm_m)

                # electrical limits → max achievable iq
                iq_max, id_best, im_best, f_e, v_cmd, v_surface_limit, v_dh_phase = model.max_iq_given_limits(
                    rpm_m, cable=cable
                )

                kt_eff = float(model.kt_effective_nm_per_arms())
                tau_extra_nm, _, _ = model.tau_extra_nm(omega_m)

                # required motor-side torque to deliver out_tq (external-only requirement)
                t_out_req_nm = ft_lbf_to_nm(float(out_tq_mag_ftlbf))
                t_gb_in_req_nm = t_out_req_nm / max(1e-12, (G * eta))

                ok_cpl, t_motor_load_req_nm, mag_slip_nm = model._mag_inverse_required_motor_torque_nm(t_gb_in_req_nm)
                if not ok_cpl:
                    return False, f"Magnetic coupler slip limit exceeded (req slip ≈ {mag_slip_nm:.2f} Nm)"

                iq_req = (float(t_motor_load_req_nm) + float(tau_extra_nm)) / max(1e-12, kt_eff)

                if float(iq_req) <= float(iq_max) + 1e-9:
                    return True, "—"

                return False, f"Iq_max={float(iq_max):.3f} Arms < Iq_req={float(iq_req):.3f} Arms"

            def _feas_row_ui_target_point() -> Tuple[List[str], str]:
                ui_rpm = float(self.params.target.out_rpm)
                ui_tq = float(self.params.target.out_torque_ftlbf)

                ok1, why1 = _point_feasible(self.model, cab1, abs(ui_rpm), abs(ui_tq))
                ok2, why2 = _point_feasible(self.model, cab2, abs(ui_rpm), abs(ui_tq))

                s1 = "✓" if ok1 else "✗"
                s2 = "✓" if ok2 else "✗"

                why = why1 if not ok1 else (why2 if not ok2 else "—")
                why = _wrap(why, width=44)

                kind = "ok" if (ok1 and ok2) else ("mixed" if (ok1 != ok2) else "bad")
                return ["UI target (point)", s1, s2, why], kind

            feas_rows = []
            row_kinds = []

            # row, kind = _feas_row("UI (current)", ui_r1, ui_r2)
            row, kind = _feas_row_ui_target_point()
            feas_rows.append(row);
            row_kinds.append(kind)

            for (name, tob, bha_tc, gb_tc, stuck, stall_tq) in presets:
                p_full = _make_params(tob, bha_tc, gb_tc, stuck=stuck, stall_tq_ftlbf=stall_tq)
                m_full = SystemModel(p_full)
                r1 = m_full.solve_target(cable_override=cab1)
                r2 = m_full.solve_target(cable_override=cab2)
                row, kind = _feas_row(name, r1, r2)
                feas_rows.append(row);
                row_kinds.append(kind)

            subtitle4 = f"UI target: {ui_rpm:.3f} RPM, {ui_tq:.0f} ft-lbf, out_dir={ui_dir}"
            col_labels4 = ["Preset", "1w", "2w", "Binding constraint"]
            col_widths4 = [0.44, 0.08, 0.08, 0.40]

            # Color-code feasibility cells (header row is r=0; data start at r=1)
            green = "#dcfce7"
            red = "#fee2e2"
            amber = "#fef9c3"
            ui_row_bg = "#e0f2fe"

            special4 = {0: ui_row_bg}  # UI row is first data row (r-1=0)

            cell_over = {}
            for r_i, row in enumerate(feas_rows, start=1):
                # 1w cell at (r_i,1), 2w at (r_i,2)
                s1 = row[1].strip()
                s2 = row[2].strip()
                cell_over[(r_i, 1)] = (green if s1 == "✓" else red)
                cell_over[(r_i, 2)] = (green if s2 == "✓" else red)

                # If mixed feasibility, tint the preset label row a bit (subtle amber)
                if row_kinds[r_i - 1] == "mixed":
                    # preset cell col 0
                    cell_over[(r_i, 0)] = amber

            tbl4 = _make_table(
                ax4,
                "Feasibility at UI target (wiring comparison)",
                subtitle4,
                col_labels4,
                feas_rows,
                col_widths4,
                left_align_cols=(0, 3),
                special_row_bg=special4,
                cell_bg_overrides=cell_over,
                fontsize=8.8,
                header_fontsize=9,
            )

            # Slightly smaller text for the constraint column; keep it left-aligned
            for r in range(1, len(feas_rows) + 1):
                cell = tbl4[(r, 3)]
                cell.get_text().set_fontsize(8.2)
                cell.get_text().set_ha("left")

            # (ax5) Stacked bar visualization of torque components (continuous presets only)
            self.preset_pane.ax5.clear()
            labels = ["Low", "Nominal", "High"]
            tob_vals = [175.0, 216.0, 216.0]
            bha_vals = [50.0, 158.0, 315.0]
            gb_vals = [55.0, 83.0, 250.0]

            x = np.arange(len(labels))
            self.preset_pane.ax5.bar(x, tob_vals, label="T_TOB")
            self.preset_pane.ax5.bar(x, bha_vals, bottom=tob_vals, label="Tc_BHA")
            bottom2 = (np.asarray(tob_vals) + np.asarray(bha_vals)).tolist()
            self.preset_pane.ax5.bar(x, gb_vals, bottom=bottom2, label="Tc_GB")

            self.preset_pane.ax5.set_xticks(x)
            self.preset_pane.ax5.set_xticklabels(labels)
            self.preset_pane.ax5.set_ylabel("Effective output torque (ft-lbf)")
            self.preset_pane.ax5.set_title("Preset component breakdown (continuous cases)")
            self.preset_pane.ax5.grid(True, axis="y", alpha=0.25)
            self.preset_pane.ax5.legend(fontsize=8, loc="best", framealpha=0.85)

            self.preset_pane.canvas.draw_idle()

    def _axis_max_abs_y(self, ax) -> float:
        """Return max(|y|) over common artists on an axes (lines, scatters, fills).

        Used to keep *all* 4-quadrant torque plots on the same symmetric y-scale.
        """
        m = 0.0

        # Lines (plot)
        try:
            for ln in getattr(ax, "lines", []):
                try:
                    y = np.asarray(ln.get_ydata(orig=False), dtype=float)
                    if y.size:
                        m = max(m, float(np.nanmax(np.abs(y))))
                except Exception:
                    pass
        except Exception:
            pass

        # Collections (scatter, fill_between -> PolyCollection, etc.)
        try:
            for col in getattr(ax, "collections", []):
                # Scatter / PathCollection
                try:
                    if hasattr(col, "get_offsets"):
                        off = np.asarray(col.get_offsets(), dtype=float)
                        if off.size and off.shape[1] >= 2:
                            m = max(m, float(np.nanmax(np.abs(off[:, 1]))))
                except Exception:
                    pass

                # PolyCollection (fill_between) and other path-based collections
                try:
                    for pth in col.get_paths():
                        v = getattr(pth, "vertices", None)
                        if v is None:
                            continue
                        v = np.asarray(v, dtype=float)
                        if v.size and v.shape[1] >= 2:
                            m = max(m, float(np.nanmax(np.abs(v[:, 1]))))
                except Exception:
                    pass
        except Exception:
            pass

        # Patches (bars / rectangles)
        try:
            for patch in getattr(ax, "patches", []):
                try:
                    y0 = float(patch.get_y())
                    h = float(patch.get_height())
                    m = max(m, abs(y0), abs(y0 + h))
                except Exception:
                    pass
        except Exception:
            pass

        if (not math.isfinite(m)) or (m <= 0.0):
            return 0.0
        return float(m)

    def _sync_quadrant_yaxis(self):
        """Sync y-axis scaling across all 4Q torque plots using the global max(|y|)."""
        axes = []

        # Envelope torque plot
        try:
            axes.append(self.mpl_env.ax_env)
        except Exception:
            pass

        # Band plots (4Q)
        try:
            axes.extend([self.band_pane.ax1, self.band_pane.ax2])
        except Exception:
            pass

        # Stonehouse preset plots (4Q)
        try:
            axes.extend([self.preset_pane.ax1, self.preset_pane.ax2])
        except Exception:
            pass

        axes = [ax for ax in axes if ax is not None]
        if not axes:
            return

        y_max = 0.0
        for ax in axes:
            y_max = max(y_max, self._axis_max_abs_y(ax))

        # Fallback if plots are empty
        if (not math.isfinite(y_max)) or (y_max <= 0.0):
            y_max = 1.0

        # Pad + round to "nice" steps so it doesn't jitter on tiny changes
        y_max *= 1.12
        y_max = max(y_max, 50.0)
        step = 50.0 if y_max <= 1500.0 else 100.0
        y_max = math.ceil(y_max / step) * step

        for ax in axes:
            ax.set_ylim(-y_max, y_max)

    def _plot_direction_risk(self, res_sel: SolveResult):
        """Render Direction Risk tab with two plots (side-by-side).

        Left: CCW margin distribution m = (Tc - TOB), including sign-flip (m=0) and guardrail (m>=T_margin)
              + inset TOB vs Tc map for intuition.
        Right: Scenario bar summary for UI + Low/Nominal/High, including UI what-if band.
        """
        if not hasattr(self, "dirrisk_pane"):
            return

        p = self.params
        pane: DirectionRiskPane = self.dirrisk_pane

        axH = pane.ax_hist
        axB = pane.ax_bars
        axH.cla()
        axB.cla()

        # -----------------------------
        # Inputs (UI)
        # -----------------------------
        bha_on = bool(getattr(p.bha, 'enabled', False))
        par_on = bool(getattr(p.parasitic, 'enabled', False))

        tob_ui = abs(float(getattr(p.bha, 'drilling_tob_ftlbf', 0.0))) if bha_on else 0.0
        tc_bha = nm_to_ft_lbf(abs(float(getattr(p.bha, 'fric_tc_nm', 0.0)))) if bha_on else 0.0
        tc_par = nm_to_ft_lbf(abs(float(getattr(p.parasitic, 'tc_nm', 0.0)))) if par_on else 0.0
        tc_ui = tc_bha + tc_par

        # Stonehouse-style points (TOB, Tc) used for quick decision context
        presets = [
            ("Low", 175.0, 50.0 + 55.0),
            ("Nominal", 216.0, 158.0 + 83.0),
            ("High", 216.0, 315.0 + 250.0),
        ]

        # Margin definition
        m_ui = tc_ui - tob_ui

        # What-if uncertainty (kept simple and explicit)
        tob_unc_pct = float(getattr(p, 'ccw_risk_tob_unc_pct', 0.25))
        tc_unc_pct = float(getattr(p, 'ccw_risk_tc_unc_pct', 0.50))
        tob_unc_pct = float(np.clip(tob_unc_pct, 0.0, 1.0))
        tc_unc_pct = float(np.clip(tc_unc_pct, 0.0, 1.0))

        tob_lo = max(0.0, tob_ui * (1.0 - tob_unc_pct))
        tob_hi = max(tob_lo, tob_ui * (1.0 + tob_unc_pct))
        tc_lo = max(0.0, tc_ui * (1.0 - tc_unc_pct))
        tc_hi = max(tc_lo, tc_ui * (1.0 + tc_unc_pct))

        # Worst/best margin across the what-if window
        m_lo = tc_lo - tob_hi
        m_hi = tc_hi - tob_lo

        # Monte Carlo distribution for the left plot
        rng = np.random.default_rng(1234)
        N = 2400
        tob_s = rng.uniform(tob_lo, tob_hi, size=N)
        tc_s = rng.uniform(tc_lo, tc_hi, size=N)
        m_s = tc_s - tob_s

        # Guardrail (engineering choice)
        T_margin = 50.0  # ft-lbf

        p_brake = float(np.mean(m_s < 0.0)) if N else 0.0
        p_guard_fail = float(np.mean(m_s < T_margin)) if N else 0.0

        # -----------------------------
        # LEFT: histogram of m
        # -----------------------------
        axH.set_title('CCW robustness: margin (Tc - TOB) with what-if uncertainty')
        axH.set_xlabel('Margin m = Tc - TOB (ft-lbf)   [m<0 → braking demanded]')
        axH.set_ylabel('Relative likelihood')
        axH.grid(True, alpha=0.25)

        if N:
            m_min = float(np.nanmin(m_s))
            m_max = float(np.nanmax(m_s))
        else:
            m_min, m_max = -150.0, 150.0

        pad = 0.15 * max(60.0, (m_max - m_min))
        m_min -= pad
        m_max += pad

        bins = np.linspace(m_min, m_max, 42)
        axH.hist(m_s, bins=bins, density=True, alpha=0.22, edgecolor='none')

        # Shade braking region + thresholds
        axH.axvspan(m_min, 0.0, alpha=0.10, hatch='///', edgecolor='none', label='CCW needs braking (m<0)')
        axH.axvline(0.0, linestyle='--', linewidth=1.3, label='Boundary: m=0 (sign flip)')
        axH.axvline(T_margin, linestyle=':', linewidth=1.3, label=f'Guardrail: m ≥ {T_margin:.0f} ft-lbf')

        # UI point marker + what-if band (drawn at a fixed y in axes coords)
        yA = 0.90
        axH.plot([m_lo, m_hi], [yA, yA], transform=axH.get_xaxis_transform(), linewidth=2.2, color='0.25')
        axH.scatter([m_ui], [yA], transform=axH.get_xaxis_transform(), marker='*', s=180, label='UI point')
        axH.text(m_ui, yA + 0.035, f"UI m={m_ui:.0f}", transform=axH.get_xaxis_transform(),
                 fontsize=9, ha='center', va='bottom')

        # Traffic-light badge
        green_th = 0.05
        yellow_th = 0.20
        if p_guard_fail <= green_th:
            badge_label = "CCW default: OK (robust)"
            badge_fc, badge_ec = "#d1fae5", "#065f46"
        elif p_guard_fail <= yellow_th:
            badge_label = "CCW default: Conditional"
            badge_fc, badge_ec = "#fef3c7", "#92400e"
        else:
            badge_label = "CCW default: NOT recommended"
            badge_fc, badge_ec = "#fee2e2", "#991b1b"

        axH.text(0.98, 0.98,
                 f"{badge_label}\n"
                 f"P(m<{T_margin:.0f})={p_guard_fail * 100:.0f}%   P(m<0)={p_brake * 100:.0f}%",
                 transform=axH.transAxes, ha='right', va='top', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.45', facecolor=badge_fc, alpha=0.95,
                           edgecolor=badge_ec, linewidth=1.2))

        axH.text(0.98, 0.82,
                 f"what-if: TOB ±{tob_unc_pct * 100:.0f}%, Tc ±{tc_unc_pct * 100:.0f}%\n"
                 f"Guardrail: m ≥ {T_margin:.0f} ft-lbf",
                 transform=axH.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.88, edgecolor='0.75'))

        axH.legend(loc='upper left', fontsize=8)

        # Inset: TOB vs Tc sign-flip map
        try:
            ins = axH.inset_axes([0.06, 0.08, 0.38, 0.33])  # [x0,y0,w,h] in axes fraction
            ins.set_title('TOB vs Tc', fontsize=9)
            ins.grid(True, alpha=0.18)

            # Limits to include UI + all preset points
            xs = [tc_ui] + [tc for (_n, _tob, tc) in presets]
            ys = [tob_ui] + [tob for (_n, tob, _tc) in presets]
            x_max = max(xs) * 1.05 + 10.0
            y_max = max(ys) * 1.05 + 10.0
            x = np.linspace(0.0, x_max, 200)

            ins.plot(x, x, linestyle='--', linewidth=1.0)
            ins.plot(x, np.maximum(0.0, x - T_margin), linestyle=':', linewidth=1.0)

            # Points
            ins.scatter([tc_ui], [tob_ui], marker='*', s=70)
            for name, tob_i, tc_i in presets:
                ins.scatter([tc_i], [tob_i], s=25)

            ins.set_xlim(0.0, x_max)
            ins.set_ylim(0.0, y_max)

        except Exception:
            pass

        # -----------------------------
        # RIGHT: scenario margin bars
        # -----------------------------
        axB.set_title('Why CCW is risky: CCW relies on an uncertain sign (Tc vs TOB)')
        axB.set_xlabel('')
        axB.set_ylabel('Margin m = Tc - TOB (ft-lbf)')
        axB.grid(True, axis='y', alpha=0.25)

        labels = ['UI', 'Low', 'Nominal', 'High']
        vals = [m_ui] + [tc - tob for (_n, tob, tc) in presets]

        def _bar_color(v: float) -> str:
            if v < 0.0:
                return '#fee2e2'  # light red
            if v < T_margin:
                return '#fef3c7'  # light yellow
            return '#d1fae5'  # light green

        colors = [_bar_color(v) for v in vals]
        x_idx = np.arange(len(labels), dtype=float)
        axB.bar(x_idx, vals, width=0.62, color=colors, edgecolor='0.20', linewidth=0.8, alpha=0.90)

        # Threshold lines
        axB.axhline(0.0, linestyle='--', linewidth=1.2, color='0.25')
        axB.axhline(T_margin, linestyle=':', linewidth=1.2, color='0.25')

        # UI what-if band (vertical whisker) + star marker
        axB.plot([x_idx[0], x_idx[0]], [m_lo, m_hi], color='0.10', linewidth=2.2)
        axB.scatter([x_idx[0]], [m_ui], marker='*', s=150)

        # Labels above bars (simple)
        for i, v in enumerate(vals):
            axB.text(x_idx[i], v + (6.0 if v >= 0 else -6.0),
                     f"{labels[i]}", ha='center', va=('bottom' if v >= 0 else 'top'), fontsize=9)

        axB.set_xticks(x_idx)
        axB.set_xticklabels(labels)

        # Y limits with padding
        y_min = min([m_lo, min(vals)])
        y_max = max([m_hi, max(vals), T_margin])
        y_pad = 0.18 * max(60.0, (y_max - y_min))
        axB.set_ylim(y_min - y_pad, y_max + y_pad)

        # Decision rule callout
        axB.text(0.98, 0.92,
                 "Decision rule: m = Tc - TOB\n"
                 "m<0 → CCW needs braking\n"
                 f"m≥{T_margin:.0f} → robust",
                 transform=axB.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.75'))

        axB.text(0.02, 0.06,
                 f"UI what-if: TOB ±{tob_unc_pct * 100:.0f}%, Tc ±{tc_unc_pct * 100:.0f}%",
                 transform=axB.transAxes, ha='left', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.75'))

        pane.canvas.draw_idle()

    def _plot_tables(self):

        """Render decision tables.

        Tables focus on quick go/no-go style comparisons:
          1) Max torque vs RPM with combined tolerance band (η ±Δη, VLL ±ΔV, L ±ΔL)
          2) Voltage + current values at the max-torque envelope points
          3) Cable copper loss at max torque
          4) Binding constraint classification (I / Vsurf / Vdh)
        """
        if not hasattr(self, "tables_pane"):
            return

        pane = self.tables_pane

        # Clear axes
        for ax in (pane.ax1, pane.ax2, pane.ax3, pane.ax4):
            ax.cla()

        # Sampling points requested
        rpm_pts = np.round(np.arange(0.0, 5.0 + 0.5, 0.5), 3).astype(float)
        out_rpm_max = float(np.max(rpm_pts))

        # Wiring overrides
        cab1 = copy.deepcopy(self.params.cable)
        cab1.wires_per_phase = 1
        cab2 = copy.deepcopy(self.params.cable)
        cab2.wires_per_phase = 2

        # Band knobs (for subtitle/context)
        try:
            d_eta = float(self.eta_band_pu.value())
        except Exception:
            d_eta = 0.0
        try:
            d_vll = float(self.vll_band_vrms.value())
        except Exception:
            d_vll = 0.0
        try:
            d_len = float(self.c_len_band.value()) if hasattr(self, "c_len_band") else 0.0
        except Exception:
            d_len = 0.0

        vll_eff = self._effective_downhole_vll_limit()
        target_tq = float(abs(getattr(self.params.target, "out_torque_ftlbf", 0.0)))

        # ---------- helpers ----------
        def _wrap(s: str, width: int) -> str:
            s = "" if s is None else str(s)
            s = s.strip()
            if not s:
                return "—"
            return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))

        def _make_table(ax, title: str, subtitle: str, col_labels: list[str], rows: list[list[str]],
                        col_widths: list[float], header_bg: str = "#f3f4f6",
                        row_bg1: str = "white", row_bg2: str = "#fafafa",
                        fontsize: float = 8.8, header_fontsize: float = 9.0,
                        left_align_cols: tuple[int, ...] = (0,),
                        special_row_bg=None,
                        cell_bg_overrides=None):
            ax.set_axis_off()
            ax.text(0.01, 0.98, title, va="top", ha="left", fontsize=11, fontweight="bold")
            if subtitle:
                ax.text(0.01, 0.91, subtitle, va="top", ha="left", fontsize=9.1, color="0.25")

            tbl = ax.table(
                cellText=rows,
                colLabels=col_labels,
                cellLoc="center",
                colLoc="center",
                loc="upper left",
                bbox=[0.01, 0.03, 0.98, 0.82],
            )
            tbl.auto_set_font_size(False)

            # Apply widths
            ncols = len(col_labels)
            for c in range(ncols):
                for r in range(len(rows) + 1):
                    cell = tbl[(r, c)]
                    cell.set_width(col_widths[c])

            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("0.15")
                cell.set_linewidth(0.8)
                if r == 0:
                    cell.set_facecolor(header_bg)
                    cell.get_text().set_fontweight("bold")
                    cell.get_text().set_fontsize(header_fontsize)
                else:
                    base_bg = row_bg1 if ((r - 1) % 2 == 0) else row_bg2
                    if special_row_bg and ((r - 1) in special_row_bg):
                        base_bg = special_row_bg[(r - 1)]
                    cell.set_facecolor(base_bg)
                    cell.get_text().set_fontsize(fontsize)

                if c in left_align_cols:
                    cell.get_text().set_ha("left")

                if cell_bg_overrides and ((r, c) in cell_bg_overrides):
                    cell.set_facecolor(cell_bg_overrides[(r, c)])

            return tbl

        def _sample_env(cab: CableParams):
            rp, tq, iq, loss = self.model.compute_envelope(
                out_rpm_max=out_rpm_max,
                n=int(len(rpm_pts)),
                cable_override=cab
            )
            rp = np.asarray(rp, dtype=float)
            tq = np.asarray(tq, dtype=float)
            iq = np.asarray(iq, dtype=float)
            loss = np.asarray(loss, dtype=float)
            if len(rp) != len(rpm_pts) or float(np.max(np.abs(rp - rpm_pts))) > 1e-6:
                tq = np.interp(rpm_pts, rp, tq)
                iq = np.interp(rpm_pts, rp, iq)
                loss = np.interp(rpm_pts, rp, loss)
            else:
                # Align explicitly
                tq = tq.copy()
                iq = iq.copy()
                loss = loss.copy()

            # Voltage + constraint classification at the same points
            G = float(self.params.gearbox.ratio())
            vll_motor = np.zeros_like(rpm_pts, dtype=float)
            vll_inv_req = np.zeros_like(rpm_pts, dtype=float)
            vll_cmd = np.zeros_like(rpm_pts, dtype=float)
            binding = []

            for i, out_rpm in enumerate(rpm_pts):
                motor_rpm = float(out_rpm) * G

                # Find operating point that achieved max torque (iq_max, id_best)
                iq_max, id_best, _, f_e, v_cmd_phase, _, v_dh_phase = self.model.max_iq_given_limits(motor_rpm, cab)

                v_motor = float(self.model.motor_voltage_required_phase_rms(motor_rpm, float(id_best), float(iq_max)))
                v_drop = float(self.model.cable_drop_phase_rms(motor_rpm, float(iq_max), cab))
                v_node = v_motor + v_drop
                v_inv = float(
                    self.model.inverter_voltage_required_phase_rms(motor_rpm, float(v_node), float(abs(iq_max))))

                vll_motor[i] = math.sqrt(3.0) * v_motor
                vll_inv_req[i] = math.sqrt(3.0) * v_inv
                vll_cmd[i] = math.sqrt(3.0) * float(v_cmd_phase)

                # Constraint classification (best-effort; multiple can be tight)
                flags = []
                i_lim = float(cab.i_phase_limit())
                if i_lim > 1e-9 and (abs(float(iq_max) - i_lim) / i_lim) < 0.01:
                    flags.append("I")

                if float(v_cmd_phase) > 1e-6:
                    if (abs(float(v_inv) - float(v_cmd_phase)) / float(v_cmd_phase)) < 0.01:
                        flags.append("Vsurf")

                if v_dh_phase is not None and float(v_dh_phase) > 1e-9:
                    if (abs(float(v_motor) - float(v_dh_phase)) / float(v_dh_phase)) < 0.01:
                        flags.append("Vdh")

                binding.append("+".join(flags) if flags else "—")

            return tq, iq, loss, vll_motor, vll_inv_req, vll_cmd, binding

        def _combined_torque_band(rpm_mag: np.ndarray, tq_mag: np.ndarray, cab: CableParams):
            # Efficiency band
            s_eta_lo, s_eta_hi = self._efficiency_torque_scalers()
            lo = np.asarray(tq_mag, dtype=float) * float(s_eta_lo)
            hi = np.asarray(tq_mag, dtype=float) * float(s_eta_hi)

            # Downhole VLL band
            vll_band = self._downhole_vll_band(rpm_mag, cable_override=cab)
            if vll_band is not None:
                lo = np.minimum(lo, np.asarray(vll_band["lo"], dtype=float))
                hi = np.maximum(hi, np.asarray(vll_band["hi"], dtype=float))

            # Cable length band
            len_band = self._cable_length_torque_band(rpm_mag, tq_mag, cable_override=cab)
            if len_band is not None:
                lo = np.minimum(lo, np.asarray(len_band["lo"], dtype=float))
                hi = np.maximum(hi, np.asarray(len_band["hi"], dtype=float))

            return lo, hi

        # ---------- compute data ----------
        tq1, iq1, loss1, vllm1, vlli1, vllc1, bind1 = _sample_env(cab1)
        tq2, iq2, loss2, vllm2, vlli2, vllc2, bind2 = _sample_env(cab2)

        tq1_lo, tq1_hi = _combined_torque_band(rpm_pts, tq1, cab1)
        tq2_lo, tq2_hi = _combined_torque_band(rpm_pts, tq2, cab2)

        # ---------- Table 1: Max torque with tolerance ----------
        col_labels = ["Wiring"] + [f"{r:g}" for r in rpm_pts]
        col_widths = [0.22] + [max(0.06, (1.0 - 0.22) / max(1, len(rpm_pts)))] * len(rpm_pts)

        def _cell_tq(nom: float, lo: float, hi: float) -> str:
            return f"{nom:.0f}\n({lo:.0f}–{hi:.0f})"

        rows_tq = [
            ["1 wire/phase"] + [_cell_tq(float(tq1[i]), float(tq1_lo[i]), float(tq1_hi[i])) for i in
                                range(len(rpm_pts))],
            ["2 wires/phase"] + [_cell_tq(float(tq2[i]), float(tq2_lo[i]), float(tq2_hi[i])) for i in
                                 range(len(rpm_pts))],
        ]

        green = "#dcfce7"
        red = "#fee2e2"
        amber = "#fef9c3"
        cell_over_1 = {}
        # Color by whether combined band meets the current UI target torque
        if target_tq > 0.0:
            for j in range(len(rpm_pts)):
                # 1w row r=1, col=1+j
                lo = float(tq1_lo[j]);
                hi = float(tq1_hi[j])
                cell_over_1[(1, 1 + j)] = (green if lo >= target_tq else (red if hi < target_tq else amber))
                lo = float(tq2_lo[j]);
                hi = float(tq2_hi[j])
                cell_over_1[(2, 1 + j)] = (green if lo >= target_tq else (red if hi < target_tq else amber))

        subtitle1 = (
            f"Cells: Nominal max torque with combined band (lo–hi). "
            f"Band knobs: Δη={d_eta:.2f} pu, ΔVLL={d_vll:.0f} Vrms, ΔL={d_len:.0f} m. "
            f"Colored vs UI target torque={target_tq:.0f} ft-lbf."
        )
        tbl1 = _make_table(
            pane.ax1,
            "Max torque vs RPM (with tolerance from bands)",
            _wrap(subtitle1, 88),
            col_labels,
            rows_tq,
            col_widths,
            left_align_cols=(0,),
            cell_bg_overrides=cell_over_1,
        )

        try:
            tbl1.scale(1.0, 1.25)
        except Exception:
            pass

        # ---------- Table 2: Voltage + current ----------
        # Build rows: Iq (1w/2w), VLL motor (1w/2w), VLL inverter required (1w/2w), VLL command (surface), VLL downhole limit
        rows_v = []

        def _fmt_arr(a, fmt):
            return [fmt(x) for x in a]

        rows_v.append(["Iq_max 1w (Arms)"] + _fmt_arr(iq1, lambda x: f"{float(x):.3f}"))
        rows_v.append(["Iq_max 2w (Arms)"] + _fmt_arr(iq2, lambda x: f"{float(x):.3f}"))
        rows_v.append(["VLL_motor 1w (Vrms)"] + _fmt_arr(vllm1, lambda x: f"{float(x):.0f}"))
        rows_v.append(["VLL_motor 2w (Vrms)"] + _fmt_arr(vllm2, lambda x: f"{float(x):.0f}"))
        rows_v.append(["VLL_inv_req 1w (Vrms)"] + _fmt_arr(vlli1, lambda x: f"{float(x):.0f}"))
        rows_v.append(["VLL_inv_req 2w (Vrms)"] + _fmt_arr(vlli2, lambda x: f"{float(x):.0f}"))
        rows_v.append(["VLL_cmd (surface) (Vrms)"] + _fmt_arr(vllc1, lambda x: f"{float(x):.0f}"))
        rows_v.append(
            ["VLL_downhole_eff (Vrms)"] + [("—" if vll_eff is None else f"{float(vll_eff):.0f}") for _ in rpm_pts])

        col_widths2 = [0.32] + [max(0.06, (1.0 - 0.32) / max(1, len(rpm_pts)))] * len(rpm_pts)

        cell_over_2 = {}
        # Highlight downhole VLL exceedance (motor terminals)
        if vll_eff is not None and float(vll_eff) > 1e-9:
            for j in range(len(rpm_pts)):
                # rows: VLL_motor 1w is row index 3 in rows_v (0-based), but table cell coords are (r,c) with header at r=0
                # rows_v indices: 0..7 => table rows r=1..8
                if float(vllm1[j]) > float(vll_eff) + 1e-6:
                    cell_over_2[(1 + 2, 1 + j)] = red  # VLL_motor 1w
                if float(vllm2[j]) > float(vll_eff) + 1e-6:
                    cell_over_2[(1 + 3, 1 + j)] = red  # VLL_motor 2w

        # Highlight surface voltage overrun (inv required > cmd)
        for j in range(len(rpm_pts)):
            if float(vlli1[j]) > float(vllc1[j]) + 1e-6:
                cell_over_2[(1 + 4, 1 + j)] = red
            if float(vlli2[j]) > float(vllc2[j]) + 1e-6:
                cell_over_2[(1 + 5, 1 + j)] = red

        subtitle2 = (
            "Values at the max-torque envelope points. "
            "VLL_motor is at motor terminals; VLL_inv_req is the surface inverter phase-voltage requirement converted to VLL."
        )
        _make_table(
            pane.ax2,
            "Voltage + current at max-torque points",
            _wrap(subtitle2, 88),
            col_labels,
            rows_v,
            col_widths2,
            left_align_cols=(0,),
            cell_bg_overrides=cell_over_2,
            fontsize=8.2,
            header_fontsize=9.0,
        )

        # ---------- Table 3: Cable copper loss ----------
        rows_loss = [
            ["Loss 1w (W)"] + [f"{float(x):.0f}" for x in loss1],
            ["Loss 2w (W)"] + [f"{float(x):.0f}" for x in loss2],
            ["2w/1w (%)"] + [("—" if float(loss1[i]) <= 1e-9 else f"{100.0 * float(loss2[i]) / float(loss1[i]):.1f}")
                             for i in range(len(rpm_pts))],
        ]
        col_widths3 = [0.26] + [max(0.06, (1.0 - 0.26) / max(1, len(rpm_pts)))] * len(rpm_pts)

        cell_over_3 = {}
        for j in range(len(rpm_pts)):
            if float(loss1[j]) > 1e-9:
                ratio = float(loss2[j]) / float(loss1[j])
                cell_over_3[(3, 1 + j)] = (green if ratio < 1.0 else red)

        subtitle3 = (
            "Cable copper loss reported by the envelope solver at the max-torque operating point."
        )
        _make_table(
            pane.ax3,
            "Cable copper loss vs RPM",
            _wrap(subtitle3, 88),
            col_labels,
            rows_loss,
            col_widths3,
            left_align_cols=(0,),
            cell_bg_overrides=cell_over_3,
        )

        # ---------- Table 4: Binding constraint classification ----------
        rows_bind = [
            ["1 wire/phase"] + [str(x) for x in bind1],
            ["2 wires/phase"] + [str(x) for x in bind2],
        ]
        col_widths4 = [0.22] + [max(0.06, (1.0 - 0.22) / max(1, len(rpm_pts)))] * len(rpm_pts)

        cell_over_4 = {}
        blue = "#e0f2fe"
        grey = "#f3f4f6"
        for j in range(len(rpm_pts)):
            b = str(bind1[j])
            if "Vdh" in b:
                cell_over_4[(1, 1 + j)] = red
            elif "Vsurf" in b:
                cell_over_4[(1, 1 + j)] = amber
            elif "I" in b:
                cell_over_4[(1, 1 + j)] = blue
            else:
                cell_over_4[(1, 1 + j)] = grey

            b = str(bind2[j])
            if "Vdh" in b:
                cell_over_4[(2, 1 + j)] = red
            elif "Vsurf" in b:
                cell_over_4[(2, 1 + j)] = amber
            elif "I" in b:
                cell_over_4[(2, 1 + j)] = blue
            else:
                cell_over_4[(2, 1 + j)] = grey

        subtitle4 = "Best-effort identification of the tight constraint at the max-torque point: I, Vsurf, Vdh (can stack)."
        _make_table(
            pane.ax4,
            "Binding constraint at max torque",
            _wrap(subtitle4, 88),
            col_labels,
            rows_bind,
            col_widths4,
            left_align_cols=(0,),
            cell_bg_overrides=cell_over_4,
        )

        pane.canvas.draw_idle()

    def _linspace_safe(self, a: float, b: float, n: int) -> np.ndarray:
        """Return a monotonic linspace with sane bounds and at least 2 points."""
        n_i = max(2, int(n))
        a_f = float(a)
        b_f = float(b)
        if b_f < a_f:
            a_f, b_f = b_f, a_f
        return np.linspace(a_f, b_f, n_i)

    def _sweep_ratios(self) -> np.ndarray:
        return self._linspace_safe(self.sw_ratio_min.value(), self.sw_ratio_max.value(), self.sw_points.value())

    def _sweep_ke(self) -> np.ndarray:
        return self._linspace_safe(self.sw_ke_min.value(), self.sw_ke_max.value(), self.sw_points.value())

    def _sweep_lengths(self) -> np.ndarray:
        return self._linspace_safe(self.sw_len_min.value(), self.sw_len_max.value(), self.sw_points.value())

    def _sweep_inv_vlim(self) -> np.ndarray:
        return self._linspace_safe(self.sw_inv_vlim_min.value(), self.sw_inv_vlim_max.value(), self.sw_points.value())

    def _sweep_basef(self) -> np.ndarray:
        return self._linspace_safe(self.sw_basef_min.value(), self.sw_basef_max.value(), self.sw_points.value())

    def _sweep_vboost(self) -> np.ndarray:
        return self._linspace_safe(self.sw_vboost_min.value(), self.sw_vboost_max.value(), self.sw_points.value())

    def _sweep_pole_pairs(self) -> np.ndarray:
        a = int(self.sw_pp_min.value())
        b = int(self.sw_pp_max.value())
        if b < a:
            a, b = b, a
        return np.arange(a, b + 1, dtype=int)

    def _sweep_dh_vphase(self) -> np.ndarray:
        return self._linspace_safe(self.sw_dh_vph_min.value(), self.sw_dh_vph_max.value(), self.sw_points.value())

    def _sweep_dh_vll(self) -> np.ndarray:
        return self._linspace_safe(self.sw_dh_vll_min.value(), self.sw_dh_vll_max.value(), self.sw_points.value())

    def _plot_architecture(self) -> None:
        '''Draw a live, readable block diagram of the *static* model configured by the UI.'''
        if not hasattr(self, "arch_pane") or self.arch_pane is None:
            return

        p = getattr(self, "params", None)
        if p is None:
            return

        pane = self.arch_pane
        ax = pane.ax_diag
        axs = pane.ax_feat

        # --- flags ---
        mag_on = bool(getattr(getattr(p, "mag_coupler", None), "enabled", True))
        paras_on = bool(getattr(getattr(p, "parasitic", None), "enabled", True))
        bha_on = bool(getattr(getattr(p, "bha", None), "enabled", True))
        stuck_on = bool(getattr(p, "stuck_mode", False))

        backdriveable = bool(getattr(getattr(p, "gearbox", None), "backdrivable", True))
        brake_path = bool(getattr(p, "braking_path_available", True))
        regen_ok = brake_path and backdriveable

        cable_regen_lim_on = bool(getattr(p, "regen_cable_limit_enabled", False))
        clamp = float(getattr(p, "regen_surface_clamp_frac", 1.0))
        brake_pcap_on = bool(getattr(p, "brake_power_limit_enabled", False))
        pcap_kw = float(getattr(p, "brake_power_kw_max", 0.0))

        enforce_vll = bool(getattr(getattr(p, "limits", None), "enforce_downhole_vll_limit", True))
        vll_lim = float(getattr(getattr(p, "limits", None), "downhole_v_ll_rms_limit", 0.0))
        enforce_vphase = bool(getattr(getattr(p, "limits", None), "enforce_downhole_vphase_limit", False))
        vph_lim = float(getattr(getattr(p, "limits", None), "downhole_v_phase_rms_limit", 0.0))

        cmd_dir = str(getattr(p, "out_dir", "CW"))
        op_mode = "STALL" if stuck_on else "SPEED"

        # --- styling (neutral, "engineering" look) ---
        ax.clear()
        axs.clear()
        ax.set_axis_off()
        axs.set_axis_off()
        pane.fig.patch.set_facecolor("white")

        enabled_face = "#ECFDF3"
        enabled_edge = "#027A48"
        disabled_face = "#F2F4F7"
        disabled_edge = "#98A2B3"
        title_color = "#101828"
        sub_color = "#475467"
        faint = "#667085"

        def _block(ax_, x, y, w, h, title, subtitle, enabled=True):
            import matplotlib.patches as patches
            face = enabled_face if enabled else disabled_face
            edge = enabled_edge if enabled else disabled_edge
            lw = 1.8 if enabled else 1.2
            box = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=lw, edgecolor=edge, facecolor=face
            )
            ax_.add_patch(box)
            ax_.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                     fontsize=12, weight="bold", color=title_color)
            ax_.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                     fontsize=9, color=sub_color, wrap=True)

        def _arrow(ax_, x0, y0, x1, y1, enabled=True):
            import matplotlib.patches as patches
            col = enabled_edge if enabled else disabled_edge
            arr = patches.FancyArrowPatch((x0, y0), (x1, y1),
                                          arrowstyle="-|>", mutation_scale=12,
                                          linewidth=1.4, color=col)
            ax_.add_patch(arr)

        # --- diagram blocks (left) ---
        blocks = [
            ("Surface\nInverter", f"{p.vf.modulation} | Vdc={p.vf.vdc_link_v:.0f} V", True),
            ("Sine\nFilter",
             f"Lf={p.sine_filter.lf_h * 1e3:.2f} mH | Cf={p.sine_filter.cf_f * 1e6:.1f} µF ({p.sine_filter.cap_connection})",
             bool(p.sine_filter.enabled)),
            ("Heptacable\n(RL lumped)", f"L={p.cable.length_m / 1000:.1f} km | {p.cable.wires_per_phase} wire/phase",
             True),
            ("PMSM\nMotor", f"{p.motor.pole_pairs} pp | Kt={p.motor.kt_nm_per_arms:.2f} Nm/Arms", True),
            ("Magnetic\nCoupler", f"Tslip={nm_to_ft_lbf(p.mag_coupler.t_slip_nm):.0f} ft-lbf", mag_on),
            ("Gearbox",
             f"G={p.gearbox.ratio():.0f}:1 | η={p.gearbox.eff_total():.3f}\nParasitics: {'ON' if paras_on else 'OFF'}",
             True),
            (
                "CCRS\nOutput", f"Cmd: {cmd_dir} | {op_mode}\nRPMcmd={0.0 if stuck_on else p.target.out_rpm:.2f}",
                True),
            ("BHA Load\n(External)",
             f"TOB={p.bha.drilling_tob_ftlbf:.0f} ft-lbf\nFric Tc={p.bha.fric_tc_nm:.0f} Nm", bha_on),
        ]

        # Axis coordinate system for diagram
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.02, 0.96, "Live Architecture (static model)", fontsize=15, weight="bold", color=title_color, va="top")
        ax.text(0.02, 0.91, "Green = included, Gray = bypassed/disabled", fontsize=9.5, color=faint, va="top")

        # Layout: single row of blocks
        left = 0.03
        right = 0.97
        y = 0.48
        h = 0.28
        n = len(blocks)
        gap = 0.012
        w = (right - left - gap * (n - 1)) / n

        for i, (name, sub, enabled) in enumerate(blocks):
            x = left + i * (w + gap)
            _block(ax, x, y, w, h, name, sub, enabled=enabled)

        # arrows
        ymid = y + h / 2
        for i in range(n - 1):
            x0 = left + (i + 1) * w + i * gap
            x1 = x0 + gap
            en = blocks[i][2] and blocks[i + 1][2]
            _arrow(ax, x0, ymid, x1, ymid, enabled=en)

        # Status pill
        import matplotlib.patches as patches
        pill_face = enabled_face if regen_ok else disabled_face
        pill_edge = enabled_edge if regen_ok else disabled_edge
        pill = patches.FancyBboxPatch((0.03, 0.10), 0.60, 0.09,
                                      boxstyle="round,pad=0.012,rounding_size=0.03",
                                      linewidth=1.6, edgecolor=pill_edge, facecolor=pill_face)
        ax.add_patch(pill)
        ax.text(0.05, 0.145, "Regen enabled" if regen_ok else "Regen disabled",
                fontsize=11.5, weight="bold", color=title_color, va="center")
        ax.text(0.24, 0.145,
                f"Brake path: {'ON' if brake_path else 'OFF'}  |  Gearbox: {'backdrivable' if backdriveable else 'non-backdrivable'}",
                fontsize=9.5, color=sub_color, va="center")

        ax.text(0.03, 0.04, "Note: This view is *static* torque/voltage bookkeeping (not dynamic control / inertia).",
                fontsize=8.5, color=faint, va="bottom")

        # --- right-side flags panel ---
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)
        axs.text(0.02, 0.96, "Active switches & limits", fontsize=14, weight="bold", color=title_color, va="top")

        def _sec(ax_, y0, title, lines):
            ax_.text(0.02, y0, title, fontsize=11, weight="bold", color=sub_color, va="top")
            y_ = y0 - 0.05
            for ln in lines:
                ax_.text(0.03, y_, ln, fontsize=9.6, color=title_color if ln.startswith("✓") else faint, va="top")
                y_ -= 0.042
            return y_ - 0.03

        ycur = 0.88
        ycur = _sec(axs, ycur, "Blocks", [
            f"{'✓' if mag_on else '–'} Magnetic coupler — Tslip={nm_to_ft_lbf(p.mag_coupler.t_slip_nm):.0f} ft-lbf",
            f"{'✓' if paras_on else '–'} Gearbox parasitic losses — Tc={nm_to_ft_lbf(p.parasitic.tc_nm):.0f} ft-lbf",
            f"{'✓' if bha_on else '–'} BHA external load block — TOB={p.bha.drilling_tob_ftlbf:.0f} ft-lbf",
            f"{'✓' if stuck_on else '–'} Stuck / stall mode",
        ])
        ycur = _sec(axs, ycur, "Operating", [
            f"✓ Commanded direction: {cmd_dir}",
            f"✓ Mode: {op_mode}",
        ])
        ycur = _sec(axs, ycur, "Braking / regen", [
            f"{'✓' if brake_path else '–'} Braking path at surface",
            f"{'✓' if backdriveable else '–'} Gearbox backdrivable",
            f"{'✓' if regen_ok else '–'} Quadrants II/IV usable",
            f"{'✓' if cable_regen_lim_on else '–'} Cable regen clamp: {clamp:.2f}",
            f"{'✓' if brake_pcap_on else '–'} Brake power cap: {pcap_kw:.0f} kW",
        ])
        _sec(axs, ycur, "Downhole limits", [
            f"{'✓' if enforce_vll else '–'} VLL,rms limit: {vll_lim:.0f} Vrms",
            f"{'✓' if enforce_vphase else '–'} Vϕ,rms limit: {vph_lim:.0f} Vrms",
        ])

        pane.canvas.draw_idle()

    def _plot_sweeps(self):
        """Render all sweep tabs."""
        self._plot_sweep_ratio_trade()
        self._plot_sweep_speed_voltage()
        self._plot_sweep_motor_design()
        self._plot_sweep_field_weakening()
        self._plot_sweep_cable_sensitivity()
        self._plot_sweep_surface_inverter()
        self._plot_sweep_motor_poles()
        self._plot_sweep_voltage_limits()
        self._plot_sweep_power_assessment()
        self._plot_sweep_load_braking()

    def _plot_sweep_ratio_trade(self):
        """Sweep ratio to show torque capability, EMF, Ke limit and η bound."""
        p = self.params
        mp = p.motor
        cab = p.cable
        eta = p.gearbox.eff_total()
        fast = bool(self.chk_fast_sweeps.isChecked())

        ax1 = self.sweep_ratio.ax_t
        ax2 = self.sweep_ratio.ax_emf
        ax3 = self.sweep_ratio.ax_ke
        ax4 = self.sweep_ratio.ax_eta
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        ratios = self._sweep_ratios()

        def max_iq(motor_rpm: float) -> float:
            if fast:
                iq, *_ = self.model.max_iq_given_limits_fast(motor_rpm, cab)
            else:
                iq, *_ = self.model.max_iq_given_limits(motor_rpm, cab)
            return float(iq)

        def max_torque_ftlbf_at_ratio(out_rpm: float, G: float) -> float:
            motor_rpm = out_rpm * G
            iq_max = max_iq(motor_rpm)
            kt_eff = self.model.kt_effective_nm_per_arms()
            tau_extra, _, _ = self.model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
            t_out = t_motor_useful * G * eta
            return nm_to_ft_lbf(t_out)

        tq_05 = np.array([max_torque_ftlbf_at_ratio(0.5, G) for G in ratios])
        tq_10 = np.array([max_torque_ftlbf_at_ratio(1.0, G) for G in ratios])

        ax1.plot(ratios, tq_05, linewidth=2, label="Max torque @ 0.5 rpm")
        ax1.plot(ratios, tq_10, linewidth=2, linestyle="--", label="Max torque @ 1.0 rpm")
        ax1.axhline(250.0, linestyle=":", linewidth=2, label="Continuous req (250 ft-lbf)")
        ax1.axhline(1000.0, linestyle="-.", linewidth=2, label="Peak req (1000 ft-lbf)")
        ax1.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
        ax1.set_title("Trade: Max Output Torque vs Total Ratio")
        ax1.set_xlabel("Total ratio (N:1)")
        ax1.set_ylabel("ft-lbf")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        # No-load back-EMF @ 1 rpm vs ratio (phase RMS)
        vph_emf_10 = []
        for G in ratios:
            motor_rpm = 1.0 * G
            omega_m = rpm_to_rad_s(motor_rpm)
            omega_e = max(1, int(mp.pole_pairs)) * omega_m
            vph = omega_e * mp.lambda_wb / math.sqrt(2.0)
            vph_emf_10.append(vph)
        vph_emf_10 = np.array(vph_emf_10)

        ax2.plot(ratios, vph_emf_10, linewidth=2, label="Back-EMF @ 1 rpm (phase RMS, no-load)")
        if p.limits.enforce_downhole_vphase_limit:
            ax2.axhline(p.limits.downhole_v_phase_rms_limit, linestyle="--", linewidth=2, label="Downhole Vphase limit")
        if p.limits.enforce_downhole_vll_limit:
            ax2.axhline(p.limits.downhole_vll_rms_limit / math.sqrt(3.0), linestyle="-.", linewidth=2,
                        label="Downhole Vll limit (as phase)")
        ax2.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
        ax2.set_title("Trade: No-load Back-EMF vs Ratio (voltage hard limit)")
        ax2.set_xlabel("Total ratio (N:1)")
        ax2.set_ylabel("Vrms (phase)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        vll_limit_eff = self._effective_downhole_vll_limit()
        if vll_limit_eff is not None:
            ke_max = (vll_limit_eff * 1000.0) / ratios
            ax3.plot(ratios, ke_max, linewidth=2, label="Ke_max allowed @ 1 rpm (from downhole V limit)")
            ax3.axhline(mp.ke_vll_rms_per_krpm, linestyle="--", linewidth=2,
                        label=f"Current Ke={mp.ke_vll_rms_per_krpm:.1f}")
            ax3.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
            ax3.set_title("Trade: Ke limit vs Ratio (for 1 rpm under downhole voltage)")
            ax3.set_xlabel("Total ratio (N:1)")
            ax3.set_ylabel("Ke (Vll_rms/krpm)")
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8, loc="best")
        else:
            ax3.set_title("Trade: Ke limit vs Ratio")
            ax3.text(0.5, 0.5, "Enable a downhole voltage limit to see Ke ceiling.", ha="center", va="center",
                     transform=ax3.transAxes)
            ax3.axis("off")

        i_lim = cab.i_phase_limit()
        tpk_nm = ft_lbf_to_nm(1000.0)
        tct_nm = ft_lbf_to_nm(250.0)
        kt_eff = self.model.kt_effective_nm_per_arms()
        out_rpm_ref = max(0.01, float(p.target.out_rpm))
        tau_extra_arr = np.array([self.model.tau_extra_nm(rpm_to_rad_s(out_rpm_ref * G))[0] for G in ratios],
                                 dtype=float)
        motor_torque_cap = np.maximum(1e-9, float(kt_eff) * max(1e-9, float(i_lim)) - tau_extra_arr)

        eta_req_pk = tpk_nm / (ratios * motor_torque_cap)
        eta_req_ct = tct_nm / (ratios * motor_torque_cap)

        ax4.plot(ratios, eta_req_pk, linewidth=2, label="η required (peak 1000 ft-lbf) [I-limit bound]")
        ax4.plot(ratios, eta_req_ct, linewidth=2, linestyle="--", label="η required (cont 250 ft-lbf) [I-limit bound]")
        ax4.axhline(eta, linestyle=":", linewidth=2, label=f"Current η_total={eta:.3f}")
        ax4.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
        ax4.set_ylim(0.0, 1.05)
        ax4.set_title("Trade: Minimum Efficiency vs Ratio (current-limited lower bound)")
        ax4.set_xlabel("Total ratio (N:1)")
        ax4.set_ylabel("η")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")

        self.sweep_ratio.fig.tight_layout(pad=2.0)
        self.sweep_ratio.canvas.draw_idle()

    def _plot_sweep_speed_voltage(self):
        """Speed ceilings vs ratio (downhole and surface), plus downhole voltage interpretation."""
        p = self.params
        mp = p.motor
        vf = p.vf
        ratios = self._sweep_ratios()

        ax1, ax2, ax3, ax4 = self.sweep_speed.ax1, self.sweep_speed.ax2, self.sweep_speed.ax3, self.sweep_speed.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        vll_dh = self._effective_downhole_vll_limit()
        if vll_dh is None:
            ax1.text(0.5, 0.5, "Enable a downhole voltage limit to view speed ceilings.", ha="center", va="center",
                     transform=ax1.transAxes)
            ax1.axis("off")
            self.sweep_speed.canvas.draw_idle()
            return

        # Downhole ceiling (simple no-load EMF): rpm_out_max = 1000*Vll / (Ke*G)
        rpm_out_max_dh = (1000.0 * vll_dh) / (max(1e-9, mp.ke_vll_rms_per_krpm) * ratios)

        # Surface ceiling (simplified): Vll_limit_surface from surface phase RMS limit
        if vf.v_limit_type.lower().startswith("line"):
            vll_surface = float(vf.v_limit_value)
        else:
            vll_surface = float(vf.v_limit_value) * math.sqrt(3.0)

        # Above base freq, Vcmd plateaus at base_v_phase_rms
        vll_cmd_plateau = float(vf.base_v_phase_rms) * math.sqrt(3.0)
        vll_surface_eff = min(vll_surface, vll_cmd_plateau)
        rpm_out_max_surface = (1000.0 * vll_surface_eff) / (max(1e-9, mp.ke_vll_rms_per_krpm) * ratios)

        ax1.plot(ratios, rpm_out_max_dh, linewidth=2, label="Max output rpm (downhole V limit, no-load EMF)")
        ax1.plot(ratios, rpm_out_max_surface, linewidth=2, linestyle="--",
                 label="Max output rpm (surface V limit/plateau, no-load EMF)")
        ax1.axhline(1.0, linestyle=":", linewidth=2, label="1.0 rpm target")
        ax1.axhline(0.5, linestyle="-.", linewidth=2, label="0.5 rpm target")
        ax1.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
        ax1.set_title("Speed Ceiling vs Ratio (simplified EMF + voltage hard limits)")
        ax1.set_xlabel("Total ratio (N:1)")
        ax1.set_ylabel("Output rpm")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        # Downhole interpretation (phase vs line-line)
        vph_dh = vll_dh / math.sqrt(3.0)
        ax2.axhline(vph_dh, linewidth=2, label=f"Effective downhole Vphase limit ≈ {vph_dh:.1f} Vrms")
        if p.limits.enforce_downhole_vphase_limit:
            ax2.axhline(p.limits.downhole_v_phase_rms_limit, linestyle="--", linewidth=2,
                        label="Motor Vphase (L-N) limit")
        if p.limits.enforce_downhole_vll_limit:
            ax2.axhline(p.limits.downhole_vll_rms_limit / math.sqrt(3.0), linestyle="-.", linewidth=2,
                        label="Contact Vll (as phase)")
        ax2.set_title("Downhole limit interpretation (all shown as phase RMS)")
        ax2.set_xlabel("(ratio independent)")
        ax2.set_ylabel("Vrms (phase)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        # Back-EMF at 0.5 and 1.0 rpm (phase RMS)
        vph_emf_05, vph_emf_10 = [], []
        for G in ratios:
            for out_rpm, arr in [(0.5, vph_emf_05), (1.0, vph_emf_10)]:
                motor_rpm = out_rpm * G
                omega_m = rpm_to_rad_s(motor_rpm)
                omega_e = max(1, int(mp.pole_pairs)) * omega_m
                arr.append(omega_e * mp.lambda_wb / math.sqrt(2.0))
        ax3.plot(ratios, np.array(vph_emf_05), linewidth=2, label="Back-EMF @ 0.5 rpm (phase RMS, no-load)")
        ax3.plot(ratios, np.array(vph_emf_10), linewidth=2, linestyle="--",
                 label="Back-EMF @ 1.0 rpm (phase RMS, no-load)")
        ax3.axhline(vph_dh, linestyle=":", linewidth=2, label="Effective downhole Vphase")
        ax3.axvline(p.gearbox.ratio(), linestyle=":", linewidth=2)
        ax3.set_title("No-load Back-EMF vs Ratio")
        ax3.set_xlabel("Total ratio (N:1)")
        ax3.set_ylabel("Vrms (phase)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")

        # Simple feasibility marker: ratio ceiling for 1 rpm
        G_max_1rpm = (1000.0 * vll_dh) / max(1e-9, mp.ke_vll_rms_per_krpm)
        ax4.axhline(G_max_1rpm, linewidth=2, label=f"G_max for 1 rpm (no-load EMF) ≈ {G_max_1rpm:.0f}")
        ax4.axhline(p.gearbox.ratio(), linestyle="--", linewidth=2, label="Current total ratio")
        ax4.set_title("Rule-of-thumb: max ratio allowed to hit 1 rpm (voltage-limited)")
        ax4.set_xlabel("(ratio independent)")
        ax4.set_ylabel("G_max (N:1)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")

        self.sweep_speed.fig.tight_layout(pad=2.0)
        self.sweep_speed.canvas.draw_idle()

    def _plot_sweep_motor_design(self):
        """Sweep Ke to show current required and speed ceiling at the selected ratio."""
        p = self.params
        mp = p.motor
        cab = p.cable
        G = p.gearbox.ratio()
        eta = p.gearbox.eff_total()
        i_lim = cab.i_phase_limit()
        vll_dh = self._effective_downhole_vll_limit()

        ax1, ax2, ax3, ax4 = self.sweep_motor.ax1, self.sweep_motor.ax2, self.sweep_motor.ax3, self.sweep_motor.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        kes = self._sweep_ke()

        # linked PMSM relation: Kt(Nm/Arms) = √3 * Ke_ll_rms_per_rad
        krpm_to_rad_s = (1000.0 * 2.0 * math.pi) / 60.0
        kt_from_ke = math.sqrt(3.0) * (kes / krpm_to_rad_s)

        # speed ceiling under downhole voltage (no-load EMF)
        if vll_dh is not None:
            rpm_out_max = (1000.0 * vll_dh) / (np.maximum(1e-9, kes) * G)
        else:
            rpm_out_max = np.full_like(kes, np.nan)

        # current required to deliver peak/cont torque (ignores voltage; purely torque constant)
        tpk_nm = ft_lbf_to_nm(1000.0)
        tct_nm = ft_lbf_to_nm(250.0)
        t_m_pk = tpk_nm / max(1e-9, (G * eta))
        t_m_ct = tct_nm / max(1e-9, (G * eta))
        i_req_pk = t_m_pk / np.maximum(1e-9, kt_from_ke)
        i_req_ct = t_m_ct / np.maximum(1e-9, kt_from_ke)

        ax1.plot(kes, rpm_out_max, linewidth=2, label="Max output rpm (downhole V limit, no-load EMF)")
        ax1.axhline(1.0, linestyle=":", linewidth=2, label="1.0 rpm target")
        ax1.axhline(0.5, linestyle="-.", linewidth=2, label="0.5 rpm target")
        ax1.axvline(mp.ke_vll_rms_per_krpm, linestyle="--", linewidth=2,
                    label=f"Current Ke={mp.ke_vll_rms_per_krpm:.1f}")
        ax1.set_title(f"Motor Design Trade @ fixed ratio G={G:.0f}: Speed ceiling vs Ke")
        ax1.set_xlabel("Ke (Vll_rms/krpm)")
        ax1.set_ylabel("Output rpm")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        ax2.plot(kes, i_req_pk, linewidth=2, label="Ireq (peak 1000 ft-lbf)")
        ax2.plot(kes, i_req_ct, linewidth=2, linestyle="--", label="Ireq (cont 250 ft-lbf)")
        ax2.axhline(i_lim, linestyle=":", linewidth=2, label=f"I_limit={i_lim:.2f} Arms")
        ax2.axvline(mp.ke_vll_rms_per_krpm, linestyle="--", linewidth=2)
        ax2.set_title("Current required vs Ke (torque-only bound)")
        ax2.set_xlabel("Ke (Vll_rms/krpm)")
        ax2.set_ylabel("Arms (phase)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        # torque ceiling at 1 rpm if current-limited and speed-feasible (simple bound)
        t_out_max_simple = nm_to_ft_lbf(kt_from_ke * i_lim * G * eta)
        if vll_dh is not None:
            feasible_speed_mask = rpm_out_max >= 1.0
            t_out_max_simple = np.where(feasible_speed_mask, t_out_max_simple, 0.0)
        ax3.plot(kes, t_out_max_simple, linewidth=2, label="Max output torque @ 1 rpm (I-limit bound + speed-feasible)")
        ax3.axhline(1000.0, linestyle="-.", linewidth=2, label="Peak req (1000)")
        ax3.axhline(250.0, linestyle=":", linewidth=2, label="Cont req (250)")
        ax3.axvline(mp.ke_vll_rms_per_krpm, linestyle="--", linewidth=2)
        ax3.set_title("Torque headroom @ 1 rpm vs Ke (fast bound)")
        ax3.set_xlabel("Ke (Vll_rms/krpm)")
        ax3.set_ylabel("ft-lbf")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")

        # ratio ceiling for 1 rpm as a function of Ke
        if vll_dh is not None:
            G_max = (1000.0 * vll_dh) / np.maximum(1e-9, kes)
            ax4.plot(kes, G_max, linewidth=2, label="G_max to hit 1 rpm (no-load EMF)")
            ax4.axhline(G, linestyle="--", linewidth=2, label="Current G")
            ax4.axvline(mp.ke_vll_rms_per_krpm, linestyle=":", linewidth=2)
            ax4.set_title("Voltage-limited gear ratio ceiling vs Ke")
            ax4.set_xlabel("Ke (Vll_rms/krpm)")
            ax4.set_ylabel("G_max (N:1)")
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=8, loc="best")
        else:
            ax4.text(0.5, 0.5, "Enable a downhole voltage limit to see G_max vs Ke.", ha="center", va="center",
                     transform=ax4.transAxes)
            ax4.axis("off")

        self.sweep_motor.fig.tight_layout(pad=2.0)
        self.sweep_motor.canvas.draw_idle()

    def _plot_sweep_field_weakening(self):
        """Sweep Id_max to visualize how much FW helps speed/torque under hard downhole voltage."""
        p = self.params
        mp = p.motor
        cab = p.cable
        G = p.gearbox.ratio()
        eta = p.gearbox.eff_total()
        i_lim = cab.i_phase_limit()
        vph_dh = self.model._effective_downhole_phase_limit()

        ax1, ax2, ax3, ax4 = self.sweep_fw.ax1, self.sweep_fw.ax2, self.sweep_fw.ax3, self.sweep_fw.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        if vph_dh is None:
            ax1.text(0.5, 0.5, "Enable a downhole motor Vphase limit to evaluate FW benefit.", ha="center", va="center",
                     transform=ax1.transAxes)
            ax1.axis("off")
            self.sweep_fw.canvas.draw_idle()
            return

        idmax_grid = np.linspace(0.0, float(i_lim), 41)

        def max_out_rpm_no_load_for_id(id_rms: float) -> float:
            # Solve for maximum motor rpm such that v_motor_phase_rms(id, iq=0) <= vph_dh
            # (ignores surface V/f and cable drop; downhole motor limit is hard)
            lo, hi = 0.0, 20000.0
            for _ in range(35):
                mid = 0.5 * (lo + hi)
                v = self.model.motor_voltage_required_phase_rms(mid, id_rms, 0.0)
                if v <= vph_dh:
                    lo = mid
                else:
                    hi = mid
            motor_rpm_max = lo
            return motor_rpm_max / max(1e-9, G)

        rpm_no_fw = np.array([max_out_rpm_no_load_for_id(0.0) for _ in idmax_grid])
        rpm_fw = np.array([max_out_rpm_no_load_for_id(-idm) for idm in idmax_grid])

        ax1.plot(idmax_grid, rpm_no_fw, linewidth=2, label="Max out rpm (no-load, Id=0)")
        ax1.plot(idmax_grid, rpm_fw, linewidth=2, linestyle="--", label="Max out rpm (no-load, Id=-Idmax)")
        ax1.axhline(1.0, linestyle=":", linewidth=2, label="1.0 rpm")
        ax1.axhline(0.5, linestyle="-.", linewidth=2, label="0.5 rpm")
        ax1.set_title("FW benefit: speed ceiling vs Id_max (downhole motor Vphase hard limit)")
        ax1.set_xlabel("Id_max (Arms, phase)")
        ax1.set_ylabel("Output rpm")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        # Torque at 1 rpm using fast/max approach while sweeping Id_max
        fw = p.fw
        old_enabled = bool(fw.enabled)
        old_id_max = float(fw.id_max_arms)

        fw.enabled = True
        tq_1rpm: List[float] = []
        iq_max_1rpm: List[float] = []
        id_best_1rpm: List[float] = []

        motor_rpm = 1.0 * G
        kt_eff = self.model.kt_effective_nm_per_arms()
        tau_extra_1rpm, _, _ = self.model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
        for idm in idmax_grid:
            fw.id_max_arms = float(idm)
            iq_max, idb, _, _, _, _, _ = self.model.max_iq_given_limits_fast(motor_rpm, cab)
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra_1rpm))
            t_out = t_motor_useful * G * eta
            tq_1rpm.append(nm_to_ft_lbf(t_out))
            iq_max_1rpm.append(iq_max)
            id_best_1rpm.append(idb)

        tq_1rpm = np.array(tq_1rpm)
        ax2.plot(idmax_grid, tq_1rpm, linewidth=2, label="Max torque @ 1 rpm (fast)")
        ax2.axhline(250.0, linestyle=":", linewidth=2, label="250 cont")
        ax2.axhline(1000.0, linestyle="-.", linewidth=2, label="1000 peak")
        ax2.set_title("FW benefit: torque capability @ 1 rpm vs Id_max")
        ax2.set_xlabel("Id_max (Arms, phase)")
        ax2.set_ylabel("ft-lbf")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        ax3.plot(idmax_grid, np.array(iq_max_1rpm), linewidth=2, label="Iq_max @ 1 rpm")
        ax3.axhline(i_lim, linestyle=":", linewidth=2, label="I_limit")
        ax3.set_title("Iq headroom @ 1 rpm vs Id_max (fast)")
        ax3.set_xlabel("Id_max (Arms, phase)")
        ax3.set_ylabel("Arms")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")

        ax4.plot(idmax_grid, np.array(id_best_1rpm), linewidth=2, label="Id used at Iq_max (fast)")
        ax4.axhline(0.0, linestyle=":", linewidth=2)
        ax4.set_title("Chosen Id at max torque point (fast) vs Id_max")
        ax4.set_xlabel("Id_max (Arms, phase)")
        ax4.set_ylabel("Id (Arms)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")

        # restore FW settings
        fw.id_max_arms = old_id_max
        fw.enabled = old_enabled

        self.sweep_fw.fig.tight_layout(pad=2.0)
        self.sweep_fw.canvas.draw_idle()

    def _plot_sweep_cable_sensitivity(self):
        """Sweep cable length to show torque capability and losses at key points."""
        p = self.params
        mp = p.motor
        G = p.gearbox.ratio()
        eta = p.gearbox.eff_total()
        fast = bool(self.chk_fast_sweeps.isChecked())

        ax1, ax2, ax3, ax4 = self.sweep_cable.ax1, self.sweep_cable.ax2, self.sweep_cable.ax3, self.sweep_cable.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        lens = self._sweep_lengths()

        tq_05, tq_10 = [], []
        vdrop_10, ploss_10, vmargin_10 = [], [], []

        for Lm in lens:
            cab = CableParams(**vars(p.cable))
            cab.length_m = float(Lm)

            def max_iq(motor_rpm: float) -> Tuple[float, float, float, float, float, float, Optional[float]]:
                if fast:
                    return self.model.max_iq_given_limits_fast(motor_rpm, cab)
                return self.model.max_iq_given_limits(motor_rpm, cab)

            # max torque @ 0.5 and 1.0 rpm
            for out_rpm, store in [(0.5, tq_05), (1.0, tq_10)]:
                motor_rpm = out_rpm * G
                iq_max, _, _, _, _, _, _ = max_iq(motor_rpm)
                kt_eff = self.model.kt_effective_nm_per_arms()
                tau_extra, _, _ = self.model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
                t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
                t_out = t_motor_useful * G * eta
                store.append(nm_to_ft_lbf(t_out))

            # at 1 rpm: cable drop and copper loss at max torque point
            motor_rpm = 1.0 * G
            iq_max, id_best, im_best, f_e, v_cmd, v_surface_limit, v_dh = max_iq(motor_rpm)
            v_motor = self.model.motor_voltage_required_phase_rms(motor_rpm, id_best, iq_max)
            v_drop = self.model.cable_drop_phase_rms(motor_rpm, im_best, cab)
            p_loss = self.model.cable_copper_loss_w(im_best, cab)
            vdrop_10.append(v_drop)
            ploss_10.append(p_loss)
            vmargin_10.append(v_cmd - (v_motor + v_drop))

        tq_05 = np.array(tq_05)
        tq_10 = np.array(tq_10)
        vdrop_10 = np.array(vdrop_10)
        ploss_10 = np.array(ploss_10)
        vmargin_10 = np.array(vmargin_10)

        ax1.plot(lens, tq_05, linewidth=2, label="Max torque @ 0.5 rpm")
        ax1.plot(lens, tq_10, linewidth=2, linestyle="--", label="Max torque @ 1.0 rpm")
        ax1.axhline(250.0, linestyle=":", linewidth=2, label="250 cont")
        ax1.axhline(1000.0, linestyle="-.", linewidth=2, label="1000 peak")
        ax1.set_title("Cable sensitivity: torque capability vs cable length")
        ax1.set_xlabel("Cable length (m)")
        ax1.set_ylabel("ft-lbf")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

        ax2.plot(lens, vdrop_10, linewidth=2, label="Cable drop @ 1 rpm (phase RMS, at max torque)")
        ax2.set_title("Cable sensitivity: voltage drop vs length")
        ax2.set_xlabel("Cable length (m)")
        ax2.set_ylabel("Vrms")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        ax3.plot(lens, ploss_10, linewidth=2, label="Cable copper loss @ 1 rpm (W, at max torque)")
        ax3.set_title("Cable sensitivity: copper loss vs length")
        ax3.set_xlabel("Cable length (m)")
        ax3.set_ylabel("W")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc="best")

        ax4.plot(lens, vmargin_10, linewidth=2, label="V margin = Vcmd - (Vmotor+Vdrop) @ 1 rpm")
        ax4.axhline(0.0, linestyle=":", linewidth=2)
        ax4.set_title("Cable sensitivity: surface voltage margin vs length")
        ax4.set_xlabel("Cable length (m)")
        ax4.set_ylabel("Vrms")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc="best")

        self.sweep_cable.fig.tight_layout(pad=2.0)
        self.sweep_cable.canvas.draw_idle()

    def _plot_sweep_surface_inverter(self):
        # Sweeps surface inverter/Vf parameters and shows impact on torque margin.
        p0 = self.params
        fast = bool(self.chk_fast_sweeps.isChecked())

        ax1, ax2, ax3, ax4 = self.sweep_inverter.ax1, self.sweep_inverter.ax2, self.sweep_inverter.ax3, self.sweep_inverter.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        G = p0.gearbox.ratio()
        eta = p0.gearbox.eff_total()
        cab = p0.cable

        def max_torque_ftlbf_at_out_rpm(model, out_rpm):
            motor_rpm = float(out_rpm) * G
            if fast:
                iq_max, idb, im, *_ = model.max_iq_given_limits_fast(motor_rpm, cab)
            else:
                iq_max, idb, im, *_ = model.max_iq_given_limits(motor_rpm, cab)
            kt_eff = model.kt_effective_nm_per_arms()
            tau_extra, _, _ = model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
            t_out_nm = t_motor_useful * G * eta
            return nm_to_ft_lbf(t_out_nm), float(iq_max), float(idb), float(im)

        # --- Sweep inverter voltage limit value (same units as UI) ---
        vlims = self._sweep_inv_vlim()
        tq1 = []
        vcmd1 = []
        for v in vlims:
            p = copy.deepcopy(p0)
            # Sweep voltage in the selected entry basis
            if p.vf.voltage_entry_basis.strip().upper().startswith('DC'):
                p.vf.vdc_link_v = float(v)
            else:
                p.vf.v_limit_value = float(v)
            model = SystemModel(p)
            t, *_ = max_torque_ftlbf_at_out_rpm(model, 1.0)
            tq1.append(t)
            # record Vcmd at 1 rpm
            motor_rpm = 1.0 * G
            _, v_cmd, _ = model._v_cmd_phase(motor_rpm)
            vcmd1.append(v_cmd)

        tq1 = np.array(tq1)
        vcmd1 = np.array(vcmd1)

        ax1.plot(vlims, tq1, linewidth=2, label='Max torque @ 1 rpm')
        ax1.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax1.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        current_v = p0.vf.vdc_link_v if p0.vf.voltage_entry_basis.strip().upper().startswith(
            'DC') else p0.vf.v_limit_value
        ax1.axvline(current_v, linestyle='--', linewidth=1.0, label='Current voltage (entry basis)')
        ax1.set_title('Inverter sweep: Vlimit → torque capability @ 1 rpm')
        ax1.set_xlabel('Vlimit value (UI units)')
        ax1.set_ylabel('ft-lbf')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')

        ax2.plot(vlims, vcmd1, linewidth=2, label='Vcmd @ 1 rpm (phase RMS)')
        ax2.axvline(current_v, linestyle='--', linewidth=1.0)
        if p0.limits.enforce_downhole_vphase_limit:
            ax2.axhline(p0.limits.downhole_v_phase_rms_limit, linestyle=':', linewidth=2,
                        label='Downhole motor Vphase limit')
        ax2.set_title('Inverter sweep: Vlimit → Vcmd at operating point')
        ax2.set_xlabel('Vlimit value (UI units)')
        ax2.set_ylabel('Vrms (phase)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')

        # --- Sweep base frequency ---
        basefs = self._sweep_basef()
        tq_basef = []
        for f in basefs:
            p = copy.deepcopy(p0)
            p.vf.base_freq_hz = float(f)
            model = SystemModel(p)
            t, *_ = max_torque_ftlbf_at_out_rpm(model, 1.0)
            tq_basef.append(t)
        tq_basef = np.array(tq_basef)

        ax3.plot(basefs, tq_basef, linewidth=2, label='Max torque @ 1 rpm')
        ax3.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax3.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        ax3.axvline(p0.vf.base_freq_hz, linestyle='--', linewidth=2, label='Current base f')
        ax3.set_title('Inverter sweep: Base freq → torque @ 1 rpm (V/Hz slope changes)')
        ax3.set_xlabel('Base frequency (Hz)')
        ax3.set_ylabel('ft-lbf')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='best')

        # --- Sweep Vboost ---
        boosts = self._sweep_vboost()
        tq_boost = []
        for vb in boosts:
            p = copy.deepcopy(p0)
            p.vf.v_boost = float(vb)
            model = SystemModel(p)
            t, *_ = max_torque_ftlbf_at_out_rpm(model, 0.5)
            tq_boost.append(t)
        tq_boost = np.array(tq_boost)

        ax4.plot(boosts, tq_boost, linewidth=2, label='Max torque @ 0.5 rpm')
        ax4.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax4.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        ax4.axvline(p0.vf.v_boost, linestyle='--', linewidth=2, label='Current Vboost')
        ax4.set_title('Inverter sweep: Vboost → torque @ 0.5 rpm')
        ax4.set_xlabel('Vboost (Vrms)')
        ax4.set_ylabel('ft-lbf')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='best')

        self.sweep_inverter.fig.tight_layout(pad=2.0)
        self.sweep_inverter.canvas.draw_idle()

    def _plot_sweep_motor_poles(self):
        # Sweeps pole pairs and shows impact on electrical freq and torque margin.
        p0 = self.params
        fast = bool(self.chk_fast_sweeps.isChecked())

        ax1, ax2, ax3, ax4 = self.sweep_poles.ax1, self.sweep_poles.ax2, self.sweep_poles.ax3, self.sweep_poles.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        G = p0.gearbox.ratio()
        eta = p0.gearbox.eff_total()
        cab = p0.cable

        pps = self._sweep_pole_pairs()

        tq_1rpm = []
        f_e_1rpm = []
        vcmd_1rpm = []
        iqmax_1rpm = []

        hold = self.sw_pp_hold.currentText()

        for pp in pps:
            p = copy.deepcopy(p0)
            p.motor.pole_pairs = int(pp)

            if hold.startswith('Hold lambda'):
                # keep lambda as-is
                pass
            elif hold.startswith('Hold Ke'):
                p.motor.link_kt_ke = True
                p.motor.motor_param_mode = 'Ke'
                # keep canonical ke value from baseline
                p.motor.ke_vll_rms_per_krpm = float(p0.motor.ke_vll_rms_per_krpm)
            else:  # Hold Kt
                p.motor.link_kt_ke = True
                p.motor.motor_param_mode = 'Kt'
                p.motor.kt_nm_per_arms = float(p0.motor.kt_nm_per_arms)

            model = SystemModel(p)

            motor_rpm = 1.0 * G
            if fast:
                iq_max, idb, im, f_e, v_cmd, v_lim, v_dh = model.max_iq_given_limits_fast(motor_rpm, cab)
            else:
                iq_max, idb, im, f_e, v_cmd, v_lim, v_dh = model.max_iq_given_limits(motor_rpm, cab)

            kt_eff = model.kt_effective_nm_per_arms()
            tau_extra, _, _ = model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
            t_out_nm = t_motor_useful * G * eta

            tq_1rpm.append(nm_to_ft_lbf(t_out_nm))
            f_e_1rpm.append(f_e)
            vcmd_1rpm.append(v_cmd)
            iqmax_1rpm.append(iq_max)

        tq_1rpm = np.array(tq_1rpm)
        f_e_1rpm = np.array(f_e_1rpm)
        vcmd_1rpm = np.array(vcmd_1rpm)
        iqmax_1rpm = np.array(iqmax_1rpm)

        ax1.plot(pps, tq_1rpm, linewidth=2, label='Max torque @ 1 rpm')
        ax1.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax1.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        ax1.axvline(p0.motor.pole_pairs, linestyle='--', linewidth=2, label='Current pp')
        ax1.set_title('Motor sweep: pole pairs → torque capability @ 1 rpm')
        ax1.set_xlabel('Pole pairs')
        ax1.set_ylabel('ft-lbf')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')

        ax2.plot(pps, f_e_1rpm, linewidth=2, label='Electrical freq @ 1 rpm')
        ax2.axvline(p0.motor.pole_pairs, linestyle='--', linewidth=2)
        ax2.set_title('Motor sweep: pole pairs → electrical frequency at operating point')
        ax2.set_xlabel('Pole pairs')
        ax2.set_ylabel('Hz')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')

        ax3.plot(pps, vcmd_1rpm, linewidth=2, label='Vcmd @ 1 rpm (phase RMS)')
        if p0.limits.enforce_downhole_vphase_limit:
            ax3.axhline(p0.limits.downhole_v_phase_rms_limit, linestyle=':', linewidth=2,
                        label='Downhole motor Vphase limit')
        ax3.axvline(p0.motor.pole_pairs, linestyle='--', linewidth=2)
        ax3.set_title('Motor sweep: pole pairs → Vcmd')
        ax3.set_xlabel('Pole pairs')
        ax3.set_ylabel('Vrms (phase)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='best')

        ax4.plot(pps, iqmax_1rpm, linewidth=2, label='Iq_max @ 1 rpm (phase RMS)')
        ax4.axhline(p0.cable.i_phase_limit(), linestyle=':', linewidth=2, label='|I| limit')
        ax4.axvline(p0.motor.pole_pairs, linestyle='--', linewidth=2)
        ax4.set_title('Motor sweep: pole pairs → Iq_max @ 1 rpm')
        ax4.set_xlabel('Pole pairs')
        ax4.set_ylabel('Arms')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='best')

        self.sweep_poles.fig.tight_layout(pad=2.0)
        self.sweep_poles.canvas.draw_idle()

    def _plot_sweep_voltage_limits(self):
        # Sweeps downhole voltage limits and shows effect on speed ceiling + torque margin.
        p0 = self.params
        fast = bool(self.chk_fast_sweeps.isChecked())

        ax1, ax2, ax3, ax4 = self.sweep_limits.ax1, self.sweep_limits.ax2, self.sweep_limits.ax3, self.sweep_limits.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        G = p0.gearbox.ratio()
        eta = p0.gearbox.eff_total()
        cab = p0.cable

        # --- sweep motor Vphase limit ---
        vphs = self._sweep_dh_vphase()
        tq_vph = []
        rpmceil_vph = []

        for vph in vphs:
            p = copy.deepcopy(p0)
            p.limits.enforce_downhole_vphase_limit = True
            p.limits.downhole_v_phase_rms_limit = float(vph)
            model = SystemModel(p)

            # torque capability @ 1 rpm
            motor_rpm = 1.0 * G
            if fast:
                iq_max, *_ = model.max_iq_given_limits_fast(motor_rpm, cab)
            else:
                iq_max, *_ = model.max_iq_given_limits(motor_rpm, cab)
            kt_eff = model.kt_effective_nm_per_arms()
            tau_extra, _, _ = model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
            t_out_nm = t_motor_useful * G * eta
            tq_vph.append(nm_to_ft_lbf(t_out_nm))

            # no-load speed ceiling from motor emf (Id=Iq=0) under motor Vphase limit
            pp = max(1, int(model.p.motor.pole_pairs))
            lam = float(model.p.motor.lambda_wb)
            # vph = omega_e*lam/sqrt(2) = (pp*omega_m)*lam/sqrt(2)
            omega_m_max = (float(vph) * math.sqrt(2.0)) / max(1e-12, (pp * lam))
            motor_rpm_max = omega_m_max * 60.0 / (2.0 * math.pi)
            rpmceil_vph.append(motor_rpm_max / max(1e-9, G))

        tq_vph = np.array(tq_vph)
        rpmceil_vph = np.array(rpmceil_vph)

        ax1.plot(vphs, tq_vph, linewidth=2, label='Max torque @ 1 rpm')
        ax1.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax1.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        ax1.axvline(p0.limits.downhole_v_phase_rms_limit, linestyle='--', linewidth=2, label='Current Vphase limit')
        ax1.set_title('Downhole sweep: motor Vphase limit → torque capability @ 1 rpm')
        ax1.set_xlabel('Downhole motor Vphase limit (Vrms)')
        ax1.set_ylabel('ft-lbf')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')

        ax2.plot(vphs, rpmceil_vph, linewidth=2, label='No-load speed ceiling')
        ax2.axhline(0.5, linestyle=':', linewidth=2, label='0.5 rpm')
        ax2.axhline(1.0, linestyle='-.', linewidth=2, label='1.0 rpm')
        ax2.axvline(p0.limits.downhole_v_phase_rms_limit, linestyle='--', linewidth=2)
        ax2.set_title('Downhole sweep: motor Vphase limit → speed ceiling (no-load EMF)')
        ax2.set_xlabel('Downhole motor Vphase limit (Vrms)')
        ax2.set_ylabel('Output rpm')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')

        # --- sweep contact Vll limit ---
        vlls = self._sweep_dh_vll()
        tq_vll = []
        rpmceil_vll = []

        for vll in vlls:
            p = copy.deepcopy(p0)
            p.limits.enforce_downhole_vll_limit = True
            p.limits.downhole_vll_rms_limit = float(vll)
            model = SystemModel(p)

            motor_rpm = 1.0 * G
            if fast:
                iq_max, *_ = model.max_iq_given_limits_fast(motor_rpm, cab)
            else:
                iq_max, *_ = model.max_iq_given_limits(motor_rpm, cab)
            kt_eff = model.kt_effective_nm_per_arms()
            tau_extra, _, _ = model.tau_extra_nm(rpm_to_rad_s(motor_rpm))
            t_motor_useful = max(0.0, float(kt_eff) * float(iq_max) - float(tau_extra))
            t_out_nm = t_motor_useful * G * eta
            tq_vll.append(nm_to_ft_lbf(t_out_nm))

            vph_eff = float(vll) / math.sqrt(3.0)
            pp = max(1, int(model.p.motor.pole_pairs))
            lam = float(model.p.motor.lambda_wb)
            omega_m_max = (vph_eff * math.sqrt(2.0)) / max(1e-12, (pp * lam))
            motor_rpm_max = omega_m_max * 60.0 / (2.0 * math.pi)
            rpmceil_vll.append(motor_rpm_max / max(1e-9, G))

        tq_vll = np.array(tq_vll)
        rpmceil_vll = np.array(rpmceil_vll)

        ax3.plot(vlls, tq_vll, linewidth=2, label='Max torque @ 1 rpm')
        ax3.axhline(250.0, linestyle=':', linewidth=2, label='250 cont')
        ax3.axhline(1000.0, linestyle='-.', linewidth=2, label='1000 peak')
        ax3.axvline(p0.limits.downhole_vll_rms_limit, linestyle='--', linewidth=2, label='Current Vll limit')
        ax3.set_title('Downhole sweep: contact Vll limit → torque capability @ 1 rpm')
        ax3.set_xlabel('Downhole contact Vll limit (Vrms L-L)')
        ax3.set_ylabel('ft-lbf')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='best')

        ax4.plot(vlls, rpmceil_vll, linewidth=2, label='No-load speed ceiling (effective phase = Vll/√3)')
        ax4.axhline(0.5, linestyle=':', linewidth=2, label='0.5 rpm')
        ax4.axhline(1.0, linestyle='-.', linewidth=2, label='1.0 rpm')
        ax4.axvline(p0.limits.downhole_vll_rms_limit, linestyle='--', linewidth=2)
        ax4.set_title('Downhole sweep: contact Vll limit → speed ceiling (no-load EMF)')
        ax4.set_xlabel('Downhole contact Vll limit (Vrms L-L)')
        ax4.set_ylabel('Output rpm')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='best')

        self.sweep_limits.fig.tight_layout(pad=2.0)
        self.sweep_limits.canvas.draw_idle()

    def _plot_sweep_power_assessment(self):
        """Power-flow and loss assessment along the max-torque envelope (current config)."""
        if not hasattr(self, "sweep_power"):
            return

        p = self.params
        cab = p.cable
        gb = p.gearbox

        G = float(gb.ratio())
        eta = float(gb.eff_total())

        fast = bool(getattr(self, "chk_fast_sweeps", None) and self.chk_fast_sweeps.isChecked())

        # Use the same RPM max as the Envelope tab (if present); otherwise default to a sensible value.
        try:
            out_rpm_max = float(self.env_out_rpm_max.value())
        except Exception:
            out_rpm_max = max(1.0, float(getattr(getattr(p, "target", None), "out_rpm", 1.0)))

        out_rpms = np.linspace(0.02, max(0.05, out_rpm_max), 140)

        P_out_w = np.zeros_like(out_rpms)
        P_out_raw_w = np.zeros_like(out_rpms)  # before parasitic output torque subtraction
        P_surface_w = np.zeros_like(out_rpms)

        P_cable_w = np.zeros_like(out_rpms)
        P_mcu_w = np.zeros_like(out_rpms)
        P_extra_w = np.zeros_like(out_rpms)  # core + viscous (modeled as opposing torque)
        P_mag_w = np.zeros_like(out_rpms)  # magnetic-coupler slip (if enabled)
        P_gear_w = np.zeros_like(out_rpms)  # gearbox loss from efficiency
        P_par_w = np.zeros_like(out_rpms)  # parasitic output torque loss

        eff_total = np.zeros_like(out_rpms)

        pf_cable = np.zeros_like(out_rpms)
        Xl_ohm = np.zeros_like(out_rpms)

        # Precompute cable R,L (speed-independent in this model)
        Rc_ohm = float(cab.effective_r_phase())
        Lc_h = float(cab.effective_l_phase())

        for k, out_rpm in enumerate(out_rpms):
            motor_rpm = float(out_rpm * G)

            # Max feasible iq at this speed (respects control strategy + FW settings).
            if fast:
                iq_max, id_best, i_mag, f_e, v_cmd, v_surf_lim, v_dh = self.model.max_iq_given_limits_fast(motor_rpm,
                                                                                                           cab)
            else:
                iq_max, id_best, i_mag, f_e, v_cmd, v_surf_lim, v_dh = self.model.max_iq_given_limits(motor_rpm, cab)

            omega_m = rpm_to_rad_s(motor_rpm)
            omega_out = abs(rpm_to_rad_s(out_rpm))

            # Cable PF is purely from its RL impedance at the electrical frequency (model ignores C).
            omega_e = 2.0 * math.pi * float(f_e)
            Xl = omega_e * Lc_h
            Xl_ohm[k] = Xl
            pf_cable[k] = Rc_ohm / max(1e-12, math.sqrt(Rc_ohm * Rc_ohm + Xl * Xl))

            # Motor electromagnetic torque capability (sinusoidal PMSM: torque ~ Kt * iq)
            kt = float(self.model.kt_effective_nm_per_arms())
            tau_em = kt * float(iq_max)

            # Extra opposing torques (core + viscous) modeled at motor shaft.
            tau_core = float(self.model.tau_core_nm(omega_m))
            tau_visc = float(self.model.tau_visc_nm(omega_m))
            tau_extra = tau_core + tau_visc

            # Useful motor shaft torque (what's left after internal opposing torque).
            tau_useful = max(0.0, tau_em - tau_extra)

            # Magnetic coupler (if enabled): can limit transmitted torque -> slip loss.
            t_gb_in, mag_slip = self.model._mag_forward_transmitted_to_gb_nm(tau_useful)

            # Output-side parasitic torque (bearings/seals) subtracts from deliverable output torque.
            tau_par = float(self.model._rot_loss_torque_nm(self.model.out_dir_sign() * omega_out, p.parasitic))

            # Torque mapping to output
            t_out_raw = float(t_gb_in) * G * eta
            t_out_cap = max(0.0, t_out_raw - abs(tau_par))

            # --- Power bookkeeping (all real, fundamental only) ---
            # Electrical -> mechanical conversion at motor:
            P_em = tau_em * abs(omega_m)  # electromagnetic airgap power
            P_extra = abs(tau_extra) * abs(omega_m)  # modeled motor internal mech loss
            P_mag = max(0.0, (tau_useful - float(t_gb_in))) * abs(omega_m)  # coupler slip (if any)

            # Gearbox: P_out_raw = eta * P_gb_in ; loss = (1-eta)*P_gb_in
            P_gb_in = float(t_gb_in) * abs(omega_m)
            P_gear = max(0.0, (1.0 - eta)) * P_gb_in

            # Output power (before and after parasitic subtraction)
            P_out_raw = t_out_raw * omega_out
            P_par = abs(tau_par) * omega_out
            P_out = t_out_cap * omega_out

            # Copper losses
            rs_eff = float(self.model.rs_effective_ohm())
            P_mcu = 3.0 * (float(i_mag) ** 2) * rs_eff
            P_cable = float(self.model.cable_loss_w(float(i_mag), cab))

            # Surface real power estimate (ignores inverter/filter losses): motor electrical + cable copper
            # Motor electrical ~= airgap + copper
            P_surface = (P_em + P_mcu) + P_cable

            # Store arrays
            P_out_w[k] = P_out
            P_out_raw_w[k] = P_out_raw
            P_surface_w[k] = P_surface

            P_cable_w[k] = P_cable
            P_mcu_w[k] = P_mcu
            P_extra_w[k] = P_extra
            P_mag_w[k] = P_mag
            P_gear_w[k] = P_gear
            P_par_w[k] = P_par

            # Total-chain efficiency to delivered output (after parasitics)
            eff_total[k] = (P_out / P_surface) if P_surface > 1e-12 else 0.0

        # ----------------- Plotting -----------------
        pane = self.sweep_power
        ax1, ax2, ax3, ax4 = pane.ax1, pane.ax2, pane.ax3, pane.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()
            ax.grid(True, alpha=0.30)

        # (1) Power flow (kW)
        ax1.plot(out_rpms, P_surface_w / 1000.0, label="Surface real power (est.)")
        ax1.plot(out_rpms, P_out_w / 1000.0, label="Delivered output power")
        ax1.plot(out_rpms, P_out_raw_w / 1000.0, label="Output power (pre-parasitic)")
        ax1.set_title("Power flow vs output speed (max-torque envelope)")
        ax1.set_xlabel("Output speed (RPM)")
        ax1.set_ylabel("Power (kW)")
        ax1.legend(loc="best", frameon=False)

        # (2) Loss breakdown (absolute) — stacked area (kW)
        loss_labels = [
            "Cable copper",
            "Motor copper",
            "Motor core+visc (modeled)",
            "Mag coupler slip",
            "Gearbox loss (η)",
            "Output parasitics",
        ]
        loss_stack_w = np.vstack([P_cable_w, P_mcu_w, P_extra_w, P_mag_w, P_gear_w, P_par_w])
        loss_total_w = np.sum(loss_stack_w, axis=0)

        ax2.stackplot(out_rpms, (loss_stack_w / 1000.0), labels=loss_labels, alpha=0.85)
        ax2.plot(out_rpms, loss_total_w / 1000.0, label="Total loss", linewidth=2.2)
        ax2.set_title("Loss breakdown (absolute, real power)")
        ax2.set_xlabel("Output speed (RPM)")
        ax2.set_ylabel("Loss power (kW)")
        ax2.legend(loc="best", frameon=False, fontsize=8)

        # (3) Loss contribution (percent of total) — 100% stacked
        ax3.clear()
        denom = np.where(loss_total_w > 1e-12, loss_total_w, np.nan)
        loss_pct = 100.0 * (loss_stack_w / denom)
        loss_pct = np.nan_to_num(loss_pct, nan=0.0, posinf=0.0, neginf=0.0)

        ax3.stackplot(out_rpms, loss_pct, labels=loss_labels, alpha=0.85)
        ax3.set_title("Loss contribution by component")
        ax3.set_xlabel("Output speed (RPM)")
        ax3.set_ylabel("Share of total loss (%)")
        ax3.set_ylim(0.0, 100.0)

        # Context line: total loss (kW) on the right axis
        ax3b = ax3.twinx()
        ax3b.plot(out_rpms, loss_total_w / 1000.0, linestyle="--", linewidth=1.6)
        ax3b.set_ylabel("Total loss (kW)")

        # Combine legends (ax3 + ax3b)
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3b.get_legend_handles_labels()
        ax3.legend(h1 + h2, l1 + l2, loc="best", frameon=False, fontsize=8, ncol=2)

        # (4) Efficiency + cable PF, with an inset showing the 5-seg thermal profile
        ax4.plot(out_rpms, 100.0 * eff_total, label="η total → delivered output")
        ax4.set_title("Efficiency and cable PF (RL-only)")
        ax4.set_xlabel("Output speed (RPM)")
        ax4.set_ylabel("Efficiency (%)")
        ax4.set_ylim(0.0, 100.0)

        ax4b = ax4.twinx()
        ax4b.plot(out_rpms, pf_cable, linestyle="--", label="Cable PF (RL only)")
        ax4b.set_ylabel("Cable PF (-)")
        ax4b.set_ylim(0.0, 1.05)

        # Combine legends
        h1, l1 = ax4.get_legend_handles_labels()
        h2, l2 = ax4b.get_legend_handles_labels()
        ax4.legend(h1 + h2, l1 + l2, loc="best", frameon=False, fontsize=8)

        # Inset: 5-seg temperature + R-multiplier profile
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax4ins = inset_axes(ax4, width="48%", height="48%", loc="center", borderpad=1.0)
            ax4ins.grid(True, alpha=0.25)

            if cab.temp_model_5seg and len(cab.temp5_seg_len_m) == 5 and len(cab.temp5_seg_temp_C) == 5:
                seg_l = [float(x) for x in cab.temp5_seg_len_m]
                seg_T = [float(x) for x in cab.temp5_seg_temp_C]
                x = [0.0]
                for L in seg_l:
                    x.append(x[-1] + L)

                yT = seg_T + [seg_T[-1]]
                ax4ins.step(x, yT, where="post", linewidth=2.0)
                ax4ins.set_xlim(0.0, max(1.0, x[-1]))
                ax4ins.set_xlabel("m", fontsize=8)
                ax4ins.set_ylabel("°C", fontsize=8)
                ax4ins.tick_params(axis="both", labelsize=8)

                # R multiplier on right axis
                a = float(getattr(cab, 'temp_alpha_per_C', getattr(cab, 'temp5_alpha_per_C', 0.00393)))
                tref = float(getattr(cab, 'temp_ref_C', getattr(cab, 'temp5_ref_C', 20.0)))
                yR = [(1.0 + a * (t - tref)) for t in seg_T] + [(1.0 + a * (seg_T[-1] - tref))]
                ax4insb = ax4ins.twinx()
                ax4insb.step(x, yR, where="post", linestyle="--", linewidth=1.6)
                ax4insb.set_ylabel("R×", fontsize=8)
                ax4insb.tick_params(axis="y", labelsize=8)

                ax4.text(0.02, 0.02, "Inset: 5-seg T(x) + R multiplier", transform=ax4.transAxes, fontsize=8, alpha=0.9)
            else:
                ax4ins.text(0.05, 0.9, "5-seg model OFF", transform=ax4ins.transAxes, fontsize=9)
                ax4ins.set_xticks([])
                ax4ins.set_yticks([])
        except Exception:
            pass
        pane.fig.tight_layout(pad=2.0)
        pane.canvas.draw_idle()

    def _plot_sweep_load_braking(self):
        """New v15 sweep tab.

        Helps interpret CCRS behavior when the load can either oppose or assist rotation.
        We decompose steady-state static-blocks torque into:
          - T_CCRS (signed): torque applied by CCRS at the output shaft
          - T_load (signed): net external torque on the shaft (T_load = -T_CCRS)
          - T_drive (signed): component of T_CCRS that *drives* in the direction of motion
          - T_brake (signed): component that *brakes* (opposes motion). If regen/braking isn't
            possible (no braking path or non-backdriveable gearbox), this term is suppressed.

        Plots are steady-state static models (no inertia), intended as an intuition aid.
        """
        p = self.params

        ax1 = self.sweep_load.ax1
        ax2 = self.sweep_load.ax2
        ax3 = self.sweep_load.ax3
        ax4 = self.sweep_load.ax4
        for ax in (ax1, ax2, ax3, ax4):
            ax.cla()

        rpm_max = float(self.env_out_rpm_max.value())
        rpms = np.linspace(-rpm_max, rpm_max, 401)

        # Regen/braking availability is a *system* property (motor/inverter may be 4Q,
        # but a self-locking gearbox blocks mechanical power flow back to the motor).
        braking_path = bool(getattr(p, 'braking_path_available', True))
        backdrivable = bool(getattr(getattr(p, 'gearbox', None), 'backdrivable', True))
        regen_ok = braking_path and backdrivable

        cab_1 = copy.deepcopy(p.cable)
        cab_1.wires_per_phase = 1
        cab_2 = copy.deepcopy(p.cable)
        cab_2.wires_per_phase = 2

        T_ccrs = []  # signed ft-lbf
        T_load = []  # signed ft-lbf
        T_drive = []  # signed ft-lbf
        T_brake = []  # signed ft-lbf (opposes motion)
        P_mech_kw = []  # kW
        brake_supp = []

        regen_cap_1 = []  # ft-lbf (magnitude vs |rpm|)
        regen_cap_2 = []

        for rpm in rpms:
            omega = rpm_to_rad_s(rpm)

            t_drive_nm, t_brake_nm, *_rest, brake_blocked = self.model._required_output_drive_torque_nm(omega)

            s = 1.0 if omega >= 0.0 else -1.0  # motion sign
            tau_ccrs_nm = s * (t_drive_nm - t_brake_nm)
            tau_load_nm = -tau_ccrs_nm

            T_ccrs.append(nm_to_ftlbf(tau_ccrs_nm))
            T_load.append(nm_to_ftlbf(tau_load_nm))

            # Signed components for clarity:
            #   drive torque aligns with motion (+ for CW, - for CCW)
            #   braking torque opposes motion
            T_drive.append(nm_to_ftlbf(s * t_drive_nm))
            T_brake.append(nm_to_ftlbf(-s * t_brake_nm))

            P_mech_kw.append((tau_ccrs_nm * omega) / 1000.0)
            brake_supp.append(bool(brake_blocked))

            if regen_ok:
                regen_cap_1.append(float(self.model.regen_cap_output_torque_ftlbf(abs(rpm), cab_1)))
                regen_cap_2.append(float(self.model.regen_cap_output_torque_ftlbf(abs(rpm), cab_2)))
            else:
                regen_cap_1.append(0.0)
                regen_cap_2.append(0.0)

        T_ccrs = np.asarray(T_ccrs)
        T_load = np.asarray(T_load)
        T_drive = np.asarray(T_drive)
        T_brake = np.asarray(T_brake)
        P_mech_kw = np.asarray(P_mech_kw)
        brake_supp = np.asarray(brake_supp)

        regen_cap_1 = np.asarray(regen_cap_1, dtype=float)
        regen_cap_2 = np.asarray(regen_cap_2, dtype=float)
        regen_cap_1[~np.isfinite(regen_cap_1)] = 0.0
        regen_cap_2[~np.isfinite(regen_cap_2)] = 0.0

        # -------- Plot 1: Net torque balance --------
        ax1.plot(rpms, T_ccrs, label='T_CCRS (signed)')
        ax1.plot(rpms, T_load, ls='--', label='T_load = -T_CCRS')
        ax1.axhline(0.0, lw=0.8)
        ax1.axvline(0.0, lw=0.8)
        ax1.set_title('Signed torque balance at output shaft')
        ax1.set_xlabel('Output RPM (CW=+, CCW=-)')
        ax1.set_ylabel('Torque (ft-lbf)')
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8, loc='best')

        # -------- Plot 2: Drive vs Brake decomposition --------
        ax2.plot(rpms, T_drive, label='T_drive (signed)')
        ax2.plot(rpms, T_brake, label='T_brake (signed, opposes motion)')
        ax2.axhline(0.0, lw=0.8)
        ax2.axvline(0.0, lw=0.8)
        ax2.set_title('How the required torque decomposes')
        ax2.set_xlabel('Output RPM (CW=+, CCW=-)')
        ax2.set_ylabel('Torque (ft-lbf)')
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8, loc='best')

        if brake_supp.any():
            ax2.text(0.02, 0.02,
                     'Note: braking/regen suppressed\n(non-backdriveable gearbox)',
                     transform=ax2.transAxes, fontsize=8, va='bottom')

        # -------- Plot 3: Regen/braking capability vs demand --------
        abs_rpm = np.abs(rpms)
        T_brake_mag = np.abs(T_brake)
        ax3.plot(abs_rpm, T_brake_mag, label='|T_brake demand|')
        ax3.plot(abs_rpm, regen_cap_1, ls='--', label='Regen cap (cable-aware, 1 wire/phase)')
        ax3.plot(abs_rpm, regen_cap_2, ls='--', label='Regen cap (cable-aware, 2 wires/phase)')
        ax3.set_title('If braking is needed, can we absorb it?')
        ax3.set_xlabel('|Output RPM|')
        ax3.set_ylabel('Torque (ft-lbf)')
        ax3.grid(True, alpha=0.25)
        ax3.legend(fontsize=8, loc='best')

        if not regen_ok:
            ax3.text(0.02, 0.98,
                     'Regen/braking not available\n(no braking path OR non-backdriveable gearbox)',
                     transform=ax3.transAxes, fontsize=8, va='top')

        # -------- Plot 4: Mechanical power flow --------
        ax4.plot(rpms, P_mech_kw, label='P_mech = T_CCRS * ω')
        ax4.axhline(0.0, lw=0.8)
        ax4.axvline(0.0, lw=0.8)
        ax4.set_title('Mechanical power (steady-state)')
        ax4.set_xlabel('Output RPM (CW=+, CCW=-)')
        ax4.set_ylabel('Power (kW)')
        ax4.grid(True, alpha=0.25)
        ax4.legend(fontsize=8, loc='best')

        self.sweep_load.canvas.draw_idle()

    # ---------- Report generation ----------
    @staticmethod
    def _fig_to_rl_image(fig: Figure, width_in: float = 7.4, dpi: int = 180) -> RLImage:
        """Convert a matplotlib Figure into a ReportLab Image flowable (PNG in-memory)."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = RLImage(buf)
        w = width_in * inch
        img.drawWidth = w
        # keep aspect
        aspect = float(img.imageHeight) / max(1.0, float(img.imageWidth))
        img.drawHeight = w * aspect
        return img

    @staticmethod
    def _fmt_bool(x: bool) -> str:
        return "ON" if bool(x) else "OFF"

    # ---------- Plot export (PNG) ----------
    @staticmethod
    def _slugify(s: str, max_len: int = 90) -> str:
        """Make a filesystem-safe slug."""
        if s is None:
            return "na"
        s = str(s).strip()
        if not s:
            return "na"
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9._\-+]+", "", s)
        s = s.lstrip(".") or "na"
        if len(s) > max_len:
            s = s[:max_len]
        return s

    def _tab_path_for_widget(self, w: QWidget) -> List[str]:
        """Return hierarchical tab path for a widget (outer->inner)."""
        path: List[str] = []
        child = w
        parent = child.parentWidget()
        while parent is not None:
            if isinstance(parent, QTabWidget):
                try:
                    idx = parent.indexOf(child)
                except Exception:
                    idx = -1
                if idx is not None and idx >= 0:
                    try:
                        path.append(parent.tabText(idx))
                    except Exception:
                        pass
            child = parent
            parent = parent.parentWidget()
        path.reverse()
        return path

    @staticmethod
    def _save_axes_png(fig: Figure, ax, out_path: str, dpi: int = 220) -> bool:
        """Save a single Axes as a cropped PNG. Returns True on success."""
        try:
            try:
                fig.canvas.draw()
            except Exception:
                pass

            renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
            if renderer is None:
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                return True

            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                return True

            try:
                bbox = bbox.expanded(1.03, 1.08)
            except Exception:
                pass

            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(out_path, dpi=dpi, bbox_inches=bbox)
            return True
        except Exception:
            return False

    def _iter_canvases(self) -> List[FigureCanvas]:
        """Collect all matplotlib canvases currently in the UI (includes sweeps)."""
        canvases: List[FigureCanvas] = []
        try:
            canvases = list(self.findChildren(FigureCanvas))
        except Exception:
            canvases = []

        # De-dup by figure id
        uniq: List[FigureCanvas] = []
        seen = set()
        for c in canvases:
            fig = getattr(c, "figure", None)
            key = id(fig) if fig is not None else id(c)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    def _save_all_plots(self):
        """Export all plots + all individual subplots from every tab to a new unique folder."""
        # Try to refresh plots so the export matches the current UI settings
        try:
            self.update_all()
        except Exception:
            pass
        try:
            self._plot_sweeps()
        except Exception:
            pass

        base_dir = QFileDialog.getExistingDirectory(self, "Select folder to save plots")
        if not base_dir:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        token = uuid.uuid4().hex[:8]
        run_dir = os.path.join(base_dir, f"NavigatorPlots_{ts}_{token}")
        figs_dir = os.path.join(run_dir, "figures")
        axes_dir = os.path.join(run_dir, "subplots")

        try:
            os.makedirs(figs_dir, exist_ok=False)
            os.makedirs(axes_dir, exist_ok=False)
        except Exception as e:
            QMessageBox.critical(self, "Save plots failed", f"Could not create output folder.\n\n{e}")
            return

        canvases = self._iter_canvases()
        if not canvases:
            QMessageBox.information(self, "Save plots", "No plot canvases were found to export.")
            return

        n_fig = 0
        n_ax = 0
        n_fail = 0

        for i, canvas in enumerate(canvases):
            fig = getattr(canvas, "figure", None)
            if fig is None:
                continue

            tab_path = self._tab_path_for_widget(canvas)
            tab_tag = "__".join([self._slugify(t) for t in tab_path]) if tab_path else "plots"

            obj = ""
            try:
                obj = str(canvas.objectName() or "").strip()
            except Exception:
                obj = ""
            obj_tag = self._slugify(obj) if obj else ""

            base = f"{i:03d}__{tab_tag}"
            if obj_tag:
                base += f"__{obj_tag}"

            fig_path = os.path.join(figs_dir, f"{base}.png")
            try:
                fig.savefig(fig_path, dpi=220, bbox_inches="tight")
                n_fig += 1
            except Exception:
                n_fail += 1
                continue

            try:
                axes = list(getattr(fig, "axes", []))
            except Exception:
                axes = []

            for j, ax in enumerate(axes):
                try:
                    if not ax.get_visible():
                        continue
                except Exception:
                    pass

                title = ""
                try:
                    title = ax.get_title() or ax.get_label() or ""
                except Exception:
                    title = ""
                if not title:
                    try:
                        title = ax.get_ylabel() or ax.get_xlabel() or ""
                    except Exception:
                        title = ""
                if not title:
                    title = f"ax{j:02d}"

                ax_tag = self._slugify(title)
                ax_path = os.path.join(axes_dir, f"{base}__ax{j:02d}__{ax_tag}.png")

                ok = self._save_axes_png(fig, ax, ax_path, dpi=220)
                if ok:
                    n_ax += 1
                else:
                    n_fail += 1

        QMessageBox.information(
            self,
            "Plots saved",
            f"Saved {n_fig} figure(s) and {n_ax} subplot(s).\n\nOutput folder:\n{run_dir}\n\nFailures: {n_fail}",
        )

    def _generate_pdf_report(self):
        """Create a PDF report with all plots embedded + interpreted."""
        # Ensure plots and results are current
        try:
            self.update_all()
        except Exception:
            pass

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"CCRS_Feasibility_Report_{ts}.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF report", default_name, "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        try:
            self._build_pdf_report(path)
        except Exception as e:
            QMessageBox.critical(self, "Report generation failed", f"Failed to create PDF report.\n\n{e}")
            return

        QMessageBox.information(self, "Report saved", f"Saved PDF report:\n{path}")

    def _build_pdf_report(self, pdf_path: str):
        p = self.params
        model = self.model

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=10))
        styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=8))
        styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=13, spaceAfter=6))
        styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=4))

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
            title="Navigator CCRSFeasibility Report",
        )

        story = []

        # ----- Title page -----
        story.append(Paragraph("Navigator Power Chain Feasibility Report", styles["H1"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Body"]))

        req_line = (
            "Required (output): 1000 ft-lbf peak, 250 ft-lbf continuous, 0.5-1.0 rpm continuous."
        )
        story.append(Paragraph(req_line, styles["Body"]))

        # constraints summary
        eff_vph = model._effective_downhole_phase_limit()
        eff_vll = self._effective_downhole_vll_limit()
        story.append(Paragraph("Key limits configured in the model:", styles["H2"]))
        lim_lines = []
        lim_lines.append(f"Control strategy: {getattr(p, 'control_strategy', 'VF')}")
        lim_lines.append(
            f"Downhole motor voltage limit (phase RMS, L-N): {p.limits.downhole_v_phase_rms_limit:.1f} Vrms (enforced={self._fmt_bool(p.limits.enforce_downhole_vphase_limit)})")
        lim_lines.append(
            f"Downhole contact-block voltage limit (line-line RMS, L-L): {p.limits.downhole_vll_rms_limit:.1f} Vrms (enforced={self._fmt_bool(p.limits.enforce_downhole_vll_limit)})")
        if eff_vph is not None:
            lim_lines.append(
                f"Effective downhole phase-limit used by solver: {eff_vph:.1f} Vrms (min of enabled limits)")
        if eff_vll is not None:
            lim_lines.append(f"Effective downhole line-line limit (for Ke ceilings): {eff_vll:.1f} Vrms")
        lim_lines.append(
            f"Cable phase-current limit basis: {p.cable.i_limit_basis}; per-conductor hard limit input: {p.cable.i_limit_arms:.3f} A")
        lim_lines.append(
            f"Field weakening: {self._fmt_bool(p.fw.enabled)} (Id_max={p.fw.id_max_arms:.3f} Arms, apply_only_above_base={self._fmt_bool(p.fw.apply_only_above_base)})")
        story.append(Paragraph("<br/>".join(lim_lines), styles["Small"]))

        # ----- Executive summary (operating points) -----
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Executive summary", styles["H2"]))

        cab1 = CableParams(**vars(p.cable));
        cab1.wires_per_phase = 1
        cab2 = CableParams(**vars(p.cable));
        cab2.wires_per_phase = 2

        operating_points = [
            (0.5, 250.0, "Continuous check"),
            (1.0, 250.0, "Continuous check"),
            (0.5, 1000.0, "Peak check"),
            (1.0, 1000.0, "Peak check"),
        ]

        rows = [["Case", "Wires/phase", "Output rpm", "Output torque (ft-lbf)", "Feasible", "Primary limiter (if any)"]]

        def primary_limiter(res: SolveResult) -> str:
            if res.feasible:
                return "—"
            if not res.reasons:
                return "Constraint violation"
            # first reason is usually most informative in our solver
            return res.reasons[0]

        for (rpm, tq, tag) in operating_points:
            for wpp, cab in [(1, cab1), (2, cab2)]:
                res = model.solve_point(rpm, tq, cable_override=cab)
                rows.append([tag, str(wpp), f"{rpm:.2f}", f"{tq:.0f}", "PASS" if res.feasible else "FAIL",
                             primary_limiter(res)])

        tbl = Table(rows, colWidths=[1.15 * inch, 0.85 * inch, 0.8 * inch, 1.15 * inch, 0.75 * inch, 3.3 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(tbl)

        story.append(Spacer(1, 0.14 * inch))
        story.append(Paragraph(
            "Note on the 'motor torque looks low' concern: with very large total ratio, output torque is multiplied by the gearbox, so motor torque at the electrical machine can be ~1 N·m even when output torque is hundreds to a thousand ft-lbf."
            " Motor feasibility is therefore usually governed by voltage and current constraints, not by motor torque magnitude alone.",
            styles["Small"],
        ))

        # ----- Inputs summary -----
        story.append(PageBreak())
        story.append(Paragraph("Model inputs used", styles["H1"]))

        in_rows = [
            ["Block", "Parameter", "Value"],
            ["Target (UI)", "Selected point", f"{p.target.out_rpm:.3f} rpm @ {p.target.out_torque_ftlbf:.0f} ft-lbf"],
            ["Gearbox", "Stages", f"{p.gearbox.stage1:.1f} x {p.gearbox.stage2:.1f} x {p.gearbox.stage3:.1f}"],
            ["Gearbox", "Total ratio", f"{p.gearbox.ratio():.1f}:1"],
            ["Gearbox", "Total efficiency",
             f"{p.gearbox.eff_total():.3f} (override={self._fmt_bool(p.gearbox.override_total_eta)})"],
            ["Motor", "Pole pairs", f"{p.motor.pole_pairs}"],
            ["Motor", "Rs", f"{p.motor.rs_ohm:.4f} ohm"],
            ["Motor", "Rs(T) scaling",
             f"en={self._fmt_bool(getattr(p.extra, 'rs_temp_enabled', False))}, ΔT_w={getattr(p.extra, 'winding_rise_C', 0.0):.1f} °C, α_R={100.0 * getattr(p.extra, 'rs_temp_coeff_per_C', 0.0):.3f} %/°C, T_w={float(getattr(p.extra, 'temp_C', 25.0)) + (getattr(p.extra, 'winding_rise_C', 0.0) if getattr(p.extra, 'rs_temp_enabled', False) else 0.0):.1f} °C"],
            ["Motor", "Ld / Lq", f"{p.motor.ld_h:.6f} H / {p.motor.lq_h:.6f} H"],
            ["Motor", "Lambda", f"{p.motor.lambda_wb:.6f} Wb"],
            ["Motor", "Ke (canonical)", f"{p.motor.ke_vll_rms_per_krpm:.2f} Vll_rms/krpm"],
            ["Motor", "Kt (canonical)", f"{p.motor.kt_nm_per_arms:.4f} Nm/Arms"],
            ["Extra torque", "Enabled", f"{self._fmt_bool(p.extra.extra_enabled)}"],
            ["Extra torque", "T (for Kt(T) / τ_visc)", f"{p.extra.temp_C:.1f} °C (Tref={p.extra.temp_ref_C:.1f} °C)"],
            ["Extra torque", "Kt(T) scaling",
             f"en={self._fmt_bool(p.extra.kt_temp_enabled)}, dKt/dT={100.0 * p.extra.kt_temp_coeff_per_C:+.3f} %/°C"],
            ["Extra torque", "τ_core",
             f"en={self._fmt_bool(p.extra.core_enabled)}, C_L={p.extra.core_cL:.4f}, exp={p.extra.core_exp:.2f}"],
            ["Extra torque", "τ_visc",
             f"en={self._fmt_bool(p.extra.visc_enabled)}, model={p.extra.visc_model}, rpm1={p.extra.visc_rpm1:.0f}, rpm2={p.extra.visc_rpm2:.0f}"],
            ["Extra torque", "τ_visc coeffs",
             f"k_c={p.extra.visc_k_couette:.4f}, k_tr={p.extra.visc_k_transition:.4f}, n_tr={p.extra.visc_n_transition:.2f}, k_tb={p.extra.visc_k_turb:.6f}"],
            ["Extra torque", "τ_visc T-scaling",
             f"{p.extra.visc_temp_scaling} (lin={100.0 * p.extra.visc_lin_coeff_per_C:+.3f} %/°C, beta={p.extra.visc_beta_per_C:.4f} 1/°C), smooth={self._fmt_bool(p.extra.smooth_transitions)}"],
            ["Cable", "Length", f"{p.cable.length_m:.1f} m"],
            ["Cable", "R per conductor", f"{p.cable.r_ohm_per_m:.6f} ohm/m"],
            ["Cable", "L per conductor", f"{p.cable.l_h_per_m:.9f} H/m"],
            ["Cable", "Wires per phase", f"{p.cable.wires_per_phase}"],
            ["Cable", "Effective R_phase", f"{p.cable.effective_r_phase():.6f} ohm"],
            ["Cable", "Effective L_phase", f"{p.cable.effective_l_phase():.9f} H"],
            ["V/f", "Limit type/value", f"{p.vf.v_limit_type} = {p.vf.v_limit_value:.1f}"],
            ["V/f", "Base f", f"{p.vf.base_freq_hz:.1f} Hz"],
            ["V/f", "Base Vphase", f"{p.vf.base_v_phase_rms:.1f} Vrms"],
            ["V/f", "Vboost", f"{p.vf.v_boost:.1f} Vrms"],
            ["Downhole", "Motor Vphase limit",
             f"{p.limits.downhole_v_phase_rms_limit:.1f} Vrms (enforced={self._fmt_bool(p.limits.enforce_downhole_vphase_limit)})"],
            ["Downhole", "Contact Vll limit",
             f"{p.limits.downhole_vll_rms_limit:.1f} Vrms (enforced={self._fmt_bool(p.limits.enforce_downhole_vll_limit)})"],
        ]
        in_tbl = Table(in_rows, colWidths=[1.1 * inch, 2.0 * inch, 4.4 * inch])
        in_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(in_tbl)

        story.append(Spacer(1, 0.16 * inch))
        story.append(Paragraph(
            "Interpretation of 70 Vrms downhole motor limit: in this tool the motor-side limit is phase-to-neutral (L-N) Vrms. "
            "A contact-block limit is typically specified as line-to-line (L-L) Vrms and is internally converted to an equivalent phase limit by dividing by √3.",
            styles["Small"],
        ))

        # ----- Recommendations -----
        story.append(Paragraph("Recommendations", styles["H1"]))

        # choose a representative "selected" point and wiring for recommendations
        sel_wires = int(p.cable.wires_per_phase)
        sel_cab = cab1 if sel_wires == 1 else cab2
        sel_res = model.solve_target(cable_override=sel_cab)

        recs = []
        if sel_res.feasible:
            recs.append(
                "The selected operating point is feasible under current limits. Use sweeps to quantify margin to peak requirement and margin to 1.0 rpm.")
        else:
            # voltage vs current diagnosis
            current_limited = (sel_res.iq_req_rms > sel_res.iq_max_rms + 1e-9)
            vph_eff = model._effective_downhole_phase_limit()
            v_need_id0 = model.motor_voltage_required_phase_rms(sel_res.motor_rpm, 0.0,
                                                                sel_res.iq_req_rms) + model.cable_drop_phase_rms(
                sel_res.motor_rpm, sel_res.iq_req_rms, sel_cab)
            voltage_limited = (v_need_id0 > sel_res.v_surface_cmd + 1e-6) or (
                    vph_eff is not None and model.motor_voltage_required_phase_rms(sel_res.motor_rpm, 0.0,
                                                                                   sel_res.iq_req_rms) > vph_eff + 1e-6)

            if voltage_limited:
                if math.isfinite(sel_res.ke_required_max_vll_krpm):
                    recs.append(
                        f"Voltage is the primary limiter at the selected point. Reduce motor Ke (or reduce total ratio) to satisfy downhole voltage at speed. "
                        f"Rule-of-thumb Ke ceiling at this operating point: Ke <= {sel_res.ke_required_max_vll_krpm:.1f} Vll_rms/krpm ({sel_res.ke_required_max_vll_krpm * math.sqrt(2.0):.1f} Vll_peak/krpm) (see Motor design and Speed & voltage sweeps)."
                    )
                else:
                    recs.append(
                        "Voltage is limiting at the selected point. Reduce Ke or reduce total ratio; enable downhole voltage limits to compute a numeric Ke ceiling.")

                if p.fw.enabled:
                    recs.append(
                        "Field weakening is enabled: use the Field weakening sweep to determine whether Id budget meaningfully increases speed ceiling without sacrificing required Iq.")
                else:
                    recs.append(
                        "Consider enabling Field weakening for trade studies (it can extend speed under a hard voltage limit, but it consumes current headroom).")

            if current_limited:
                recs.append(
                    f"Current is limiting at the selected point. With the present cable basis, the minimum effective Kt(T) required (incl τ_extra) is {sel_res.kt_required_min:.4f} Nm/Arms "
                    f"({sel_res.kt_required_min / LBIN_TO_NM:.2f} lb-in/Arms) at T={sel_res.temp_C:.1f}°C. "
                    "The most direct lever is increasing available phase current (e.g., more conductors per phase), improving gearbox efficiency, or increasing ratio (while checking voltage)."
                )

            if (not voltage_limited) and (not current_limited):
                recs.append(
                    "Multiple constraints interact; consult sweeps to isolate the dominant lever (ratio, Ke, downhole limits, cable length, inverter plateau).")

        # motor-length note
        recs.append(
            "Motor length impacts torque constant and back-EMF together: increasing active length typically increases both Kt and Ke roughly proportionally (via flux linkage). "
            "That helps torque under a current limit but makes voltage margin tighter at high speed. Use the Ke sweeps to decide whether added length is helpful or harmful under the 70 Vrms hard limit."
        )

        story.append(Paragraph("<br/>".join([f"• {r}" for r in recs]), styles["Body"]))

        # ----- Plots -----
        story.append(PageBreak())
        story.append(Paragraph("Plots and interpretation", styles["H1"]))

        def add_plot(title: str, fig: Figure, caption: str):
            story.append(Paragraph(title, styles["H2"]))
            story.append(self._fig_to_rl_image(fig, width_in=7.6, dpi=200))
            story.append(Paragraph(caption, styles["Small"]))
            story.append(Spacer(1, 0.18 * inch))

        add_plot(
            "Envelope",
            self.mpl_env.fig,
            "Shows max torque envelope for 1-wire and 2-wire per phase, plus voltage budget at the selected point (motor required + cable drop vs commanded V/f). "
            "If the voltage bars exceed the command or downhole limits, the system is voltage-limited; if Iq_req exceeds Iq_max, the system is current-limited.",
        )

        add_plot(
            "Sweep - Ratio trade",
            self.sweep_ratio.fig,
            "Sweeps total ratio to show how torque headroom and voltage/Ke ceilings move. Use this to identify a ratio range that can hit 1.0 rpm under the downhole voltage limit while still meeting peak torque with available current and efficiency.",
        )

        add_plot(
            "Sweep - Speed and voltage",
            self.sweep_speed.fig,
            "Highlights simplified speed ceilings driven by hard voltage limits (downhole and surface plateau). If the 1.0 rpm line is above the ceiling at your current ratio, either reduce Ke, reduce ratio, or use field weakening (with current tradeoff).",
        )

        add_plot(
            "Sweep - Motor design (Ke)",
            self.sweep_motor.fig,
            "Shows how Ke drives speed ceiling (voltage) and current required (torque constant link). The feasible region is typically a window: too high Ke fails voltage at speed; too low Ke fails torque under current limit.",
        )

        add_plot(
            "Sweep - Field weakening",
            self.sweep_fw.fig,
            "Quantifies benefit of allocating negative Id under a hard downhole voltage limit. Useful only if voltage is the limiter and you have enough current magnitude left for required Iq.",
        )

        add_plot(
            "Sweep - Cable sensitivity",
            self.sweep_cable.fig,
            "Shows how cable length/resistance/inductance affects available current and voltage margin. If cable loss or drop dominates, increase conductor count per phase, reduce length, or adjust electrical frequency/pole-pairs.",
        )

        add_plot(
            "Sweep - Surface inverter and V/f parameters",
            self.sweep_inverter.fig,
            "Shows sensitivity to surface inverter limit and V/f plateau (base V and base frequency). If surface command is limiting, increase base Vphase or inverter phase limit, or reduce required electrical frequency (via pole pairs or ratio).",
        )

        add_plot(
            "Sweep - Motor pole pairs",
            self.sweep_poles.fig,
            "Motor pole pairs affect electrical frequency at a given rpm, impacting inductive drop and voltage requirements. Lower pole pairs reduce electrical frequency and typically improve voltage margin, but the Kt/Ke linkage must remain consistent.",
        )

        add_plot(
            "Sweep - Downhole voltage limits",
            self.sweep_limits.fig,
            "Sensitivity to downhole motor Vphase and contact Vll limits. Use this to translate a requirements discussion (e.g., 70 Vrms phase limit) into concrete feasibility margins.",
        )

        story.append(Paragraph(
            "Model caveats: The dq voltage magnitude model is a steady-state approximation suitable for early trade studies. For final design, validate against detailed motor/inverter models (including PWM, harmonics, temperature-dependent Rs, saturation, and control-loop limits).",
            styles["Small"],
        ))

        doc.build(story);


def main():
    app = QApplication([])
    w = NavigatorVfWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
