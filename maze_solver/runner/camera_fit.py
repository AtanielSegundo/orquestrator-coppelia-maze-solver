"""
Pose correction from the camera: fit (longitudinal, lateral, yaw) so the
predicted floor-contact profiles of every view match the measured ones.

Unknowns
--------
delta = (a, b, psi) in the frame of the heading frozen at capture: `a` along
it, `b` to its left, `psi` the kart's yaw error. The candidate pose is

    P(delta) = (x0 + c*a - s*b,  y0 + s*a + c*b,  theta0 + psi)

-- the same parameterisation the caller uses to apply it (SimMirror.reanchor).

Each side view also gets a pan nuisance `eta` (bounded, with a prior): a hobby
servo does not land on the same degree twice, and an unmodelled pan error would
otherwise be read as kart yaw. It enters as an extra rotation of that view.

Residuals
---------
Per column, the signed gap between the measured contact interval and the
predicted one (runner/contact_model.py), in units of the pixel sigma, through a
Huber loss truncated at CAMERA_FIX_TRUNCATE_K -- a person or a bag in view is a
bounded outlier rather than a lever. The data cost is divided by
CAMERA_FIX_COLUMN_CORRELATION (neighbouring columns see the same edge and are
not independent evidence). Priors: yaw (honest, CAMERA_FIX_YAW_PRIOR_DEG), a
weak one on translation (so an axis the data does not see settles at zero), and
each nuisance's.

Search: a grid to find the basin, Levenberg-Marquardt to finish it
-----------------------------------------------------------------
The cost has real local minima -- the next cell's walls look like this cell's --
so it starts with an exhaustive coarse grid (11x11 translations x 13 yaws, each
nuisance minimised out). Scored at a WIDENED sigma (CAMERA_FIX_COARSE_SIGMA_
SCALE): near a side wall a contact moves ~4 px per cm, so at the real sigma a
2 cm grid step would put every candidate but the exact one on the truncated
plateau.

The best few local minima of that grid are then each finished by robust
Levenberg-Marquardt over (a, b, psi, every eta), annealing sigma from the coarse
value down to the real one. Not a finer grid: at the real sigma the basin is a
narrow DIAGONAL valley (translation along a view's line of sight trades against
rotation), and a 5 mm / 0.25 deg grid lands in it only by luck -- measured at
tcc.maze's start cell, the grid kept a point costing twice the truth's. Gauss-
Newton walks along the valley.

Observability, then gates
-------------------------
The Gauss-Newton information matrix of the DATA (nuisance priors included,
kart priors not) gives the covariance; its (a, b, psi) block is the MARGINAL
covariance with every nuisance integrated out. An axis whose marginal std is
too large is reported as None. Marginal, not per-axis: in front of a single
wall seen face-on, lateral and yaw slide together along a valley that a
one-axis rise test does not see.

Then the gates (`_judge`): enough columns, residuals the model explains, the
optimum not on the edge of the window, no second lateral hypothesis nearly as
good, plausible magnitudes, and a real improvement over the uncorrected pose.

Everything here is plain numpy and touches no I/O, so it is tested against
synthetic frames with no kart and no simulator.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from camera_link.contact_profile import HIT, ContactProfile
from runner.constants import (
    CAMERA_FIX_AMBIGUITY_MIN_DELTA, CAMERA_FIX_AMBIGUITY_SEPARATION_M,
    CAMERA_FIX_COARSE_SIGMA_SCALE, CAMERA_FIX_COARSE_STEP_M,
    CAMERA_FIX_COARSE_STEP_YAW_DEG, CAMERA_FIX_COLUMN_CORRELATION,
    CAMERA_FIX_CONFIRM_M, CAMERA_FIX_CONFIRM_YAW_DEG, CAMERA_FIX_FINE_STARTS,
    CAMERA_FIX_HUBER_K, CAMERA_FIX_LM_ITERATIONS,
    CAMERA_FIX_MAX_LATERAL_M, CAMERA_FIX_MAX_LONG_M, CAMERA_FIX_MAX_RMS_PX,
    CAMERA_FIX_MAX_STD_LAT_M, CAMERA_FIX_MAX_STD_LONG_M,
    CAMERA_FIX_MAX_STD_YAW_DEG, CAMERA_FIX_MAX_YAW_DEG,
    CAMERA_FIX_MIN_HITS_PER_VIEW, CAMERA_FIX_MIN_HITS_TOTAL,
    CAMERA_FIX_MIN_IMPROVEMENT, CAMERA_FIX_MIN_INLIER_FRACTION,
    CAMERA_FIX_PAN_NUISANCE_HALF_DEG, CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG,
    CAMERA_FIX_PIXEL_SIGMA_PX, CAMERA_FIX_SEARCH_HALF_M,
    CAMERA_FIX_SEARCH_HALF_YAW_DEG, CAMERA_FIX_TRANSLATION_PRIOR_M,
    CAMERA_FIX_TRUNCATE_K, CAMERA_FIX_YAW_PRIOR_DEG, SCAN_FIT_YIELD_S,
)
from runner.contact_model import near_walls, predict_intervals
from runner.view_geometry import ViewGeometry
from runner.wall_world import WallWorld

CORRECTED = "corrigido"
CONFIRMED = "confirmado"
REFUSED = "recusado"


@dataclass
class FitParams:
    """Every knob of the fit, defaulting to runner/constants.py. A dataclass so
    tests can loosen one thing without monkeypatching the module."""
    sigma_px: float = CAMERA_FIX_PIXEL_SIGMA_PX
    huber_k: float = CAMERA_FIX_HUBER_K
    trunc_k: float = CAMERA_FIX_TRUNCATE_K
    corr: float = CAMERA_FIX_COLUMN_CORRELATION
    half_m: float = CAMERA_FIX_SEARCH_HALF_M
    half_yaw_deg: float = CAMERA_FIX_SEARCH_HALF_YAW_DEG
    coarse_m: float = CAMERA_FIX_COARSE_STEP_M
    coarse_yaw_deg: float = CAMERA_FIX_COARSE_STEP_YAW_DEG
    coarse_sigma_scale: float = CAMERA_FIX_COARSE_SIGMA_SCALE
    n_starts: int = CAMERA_FIX_FINE_STARTS
    lm_iterations: int = CAMERA_FIX_LM_ITERATIONS
    yaw_prior_deg: float = CAMERA_FIX_YAW_PRIOR_DEG
    trans_prior_m: float = CAMERA_FIX_TRANSLATION_PRIOR_M
    eta_sigma_deg: float = CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG
    eta_half_deg: float = CAMERA_FIX_PAN_NUISANCE_HALF_DEG
    min_hits_view: int = CAMERA_FIX_MIN_HITS_PER_VIEW
    min_hits_total: int = CAMERA_FIX_MIN_HITS_TOTAL
    min_inlier: float = CAMERA_FIX_MIN_INLIER_FRACTION
    max_rms_px: float = CAMERA_FIX_MAX_RMS_PX
    max_std_lat: float = CAMERA_FIX_MAX_STD_LAT_M
    max_std_long: float = CAMERA_FIX_MAX_STD_LONG_M
    max_std_yaw_deg: float = CAMERA_FIX_MAX_STD_YAW_DEG
    max_lat: float = CAMERA_FIX_MAX_LATERAL_M
    max_long: float = CAMERA_FIX_MAX_LONG_M
    max_yaw_deg: float = CAMERA_FIX_MAX_YAW_DEG
    ambig_sep_m: float = CAMERA_FIX_AMBIGUITY_SEPARATION_M
    ambig_min_delta: float = CAMERA_FIX_AMBIGUITY_MIN_DELTA
    min_improvement: float = CAMERA_FIX_MIN_IMPROVEMENT
    confirm_m: float = CAMERA_FIX_CONFIRM_M
    confirm_yaw_deg: float = CAMERA_FIX_CONFIRM_YAW_DEG
    yield_s: float = SCAN_FIT_YIELD_S


class ViewMeasurement(NamedTuple):
    geom: ViewGeometry
    profile: ContactProfile
    veto: np.ndarray                 # (n,) columns not to use

    @property
    def name(self) -> str:
        return self.geom.name


class CameraFit(NamedTuple):
    outcome: str                     # corrigido | confirmado | recusado
    reason: str
    longitudinal: Optional[float]    # m, None = not observable / refused
    lateral: Optional[float]         # m, + = left of the capture heading
    yaw: Optional[float]             # rad, + = CCW
    estimate: Tuple[float, float, float]
    std: Tuple[float, float, float]  # marginal: m, m, rad
    etas: Dict[str, float]           # per-view pan nuisance, rad
    cost: float
    cost_zero: float
    s2: float
    n_columns: int
    n_hits: int
    inlier_fraction: float
    rms_px: float
    views: Tuple[str, ...]

    def describe(self) -> str:
        def cm(v):
            return "n/d" if v is None else f"{v * 100:+.1f} cm"
        yaw = "n/d" if self.yaw is None else f"{math.degrees(self.yaw):+.1f} deg"
        head = (f"lat {cm(self.lateral)}, yaw {yaw}, long {cm(self.longitudinal)} "
                f"({len(self.views)} vista(s), {self.n_hits} col, "
                f"rms {self.rms_px:.1f} px)")
        if self.outcome == REFUSED:
            return f"recusado: {self.reason} | {head}"
        if self.outcome == CONFIRMED:
            return f"confirmado | {head}"
        return head


def _rho(e: np.ndarray, k: float, trunc: float) -> np.ndarray:
    e = np.minimum(np.abs(e), trunc)
    return np.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k))


# A column whose residual moves more than this (px) over one Jacobian step did
# not slope, it JUMPED: its contact moved to another wall across a corner. One
# 1 mm / 0.06 deg step moves a smooth contact by under a pixel.
_JUMP_PX = 2.5


def _difference(r_step: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """Forward difference of residuals (px), zero where it is not a slope.

    Two cases are not: a column that becomes UNKNOWN on one side of the step
    (a change in what the model can say), and a column whose contact jumps to
    another wall across a corner. Taken as slopes, both are derivatives of
    pixels-per-millimetre-of-nothing that swamp every real column in the
    normal equations -- measured at tcc.maze (0, 3), the LM stalled 0.6 cost
    units above the truth with lambda pinned at its ceiling. The cost itself
    still counts them; only the local linear model leaves them out.
    """
    diff = r_step - r
    d = diff / h
    bad = np.isnan(r_step) | np.isnan(r) | (np.abs(diff) > _JUMP_PX)
    return np.where(bad, 0.0, d)


def _grid(center: float, half: float, step: float, lo: float, hi: float) -> np.ndarray:
    n = int(round(half / step)) if step > 0 else 0
    vals = center + step * np.arange(-n, n + 1)
    vals = vals[(vals >= lo - 1e-9) & (vals <= hi + 1e-9)]
    return vals if len(vals) else np.array([min(max(center, lo), hi)])


class _Stage(NamedTuple):
    a: np.ndarray
    b: np.ndarray
    psi: np.ndarray
    J_data: np.ndarray               # (A, B, P)
    J_map: np.ndarray
    etas: Dict[str, np.ndarray]      # view -> (A, B, P) chosen eta


class _Solution(NamedTuple):
    x: np.ndarray                    # a, b, psi, eta_1 ... eta_k
    cost_map: float                  # robust data cost + priors, real sigma
    cost_data: float
    residuals: np.ndarray            # final data residuals, sigma units
    H_data: np.ndarray               # Gauss-Newton information, data + eta priors


class CameraFitter:
    """Incremental: `add_view` as each view's measurement arrives (its coarse
    table is computed right away, overlapping the next view's servo move),
    `solve` once all are in."""

    def __init__(self, world: WallWorld, capture_pose, params: FitParams = None):
        self.world = world
        self.capture = tuple(capture_pose)
        self.p = params or FitParams()
        self.views: List[ViewMeasurement] = []
        self.dropped: Dict[str, str] = {}
        self._coarse_tables: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        reach = 1.2 + 2.0 * self.p.half_m
        self.walls = near_walls(world, self.capture, reach)
        self._coarse_axes = self._coarse_grid()

    # -- grids -------------------------------------------------------------
    def _coarse_grid(self):
        p = self.p
        a = _grid(0.0, p.half_m, p.coarse_m, -p.half_m, p.half_m)
        b = a.copy()
        step = math.radians(p.coarse_yaw_deg)
        half = math.radians(p.half_yaw_deg)
        psi = _grid(0.0, half, step, -half, half)
        return a, b, psi, step

    def _eta_values(self, step: float, nuisance: bool):
        if not nuisance or self.p.eta_sigma_deg <= 0:
            return np.array([0.0])
        lim = math.radians(self.p.eta_half_deg)
        return _grid(0.0, lim, step, -lim, lim)

    # -- per-view evaluation -------------------------------------------------
    def _pose(self, a, b):
        x0, y0, th0 = self.capture
        c, s = math.cos(th0), math.sin(th0)
        return x0 + c * a - s * b, y0 + s * a + c * b

    @staticmethod
    def _used(meas: ViewMeasurement):
        m_lo, m_hi = meas.profile.intervals()
        use = ~meas.veto & ~np.isnan(m_lo)
        return m_lo[use], m_hi[use], use

    def _signed_gaps(self, meas, xs, ys, theta) -> np.ndarray:
        """(B, n_used) signed gap in pixels: positive when the measured contact
        is further down the image than the prediction allows, negative when
        further up, 0 inside. NaN where the model has nothing to say."""
        m_lo, m_hi, use = self._used(meas)
        p_lo, p_hi, _kind = predict_intervals(meas.geom, xs, ys, theta,
                                              self.world, self.walls)
        p_lo, p_hi = p_lo[:, use], p_hi[:, use]
        with np.errstate(invalid="ignore"):
            pos = m_lo[None, :] - p_hi
            neg = p_lo - m_hi[None, :]
            out = np.where(pos > 0, pos, np.where(neg > 0, -neg, 0.0))
        undefined = np.isnan(p_lo) | np.isnan(p_hi)
        return np.where(undefined, np.nan, out)

    def _view_table(self, meas: ViewMeasurement, a_vals, b_vals, psi_eff,
                    sigma: float) -> np.ndarray:
        """Data cost (already divided by the column correlation) of one view
        over a translation grid x a list of effective headings."""
        th0 = self.capture[2]
        aa, bb = np.meshgrid(a_vals, b_vals, indexing="ij")
        xs, ys = self._pose(aa.ravel(), bb.ravel())
        out = np.zeros((len(a_vals), len(b_vals), len(psi_eff)))
        if not np.any(self._used(meas)[2]):
            return out
        for k, psi in enumerate(psi_eff):
            gap = self._signed_gaps(meas, xs, ys, th0 + psi)
            rho = _rho(gap / sigma, self.p.huber_k, self.p.trunc_k)
            out[..., k] = np.nansum(rho, axis=1).reshape(aa.shape) / self.p.corr
            if self.p.yield_s:
                time.sleep(self.p.yield_s)
        return out

    def _stage(self, a_vals, b_vals, psi_vals, step, coarse: bool) -> _Stage:
        """Costs on a (a, b, psi) grid, every view's nuisance minimised out.
        `coarse` scores with the widened sigma (see the module docstring)."""
        p = self.p
        sigma = p.sigma_px * (p.coarse_sigma_scale if coarse else 1.0)
        sig_eta = math.radians(max(p.eta_sigma_deg, 1e-9))
        J_data = np.zeros((len(a_vals), len(b_vals), len(psi_vals)))
        etas = {}
        for meas in self.views:
            name = meas.name
            eta_vals = self._eta_values(step, meas.geom.nuisance)
            offs = np.unique(np.round((psi_vals[:, None] + eta_vals[None, :]) / step).astype(int))
            cached = self._coarse_tables.get(name) if coarse else None
            if (cached is not None and np.array_equal(a_vals, self._coarse_axes[0])
                    and np.array_equal(b_vals, self._coarse_axes[1])):
                table_offs, table = cached
                lut = {o: i for i, o in enumerate(table_offs)}
                T = np.stack([table[..., lut[o]] for o in offs], axis=-1)
            else:
                T = self._view_table(meas, a_vals, b_vals, offs * step, sigma)
            index = {o: i for i, o in enumerate(offs)}
            best = np.full(J_data.shape, np.inf)
            arg = np.zeros(J_data.shape)
            for eta in eta_vals:
                prior = (0.5 * (eta / sig_eta) ** 2) if meas.geom.nuisance else 0.0
                for i, psi in enumerate(psi_vals):
                    cand = T[..., index[int(round((psi + eta) / step))]] + prior
                    better = cand < best[..., i]
                    best[..., i] = np.where(better, cand, best[..., i])
                    arg[..., i] = np.where(better, eta, arg[..., i])
            J_data += best
            etas[name] = arg
        aa, bb, pp = np.meshgrid(a_vals, b_vals, psi_vals, indexing="ij")
        J_map = (J_data
                 + 0.5 * (pp / math.radians(p.yaw_prior_deg)) ** 2
                 + 0.5 * (aa ** 2 + bb ** 2) / p.trans_prior_m ** 2)
        return _Stage(a_vals, b_vals, psi_vals, J_data, J_map, etas)

    # -- public ------------------------------------------------------------
    def add_view(self, meas: ViewMeasurement) -> bool:
        """Accept one view's measurement. False (and the reason in `dropped`)
        when it has too few contacts to be worth a table."""
        hits = int(np.count_nonzero((meas.profile.kind == HIT) & ~meas.veto))
        if hits < self.p.min_hits_view:
            self.dropped[meas.name] = f"{hits} contatos"
            return False
        self.views.append(meas)
        a, b, psi, step = self._coarse_axes
        eta = self._eta_values(step, meas.geom.nuisance)
        offs = np.unique(np.round((psi[:, None] + eta[None, :]) / step).astype(int))
        sigma = self.p.sigma_px * self.p.coarse_sigma_scale
        self._coarse_tables[meas.name] = (offs, self._view_table(meas, a, b,
                                                                 offs * step, sigma))
        return True

    def solve(self) -> CameraFit:
        if not self.views:
            return self._refused("nenhuma vista util" + self._dropped_note())
        hits = sum(int(np.count_nonzero((m.profile.kind == HIT) & ~m.veto))
                   for m in self.views)
        if hits < self.p.min_hits_total:
            return self._refused(f"so {hits} contatos (minimo {self.p.min_hits_total})")

        a, b, psi, step = self._coarse_axes
        coarse = self._stage(a, b, psi, step, coarse=True)
        solutions = [self._lm(self._start_vector(coarse, idx))
                     for idx in self._basins(coarse)]
        solutions.sort(key=lambda sol: sol.cost_map)
        best = solutions[0]
        return self._judge(best, solutions[1:], hits)

    # -- internals ---------------------------------------------------------
    def _dropped_note(self) -> str:
        if not self.dropped:
            return ""
        return " (" + ", ".join(f"{k}: {v}" for k, v in self.dropped.items()) + ")"

    def _basins(self, stage: _Stage) -> List[tuple]:
        """Grid indices of the best few LOCAL minima of the coarse stage.

        The coarse grid samples the truth up to half a step away, and the
        nuisance pans let a neighbouring wrong basin (lateral a step off, made
        up by yaw and pan) undercut the right one by a little. Finishing only
        the global coarse minimum then converges on the wrong basin; finishing
        each of the best few and keeping the cheapest result does not.
        """
        J = stage.J_map
        padded = np.pad(J, 1, constant_values=np.inf)
        is_min = np.ones(J.shape, dtype=bool)
        A, B, P = J.shape
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dp in (-1, 0, 1):
                    if da == db == dp == 0:
                        continue
                    nb = padded[1 + da:1 + da + A, 1 + db:1 + db + B, 1 + dp:1 + dp + P]
                    is_min &= J <= nb
        idx = np.argwhere(is_min)
        order = np.argsort(J[tuple(idx.T)])
        best = float(J[tuple(idx[order[0]])])
        out = []
        for k in order[:self.p.n_starts]:
            cand = tuple(int(v) for v in idx[k])
            # A start far worse than the best is not a rival basin.
            if out and J[cand] > 2.0 * best + 10.0:
                break
            out.append(cand)
        return out

    def _nuisance_views(self) -> List[ViewMeasurement]:
        if self.p.eta_sigma_deg <= 0:
            return []
        return [m for m in self.views if m.geom.nuisance]

    def _start_vector(self, stage: _Stage, idx) -> np.ndarray:
        x = [stage.a[idx[0]], stage.b[idx[1]], stage.psi[idx[2]]]
        x += [float(stage.etas[m.name][idx]) for m in self._nuisance_views()]
        return np.array(x, dtype=float)

    def _residuals(self, x: np.ndarray, only: Optional[str] = None,
                   base: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Signed data residuals in PIXELS per view, at parameters `x`. With
        `only`, recompute just that view and take the rest from `base`."""
        a, b, psi = x[:3]
        eta = {m.name: x[3 + i] for i, m in enumerate(self._nuisance_views())}
        X, Y = self._pose(a, b)
        out = dict(base) if base is not None else {}
        th0 = self.capture[2]
        for meas in self.views:
            if only is not None and meas.name != only:
                continue
            th = th0 + psi + eta.get(meas.name, 0.0)
            out[meas.name] = self._signed_gaps(meas, [X], [Y], th)[0]
        return out

    def _priors(self, x: np.ndarray, with_kart: bool = True):
        """Prior residuals and their (diagonal) Jacobian."""
        p = self.p
        n = len(x)
        sig = [p.trans_prior_m, p.trans_prior_m, math.radians(p.yaw_prior_deg)]
        sig += [math.radians(max(p.eta_sigma_deg, 1e-9))] * (n - 3)
        sig = np.array(sig)
        keep = np.ones(n, dtype=bool)
        if not with_kart:
            keep[:3] = False
        r = np.where(keep, x / sig, 0.0)
        Jp = np.diag(np.where(keep, 1.0 / sig, 0.0))
        return r, Jp

    def _robust(self, r_px: np.ndarray, sigma: float):
        """(cost, weights) of residuals in pixels at `sigma`, NaN-safe."""
        e = np.nan_to_num(r_px / sigma, nan=0.0)
        k, t = self.p.huber_k, self.p.trunc_k
        a = np.abs(e)
        w = np.where(a <= k, 1.0, np.where(a <= t, k / np.maximum(a, 1e-12), 0.0))
        w = np.where(np.isnan(r_px), 0.0, w)
        return float(np.sum(_rho(e, k, t))) / self.p.corr, w

    def _lm(self, x0: np.ndarray) -> _Solution:
        """Robust Levenberg-Marquardt from `x0`, annealing sigma down to the
        real one. Forward-difference Jacobian; a nuisance perturbation only
        re-evaluates its own view."""
        p = self.p
        views = [m.name for m in self.views]
        nuis = [m.name for m in self._nuisance_views()]
        n = len(x0)
        steps = np.array([1e-3, 1e-3, 1e-3] + [1e-3] * (n - 3))
        lo = np.array([-p.half_m, -p.half_m, -math.radians(p.half_yaw_deg)]
                      + [-math.radians(p.eta_half_deg)] * (n - 3))
        hi = -lo
        x = x0.copy()

        def stack(res):
            return np.concatenate([res[v] for v in views])

        def total(res, xx, sigma):
            c, _w = self._robust(stack(res), sigma)
            rp, _ = self._priors(xx)
            return c + 0.5 * float(rp @ rp), c

        res = self._residuals(x)
        schedule = [p.sigma_px * p.coarse_sigma_scale, p.sigma_px * 2.0, p.sigma_px]
        J = None
        for sigma in schedule:
            lam = 1e-3
            cost, _ = total(res, x, sigma)
            for _it in range(p.lm_iterations):
                r = stack(res)
                cols = []
                for i in range(n):
                    xp = x.copy()
                    xp[i] += steps[i]
                    if i >= 3:
                        rp_ = self._residuals(xp, only=nuis[i - 3], base=res)
                    else:
                        rp_ = self._residuals(xp)
                    cols.append(_difference(stack(rp_), r, steps[i]))
                J = np.stack(cols, axis=1) / sigma
                _c, w = self._robust(r, sigma)
                e = np.nan_to_num(r / sigma)
                rp, Jp = self._priors(x)
                H = (J.T * w) @ J / p.corr + Jp.T @ Jp
                g = (J.T * w) @ e / p.corr + Jp.T @ rp
                improved = False
                for _try in range(6):
                    A = H + lam * np.diag(np.diag(H) + 1e-12)
                    try:
                        dx = -np.linalg.solve(A, g)
                    except np.linalg.LinAlgError:
                        lam *= 10.0
                        continue
                    xn = np.clip(x + dx, lo, hi)
                    res_n = self._residuals(xn)
                    cost_n, _ = total(res_n, xn, sigma)
                    if cost_n < cost:
                        x, res, cost = xn, res_n, cost_n
                        lam = max(lam / 10.0, 1e-7)
                        improved = True
                        break
                    lam *= 10.0
                if self.p.yield_s:
                    time.sleep(self.p.yield_s)
                if not improved or np.max(np.abs(dx) / steps) < 1e-2:
                    break

        # Information of the data (plus the nuisance priors, which are part of
        # the measurement model) at the real sigma.
        r = stack(res)
        _c, w = self._robust(r, p.sigma_px)
        cols = []
        for i in range(n):
            xp = x.copy()
            xp[i] += steps[i]
            rp_ = (self._residuals(xp, only=nuis[i - 3], base=res) if i >= 3
                   else self._residuals(xp))
            cols.append(_difference(stack(rp_), r, steps[i]))
        J = np.stack(cols, axis=1) / p.sigma_px
        _rp, Jp_eta = self._priors(x, with_kart=False)
        H_data = (J.T * w) @ J / p.corr + Jp_eta.T @ Jp_eta
        cost_map, cost_data = total(res, x, p.sigma_px)
        return _Solution(x, cost_map, cost_data, r / p.sigma_px, H_data)

    def _marginal_std(self, sol: _Solution, s2: float) -> Tuple[float, float, float]:
        lam, V = np.linalg.eigh(sol.H_data)
        lam = np.maximum(lam, s2 / 1.0)       # floor = 1 m / 1 rad of std
        cov = (V * (s2 / lam)) @ V.T
        return tuple(float(math.sqrt(max(cov[i, i], 0.0))) for i in range(3))

    def _cost_at(self, x: np.ndarray) -> float:
        """Robust data cost at real sigma, each nuisance view's pan minimised
        over its range on a 0.25 deg lattice -- the measure the improvement
        gate compares, so the uncorrected pose gets the same benefit of the
        doubt about its pans as the fit did."""
        step = math.radians(0.25)
        th0 = self.capture[2]
        X, Y = self._pose(x[0], x[1])
        total = 0.0
        for meas in self.views:
            etas = self._eta_values(step, meas.geom.nuisance)
            sig_eta = math.radians(max(self.p.eta_sigma_deg, 1e-9))
            best = math.inf
            for eta in etas:
                g = self._signed_gaps(meas, [X], [Y], th0 + x[2] + eta)[0]
                c, _ = self._robust(g, self.p.sigma_px)
                prior = 0.5 * (eta / sig_eta) ** 2 if meas.geom.nuisance else 0.0
                best = min(best, c + prior)
            total += best
        return total

    def _judge(self, sol: _Solution, others: Sequence[_Solution], hits: int) -> CameraFit:
        p = self.p
        a, b, psi = (float(v) for v in sol.x[:3])
        est = (a, b, psi)
        etas = {m.name: float(sol.x[3 + i]) for i, m in enumerate(self._nuisance_views())}
        n_cols = int(np.count_nonzero(~np.isnan(sol.residuals)))
        n_eff = max(n_cols / p.corr, 4.0)
        s2 = max(1.0, 2.0 * sol.cost_data / max(n_eff - 3.0, 1.0))
        std = self._marginal_std(sol, s2)
        e = sol.residuals[~np.isnan(sol.residuals)]
        inl = np.abs(e) < p.trunc_k
        frac = float(np.mean(inl)) if e.size else 0.0
        rms = float(np.sqrt(np.mean(e[inl] ** 2)) * p.sigma_px) if np.any(inl) else math.inf
        cost_zero = self._cost_at(np.zeros(3))
        cost = self._cost_at(sol.x[:3])
        base = dict(estimate=est, std=std, etas=etas, cost=cost,
                    cost_zero=cost_zero, s2=s2, n_columns=n_cols, n_hits=hits,
                    inlier_fraction=frac, rms_px=rms,
                    views=tuple(m.name for m in self.views))

        def refuse(reason):
            return CameraFit(REFUSED, reason, None, None, None, **base)

        long_obs = std[0] <= p.max_std_long
        lat_obs = std[1] <= p.max_std_lat
        yaw_obs = std[2] <= math.radians(p.max_std_yaw_deg)
        if not lat_obs and not yaw_obs:
            return refuse("lateral e yaw nao observaveis daqui")

        if frac < p.min_inlier or rms > p.max_rms_px:
            return refuse(f"modelo nao explica a imagem (inliers {frac * 100:.0f}%, "
                          f"rms {rms:.1f} px)")

        # Edge of the window, on the axes that are actually estimated.
        edge_m = p.half_m - p.coarse_m + 1e-6
        edge_y = math.radians(p.half_yaw_deg - p.coarse_yaw_deg) + 1e-9
        if ((lat_obs and abs(b) > edge_m) or (long_obs and abs(a) > edge_m)
                or (yaw_obs and abs(psi) > edge_y)):
            return refuse("otimo na borda da janela (pose perdida?)")

        reasons = []
        if lat_obs:
            # A second lateral hypothesis: another start that converged at
            # least `ambig_sep_m` away must cost clearly more, in noise units.
            rivals = [o for o in others if abs(o.x[1] - b) >= p.ambig_sep_m]
            if rivals:
                gap = min(o.cost_map for o in rivals) - sol.cost_map
                if gap / s2 < p.ambig_min_delta:
                    lat_obs = False
                    reasons.append("lateral ambiguo")
        if not lat_obs and not yaw_obs:
            return refuse(", ".join(reasons) or "nada observavel")

        if lat_obs and abs(b) > p.max_lat:
            return refuse(f"lateral {b * 100:+.1f} cm grande demais (celula errada?)")
        if long_obs and abs(a) > p.max_long:
            long_obs = False
        if yaw_obs and abs(psi) > math.radians(p.max_yaw_deg):
            return refuse(f"yaw {math.degrees(psi):+.1f} deg grande demais")

        lat = b if lat_obs else None
        lng = a if long_obs else None
        yaw = psi if yaw_obs else None
        small = ((lat is None or abs(lat) < p.confirm_m)
                 and (lng is None or abs(lng) < p.confirm_m)
                 and (yaw is None or abs(yaw) < math.radians(p.confirm_yaw_deg)))
        reason = ", ".join(reasons)
        if small:
            return CameraFit(CONFIRMED, reason, lng, lat, yaw, **base)
        if cost > p.min_improvement * cost_zero:
            return refuse(f"sem ganho (custo {cost_zero:.1f} -> {cost:.1f})")
        return CameraFit(CORRECTED, reason, lng, lat, yaw, **base)

    def _refused(self, reason) -> CameraFit:
        return CameraFit(REFUSED, reason, None, None, None, (0.0, 0.0, 0.0),
                         (math.inf,) * 3, {}, 0.0, 0.0, 1.0, 0, 0, 0.0,
                         math.inf, tuple(m.name for m in self.views))


def fit_camera_pose(measurements: Sequence[ViewMeasurement], world: WallWorld,
                    capture_pose, params: FitParams = None) -> CameraFit:
    """One-shot convenience over CameraFitter."""
    fitter = CameraFitter(world, capture_pose, params)
    for m in measurements:
        fitter.add_view(m)
    return fitter.solve()
