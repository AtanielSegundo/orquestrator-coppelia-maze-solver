"""
Box edges, real vs simulated, and the parameters that line them up: the
measurement of the camera pose fix v2 (runner/camera_fix_v2.py) and of the
bench calibration of its views (camera_calibrate.py extrinsics).

The edges
---------
Every boundary between a box and anything else in the maze ray caster's image
(runner/reference_renderer.py) is an edge of a box: its top against the room
(seen from the "cima" tilt), its floor contact (the "baixo" tilt), or a
vertical corner. Points are taken along those boundaries every
CAMERA_V2_EDGE_STEP_PX, each with its normal (pointing out of the box) and its
3-D point -- the wall hit of the pixel on the box side.

The real image is reduced to a WALLNESS field: +1 inside the boxes, -1
outside, from the per-view colour model (camera_link/floor_segmenter.py),
lightly blurred. For every simulated point the real edge is searched along
its normal -- the nearest crossing from + to - within the search radius -- and
the signed distance is that point's DELTA: how far the real edge sits from
where the simulation expects it, in pixels.

The sensitivities
-----------------
Box edges are straight 3-D lines that do not move when the kart does. So the
derivative of a delta with respect to any parameter -- the kart's yaw,
longitudinal or lateral offset, a servo's pan, a tilt's pitch, the lens
height -- is the motion of that 3-D point's projection along the normal:
projecting a few thousand points, never rendering. Anything with a
`build(x) -> {view: (pinhole, pose)}` is a parameter set.

Groups
------
The centre pan's points are split into the LEFT and RIGHT half planes of the
image (CAMERA_V2_SPLIT_PAN); every other pan is one group; each tilt is kept
apart, so the box tops ("cima") and the floor contacts ("baixo") of a plane
are separate groups. Per group, the best 2-D shift of its real edges against
the simulated ones is reported: in the centre pan, a horizontal shift common
to both halves is yaw, halves shifting apart or together is a longitudinal
offset; the side pans shift the same way for yaw and opposite ways for a
longitudinal offset. The solve uses every point with its own sensitivity --
the quantitative form of the same reading -- and the groups are also its
outlier unit: a group the result leaves far off (a person in view, a bag, a
misread colour) is dropped and the solve redone once.

The solve
---------
Robust Gauss-Newton: deltas modelled as J dx, Huber, points far off ignored,
a Gaussian prior per parameter. Every pass re-renders at the current x and
re-matches, with the robust scale annealed from wide to CAMERA_V2_HUBER_PX:
on the first pass the deltas are as large as the error being measured.

Plain numpy + OpenCV, no I/O: tested on synthetic frames.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from runner.constants import (
    CAMERA_V2_EDGE_BORDER_PX, CAMERA_V2_EDGE_STEP_PX, CAMERA_V2_GROUP_MAX_RESIDUAL_PX,
    CAMERA_V2_HUBER_PX, CAMERA_V2_INNER_ITERATIONS, CAMERA_V2_MAX_EDGE_RANGE_M,
    CAMERA_V2_MIN_POINTS, CAMERA_V2_MIN_POINTS_PER_GROUP, CAMERA_V2_OUTLIER_K,
    CAMERA_V2_POINT_CORRELATION, CAMERA_V2_SEARCH_PX, CAMERA_V2_SPLIT_PAN,
)
from runner.reference_renderer import WALL, render_labels
from runner.wall_world import WallWorld

# (pinhole, pose) per view name.
Views = Dict[str, Tuple[object, Tuple[float, float, float]]]


class ViewSpec(NamedTuple):
    name: str                    # "frente_cima"
    pan: str                     # "frente"
    tilt: str                    # "cima"
    ignore_bottom_px: int        # kart body band at the bottom of the frame


class RealView(NamedTuple):
    spec: ViewSpec
    field: np.ndarray            # float32 (H, W): + inside a box, - outside
    valid: np.ndarray            # bool (H, W): pixels the comparison may use
    accuracy: float              # colour model's training accuracy


class SimEdges(NamedTuple):
    u: np.ndarray                # (N,) sub-pixel edge position
    v: np.ndarray
    nu: np.ndarray               # (N,) unit normal, out of the box
    nv: np.ndarray
    X: np.ndarray                # (N, 3) the edge's 3-D point, world frame
    group: np.ndarray            # (N,) object: group name


class GroupStat(NamedTuple):
    n: int                       # points matched at the starting parameters
    shift: Tuple[float, float]   # px: best 2-D shift, real edges vs simulated
    mean_delta: float            # px, mean signed delta along the normals
    residual_px: float           # median |delta| left at the fitted parameters


class EdgeFit(NamedTuple):
    ok: bool
    reason: str
    x: np.ndarray
    std: np.ndarray              # per parameter
    n_points: int
    groups: Dict[str, GroupStat]
    dropped: Dict[str, str]      # group -> why it was left out
    rms_px: float                # inliers, at the fitted parameters
    edges: Dict[str, SimEdges]   # at the fitted parameters (debug images)
    deltas: Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# geometry helpers
def _to_centre(Xw: np.ndarray, pose) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    dx, dy = Xw[:, 0] - x, Xw[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy, Xw[:, 2]], axis=1)


def _to_world(Xc: np.ndarray, pose) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return np.stack([x + c * Xc[:, 0] - s * Xc[:, 1],
                     y + s * Xc[:, 0] + c * Xc[:, 1], Xc[:, 2]], axis=1)


def project(pinhole, pose, Xw: np.ndarray):
    """World points -> (u, v, depth) in the view of `pinhole` at kart `pose`."""
    return pinhole.project(_to_centre(Xw, pose))


def wall_field(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """+1 inside, -1 outside, Gaussian-blurred: its zero crossing is the
    boundary with sub-pixel resolution."""
    f = mask.astype(np.float32) * 2.0 - 1.0
    return cv2.GaussianBlur(f, (0, 0), sigma)


def group_names(spec: ViewSpec, u: np.ndarray, cx: float,
                split_pan: str = CAMERA_V2_SPLIT_PAN) -> np.ndarray:
    if spec.pan == split_pan:
        return np.array([f"{spec.pan}_{'esq' if uu < cx else 'dir'}_{spec.tilt}"
                         for uu in u], dtype=object)
    return np.full(len(u), f"{spec.pan}_{spec.tilt}", dtype=object)


# ---------------------------------------------------------------------------
# edges and matching
def find_crossings(field: np.ndarray, u, v, nu, nv, radius: float,
                   step: float = 0.5, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Signed distance (px) along the normal from each point to the nearest
    + to - crossing of `field`; NaN when there is none within `radius` (or it
    falls outside `valid`)."""
    n = len(u)
    if n == 0:
        return np.zeros(0)
    ts = np.arange(-radius, radius + 1e-9, step, dtype=np.float32)
    mx = (np.asarray(u, np.float32)[:, None] + ts[None, :] * np.asarray(nu, np.float32)[:, None])
    my = (np.asarray(v, np.float32)[:, None] + ts[None, :] * np.asarray(nv, np.float32)[:, None])
    H, W = field.shape
    inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    vals = cv2.remap(field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if valid is not None:
        ok = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        inside &= ok
    a, b = vals[:, :-1], vals[:, 1:]
    cross = (a > 0) & (b <= 0) & inside[:, :-1] & inside[:, 1:]
    frac = a / np.where(cross, a - b, 1.0)
    t = np.where(cross, ts[None, :-1] + step * frac, np.inf)
    best = t[np.arange(n), np.argmin(np.abs(t), axis=1)]
    return np.where(np.isfinite(best), best, np.nan).astype(float)


def sim_edges(spec: ViewSpec, pinhole, pose, world: WallWorld, *,
              valid: Optional[np.ndarray] = None,
              obstacles: Optional[np.ndarray] = None,
              step: int = CAMERA_V2_EDGE_STEP_PX,
              border: int = CAMERA_V2_EDGE_BORDER_PX,
              max_range: float = CAMERA_V2_MAX_EDGE_RANGE_M,
              guard_px: int = 3,
              split_pan: str = CAMERA_V2_SPLIT_PAN,
              return_field: bool = False):
    """The simulated box edges of one view, at `pose`.

    Left out: points near the image border or the kart body band, near pixels
    the real frame cannot show (`valid` False, e.g. the undistortion's black
    corners), on or next to a DISCOVERED obstacle (veto only, as in v1 --
    the map's obstacles are a belief, never evidence), and on boxes further
    than `max_range` from the lens. `return_field` also returns the
    simulated wallness field."""
    labels, t_hit = render_labels(pinhole, pose, world, return_t=True)
    wall = labels == WALL
    H, W = wall.shape
    usable = np.zeros((H, W), dtype=bool)
    usable[border:H - border, border:W - border] = True
    if spec.ignore_bottom_px:
        usable[max(0, H - 1 - spec.ignore_bottom_px - guard_px):, :] = False
    k = np.ones((2 * guard_px + 1, 2 * guard_px + 1), np.uint8)
    if valid is not None:
        usable &= cv2.erode(valid.astype(np.uint8), k, borderValue=0).astype(bool)
    if obstacles is not None and len(obstacles):
        with_obs = render_labels(pinhole, pose, world, extra_boxes=obstacles) == WALL
        diff = with_obs != wall
        if diff.any():
            usable &= ~cv2.dilate(diff.astype(np.uint8), k).astype(bool)

    F = wall_field(wall)
    gx = cv2.Sobel(F, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(F, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(wall.astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_REPLICATE).astype(bool)
    ys, xs = np.nonzero(wall & ~inner & usable)
    empty = SimEdges(*(np.zeros(0) for _ in range(4)), np.zeros((0, 3)),
                     np.zeros(0, dtype=object))
    if len(xs) == 0:
        return (empty, F) if return_field else empty
    gxs, gys = gx[ys, xs], gy[ys, xs]
    mag = np.hypot(gxs, gys)
    keep = mag > 1e-3
    nu = -gxs / np.maximum(mag, 1e-9)
    nv = -gys / np.maximum(mag, 1e-9)
    # Uniform density along every edge: a near-horizontal edge is sampled
    # every `step` columns, a near-vertical one every `step` rows.
    horiz = np.abs(nv) >= np.abs(nu)
    keep &= np.where(horiz, xs % step == 0, ys % step == 0)
    tt = t_hit[ys, xs]
    keep &= np.isfinite(tt)
    xs, ys, nu, nv, tt = xs[keep], ys[keep], nu[keep], nv[keep], tt[keep]
    if len(xs) == 0:
        return (empty, F) if return_field else empty
    rays = pinhole.rays(xs.astype(float), ys.astype(float))
    Xc = pinhole.C[None, :] + rays * tt[:, None]
    near = np.linalg.norm(Xc - pinhole.C[None, :], axis=1) <= max_range
    # Sub-pixel: the simulated boundary itself, on its own field.
    tc = find_crossings(F, xs, ys, nu, nv, radius=2.0, step=0.25)
    keep = near & np.isfinite(tc)
    xs, ys, nu, nv, tc, Xc = xs[keep], ys[keep], nu[keep], nv[keep], tc[keep], Xc[keep]
    u = xs + tc * nu
    v = ys + tc * nv
    out = SimEdges(u, v, nu, nv, _to_world(Xc, pose),
                   group_names(spec, u, pinhole.cx, split_pan))
    return (out, F) if return_field else out


def real_edges(real: RealView, step: int = CAMERA_V2_EDGE_STEP_PX,
               border: int = CAMERA_V2_EDGE_BORDER_PX, guard_px: int = 3):
    """(u, v, nu, nv) along the REAL box boundaries -- the other half of a
    symmetric comparison: a real edge the simulation does not explain costs
    as much as a simulated edge the frame does not show."""
    mask = real.field > 0
    H, W = mask.shape
    usable = np.zeros((H, W), dtype=bool)
    usable[border:H - border, border:W - border] = True
    if real.spec.ignore_bottom_px:
        usable[max(0, H - 1 - real.spec.ignore_bottom_px - guard_px):, :] = False
    k = np.ones((2 * guard_px + 1, 2 * guard_px + 1), np.uint8)
    usable &= cv2.erode(real.valid.astype(np.uint8), k, borderValue=0).astype(bool)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_REPLICATE).astype(bool)
    ys, xs = np.nonzero(mask & ~inner & usable)
    gx = cv2.Sobel(real.field, cv2.CV_32F, 1, 0, ksize=3)[ys, xs]
    gy = cv2.Sobel(real.field, cv2.CV_32F, 0, 1, ksize=3)[ys, xs]
    mag = np.hypot(gx, gy)
    nu, nv = -gx / np.maximum(mag, 1e-9), -gy / np.maximum(mag, 1e-9)
    horiz = np.abs(nv) >= np.abs(nu)
    keep = (mag > 1e-3) & np.where(horiz, xs % step == 0, ys % step == 0)
    return (xs[keep].astype(float), ys[keep].astype(float), nu[keep], nv[keep])


def point_jacobian(build: Callable[[np.ndarray], Views], x: np.ndarray,
                   steps: np.ndarray, edges: Dict[str, SimEdges]) -> Dict[str, np.ndarray]:
    """d(delta)/dx per point, (N, P) per view: the projection of each edge's
    3-D point moving along its normal as each parameter moves."""
    base = build(x)
    ref = {}
    for name, e in edges.items():
        pin, pose = base[name]
        u0, v0, _z = project(pin, pose, e.X)
        ref[name] = (u0, v0)
    out = {name: np.zeros((len(e.u), len(x))) for name, e in edges.items()}
    for i in range(len(x)):
        xi = np.array(x, dtype=float)
        xi[i] += steps[i]
        moved = build(xi)
        for name, e in edges.items():
            pin, pose = moved[name]
            ui, vi, _z = project(pin, pose, e.X)
            u0, v0 = ref[name]
            out[name][:, i] = ((ui - u0) * e.nu + (vi - v0) * e.nv) / steps[i]
    return out


def group_shift(nu, nv, d, w=None, ridge: float = 1e-3) -> Tuple[float, float]:
    """The 2-D shift (sx, sy) that best explains deltas `d` along normals
    (nu, nv). An edge only sees the shift across it, so a group of horizontal
    edges has sx ~ 0 (ridge)."""
    w = np.ones(len(d)) if w is None else w
    A = np.array([[np.sum(w * nu * nu), np.sum(w * nu * nv)],
                  [np.sum(w * nu * nv), np.sum(w * nv * nv)]]) + ridge * np.eye(2)
    b = np.array([np.sum(w * nu * d), np.sum(w * nv * d)])
    sx, sy = np.linalg.solve(A, b)
    return float(sx), float(sy)


def _huber_w(r: np.ndarray, k: float, cut: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, np.where(a <= cut, k / np.maximum(a, 1e-12), 0.0))


# ---------------------------------------------------------------------------
def match_all(reals: Dict[str, RealView], world: WallWorld, views: Views,
              search_px: float, obstacles=None, **edge_kw):
    """Simulated edges of every view at `views`, and their deltas against the
    real fields."""
    edges, deltas = {}, {}
    for name, real in reals.items():
        pin, pose = views[name]
        e = sim_edges(real.spec, pin, pose, world, valid=real.valid,
                      obstacles=obstacles, **edge_kw)
        edges[name] = e
        deltas[name] = find_crossings(real.field, e.u, e.v, e.nu, e.nv, search_px,
                                      valid=real.valid)
    return edges, deltas


def matching_cost(reals: Dict[str, RealView], world: WallWorld, views: Views,
                  search_px: float, trunc_px: float, obstacles=None,
                  real_pts: Optional[Dict[str, tuple]] = None, **edge_kw) -> float:
    """Score of a coarse grid search (camera_calibrate.py extrinsics), per view
    and SYMMETRIC: the truncated squared distance from every simulated edge
    point to the real edges, plus from every real edge point to the simulated
    edges, an unmatched point counting as truncated. One-sided, a candidate
    that predicts no edges at all (the camera pitched until the floor leaves
    the frame) scored a perfect zero -- measured on the real kart, it walked
    the "baixo" pitch and a side pan to the edge of the search.
    `real_pts`: the real edge points per view, if already extracted."""
    total = 0.0
    T2 = trunc_px ** 2
    for name, real in reals.items():
        pin, pose = views[name]
        e, F = sim_edges(real.spec, pin, pose, world, valid=real.valid,
                         obstacles=obstacles, return_field=True, **edge_kw)
        d = find_crossings(real.field, e.u, e.v, e.nu, e.nv, search_px, valid=real.valid)
        ru, rv, rnu, rnv = (real_pts[name] if real_pts is not None and name in real_pts
                            else real_edges(real))
        back = find_crossings(F, ru, rv, rnu, rnv, search_px)
        parts = []
        for dd in (d, back):
            if len(dd):
                parts.append(np.where(np.isfinite(dd), np.minimum(dd * dd, T2), T2))
        if parts:
            total += float(np.mean(np.concatenate(parts)))
        else:
            total += T2
    return total


def solve_edges(reals: Dict[str, RealView], world: WallWorld,
                build: Callable[[np.ndarray], Views], x0: Sequence[float],
                prior_sigma: Sequence[float], steps: Sequence[float], *,
                obstacles=None,
                iterations: int = CAMERA_V2_INNER_ITERATIONS,
                search_px: float = CAMERA_V2_SEARCH_PX,
                huber_px: float = CAMERA_V2_HUBER_PX,
                outlier_k: float = CAMERA_V2_OUTLIER_K,
                corr: float = CAMERA_V2_POINT_CORRELATION,
                min_points_group: int = CAMERA_V2_MIN_POINTS_PER_GROUP,
                min_points: int = CAMERA_V2_MIN_POINTS,
                group_max_residual: float = CAMERA_V2_GROUP_MAX_RESIDUAL_PX,
                coarse: Optional[Tuple[int, Sequence[float]]] = None,
                coarse_trunc_px: float = 6.0,
                max_step: Optional[Sequence[float]] = None,
                **edge_kw) -> EdgeFit:
    """Parameters x (prior N(x0, prior_sigma)) that line the simulated edges
    up with the real ones. See the module docstring.

    `coarse` = (index, values): first try each value of that one parameter
    (the others at x0) and start from the one whose truncated matching cost is
    lowest. A few degrees of yaw move every edge by more than the robust scale
    of the first pass, and the nearest real edge can then be the wrong one;
    the grid finds the right basin first.

    `max_step` bounds each pass's change per parameter (the whole step is
    scaled down to fit): a trust region, so a pass on bad correspondences
    cannot throw a weakly held parameter across the scene."""
    x0 = np.asarray(x0, dtype=float)
    start = x0.copy()
    if coarse is not None:
        idx, values = coarse
        best = (math.inf, x0[idx])
        for val in values:
            xc = x0.copy()
            xc[idx] = val
            c = matching_cost(reals, world, build(xc), search_px, coarse_trunc_px,
                              obstacles, **edge_kw)
            if c < best[0]:
                best = (c, val)
        start[idx] = best[1]
    P_inv = np.diag(1.0 / np.square(np.asarray(prior_sigma, dtype=float)))
    steps = np.asarray(steps, dtype=float)
    dropped: Dict[str, str] = {}
    # Annealed robust scale: wide first (the deltas are as big as the error),
    # the real one on the last pass -- never wider than 8x: with many passes
    # an uncapped 2^n start (256 px at 8 passes) is plain least squares, and
    # on the real kart it threw a pan 25 deg off.
    scales = [huber_px * 2.0 ** max(0, min(3, iterations - 1 - i)) for i in range(iterations)]
    limit = None if max_step is None else np.asarray(max_step, dtype=float)
    groups0: Dict[str, GroupStat] = {}

    def stack(edges, deltas, jac):
        r, J, g, nu, nv = [], [], [], [], []
        for name, e in edges.items():
            d = deltas[name]
            use = np.isfinite(d) & np.array([gg not in dropped for gg in e.group], dtype=bool)
            r.append(d[use]); J.append(jac[name][use]); g.append(e.group[use])
            nu.append(e.nu[use]); nv.append(e.nv[use])
        if not r:
            return (np.zeros(0), np.zeros((0, len(x0))), np.zeros(0, dtype=object),
                    np.zeros(0), np.zeros(0))
        return (np.concatenate(r), np.vstack(J), np.concatenate(g),
                np.concatenate(nu), np.concatenate(nv))

    # The groups at the INFORMED parameters x0 (before the coarse grid moved
    # anything): what reads as "this half plane's edges sit N px to that side".
    edges, deltas = match_all(reals, world, build(x0), search_px, obstacles, **edge_kw)
    zero = {name: np.zeros((len(e.u), len(x0))) for name, e in edges.items()}
    r, _J, g, nu, nv = stack(edges, deltas, zero)
    w0 = _huber_w(r, scales[0], outlier_k * scales[0])
    for name in sorted(set(g)):
        m = g == name
        if np.count_nonzero(m) < min_points_group:
            dropped[name] = f"{int(np.count_nonzero(m))} pontos"
            continue
        groups0[name] = GroupStat(int(np.count_nonzero(m)),
                                  group_shift(nu[m], nv[m], r[m], w0[m]),
                                  float(np.mean(r[m])), math.nan)

    x = start.copy()
    H = P_inv.copy()
    for attempt in range(2):
        x = start.copy()
        for it in range(iterations + 1):
            views = build(x)
            edges, deltas = match_all(reals, world, views, search_px, obstacles, **edge_kw)
            jac = point_jacobian(build, x, steps, edges)
            r, J, g, nu, nv = stack(edges, deltas, jac)
            if it == iterations:
                break                                   # final evaluation only
            k = scales[it]
            w = _huber_w(r, k, outlier_k * k)
            A = (J.T * w) @ J / (corr * huber_px ** 2) + P_inv
            b = (J.T * w) @ r / (corr * huber_px ** 2) - P_inv @ (x - x0)
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            if limit is not None:
                ratio = float(np.max(np.abs(dx) / limit))
                if ratio > 1.0:
                    dx = dx / ratio
            x = x + dx
        # Final residuals per group, at the fitted x.
        new_drop = False
        for name in sorted(set(g)):
            m = g == name
            res = float(np.median(np.abs(r[m]))) if np.any(m) else math.nan
            if name in groups0:
                groups0[name] = groups0[name]._replace(residual_px=res)
            if attempt == 0 and np.isfinite(res) and res > group_max_residual:
                dropped[name] = f"fora do modelo ({res:.1f} px)"
                new_drop = True
        if not new_drop:
            break

    w = _huber_w(r, huber_px, outlier_k * huber_px)
    inl = w > 0
    n = int(np.count_nonzero(inl))
    H = (J.T * w) @ J / (corr * huber_px ** 2) + P_inv
    rms = float(np.sqrt(np.mean(r[inl] ** 2))) if n else math.inf
    dof = max(n / corr - len(x), 1.0)
    s2 = max(1.0, float(np.sum(w * r * r)) / (corr * huber_px ** 2) / dof)
    try:
        cov = np.linalg.inv(H) * s2
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        std = np.full(len(x), math.inf)
    ok, reason = True, ""
    if n < min_points:
        ok, reason = False, f"so {n} pontos de borda casados (minimo {min_points})"
    return EdgeFit(ok, reason, x, std, n, groups0, dropped, rms, edges, deltas)


def describe_groups(groups: Dict[str, GroupStat], split_pan: str = CAMERA_V2_SPLIT_PAN) -> List[str]:
    """One line per group, then the centre pan's common / differential
    horizontal shift per tilt -- the user-facing reading of the deltas."""
    lines = [f"{name}: {st.n} pts, desvio ({st.shift[0]:+.1f}, {st.shift[1]:+.1f}) px, "
             f"resto {st.residual_px:.1f} px" for name, st in sorted(groups.items())]
    tilts = sorted({name.rsplit("_", 1)[1] for name in groups if name.startswith(split_pan + "_")})
    for t in tilts:
        le, ri = groups.get(f"{split_pan}_esq_{t}"), groups.get(f"{split_pan}_dir_{t}")
        if le is None or ri is None:
            continue
        common = 0.5 * (le.shift[0] + ri.shift[0])
        diff = 0.5 * (ri.shift[0] - le.shift[0])
        lines.append(f"{split_pan} {t}: comum {common:+.1f} px (yaw), "
                     f"afastamento {diff:+.1f} px (longitudinal)")
    return lines
