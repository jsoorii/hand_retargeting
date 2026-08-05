"""
thumb_retarget.py
=================
Retargeting of a human thumb (glove 6-DOF nail pose) onto a robot thumb with a
yaw + parallel-3R structure (4 independently actuated joints).

Pipeline (runs at CONTROL RATE; glove samples are simply held between updates,
the One-Euro filter does the smoothing/interpolation):

    glove (T_world_dorsum, T_world_nail)
      -> relative pose in dorsum frame
      -> nail -> pad constant offset            [TUNE]
      -> dorsum frame -> robot CMC base frame   [calibration]
      -> position scaling                       [calibration]
      -> task-space filtering (One-Euro on p, geodesic LPF on R)
      -> soft workspace saturation
      -> IK: closed-form yaw + windowed beta scan with 2R closed form
      -> joint clamp + rate limit
      -> FK recompute of ACHIEVED pose (saturation feedback)

Frame / sign conventions
------------------------
Base frame is at the CMC (yaw axis origin).
  q0 : yaw about +z
  q1,q2,q3 : parallel revolute joints, axis = -y after the yaw rotation.
             Equivalently: planar angles measured CCW in the (x', z) plane,
             where x' is the post-yaw forward axis.
  beta = q1 + q2 + q3   (the only quantity the tip orientation depends on)

Because the 3R axes are parallel, the reachable tip orientation set is exactly
    R(q) = Rz(q0) * Rplanar(beta) * R_off
which is a 2-parameter family. One body-fixed axis (`tip_axis_local`) can be
pointed anywhere on S^2; the roll about that axis is structurally uncontrollable
(0 DOF). We therefore never put any cost on roll.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

# ----------------------------------------------------------------------------
# small SO(3) helpers (no scipy dependency required)
# ----------------------------------------------------------------------------


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_planar(b: float) -> np.ndarray:
    """Rotation by +b CCW in the (x, z) plane, i.e. about the -y axis."""
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def log_so3(Rm: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector."""
    cos_t = np.clip((np.trace(Rm) - 1.0) * 0.5, -1.0, 1.0)
    t = np.arccos(cos_t)
    if t < 1e-8:
        return np.array([Rm[2, 1] - Rm[1, 2],
                         Rm[0, 2] - Rm[2, 0],
                         Rm[1, 0] - Rm[0, 1]]) * 0.5
    w = np.array([Rm[2, 1] - Rm[1, 2],
                  Rm[0, 2] - Rm[2, 0],
                  Rm[1, 0] - Rm[0, 1]])
    return w * (t / (2.0 * np.sin(t)))


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix."""
    t = np.linalg.norm(w)
    if t < 1e-12:
        return np.eye(3)
    k = w / t
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def make_T(Rm: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Ri = T[:3, :3].T
    return make_T(Ri, -Ri @ T[:3, 3])


# ----------------------------------------------------------------------------
# task-space filters (run at control rate, BEFORE IK)
# ----------------------------------------------------------------------------


class OneEuroFilter3D:
    """One-Euro filter on a 3-vector. Low cutoff at low speed (kills jitter),
    high cutoff at high speed (kills lag)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev = np.zeros(3)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-9))

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = np.zeros(3)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            return x.copy()
        dx = (x - self.x_prev) / max(dt, 1e-9)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev.copy()


class RotationLPF:
    """Geodesic first-order low-pass on SO(3):
        R <- R * exp(alpha * log(R^T R_target))
    """

    def __init__(self, cutoff: float = 2.0):
        self.cutoff = cutoff
        self.R_prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.R_prev = None

    def __call__(self, Rm: np.ndarray, dt: float) -> np.ndarray:
        if self.R_prev is None:
            self.R_prev = Rm.copy()
            return Rm.copy()
        tau = 1.0 / (2.0 * np.pi * max(self.cutoff, 1e-6))
        a = 1.0 / (1.0 + tau / max(dt, 1e-9))
        dR = self.R_prev.T @ Rm
        self.R_prev = self.R_prev @ exp_so3(a * log_so3(dR))
        return self.R_prev.copy()


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------


@dataclass
class ThumbCalibration:
    """Everything that is user- or hardware-specific."""

    # --- hardware constant: glove nail frame -> pad contact frame -------------
    # TUNE LATER. Typically ~15-20 mm from nail to pad, along the tip -z.
    T_nail_pad: np.ndarray = field(
        default_factory=lambda: make_T(np.eye(3), np.array([0.0, 0.0, -0.018]))
    )

    # --- from calibration sweep + functional poses ---------------------------
    # dorsum frame -> robot CMC base frame (rotation + human CMC offset)
    T_dorsum_cmc: np.ndarray = field(default_factory=lambda: np.eye(4))
    # L_robot / L_human, from spherical fit + Procrustes on functional poses
    scale: float = 1.0
    # constant rotation between glove pad frame and robot tip frame
    R_pad_tip: np.ndarray = field(default_factory=lambda: np.eye(3))

    def sane(self) -> bool:
        """Validation gate: reject a bad calibration instead of silently
        poisoning the whole session."""
        return 0.7 <= self.scale <= 1.4


# ----------------------------------------------------------------------------
# kinematics
# ----------------------------------------------------------------------------


@dataclass
class ThumbKinematics:
    L1: float
    L2: float
    L3: float
    d: float = 0.0                      # lateral offset of the planar chain
    R_off: np.ndarray = field(default_factory=lambda: np.eye(3))
    # body-fixed axis whose direction we try to match (e.g. pad normal)
    tip_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0])
    )
    q_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad([-40.0, -15.0, 0.0, 0.0])
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad([40.0, 90.0, 100.0, 80.0])
    )
    # +1 or -1. Flexion-only branch selection (see notes in README section).
    elbow_sign: float = 1.0

    # ---- forward kinematics ------------------------------------------------
    def fk(self, q: np.ndarray):
        q0, q1, q2, q3 = q
        b = q1 + q2 + q3
        u = (self.L1 * np.cos(q1)
             + self.L2 * np.cos(q1 + q2)
             + self.L3 * np.cos(b))
        v = (self.L1 * np.sin(q1)
             + self.L2 * np.sin(q1 + q2)
             + self.L3 * np.sin(b))
        Rz = rotz(q0)
        p = Rz @ np.array([u, self.d, v])
        R_tip = Rz @ rot_planar(b) @ self.R_off
        return p, R_tip

    def tip_axis(self, q0: float, beta: float) -> np.ndarray:
        return rotz(q0) @ rot_planar(beta) @ self.R_off @ self.tip_axis_local

    # ---- workspace ---------------------------------------------------------
    @property
    def r_max(self) -> float:
        return self.L1 + self.L2 + self.L3

    @property
    def r_min(self) -> float:
        # crude fallback only; prefer the envelope below
        return max(0.25 * self.r_max, abs(self.d) + 1e-3)

    def envelope(self) -> "ReachEnvelope":
        if getattr(self, "_env", None) is None:
            self._env = build_reach_envelope(self)
        return self._env


@dataclass
class ReachEnvelope:
    """Reachable radius as a function of elevation, in the digit's flexion
    plane, derived from the ACTUAL joint limits.

    A scalar [r_min, r_max] is badly wrong for a 3R chain: the inner boundary
    is set by how far the chain can fold, which depends strongly on elevation
    and is usually far larger than any fraction of the outer reach. Using a
    scalar leaves a whole band of targets that pass the workspace projection
    and then fail IK, and a digit whose IK fails just holds -- which the
    operator experiences as the hand refusing to follow in one direction.
    """

    theta: np.ndarray
    r_lo: np.ndarray
    r_hi: np.ndarray

    def bounds(self, th: float) -> tuple[float, float]:
        t = float(np.clip(th, self.theta[0], self.theta[-1]))
        return (float(np.interp(t, self.theta, self.r_lo)),
                float(np.interp(t, self.theta, self.r_hi)))

    @property
    def theta_range(self) -> tuple[float, float]:
        return float(self.theta[0]), float(self.theta[-1])


def build_reach_envelope(kin: "ThumbKinematics", n_grid: int = 21,
                         n_bins: int = 48) -> ReachEnvelope:
    """One-time sampling of the planar chain over its joint-limit box."""
    g = [np.linspace(kin.q_min[i], kin.q_max[i], n_grid) for i in (1, 2, 3)]
    Q1, Q2, Q3 = np.meshgrid(*g, indexing="ij")
    b = Q1 + Q2 + Q3
    u = (kin.L1 * np.cos(Q1) + kin.L2 * np.cos(Q1 + Q2) + kin.L3 * np.cos(b))
    v = (kin.L1 * np.sin(Q1) + kin.L2 * np.sin(Q1 + Q2) + kin.L3 * np.sin(b))
    th = np.arctan2(v, u).ravel()
    r = np.hypot(u, v).ravel()

    edges = np.linspace(th.min(), th.max(), n_bins + 1)
    idx = np.clip(np.digitize(th, edges) - 1, 0, n_bins - 1)
    centers, lo, hi = [], [], []
    for k in range(n_bins):
        m = idx == k
        if not np.any(m):
            continue
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        lo.append(r[m].min())
        hi.append(r[m].max())
    if len(centers) < 2:
        raise ValueError("degenerate reach envelope; check joint limits")
    return ReachEnvelope(np.array(centers), np.array(lo), np.array(hi))


# ----------------------------------------------------------------------------
# workspace soft saturation (no dead zone)
# ----------------------------------------------------------------------------


def soft_saturate_radius(p: np.ndarray, r_min: float, r_max: float,
                         knee: float = 0.9) -> tuple[np.ndarray, bool]:
    """Compress the radius near the outer boundary with a tanh so the mapping
    stays C1 and the user feels the limit as 'heaviness' rather than a dead
    zone. Also enforces an inner shell."""
    r = float(np.linalg.norm(p))
    if r < 1e-9:
        return np.array([r_min, 0.0, 0.0]), True
    u = p / r
    r0 = knee * r_max
    saturated = False
    if r > r0:
        r_new = r0 + (r_max - r0) * np.tanh((r - r0) / (r_max - r0))
        saturated = True
    elif r < r_min:
        r_new = r_min
        saturated = True
    else:
        r_new = r
    return u * r_new, saturated


def soft_saturate_planar(p: np.ndarray, env: "ReachEnvelope", d: float = 0.0,
                         knee: float = 0.9,
                         margin: float = 0.002) -> tuple[np.ndarray, bool]:
    """Saturate the target into the digit's true reachable region.

    Works in the flexion plane (rho, zeta), compressing BOTH the elevation and
    the radius against the envelope, so every direction degrades smoothly and
    no target is ever handed to the IK that the IK cannot solve.

    `margin` pulls the clamp strictly INSIDE the boundary. Landing exactly on
    the envelope makes the solution set measure-zero, and the 1-degree beta
    grid then steps straight over it -- the IK reports failure on a target that
    is nominally reachable. A couple of millimetres of margin costs nothing and
    removes that whole failure class.
    """
    rho2 = p[0] ** 2 + p[1] ** 2 - d * d
    rho = np.sqrt(max(rho2, 0.0))
    zeta = p[2]
    r = float(np.hypot(rho, zeta))
    if r < 1e-9:
        return p, True
    th = float(np.arctan2(zeta, rho))
    sat = False

    th_lo, th_hi = env.theta_range
    th_m = min(margin / max(r, 1e-6), 0.25 * (th_hi - th_lo))
    if th < th_lo + th_m or th > th_hi - th_m:
        th = float(np.clip(th, th_lo + th_m, th_hi - th_m))
        sat = True

    r_lo, r_hi = env.bounds(th)
    r_lo, r_hi = r_lo + margin, r_hi - margin
    if r_hi <= r_lo:
        r_lo = r_hi = 0.5 * (r_lo + r_hi)
    r0 = r_lo + knee * (r_hi - r_lo)
    if r > r0 and r_hi > r0:
        r = r0 + (r_hi - r0) * np.tanh((r - r0) / (r_hi - r0))
        sat = True
    elif r < r_lo:
        r, sat = r_lo, True

    rho_n, zeta_n = r * np.cos(th), r * np.sin(th)
    # rebuild the 3D point on the same azimuth
    azim = np.arctan2(p[1], p[0]) - np.arctan2(d, max(rho, 1e-9))
    q = rotz(azim) @ np.array([rho_n, d, zeta_n])
    return q, sat


def soft_saturate_yaw(p: np.ndarray, q0_min: float, q0_max: float,
                      d: float = 0.0, knee: float = 0.8) -> tuple[np.ndarray, bool]:
    """Compress the target's AZIMUTH toward the first joint's limits.

    Why this exists: the radial saturation above only handles reach. The yaw /
    AA joint has its own, usually much tighter, limit, and a hard clamp there
    produces exactly the symptom that shows up when the thumb swings toward the
    pinky -- the digit tracks fine and then simply stops moving laterally, a
    dead zone in one direction while the other axes keep responding. That reads
    to the operator as "the model can't follow", when really it is a boundary
    with no gradient.

    Compressing the azimuth instead keeps the mapping monotonic: lateral motion
    keeps producing lateral motion, just progressively less of it, so the user
    feels the limit rather than hitting a wall.
    """
    rho2 = p[0] ** 2 + p[1] ** 2 - d * d
    if rho2 <= 1e-12:
        return p, False
    rho = np.sqrt(rho2)
    phi = np.arctan2(p[1], p[0]) - np.arctan2(d, rho)      # required q0

    mid = 0.5 * (q0_min + q0_max)
    half = 0.5 * (q0_max - q0_min)
    if half <= 1e-9:
        return p, False
    e = phi - mid
    e0 = knee * half
    if abs(e) <= e0:
        return p, False
    e_new = np.sign(e) * (e0 + (half - e0) * np.tanh((abs(e) - e0) / (half - e0)))
    return rotz(e_new - e) @ p, True


# ----------------------------------------------------------------------------
# IK: closed-form yaw + windowed beta scan
# ----------------------------------------------------------------------------


@dataclass
class IKConfig:
    w_orient: float = 1.0            # weight on tip-axis pointing error [rad^2]
    w_smooth: float = 0.3            # weight on ||q - q_prev||^2 [rad^2]
    branch_bonus: float = 0.05       # hysteresis: bonus for staying near q_prev
    beta_step: float = np.deg2rad(1.0)
    beta_window: float = np.deg2rad(25.0)   # local window around beta_prev
    pos_tol: float = 1e-4


@dataclass
class IKResult:
    q: np.ndarray
    beta: float
    ok: bool
    cost: float


class ThumbIK:
    def __init__(self, kin: ThumbKinematics, cfg: IKConfig = IKConfig(),
                 self_collision_fn: Optional[Callable[[np.ndarray], bool]] = None):
        self.kin = kin
        self.cfg = cfg
        # #9 self-collision hook: return True if configuration q is COLLIDING.
        # Approximate each link as a capsule (segment + radius) and the palm as
        # a box; also test against the index finger's current capsules. Because
        # this is evaluated inside the beta scan, colliding candidates are just
        # dropped and the next-best beta is used -- essentially free.
        self.self_collision_fn = self_collision_fn

    # -- planar 2R closed form ------------------------------------------------
    def _solve_2r(self, uw: float, vw: float):
        L1, L2 = self.kin.L1, self.kin.L2
        r2 = uw * uw + vw * vw
        c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        if abs(c2) > 1.0:
            return None
        q2 = self.kin.elbow_sign * np.arccos(np.clip(c2, -1.0, 1.0))
        q1 = np.arctan2(vw, uw) - np.arctan2(L2 * np.sin(q2),
                                             L1 + L2 * np.cos(q2))
        return q1, q2

    def solve(self, p_des: np.ndarray, R_des: np.ndarray,
              q_prev: Optional[np.ndarray]) -> IKResult:
        kin, cfg = self.kin, self.cfg
        px, py, pz = p_des

        # ---- yaw: closed form ---------------------------------------------
        rho2 = px * px + py * py - kin.d * kin.d
        if rho2 < 0.0:
            # inside the lateral-offset cylinder: project onto its surface
            rho, q0 = 0.0, np.arctan2(py, px)
        else:
            rho = np.sqrt(rho2)
            q0 = np.arctan2(py, px) - np.arctan2(kin.d, rho)

        # The first joint has a small range on fingers (AA / spread is typically
        # only +-15..20 deg), so it saturates often. Clamp it and RE-PROJECT the
        # target into the plane that the clamped yaw actually reaches, instead
        # of handing the planar solver a point it can never hit. The residual
        # lateral error then shows up honestly in `pos_error`.
        q0c = float(np.clip(q0, kin.q_min[0], kin.q_max[0]))
        if abs(q0c - q0) > 1e-12:
            p_planar = rotz(-q0c) @ p_des
            rho = float(p_planar[0])
        q0 = q0c

        # desired direction of the one controllable tip axis
        d_star = R_des @ kin.tip_axis_local
        d_star = d_star / max(np.linalg.norm(d_star), 1e-9)

        # ---- beta scan ------------------------------------------------------
        if q_prev is None:
            betas = np.arange(-np.pi, np.pi, cfg.beta_step)   # global scan
            beta_prev = None
        else:
            beta_prev = float(q_prev[1] + q_prev[2] + q_prev[3])
            betas = np.arange(beta_prev - cfg.beta_window,
                              beta_prev + cfg.beta_window + 1e-9,
                              cfg.beta_step)

        best: Optional[IKResult] = None
        for b in betas:
            uw = rho - kin.L3 * np.cos(b)
            vw = pz - kin.L3 * np.sin(b)
            sol = self._solve_2r(uw, vw)
            if sol is None:
                continue
            q1, q2 = sol
            q3 = b - q1 - q2
            q = np.array([q0, q1, q2, q3])

            if np.any(q < kin.q_min - 1e-9) or np.any(q > kin.q_max + 1e-9):
                continue
            if self.self_collision_fn is not None and self.self_collision_fn(q):
                continue

            # orientation cost: angle between achieved and desired tip axis.
            # NOTE: roll about that axis is structurally uncontrollable -> not
            # penalised at all (this is the eps = 0 choice).
            d_ach = kin.tip_axis(q0, b)
            ang = np.arccos(np.clip(float(d_ach @ d_star), -1.0, 1.0))
            cost = cfg.w_orient * ang * ang
            if q_prev is not None:
                dq = q - q_prev
                cost += cfg.w_smooth * float(dq @ dq)
                if beta_prev is not None and abs(b - beta_prev) < cfg.beta_step:
                    cost -= cfg.branch_bonus
            if best is None or cost < best.cost:
                best = IKResult(q, float(b), True, float(cost))

        if best is None:
            # window empty -> retry once globally (re-engage / large jump)
            if q_prev is not None:
                res = self.solve(p_des, R_des, None)
                if res.ok:
                    return res
                # still nothing: HOLD. Returning a fresh configuration here
                # would make the digit snap across the workspace at exactly the
                # moment tracking is hardest.
                return IKResult(q_prev.copy(), float(q_prev[1:].sum()),
                                False, np.inf)
            return IKResult(np.zeros(4), 0.0, False, np.inf)
        return best


# ----------------------------------------------------------------------------
# top-level retargeter
# ----------------------------------------------------------------------------


@dataclass
class RetargetOutput:
    q: np.ndarray                 # commanded joint angles [rad]
    p_cmd: np.ndarray             # filtered/projected pad target, base frame
    p_achieved: np.ndarray        # FK of q AFTER clamp + rate limit
    pos_error: float              # ||p_cmd - p_achieved||  -> feed back to user
    axis_error: float             # tip-axis angle error [rad]
    saturated_workspace: bool
    saturated_joint: bool
    ik_ok: bool


class ThumbRetargeter:
    def __init__(self, kin: ThumbKinematics, calib: ThumbCalibration,
                 ik_cfg: IKConfig = IKConfig(),
                 qd_max: Optional[np.ndarray] = None,
                 self_collision_fn: Optional[Callable[[np.ndarray], bool]] = None):
        self.kin = kin
        self.calib = calib
        self.ik = ThumbIK(kin, ik_cfg, self_collision_fn)
        self.pos_filter = OneEuroFilter3D(min_cutoff=1.0, beta=0.05)
        self.rot_filter = RotationLPF(cutoff=2.0)
        self.qd_max = (qd_max if qd_max is not None
                       else np.deg2rad(np.array([360.0, 360.0, 360.0, 360.0])))
        self.q_prev: Optional[np.ndarray] = None

        # --- #11 engage state -------------------------------------------------
        # Mapping is ABSOLUTE (no mouse-style re-anchoring): on re-engage the
        # robot ramps from its current pose to the human's current pose over
        # `ramp_time`. Disengage freezes the last commanded q.
        self.ramp_time = 1.5
        self._ramp_t = 0.0
        self._engaged = False

    # ---- session control ----------------------------------------------------
    def engage(self) -> None:
        self._engaged = True
        self._ramp_t = 0.0
        self.pos_filter.reset()
        self.rot_filter.reset()

    def disengage(self) -> None:
        self._engaged = False

    def reset(self) -> None:
        self.q_prev = None
        self.pos_filter.reset()
        self.rot_filter.reset()

    # ---- one control-rate step ---------------------------------------------
    def step(self, T_world_dorsum: np.ndarray, T_world_nail: np.ndarray,
             dt: float) -> RetargetOutput:
        """Convenience entry for sources that give two ABSOLUTE poses."""
        return self.step_relative(inv_T(T_world_dorsum) @ T_world_nail, dt)

    def step_relative(self, T_dorsum_nail: np.ndarray,
                      dt: float) -> RetargetOutput:
        """Primary entry point. `T_dorsum_nail` is the nail/sensor pose
        expressed in the dorsum frame -- exactly what a glove's raw per-sensor
        stream gives, with no skeleton solver in between."""
        c = self.calib

        # 1) nail -> pad
        T_dorsum_pad = T_dorsum_nail @ c.T_nail_pad

        # 2) into the robot CMC base frame
        T_cmc_pad = c.T_dorsum_cmc @ T_dorsum_pad
        p_raw = T_cmc_pad[:3, 3] * c.scale          # position scaling
        R_raw = T_cmc_pad[:3, :3] @ c.R_pad_tip     # glove tip -> robot tip

        # 3) task-space filtering (also acts as the glove->control-rate
        #    interpolator: held glove samples are smoothed into a continuous
        #    trajectory at the control rate)
        p_f = self.pos_filter(p_raw, dt)
        R_f = self.rot_filter(R_raw, dt)
        return self.solve_target(p_f, R_f, dt)

    def solve_target(self, p_f: np.ndarray, R_f: np.ndarray,
                     dt: float) -> RetargetOutput:
        """Saturation + IK + clamp + rate limit + FK check, given a target that
        is ALREADY in this digit's base frame and already filtered.

        HandRetargeter uses this entry point because inter-finger blending has
        to happen in a common palm frame, before the per-digit base transform.
        In that path this object's own filters are bypassed.
        """
        # 4) soft saturation: azimuth first (it changes rho), then the
        #    reachable envelope in the flexion plane
        p_cmd, yaw_sat = soft_saturate_yaw(p_f, self.kin.q_min[0],
                                           self.kin.q_max[0], self.kin.d)
        p_cmd, plane_sat = soft_saturate_planar(p_cmd, self.kin.envelope(),
                                                self.kin.d)
        ws_sat = yaw_sat or plane_sat

        # 5) IK
        res = self.ik.solve(p_cmd, R_f, self.q_prev)
        q = res.q.copy()

        # 6) engage ramp
        if self._engaged and self._ramp_t < self.ramp_time and self.q_prev is not None:
            self._ramp_t += dt
            a = np.clip(self._ramp_t / self.ramp_time, 0.0, 1.0)
            q = (1.0 - a) * self.q_prev + a * q

        # 7) joint limit clamp + rate limit
        q_clamped = np.clip(q, self.kin.q_min, self.kin.q_max)
        if self.q_prev is not None:
            dq_max = self.qd_max * dt
            q_clamped = np.clip(q_clamped, self.q_prev - dq_max,
                                self.q_prev + dq_max)
        joint_sat = bool(np.any(np.abs(q_clamped - q) > 1e-9))

        # 8) FK recompute of what is ACTUALLY achieved, so position accuracy
        #    never degrades silently. Surface pos_error to the operator
        #    (visual cue / haptic).
        p_ach, R_ach = self.kin.fk(q_clamped)
        d_ach = R_ach @ self.kin.tip_axis_local
        d_star = R_f @ self.kin.tip_axis_local
        d_star = d_star / max(np.linalg.norm(d_star), 1e-9)
        axis_err = float(np.arccos(np.clip(d_ach @ d_star, -1.0, 1.0)))

        self.q_prev = q_clamped
        return RetargetOutput(
            q=q_clamped,
            p_cmd=p_cmd,
            p_achieved=p_ach,
            pos_error=float(np.linalg.norm(p_cmd - p_ach)),
            axis_error=axis_err,
            saturated_workspace=ws_sat,
            saturated_joint=joint_sat,
            ik_ok=res.ok,
        )


# ----------------------------------------------------------------------------
# #10 CONTACT HANDLING -- simulation only for now. TODO before hardware:
#
#   Once the thumb closes on an object the commanded pose becomes permanently
#   unreachable, so a pure position controller will hold a large tracking error
#   and drive current until it thermals out. Before running on real hardware,
#   add ONE of:
#     (a) current/torque saturation per joint, with a windup guard on the
#         position loop when |pos_error| stays above a threshold for > ~0.2 s;
#     (b) impedance / admittance control -- command a virtual equilibrium pose
#         and let stiffness set the grip force;
#     (c) explicit contact detection (current spike or tactile) that latches
#         the joint into a force-controlled mode.
#   `RetargetOutput.pos_error` is already the natural trigger signal for this.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# #12 METRIC v1: position error of the thumb pad only.
# ----------------------------------------------------------------------------


class PositionErrorMetric:
    def __init__(self):
        self.errs: list[float] = []

    def update(self, out: RetargetOutput) -> None:
        self.errs.append(out.pos_error)

    def summary(self) -> dict:
        e = np.asarray(self.errs)
        if e.size == 0:
            return {}
        return {
            "mean_mm": float(e.mean() * 1e3),
            "p95_mm": float(np.percentile(e, 95) * 1e3),
            "max_mm": float(e.max() * 1e3),
            "sat_ratio": float((e > 1e-3).mean()),
        }


if __name__ == "__main__":
    kin = ThumbKinematics(L1=0.030, L2=0.025, L3=0.020, d=0.0)
    calib = ThumbCalibration(scale=1.0)
    rt = ThumbRetargeter(kin, calib)
    rt.engage()

    dt = 1.0 / 500.0
    metric = PositionErrorMetric()
    T_dorsum = np.eye(4)
    for i in range(500):
        t = i * dt
        p = np.array([0.050 + 0.010 * np.sin(2 * np.pi * 0.5 * t),
                      0.005 * np.sin(2 * np.pi * 0.3 * t),
                      0.010 * np.sin(2 * np.pi * 0.4 * t)])
        T_nail = make_T(exp_so3(np.array([0.0, 0.4 * np.sin(t), 0.0])), p)
        out = rt.step(T_dorsum, T_nail, dt)
        metric.update(out)
    print(metric.summary())
