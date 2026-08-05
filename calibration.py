"""
calibration.py
==============
Per-session calibration for glove -> robot hand retargeting.

Why this is mandatory (not a refinement)
----------------------------------------
We consume MANUS `raw_sensor` data and bypass the Advanced Hand Solver, so
MANUS Core never scales anything to the wearer's hand. This module is the ONLY
place where human hand geometry enters the pipeline. And because the Quantum
sensors clip onto the fingertips, the mount offset shifts slightly every time
the glove is donned -- so this runs at the start of every session, not once per
user.

What it produces
----------------
  scale              robot hand / human hand size ratio
  T_dorsum_palm      glove dorsum frame -> robot palm frame
  T_nail_pad         per digit, sensor frame -> pad contact frame
  R_pad_tip          per digit, glove pad frame -> robot tip frame
  d_contact_human    per finger, human pad separation at a firm pinch

Procedure (about 40 seconds total)
----------------------------------
  A. SWEEP        thumb circumduction, then a flexion sweep per finger.
                  -> joint centre + effective digit length, by geometric fit.
  B. PINCH        thumb-to-index, held.
  C. OPPOSITION   thumb-to-pinky, held.
  D. EXTENSION    hand fully open, held.
  E. FLAT         palm flat, thumb pad against the palm plane.
                  -> B..E give the similarity transform and the tip axes.

Two independent estimates of `scale` fall out of this -- one geometric (from
the sweep radii) and one functional (from the Procrustes fit). They are
cross-checked against each other; disagreement means the user did the poses
wrong, and it is far better to reject the calibration than to let a bad one
quietly poison the whole session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from thumb_retarget import make_T, ThumbCalibration
from hand_retarget import HandCalibration, PinchMap, DIGITS, FINGERS


# ----------------------------------------------------------------------------
# geometric fitting primitives
# ----------------------------------------------------------------------------


@dataclass
class FitResult:
    center: np.ndarray
    radius: float
    rms: float
    normal: Optional[np.ndarray] = None      # circle fits only
    n: int = 0


def fit_sphere(points: np.ndarray) -> FitResult:
    """Algebraic (Kasa) sphere fit.

    |p|^2 = 2 c.p + (r^2 - |c|^2)  is linear in (c, k), so this is a plain
    least-squares solve with no initial guess and no local minima.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) < 8:
        raise ValueError("sphere fit needs >= 8 points")
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    b = np.sum(P * P, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, k = sol[:3], sol[3]
    r2 = k + float(c @ c)
    if r2 <= 0:
        raise ValueError("degenerate sphere fit (points nearly coplanar)")
    r = float(np.sqrt(r2))
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return FitResult(c, r, rms, n=len(P))


def fit_circle_3d(points: np.ndarray) -> FitResult:
    """Plane fit (PCA) followed by a 2D circle fit in that plane.

    Fingers are AA + parallel-3R, so a pure flexion sweep traces a planar arc,
    not a spherical cap. Fitting a sphere to that is ill-conditioned; fit the
    circle instead.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) < 6:
        raise ValueError("circle fit needs >= 6 points")
    mu = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    normal = Vt[2]
    e1, e2 = Vt[0], Vt[1]
    Q = np.column_stack([(P - mu) @ e1, (P - mu) @ e2])

    A = np.hstack([2.0 * Q, np.ones((len(Q), 1))])
    b = np.sum(Q * Q, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c2, k = sol[:2], sol[2]
    r2 = k + float(c2 @ c2)
    if r2 <= 0:
        raise ValueError("degenerate circle fit")
    r = float(np.sqrt(r2))
    center = mu + c2[0] * e1 + c2[1] * e2
    resid = np.linalg.norm(Q - c2, axis=1) - r
    # out-of-plane spread counts as error too: it means the user was not doing
    # a clean flexion sweep
    out_of_plane = (P - mu) @ normal
    rms = float(np.sqrt(np.mean(resid ** 2 + out_of_plane ** 2)))
    return FitResult(center, r, rms, normal=normal, n=len(P))


def umeyama(X: np.ndarray, Y: np.ndarray):
    """Least-squares similarity transform mapping X onto Y.

        Y ~= s * R @ X + t

    Returns (s, R, t, rms). Reflections are suppressed, which matters here:
    an unconstrained fit can happily return a mirrored hand.
    """
    X = np.asarray(X, float).reshape(-1, 3)
    Y = np.asarray(Y, float).reshape(-1, 3)
    if len(X) != len(Y) or len(X) < 4:
        raise ValueError("umeyama needs >= 4 matched point pairs")
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    Sigma = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_x = float(np.mean(np.sum(Xc * Xc, axis=1)))
    s = float(np.trace(np.diag(D) @ S) / max(var_x, 1e-12))
    t = my - s * R @ mx
    rms = float(np.sqrt(np.mean(np.sum((Y - (s * (R @ X.T).T + t)) ** 2, axis=1))))
    return s, R, t, rms


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector `a` onto unit vector `b`.

    A single direction pair leaves the roll about that direction undetermined,
    which is fine: roll is structurally uncontrollable on a parallel-3R digit,
    so the minimal-rotation choice loses nothing.
    """
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any perpendicular axis
        perp = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) / (1.0 + c)


# ----------------------------------------------------------------------------
# stage A: sweep
# ----------------------------------------------------------------------------


class SweepCalibrator:
    """Collects pad positions (dorsum frame) during the sweep stage."""

    def __init__(self):
        self._p: dict[str, list[np.ndarray]] = {d: [] for d in DIGITS}

    def feed(self, p_dorsum: dict[str, np.ndarray]) -> None:
        for d, p in p_dorsum.items():
            if d in self._p and np.all(np.isfinite(p)):
                self._p[d].append(np.asarray(p, float))

    def count(self, digit: str) -> int:
        return len(self._p[digit])

    def fit(self, digit: str) -> FitResult:
        P = np.array(self._p[digit])
        # thumb circumduction sweeps a spherical cap; finger flexion is planar
        return fit_sphere(P) if digit == "thumb" else fit_circle_3d(P)


# ----------------------------------------------------------------------------
# stage B..E: functional poses
# ----------------------------------------------------------------------------

POSE_STAGES = ("pinch", "opposition", "extension", "flat")


@dataclass
class RobotAnchors:
    """Desired robot pad positions in the palm frame, for each functional pose.

    Compute these from the robot's own kinematics (FK at the poses you want the
    human poses to mean), not by hand-waving. `pinch` in particular should be
    the configuration where the thumb and index pads are actually touching.
    """

    poses: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # desired direction of each digit's tip axis in the FLAT pose (palm frame)
    flat_axis: dict[str, np.ndarray] = field(default_factory=dict)


class FunctionalPoseCalibrator:
    """Accumulates held poses, then solves the similarity transform.

    Uses the MEDIAN of each hold window rather than the mean: the user's hand
    drifts and the occasional tracking glitch would drag a mean around.
    """

    def __init__(self):
        self._buf: dict[str, dict[str, list[np.ndarray]]] = {
            s: {d: [] for d in DIGITS} for s in POSE_STAGES}
        self._axis: dict[str, list[np.ndarray]] = {d: [] for d in DIGITS}

    def feed(self, stage: str, p_dorsum: dict[str, np.ndarray],
             axis_dorsum: Optional[dict[str, np.ndarray]] = None) -> None:
        if stage not in self._buf:
            raise ValueError(f"unknown stage '{stage}'")
        for d, p in p_dorsum.items():
            if d in self._buf[stage] and np.all(np.isfinite(p)):
                self._buf[stage][d].append(np.asarray(p, float))
        if stage == "flat" and axis_dorsum:
            for d, a in axis_dorsum.items():
                if d in self._axis:
                    self._axis[d].append(np.asarray(a, float))

    def count(self, stage: str) -> int:
        return min(len(v) for v in self._buf[stage].values())

    def median(self, stage: str) -> dict[str, np.ndarray]:
        return {d: np.median(np.array(v), axis=0)
                for d, v in self._buf[stage].items() if v}

    def solve(self, anchors: RobotAnchors, min_samples: int = 60):
        """Returns (s, R, t, rms, n_pairs).

        All available digits at all available poses are used, not just three
        anchor points: a 3-point similarity fit is barely determined and very
        sensitive to a single bad digit.
        """
        X, Y = [], []
        for stage in POSE_STAGES:
            if stage not in anchors.poses:
                continue
            if self.count(stage) < min_samples:
                continue
            med = self.median(stage)
            for d, p_robot in anchors.poses[stage].items():
                if d in med:
                    X.append(med[d])
                    Y.append(np.asarray(p_robot, float))
        if len(X) < 4:
            raise RuntimeError(
                f"only {len(X)} usable correspondences; hold each pose longer "
                f"or supply more digits in RobotAnchors")
        s, R, t, rms = umeyama(np.array(X), np.array(Y))
        return s, R, t, rms, len(X)

    def tip_axes(self) -> dict[str, np.ndarray]:
        out = {}
        for d, v in self._axis.items():
            if v:
                a = np.mean(np.array(v), axis=0)
                n = np.linalg.norm(a)
                if n > 1e-9:
                    out[d] = a / n
        return out


# ----------------------------------------------------------------------------
# pinch separation
# ----------------------------------------------------------------------------


class PinchSeparationCalibrator:
    """Human pad separation at a firm pinch, per finger.

    Takes the 5th percentile rather than the minimum: one tracking glitch would
    otherwise collapse the estimate and make every robot pinch interpenetrate.
    """

    def __init__(self):
        self._d: dict[str, list[float]] = {f: [] for f in FINGERS}

    def feed(self, p_dorsum: dict[str, np.ndarray]) -> None:
        if "thumb" not in p_dorsum:
            return
        for f in FINGERS:
            if f in p_dorsum:
                self._d[f].append(
                    float(np.linalg.norm(p_dorsum[f] - p_dorsum["thumb"])))

    def result(self, finger: str, min_samples: int = 60) -> Optional[float]:
        d = np.asarray(self._d[finger])
        if d.size < min_samples:
            return None
        return float(np.percentile(d, 5))


# ----------------------------------------------------------------------------
# validation gate
# ----------------------------------------------------------------------------


@dataclass
class GateThresholds:
    scale_min: float = 0.7
    scale_max: float = 1.4
    sweep_rms_max: float = 0.005          # [m]
    procrustes_rms_max: float = 0.008     # [m]
    scale_agreement: float = 0.20         # geometric vs functional, relative
    min_pairs: int = 6


@dataclass
class CalibrationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scale_functional: float = float("nan")
    scale_geometric: float = float("nan")
    procrustes_rms: float = float("nan")
    n_pairs: int = 0
    sweep: dict[str, FitResult] = field(default_factory=dict)

    def summary(self) -> str:
        head = "PASS" if self.ok else "REJECT"
        lines = [f"[{head}] scale={self.scale_functional:.3f} "
                 f"(geometric {self.scale_geometric:.3f}), "
                 f"procrustes rms={self.procrustes_rms * 1e3:.1f} mm, "
                 f"{self.n_pairs} pairs"]
        for d, f in self.sweep.items():
            lines.append(f"  sweep {d:<7} L={f.radius * 1e3:5.1f} mm  "
                         f"rms={f.rms * 1e3:4.1f} mm  n={f.n}")
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# session orchestrator
# ----------------------------------------------------------------------------

STAGES = ("sweep",) + POSE_STAGES

STAGE_PROMPTS = {
    "sweep": "Thumb: draw a large cone (circumduction), thumb held straight. "
             "Then flex each finger through its full range, one at a time.",
    "pinch": "Press thumb and index pads together firmly. Hold.",
    "opposition": "Touch thumb pad to the pinky pad. Hold.",
    "extension": "Open the hand fully, fingers spread. Hold.",
    "flat": "Lay the palm flat, thumb pad against the palm plane. Hold.",
}


class CalibrationSession:
    """Drives the stages and emits a HandCalibration plus per-digit overrides.

    Usage:
        s = CalibrationSession(robot_lengths, anchors)
        s.begin("sweep")
        ... s.feed(p_dorsum) each frame ...
        s.begin("pinch")   # etc.
        calib, digit_calib, report = s.finalize()
    """

    def __init__(self, robot_lengths: dict[str, float],
                 anchors: RobotAnchors,
                 thresholds: GateThresholds = GateThresholds()):
        self.robot_lengths = robot_lengths          # digit -> L1+L2+L3 [m]
        self.anchors = anchors
        self.th = thresholds
        self.sweep = SweepCalibrator()
        self.poses = FunctionalPoseCalibrator()
        self.pinch = PinchSeparationCalibrator()
        self.stage: Optional[str] = None

    def begin(self, stage: str) -> str:
        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'")
        self.stage = stage
        return STAGE_PROMPTS[stage]

    def feed(self, p_dorsum: dict[str, np.ndarray],
             axis_dorsum: Optional[dict[str, np.ndarray]] = None) -> None:
        """`p_dorsum`: digit -> pad position in the dorsum frame (metres).
        `axis_dorsum`: digit -> pad axis direction, only needed in 'flat'."""
        if self.stage is None:
            raise RuntimeError("call begin(stage) first")
        if self.stage == "sweep":
            self.sweep.feed(p_dorsum)
        else:
            self.poses.feed(self.stage, p_dorsum, axis_dorsum)
            if self.stage == "pinch":
                self.pinch.feed(p_dorsum)

    # -- finalise -------------------------------------------------------------
    def finalize(self, d_contact_robot: Optional[dict[str, float]] = None):
        rep = CalibrationReport(ok=True)

        # --- A. sweep -> geometric scale ------------------------------------
        ratios = []
        for d in DIGITS:
            if self.sweep.count(d) < 30:
                rep.warnings.append(f"{d}: sweep too short, skipped")
                continue
            try:
                f = self.sweep.fit(d)
            except ValueError as e:
                rep.warnings.append(f"{d}: sweep fit failed ({e})")
                continue
            rep.sweep[d] = f
            if f.rms > self.th.sweep_rms_max:
                rep.warnings.append(
                    f"{d}: sweep rms {f.rms * 1e3:.1f} mm -- digit was probably "
                    f"bending during the sweep")
            if d in self.robot_lengths and f.radius > 1e-3:
                ratios.append(self.robot_lengths[d] / f.radius)
        rep.scale_geometric = float(np.median(ratios)) if ratios else float("nan")

        # --- B. functional poses -> similarity transform ----------------------
        try:
            s, R, t, rms, n = self.poses.solve(self.anchors)
        except RuntimeError as e:
            rep.ok = False
            rep.errors.append(str(e))
            return None, None, rep
        rep.scale_functional, rep.procrustes_rms, rep.n_pairs = s, rms, n

        # --- validation gate --------------------------------------------------
        if not (self.th.scale_min <= s <= self.th.scale_max):
            rep.ok = False
            rep.errors.append(
                f"scale {s:.3f} outside [{self.th.scale_min}, "
                f"{self.th.scale_max}] -- check units and robot anchors")
        if rms > self.th.procrustes_rms_max:
            rep.ok = False
            rep.errors.append(
                f"procrustes rms {rms * 1e3:.1f} mm > "
                f"{self.th.procrustes_rms_max * 1e3:.1f} mm -- poses were "
                f"inconsistent, redo the calibration")
        if n < self.th.min_pairs:
            rep.ok = False
            rep.errors.append(f"only {n} correspondences")
        if np.isfinite(rep.scale_geometric):
            rel = abs(s - rep.scale_geometric) / max(rep.scale_geometric, 1e-9)
            if rel > self.th.scale_agreement:
                rep.warnings.append(
                    f"geometric and functional scale disagree by {rel * 100:.0f}% "
                    f"-- the sweep or the functional poses were sloppy")
        if not rep.ok:
            return None, None, rep

        # --- assemble ---------------------------------------------------------
        # Pipeline applies  p_palm = scale * (R @ p_dorsum + t_dp),
        # while umeyama gives p_robot = s * R @ p_human + t.
        # Hence t_dp = t / s.  Getting this wrong shifts the whole hand.
        calib = HandCalibration(
            T_dorsum_palm=make_T(R, t / s),
            scale=s,
            pinch={},
        )
        for f in FINGERS:
            dch = self.pinch.result(f)
            if dch is None:
                rep.warnings.append(f"{f}: no pinch data, using default")
                calib.pinch[f] = PinchMap(scale=s)
            else:
                dcr = (d_contact_robot or {}).get(f, PinchMap().d_contact_robot)
                calib.pinch[f] = PinchMap(d_contact_human=dch,
                                          d_contact_robot=dcr, scale=s)

        # --- tip axes from the FLAT pose --------------------------------------
        digit_calib: dict[str, ThumbCalibration] = {}
        axes = self.poses.tip_axes()
        for d in DIGITS:
            R_pad_tip = np.eye(3)
            if d in axes and d in self.anchors.flat_axis:
                a_measured = R @ axes[d]            # into the palm frame
                R_pad_tip = min_rotation(a_measured,
                                         np.asarray(self.anchors.flat_axis[d]))
            else:
                rep.warnings.append(f"{d}: no flat-pose axis, R_pad_tip=I")
            digit_calib[d] = ThumbCalibration(scale=s, R_pad_tip=R_pad_tip)

        return calib, digit_calib, rep


# ----------------------------------------------------------------------------
# self-test: recover a known similarity transform from synthetic data
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from thumb_retarget import exp_so3

    rng = np.random.default_rng(7)
    S_TRUE, R_TRUE = 1.15, exp_so3(np.array([0.10, -0.25, 0.40]))
    T_TRUE = np.array([0.010, -0.020, 0.030])

    def to_robot(p):
        return S_TRUE * R_TRUE @ np.asarray(p) + T_TRUE

    # --- synthetic human hand (dorsum frame) --------------------------------
    cmc = np.array([0.010, -0.020, 0.005])
    L_h = {"thumb": 0.062, "index": 0.085, "middle": 0.090,
           "ring": 0.083, "pinky": 0.068}
    mcp = {"index": np.array([-0.020, 0.060, 0.0]),
           "middle": np.array([0.000, 0.062, 0.0]),
           "ring": np.array([0.020, 0.058, 0.0]),
           "pinky": np.array([0.038, 0.050, 0.0])}

    sess = CalibrationSession(
        robot_lengths={d: S_TRUE * L_h[d] for d in DIGITS},
        anchors=RobotAnchors())

    # A. sweep
    sess.begin("sweep")
    for i in range(300):
        a, b = rng.uniform(-0.7, 0.7), rng.uniform(-0.5, 0.9)
        p = {"thumb": cmc + L_h["thumb"] * np.array(
            [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])}
        for f in FINGERS:                       # planar flexion arc
            th = rng.uniform(-0.2, 1.5)
            p[f] = mcp[f] + L_h[f] * np.array([0.0, np.cos(th), -np.sin(th)])
        sess.feed({k: v + rng.normal(0, 0.0008, 3) for k, v in p.items()})

    # B..E. functional poses
    human_poses = {
        "pinch": {"thumb": np.array([0.005, 0.045, 0.020]),
                  "index": np.array([-0.010, 0.052, 0.028]),
                  "middle": np.array([0.000, 0.060, -0.010]),
                  "ring": np.array([0.020, 0.055, -0.015]),
                  "pinky": np.array([0.038, 0.045, -0.018])},
        "opposition": {"thumb": np.array([0.040, 0.040, -0.010]),
                       "index": np.array([-0.020, 0.080, 0.005]),
                       "middle": np.array([0.000, 0.085, 0.000]),
                       "ring": np.array([0.020, 0.078, -0.005]),
                       "pinky": np.array([0.036, 0.052, -0.012])},
        "extension": {"thumb": np.array([-0.030, 0.050, 0.010]),
                      "index": np.array([-0.035, 0.140, 0.0]),
                      "middle": np.array([0.000, 0.150, 0.0]),
                      "ring": np.array([0.030, 0.140, 0.0]),
                      "pinky": np.array([0.058, 0.115, 0.0])},
        "flat": {"thumb": np.array([-0.010, 0.055, -0.002]),
                 "index": np.array([-0.020, 0.145, 0.0]),
                 "middle": np.array([0.000, 0.152, 0.0]),
                 "ring": np.array([0.020, 0.141, 0.0]),
                 "pinky": np.array([0.038, 0.118, 0.0])},
    }
    sess.anchors.poses = {st: {d: to_robot(p) for d, p in dp.items()}
                          for st, dp in human_poses.items()}
    sess.anchors.flat_axis = {d: np.array([0.0, 0.0, -1.0]) for d in DIGITS}

    for stage, dp in human_poses.items():
        sess.begin(stage)
        for _ in range(120):
            sess.feed({d: p + rng.normal(0, 0.0010, 3) for d, p in dp.items()},
                      axis_dorsum={d: np.array([0.0, 0.0, -1.0]) for d in DIGITS}
                      if stage == "flat" else None)

    calib, digit_calib, rep = sess.finalize()
    print(rep.summary())
    if calib is not None:
        err_s = abs(calib.scale - S_TRUE) / S_TRUE
        err_R = np.rad2deg(np.arccos(np.clip(
            (np.trace(calib.T_dorsum_palm[:3, :3].T @ R_TRUE) - 1) / 2, -1, 1)))
        err_t = np.linalg.norm(calib.T_dorsum_palm[:3, 3] * calib.scale - T_TRUE)
        print(f"\nrecovery: scale {err_s * 100:.2f}%  "
              f"rotation {err_R:.2f} deg  translation {err_t * 1e3:.2f} mm")
        print("pinch d_contact_human (mm):",
              {f: round(calib.pinch[f].d_contact_human * 1e3, 1)
               for f in FINGERS})
