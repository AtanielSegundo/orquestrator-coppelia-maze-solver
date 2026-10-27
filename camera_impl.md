Camera pose fix: implementation plan
0. Verdict on the proposed approach
Most of the design holds: the floor-contact feature, the undistort to an ideal pinhole, the δ frame frozen at capture, the session pattern, reanchor, the pan-servo views and sweep as fallback. Seven points are wrong or risky, roughly in order of impact.

The proposed analytic renderer is wrong whenever the camera is tilted. For a camera pitched down by φ, image columns are not vertical planes.

All column floor traces meet at the nadir point N, which sits h·tanφ behind the camera foot F (about 8 cm at h=0.15 m, φ=30°).
Visibility is decided from the camera at F, not along the trace. Near box corners, the first wall pixel in a column is a box's vertical edge, not a floor contact. Marching the trace misses that, and the error grows toward the image edges (traces sit up to ~4 cm beside F).
Fix: compute the 2-D visibility polygon from F against the wall boxes. This is exact as long as the boxes are taller than the camera. The first trace point outside that polygon is the predicted contact (§4).
Validate it against an independent per-pixel 3-D ray caster. Never test the model against itself.
Rendering in Coppelia for every search candidate adds no accuracy and costs a lot. The scene contains only the same perfect 0.30 m cubes the analytic model uses.

A render cannot say "unknown here" for floor outside the maze, and tcc.maze has no perimeter walls, so border cells look into the room. It also cannot exclude discovered obstacles.
Every render is ~3 ZMQ round trips plus a render on CoppeliaSim's thread, which competes with the 60 FPS loop's setObjectMatrix calls.
It forces a budget-limited coarse search and an approximate yaw.
Recommendation: use the analytic model for the search. Keep Coppelia for camera-convention validation, calibration labels, the debug overlay, and a cheap post-fit check (2 renders per view at the believed and corrected poses). A forward_model switch keeps an in-loop Coppelia mode, with its budget in §5.6.
"Yaw warp, ~2 mm/deg, negligible" does not hold. With the camera ~10 cm ahead of the rotation centre, 5° is ~9 mm, which matters against a 1 cm goal and a 3 cm threshold. Warping a profile is also invalid near occlusion edges. This is moot with the analytic model, which handles yaw exactly. In the Coppelia mode, warp the mask (not the profile) and fold the lever-arm translation (R(ψ)−I)·c into the translation index.

Do not take the camera pose from the sim's /camera relative to kart_base.

kart_base's frame carries KART_CENTER_OFFSET_LOCAL = (0.02, 0.12), a hand-tuned visual fudge.
The sim pan joint follows CameraMirror's rough map of 0.9°/unit. The real CMD_SERVO unit is about 1°: servo.py pulse = 500 + (angle+10)/0.09 µs, so a "percent" is really a servo degree.
Calibrate the camera pose per view directly, relative to the odometry centre on the floor (T_center_cam). Place the probe in world coordinates as T_world_center(x, y, θ)·T_center_cam. No base_matrix_at factoring is needed.
Replace "NaN plus a penalty when only one side is valid" with censored interval residuals.

Each column becomes an interval on both the measured and the model side: HIT, FREE (contact beyond the far limit), NEAR (contact hidden below the kart body) and, model side only, UNKNOWN (ray leaves the maze before any wall).
Columns that cross a discovered obstacle are vetoed: discovered obstacles may remove evidence but never supply it, the same trust boundary as range_fix.
The residual is the gap between the two intervals, which is continuous in pose and handles every case with one rule.
A yaw prior of σ≈1.5° is tighter than reality (1–5° per turn). Against a single fronto-parallel wall, lateral and yaw trade off along b ≈ −D·ψ. An over-tight prior turns real yaw error into a confidently wrong lateral correction.

Use an honest prior: σ ≈ 3°, or grow it with the turns since the last yaw fix.
Gate observability on the data-only information (no kart-yaw prior). Inflate it for column-to-column correlation.
The direct goto to the cell centre has two problems.

For a mostly-lateral 3 cm offset, the server's turn1 faces the point (drive_controller.py ~l.441), which is a ~90° turn, then a ~90° turn2. Those are exactly the turns that create the 1–5° heading error, done in place while off-centre, with a 0.28 m kart (--kart-lenght) in a 0.30 m cell.
The next leg's absolute goto already removes lateral error with a ~6° turn1 (3 cm over 30 cm).
Recommendation: use "back-off then goto". drive_distance(−0.12) keeps heading on the gyro, and then turn1 is about atan(lat/0.12) ≈ 15°.
Only move when the next leg starts with an in-place turn of 45° or more, or at the goal. Otherwise let the next leg absorb it.
This narrows your decision #1, so it is a constant you can set back to "always".
Smaller corrections:

Freshness. Waiting for the MJPEG count to advance is not enough. Add receive timestamps plus a pipeline-latency margin, confirm the servo command via KartLink.servo_telemetry(), and always approach each view from the same direction (servo backlash).
Tilt. At the home tilt (37, about horizontal), floor contacts closer than ~0.33 m are below the image. A dedicated downward fix tilt is required. Hold it for the whole run so tilt repeatability stops mattering.
Longitudinal. Simpler rule: keep the camera's longitudinal value aside and apply it only if Recenter takes no measurement. When a centring move happens, apply all axes before the goto, because the goto centres both. Log camera-vs-ultrasonic longitudinal disagreement as a calibration health metric.
Arrival metric. The belief at arrival is the goto target, so hypot(belief − centre) is about 0 by construction. The metric must use the camera-corrected estimate.
Calibration at one carefully placed pose puts that placement error into every later fix. Use 2–3 placements and a joint fit with placement priors.
Environment.
OpenCV here is 5.0.0 with no cv2.ml, so the colour GMM must be numpy. scipy 1.17 is available for the offline tool.
get_cam.py stretches the 4:3 real frame to 16:9, so its current side-by-side is not pixel-comparable.
Minor stale comments in constants.py. SWEEP_EVERY_N_LEGS is 3 while its comment argues for 1. The RANGE_FIX_MAX_TRUSTED_M comment says 0.45 m with 0.25 m cells, but the value is 0.5 with 0.30 m cells.
1. Architecture and data flow

main thread (60 FPS)                     worker threads
LiveMazeRunner._advance
  "arrived" -> _begin_arrival() --------> MjpegCamera thread (persistent): decode, (frame,count,t_recv)
     CameraFix.poll()  (servo cmds, timers, frame refs)
        per view bundle -----queue-----> CameraFix worker: undistort -> segment -> contact profile
                                           -> per-view cost table (analytic model, numpy)
                                         after last view: combine views (nuisance) -> fine grid
                                           -> Hessian/gates -> [ProbeRenderer check] -> result
                                         ProbeRenderer thread (own RemoteAPIClient):
                                           probe vision sensor + walls-only collection
     _apply_camera_fix: SimMirror.reanchor(rel_at_capture, P_corr)
     [CenterMove: back-off + goto] -> CameraFix(verify) -> reanchor
     Recenter (unchanged) -> stashed camera longitudinal if Recenter measured nothing
     Sweep only if camera lateral missing AND _sweep_worth_trying AND cadence
     ArrivalLog row (CSV) + HUD + optional debug PNGs
If camera_calib.json is missing or invalid, the runner behaves exactly as today: Recenter, then sweep.

2. Files: new and modified
shadow/
sim_mirror.py (modify):

@staticmethod
def relative(odom_pose, odom_ref): return compose(inverse(odom_ref), odom_pose)
def reanchor(self, rel, corrected_world_pose) -> None:
    """initial = corrected (-) rel, so world_pose(rel) == corrected exactly.
    Translation-only == shift_anchor; a yaw change rotates all FUTURE relative
    motion about the current position (the persistent fix for turn error)."""
    self.initial = compose(corrected_world_pose, inverse(rel))
world_pose can call relative(), with no behaviour change.
camera_link/ (camera-generic, no maze knowledge)
mjpeg_camera.py (modify): store self._stamp = time.monotonic() with each decoded frame and add latest_stamped() -> (frame, count, t_recv). latest() is unchanged.
servo_head.py (new): ServoHead(command_url, pan_ch=5, tilt_ch=1).send(pan, tilt), the fire-and-forget POST pulled out of CameraMirror. CameraMirror delegates to it, so the calibration tool and get_cam can drive the head without a sim. Optionally make the home tilt per instance so the 'C' key returns to the fix tilt during runs.
view_sequence.py (new): view_sequence(calib, wanted) -> [(name, pan_pct)]. Views sorted by increasing pan (right → forward → left), plus a preposition move to min_pan − CAMERA_FIX_PREPOSITION_PCT, so every view is approached in the same direction. Shared by the runtime and the calibration tool, so hysteresis matches calibration.
view_calibration.py / camera_calibration.py (new): dataclasses plus CameraCalibration.load(path) / .save(path) and validation: stream size matches, the boxes are taller than the camera, and the version matches.
pinhole_view.py (new): PinholeView(K_ideal, T_center_cam) with project(pts_center), floor_point(u, v), the camera foot and the horizon row. Undistortion maps (cv2.initUndistortRectifyMap) are cached on the calibration.
gaussian_mixture.py (new): numpy EM (k ≤ 3, 3-D) with fit(X, k) and loglik(X).
floor_segmenter.py (new): FloorSegmenter(model) with:
llr(frame_undist) -> float32 HxW, the wall-vs-floor log-likelihood ratio on features (a*, b*, L*/k_L). Down-weighting L* reduces shadow and exposure sensitivity.
adapt(frame, hint_labels), a per-frame EM refit seeded by the calibrated model. Labels come only from pixels ≥20 px away from the predicted boundary, which handles auto-exposure and white-balance drift.
quality(llr), the fraction of uncertain pixels.
contact_profile.py (new):

class ContactProfile(NamedTuple):
    columns: np.ndarray; rows: np.ndarray        # HIT row (sub-pixel) else nan
    kind: np.ndarray                             # 0 HIT, 1 FREE, 2 NEAR, -1 INVALID
    v_near: np.ndarray; v_far: np.ndarray; quality: float
def extract_contact_profile(llr_or_mask, columns, v_near, v_far, min_run_px) -> ContactProfile
def median_profile(profiles) -> ContactProfile   # median rows, majority kind
Each column is scanned upward from v_near (top of the ignore band). The first run of ≥ min_run_px wall pixels gives the contact, refined to the LLR zero crossing. No wall up to v_far gives FREE; a wall already at v_near gives NEAR. The same function is used for real masks and for Coppelia depth masks.
probe_renderer.py (new): ProbeRenderer(calib, maze_tag) with invalidate(tag), render_masks(T_world_cam_list, timeout) -> list|None (blocking; worker threads only) and stop(). Its own thread owns its own RemoteAPIClient, the sensor and the collection.
Coppelia/helpers/ (one function per module)
original_wall_handles.py: original_wall_handles(sim, tag) returns children of /{tag} whose alias matches ^{tag}_\d+_\d+$, which excludes _start, _goal and _obs_*.
create_probe_sensor.py: create_probe_sensor(sim, alias, res, angle, near, far, collection). Options 1|2|4 (explicit, perspective, hidden), visionintparam_entity_to_render = collection, parent −1 (world).
maze_solver/runner/
wall_world.py: WallWorld(walls (N,4), bounds (4,), obstacles (M,4), box_h) and build_wall_world(view, origin, cell, inset). Built from is_original_wall and view.obstacles (via getattr so test stubs work). Rebuilt in _build_maze; obstacles are snapshotted per session.
floor_visibility.py: vectorised 2-D slab ray caster: first_hits(foot (B,2), bearings (B,M), boxes) -> (B,M) and exit_distance(...).
view_geometry.py: ViewGeometry.from_calibration(calib, view, max_range, col_step, row_step). Precomputes, in the centre frame: columns, v_near/v_far, foot F, and the trace sample table (rows, r, φ) per column relative to F.
contact_model.py: predict_intervals(geom, poses (B,3), world) -> (lo, hi, masked) (B × n_cols, in rows). This is the exact analytic renderer (§4).
reference_renderer.py: brute-force per-pixel 3-D raycast (floor, boxes of height H, maze bounds) returning a label image (floor, wall, unknown). Used by tests, the calibration labelling and synthetic frames.
camera_fit.py: view_cost_table(...), combine_views(...), fit_camera_pose(geoms, profiles, world, capture_pose, yaw_sigma, yield_every) -> Optional[CameraFit], and judge_camera_fit(fit) -> (verdict, reason).
camera_fix.py: the CameraFix session and CameraFixResult.
center_move.py: the CenterMove session (back-off, then goto).
arrival_policy.py: pure plan_after_camera(fit, offset_from_centre, pass_name, next_leg_turns, at_goal) -> CameraDecision(apply_lat, apply_yaw, apply_long_now, stash_long, move).
arrival_log.py: ArrivalLog(path) with ArrivalRecord, CSV under maze_solver/logs/ (*.csv is already gitignored; add maze_solver/logs/ for the PNGs) and a run summary.
constants.py: new section (§8). live_maze_runner.py: wiring (§7). build_parser.py / main.py / COMMANDS / HELP: --camera-calib PATH, --no-camera-fix, a camfix dry-run command, and an optional medir <long_cm> <lat_cm> command for ground truth.
recenter.py: no change needed with the stash rule.
Tools
camera_calibrate.py (repo root, new): subcommands intrinsics | servo | tilt | extrinsics | check (§9).
get_cam.py (modify): --overlay --maze tcc.maze --pose r c heading --calib .... The live pose is the declared pose composed with odometry since the first sample. It draws the real mask and profile, the analytic profile at that pose, and optionally the probe profile. Panels use the calibrated 4:3 ideal camera.
maze_solver/tools/camera_sim_check.py (new): Coppelia sim-vs-sim checks: conventions, degradations, timing.
Optional: maze_solver/tools/fake_camera_kart.py, a fake kart server like test_get_cam.py plus POST /command and synthetic frames from its true pose with injected turn error, for an end-to-end runner test without hardware.
3. Measurement (real side)
Freshness. After the servo ack plus SERVO_SETTLE_S, accept frames with t_recv ≥ t_settled + FRAME_LATENCY_S and a new count. Take 3, with a timeout per view.
Undistort. cv2.remap to K_ideal: centred principal point, square pixels, alpha≈0 so there are no black borders that could be classified as walls.
Segment. LLR from the calibrated GMM (2–3 components per class), optional per-frame adapt, light Gaussian blur, threshold at 0, open 3×3, close 5×5.
Profile. extract_contact_profile per frame, then median_profile.
Per-view quality gate. Segmenter quality, ≥ MIN_HITS_PER_VIEW usable columns, and no columns lost to undistortion borders.
Debug. The worker writes a PNG: undistorted frame, wall tint, measured contacts (HIT green / FREE yellow / NEAR magenta), model at the belief (red) and at the corrected pose (cyan), ignore band, and text.
4. Forward model (exact analytic) and the Coppelia probe's role
Per view, precomputed once (centre frame: origin on the floor under the odometry point, x forward, y left, z up):

Camera centre C and foot F.
For each column j and each sample row v_jk from v_near up to v_far (2 px step): back-project to the floor Q_jk, then store r_jk = |Q−F| and φ_jk = bearing from F.
Per candidate pose P = (x, y, θ):

F_w = P ⊕ F.
On a uniform bearing grid (0.05°) covering θ + [φ_min, φ_max], compute:
R(β): first entry into a wall box.
E(β): exit from the maze rectangle.
O(β): first entry into a discovered obstacle.
Sample at θ+φ_jk (r and φ are invariant under translation, so only R changes).
A sample is visible iff r_jk < R. The first invisible k gives the contact; interpolate r−R between k−1 and k to get a sub-row v_model.
Model interval in rows (smaller v = farther):
HIT: [v_m, v_m], clamped to (−∞, v_far] if beyond and [v_near, ∞) if nearer.
The trace crosses E before the contact → UNKNOWN beyond: (−∞, v_exit].
O < R before the contact → column masked.
Measured interval: HIT [v, v]; FREE (−∞, v_far]; NEAR [v_near, ∞).
Residual = gap between the two intervals (0 if they overlap) / σ_px.
Vectorise over the translation grid for each yaw slice (B=121): about 121×1400 bearings × ~20 boxes for R, then a gather over ~10k samples. That is about 5–10 ms per slice per view.

Assumptions to assert at load: boxes are taller than the camera (box_h > C_z), and walls fill their cells up to a calibrated wall_inset_m (default 0; this also models boxes smaller than 30 cm).

Coppelia probe roles:

(a) Convention test: axis map between the CV frame and the sensor frame, the vertical flip, which dimension the perspective angle applies to (expected: the larger), and the principal point.
(b) Optional colour labels during calibration.
(c) Overlay.
(d) Post-fit check: render at the believed and corrected pose for each view, extract the profile from the depth mask (depth < far−ε; getVisionSensorDepth(h, 1) returns metres on CoppeliaSim 4.6+), and compute the same interval cost. Require cost(corrected) ≤ cost(believed). Separately, require analytic-vs-sim profile agreement within ≤ 2 px median at the corrected pose; otherwise log a warning and refuse, per a constant. Columns the analytic model masks or marks UNKNOWN are excluded.
Perspective angle = 2·atan((max(W,H)/2)/f_ideal).

5. Estimator (camera_fit.py)
5.1 Unknowns. δ = (a long, b lat, ψ kart yaw) in the frame of the frozen capture heading θ0. P(δ) = (x0 + c·a − s·b, y0 + s·a + c·b, θ0 + ψ). Plus one pan nuisance η_v per view.

5.2 Loss. ρ = Huber(k=2) on e = gap/σ_px with σ_px=1.5, capped at ρ(6) (truncated, so people, bags and stray boxes become bounded outliers).

C_v(a, b, ψ_eff) = Σ_cols ρ(e) per view.

5.3 Combining views with a pan nuisance. No extra raycasts: each view's table is computed on ψ_eff = ψ + η over the ψ range ±2°.

J_data(a,b,ψ) = Σ_v min_{η∈[−2°,2°]} [C_v(a,b,ψ+η) + ½(η/σ_η)²]

This is a 1-D min-convolution along ψ. σ_η is measured on the bench; with σ_η=0 it reduces to calibrated biases only. MAP cost: J = J_data + ½(ψ/σ_ψ)² + ½(|a,b|/0.08)². The weak translation prior only damps flat valleys.

5.4 Search.

Coarse: a, b ∈ ±0.10 m in 2 cm steps (11×11); ψ ∈ ±6° in 1° steps; ψ_eff grid ±8°.
Find the best two distinct basins (≥3 cm or ≥2° apart) for the ambiguity ratio.
Fine: ±2 cm in 0.5 cm steps (9×9); ψ ±1.5° in 0.25° steps (η also at 0.25°). Parabolic sub-grid refinement per axis.
Per-view coarse tables are computed as each view arrives, overlapping the servo motion, so after the last view only its table, the combine and the fine pass remain (~0.3–0.5 s).
Sleep SCAN_FIT_YIELD_S between yaw slices, as scan_fit does.
5.5 Uncertainty and observability.

Fit a quadratic to the fine J_data points within J−J_min ≤ 4.5 to get H. Σ = H⁻¹ · s², where s² = max(1, 2J_min/(n_eff−3)), n_eff = n_cols / CAMERA_FIX_COLUMN_CORRELATION.
Per-axis marginal std from diag(Σ). This catches diagonal valleys such as the lateral/yaw trade-off, which the per-axis rise test in scan_fit misses.
An axis whose std exceeds its limit becomes None. Then project δ onto the surviving axes and re-check from the projected point, mirroring scan_fit.
If H is not positive definite, the eigen-directions with tiny curvature are unobservable.
5.6 Gates (judge_camera_fit, Portuguese reasons). Checked in order:

At least 1 view, ≥ MIN_HITS_PER_VIEW per used view, total ≥ MIN_HITS_TOTAL.
Inlier fraction (|e|<6) ≥ 0.6 and inlier RMS ≤ 2.0 px ("modelo nao explica a imagem").
Optimum not within one coarse step of the window edge.
Ambiguity: J_best ≤ 0.7·J_second.
|lat| ≤ SCAN_FIT_MAX_LATERAL_M, |long| ≤ 0.10, |ψ| ≤ 5°.
Confirmed: observable and |δ| < 0.8 cm / 0.4°. This counts as success with no correction, and the sweep is not needed.
Otherwise improvement J(δ*) ≤ 0.8·J(0).
Optional Coppelia check.
Outcomes: corrigido / confirmado / recusado.

In-loop Coppelia budget (if CAMERA_FIX_FORWARD_MODEL="coppelia"). 200×150 probe, 5×5 coarse at 2.5 cm plus 3×3 refine at 1 cm = 34 renders per view, about 100 per fix. At ~8–15 ms each that is 1–1.5 s. Yaw is handled by mask homography plus the lever-arm re-index. Expect hitches in the main loop's sim calls; space the renders by 5 ms.

6. Sessions
CameraFix (runner/camera_fix.py)

class CameraFix:
    def __init__(self, calib, geoms, camera: CameraMirror, stream: MjpegCamera, link,
                 world: WallWorld, segmenter, wx, wy, wth, odom_pose, odom_ref,
                 purpose="chegada", views=None, probe=None, clock=time.monotonic): ...
    def poll(self, drive_tel: dict, odom_pose) -> str   # "pending" | "done" | "refused"
    status: str; result: Optional[CameraFixResult]; def summary(self) -> str
class CameraFixResult(NamedTuple):
    outcome: str; reason: str; fit: Optional[CameraFit]
    capture_pose: tuple; capture_rel: tuple; views_used: tuple
    sim_check: Optional[tuple]; duration_s: float; debug_paths: tuple
States (status strings in Portuguese):

assentando_kart: wait KART_SETTLE_S with goal_active False; freeze the capture pose and capture_rel.
preposicionando: camera.set_percents(min_pan − PREPOSITION, tilt_fix); wait for the ack (link.servo_telemetry()['5']) or timeout, then 0.25 s.
For each view: mirando(v): set_percents(pan_v, tilt_fix) → aguardando_servo(v): ack, then SERVO_SETTLE_S → capturando(v): fresh frames; push the bundle to the worker queue; on timeout skip the view.
processando: send set_percents(PAN_HOME, tilt_fix) immediately; wait for the worker's Event.
done / refused.
Abort rules:

Odometry moved > 5 mm or 0.5° since capture → "kart se mexeu".
All views lost → "sem frames".
Worker exception → stored like Sweep._fit_error; still restores the pan.
Views: chosen by a cheap pre-check, the analytic predicted HIT count at the belief; views with nothing in view are skipped. The runner ignores nudge / look_forward while a fix is active.

CenterMove (runner/center_move.py)
CenterMove(link, mirror, odom_ref, view, centre_xy, axis_theta, mode, backoff_m, clock):

recuando: only if the cell behind (centre − axis·cell) is in bounds and not view.is_wall; otherwise fall back to a direct goto. Uses drive_distance(−backoff, RECENTER_SPEED).
indo: the goto target is recomputed now via mirror.odom_target(...) from the live anchor, never cached (the same rule as Leg).
assentando, then done, or blocked if hypot(target − belief) > 3 cm.
Uses the same goal_active edge and grace logic as Leg and Recenter.
7. Runner integration (live_maze_runner.py)
__init__:

Load the calibration; if loading fails, camera_ready=False, print one warning, and show a HUD line ("camera: sem calibracao").
Create the stream: reuse CameraMirror's MjpegCamera when the backend is client, otherwise create a dedicated persistent one.
Create the FloorSegmenter, the ViewGeometrys, the optional ProbeRenderer and the ArrivalLog.
New state: camera_fix, center_move, _arrival, _camera_lateral_ok, _camera_pending_long, _camera_pass, _needs_start_fix, last_camera_summary.
_build_maze: self.wall_world = build_wall_world(...); probe.invalidate(tag).

confirm_ok:

_needs_start_fix = CAMERA_FIX_AT_START. This corrects operator placement and doubles as a calibration health check.
If CAMERA_FIX_HOLD_TILT, set the tilt to calib.tilt_pct.
_advance(drive_tel, odom, wx, wy, wth) (step passes pose):

If camera_fix → _poll_camera_fix; if center_move → _poll_center_move; then recenter and sweep as today.
When the leg is None:
If _needs_start_fix, call _begin_arrival("inicio").
If _arrival is set, _finalize_arrival(). This is the single completion point.
Then replan, the step gate and the leg, unchanged.
On "arrived": _begin_arrival("chegada") instead of _start_recenter.
_begin_arrival: reset the per-arrival flags. If the camera is ready and the pre-check passes, start CameraFix(pass="medicao"); otherwise go to _start_recenter.

_poll_camera_fix → _apply_camera_fix(res, decision):

decision = plan_after_camera(...), where:
offset_from_centre is the corrected pose's lateral offset from the cell centre, not δ.
next_leg_turns is the next leg's heading change ≥ 45°.
at_goal is plan_index+1 >= len(plan_path).
P_corr = capture ⊕ (a if apply_long_now else 0, b if apply_lat else 0, CAMERA_FIX_YAW_GAIN·ψ if apply_yaw else 0), then mirror.reanchor(res.capture_rel, P_corr).
Then apply_world_pose, last_world_pose and _update_kart_view, the same as _apply_range_fix.
_camera_lateral_ok = fit.lateral is not None (applied or confirmed).
stash_long sets _camera_pending_long = (a, θ0).
If decision.move (pass "medicao" only), start CenterMove; otherwise _start_recenter.
A verify pass never moves.
_poll_center_move: done → CameraFix(pass="verificacao") if CAMERA_FIX_REVERIFY, else Recenter. Blocked → log it, then Recenter.

_poll_recenter (modify):

If self.recenter.first_error_m is None (refused, or never had a valid reading) and _camera_pending_long is set → shift the anchor along θ0 by a. Log "camera long aplicado (recentragem sem leitura)".
Then _start_sweep only if not _camera_lateral_ok.
Rename _legs_since_sweep to _legs_since_lateral_fix and reset it on a successful camera lateral. The existing SWEEP_EVERY_N_LEGS and _sweep_worth_trying gates are kept for the fallback.
HUD fix line precedence: camera: {status} → centralizando: ... → recentrando → varredura → last camera summary (e.g. camera: lat +2.1 cm, yaw -1.4 deg, long n/d (3 vistas, 214 col, rms 1.1 px)) → last sweep → range-fix.

Arrival metric (ArrivalRecord):

arrival_err_long/lat/xy/yaw = (capture ⊕ full δ) − cell centre, including the longitudinal value even when it was not applied. If longitudinal is unobservable, Recenter's first_error_m fills it, with a source column.
Also: δ and std, outcome/reason, views, hits, rms, inlier fraction, sim check, move mode/blocked, verify δ and verify_err_xy, Recenter first/last/moves, cam_long_minus_us_cm, sweep used/lateral, applied dx/dy/dyaw, duration, and optional truth_long/lat from medir.
At goal or stop, print a summary: coverage, mean/p90/max of arrival_err_xy, max |lat|, refusal and fallback counts, mean pipeline time.
Phase 0 logs this from the existing stack (Recenter and sweep estimates) to get a baseline.
8. Constants to add (runner/constants.py, section "Camera pose fix")
Starting values; the bench ones get re-measured with a rationale comment.


CAMERA_FIX_ENABLED = True; CAMERA_FIX_AT_START = True; CAMERA_FIX_HOLD_TILT = True
CAMERA_FIX_FORWARD_MODEL = "analytic"          # "analytic" | "coppelia"
CAMERA_FIX_KART_SETTLE_S = 0.30                # same reasoning as RECENTER_SETTLE_S
CAMERA_FIX_SERVO_SETTLE_S = 0.50               # bench B2 (ringing of the head)
CAMERA_FIX_FRAME_LATENCY_S = 0.20              # bench B2 (capture->encode->WiFi->decode)
CAMERA_FIX_SERVO_ACK_TIMEOUT_S = 1.0; CAMERA_FIX_VIEW_TIMEOUT_S = 2.5
CAMERA_FIX_FRAMES_PER_VIEW = 3
CAMERA_FIX_SIDE_PAN = 45                        # CMD_SERVO units ~= deg; HFOV~54 -> +-72 deg coverage
CAMERA_FIX_PREPOSITION_PCT = 6
CAMERA_FIX_MAX_FLOOR_RANGE_M = 0.90             # dv/dr ~ f*h/r^2: ~0.6 px/cm at 1 m -> not worth it
CAMERA_FIX_COLUMN_STEP_PX = 4; CAMERA_FIX_ROW_STEP_PX = 2; CAMERA_FIX_MIN_RUN_PX = 3
CAMERA_FIX_PIXEL_SIGMA_PX = 1.5; CAMERA_FIX_HUBER_K = 2.0; CAMERA_FIX_TRUNCATE_K = 6.0
CAMERA_FIX_COLUMN_CORRELATION = 3.0            # variance inflation, tune at B5
CAMERA_FIX_MIN_HITS_PER_VIEW = 12; CAMERA_FIX_MIN_HITS_TOTAL = 30
CAMERA_FIX_MIN_INLIER_FRACTION = 0.6; CAMERA_FIX_MAX_RMS_PX = 2.0
CAMERA_FIX_SEARCH_HALF_M = 0.10                 # < half cell, like SCAN_FIT_MAX_SHIFT_M
CAMERA_FIX_SEARCH_HALF_YAW_DEG = 6.0
CAMERA_FIX_COARSE_STEP_M = 0.02; CAMERA_FIX_COARSE_STEP_YAW_DEG = 1.0
CAMERA_FIX_FINE_STEP_M = 0.005; CAMERA_FIX_FINE_STEP_YAW_DEG = 0.25
CAMERA_FIX_YAW_PRIOR_DEG = 3.0                  # honest vs 1-5 deg/turn (see SWEEP_EVERY_N_LEGS)
CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG = 1.0         # bench B2; 0 disables
CAMERA_FIX_PAN_NUISANCE_HALF_DEG = 2.0
CAMERA_FIX_MAX_STD_LAT_M = 0.008; CAMERA_FIX_MAX_STD_LONG_M = 0.010; CAMERA_FIX_MAX_STD_YAW_DEG = 0.6
CAMERA_FIX_MAX_LATERAL_M = SCAN_FIT_MAX_LATERAL_M; CAMERA_FIX_MAX_LONG_M = 0.10
CAMERA_FIX_MAX_YAW_DEG = 5.0; CAMERA_FIX_AMBIGUITY_RATIO = 0.7; CAMERA_FIX_MIN_IMPROVEMENT = 0.8
CAMERA_FIX_CONFIRM_M = 0.008; CAMERA_FIX_CONFIRM_YAW_DEG = 0.4
CAMERA_FIX_YAW_GAIN = 1.0                       # prior already shrinks; don't double-shrink
CAMERA_FIX_LATERAL_MOVE_M = 0.03
CAMERA_FIX_MOVE_POLICY = "before_turns"         # "always" (decision #1) | "before_turns" | "never"
CAMERA_FIX_MOVE_MODE = "backoff"                # "backoff" | "goto"
CAMERA_FIX_BACKOFF_M = 0.12; CAMERA_FIX_REVERIFY = True
CAMERA_FIX_KART_MOVED_M = 0.005; CAMERA_FIX_KART_MOVED_DEG = 0.5
CAMERA_FIX_SIM_VERIFY = True; CAMERA_FIX_SIM_TIMEOUT_S = 2.0; CAMERA_FIX_SIM_MAX_DISAGREE_PX = 2.0
CAMERA_FIX_DEBUG_DUMP = True
9. Calibration
camera_calib.json (default camera_link/camera_calib.json)

{ "version": 1, "created": "...",
  "stream": {"width": 400, "height": 300},
  "intrinsics": {"K": [[...]], "dist": [k1,k2,p1,p2,k3], "rms_px": 0.3, "n_images": 30},
  "ideal": {"width": 400, "height": 300, "f": 398.0, "cx": 199.5, "cy": 149.5},
  "probe": {"perspective_angle_rad": 0.93, "axis_map": [[-1,0,0],[0,-1,0],[0,0,1]], "flip_v": true},
  "tilt_pct": 62, "pan_home_pct": 50,
  "views": [ {"name": "direita", "pan_pct": 5,
              "T_center_cam": [[...4x4, CV axes: x right, y down, z fwd -> centre frame...]],
              "params": {"pan_deg": -44.1, "tilt_deg": -31.2, "roll_deg": 0.4, "xyz_m": [0.11,-0.01,0.142]},
              "v_near": [ ...W ints... ], "pan_sigma_deg": 0.6, "tilt_sigma_deg": 0.2, "fit_rms_px": 1.1 },
             {"name": "frente", ...}, {"name": "esquerda", ...} ],
  "color_model": {"features": "a,b,L/3", "wall": {"w": [], "mu": [], "cov": []},
                  "floor": {...}, "prior_wall": 0.4, "separability": {"bhattacharyya": 2.1, "err_rate": 0.02}},
  "world": {"box_height_m": 0.30, "wall_inset_m": 0.0, "cell_length_m": 0.30},
  "placements": [{"maze": "tcc.maze", "cell": [2,0], "heading_deg": 0, "fitted_delta": [0.002,-0.001,0.3]}]
}
axis_map and perspective_angle dimension come from the convention test, not from assumption.

camera_calibrate.py
intrinsics: frames from /video at the stream resolution (Picamera2 mode and crop decide K), findChessboardCornersSB, about 30 captures with coverage feedback, calibrateCamera (k3 fixed for a mild lens). Accept RMS < 0.5 px. Compute K_ideal.
servo: kart static in front of a textured scene; run view_sequence 20 times.
Per view, measure the shift against the first capture (phaseCorrelate on undistorted grey) to get pan/tilt σ.
Measure frame-to-frame motion vs time after a command (settle time) and the time from command to the first changed frame (latency).
Print the recommended constants.
tilt: live view with floor iso-range lines (0.15/0.3/0.6/0.9 m) from the provisional extrinsics (ruler-measured height and offset). Arrow keys pick the tilt so that 0.13–0.9 m is visible above the kart body. ENTER saves.
extrinsics: 2–3 placements on taped cell-centre crosses (different cells and headings), each captured with the runtime sequence.
Initial labels: k-means(2) in Lab, with "floor" being the cluster that dominates just above the body.
scipy.optimize.least_squares(loss="soft_l1") over per-view (pan, tilt, roll), a shared z, ruler x,y with a prior (σ 5 mm), and per-placement δ with priors (σ 5 mm / 1°), whose mean fixes the gauge. The cost is the §4 interval cost.
Relabel with reference_renderer (or the probe), erode 6 px, train the GMM, refit. Two iterations.
Ignore band: pixels with low temporal variance across placements and connected to the bottom, minus 3 px, with keys for manual adjustment.
Report per-view RMS, inlier fraction, separability (go/no-go: if the floor is wood-brown like the boxes, stop and add a contrasting floor mat, or switch to an edge-based feature) and parameter stds.
check: run the full runtime estimator at a placement (δ should be about 0) and show the overlay.
10. Threading and ZMQ
Main thread: only polls (timers, servo POSTs already threaded, frame references, servo_telemetry). No cv2 or numpy-heavy work, no ZMQ beyond today's calls.
CameraFix worker (daemon, one per session, fed by queue.Queue): cv2 remap and segmentation (GIL released), numpy cost tables (large ufuncs release the GIL), SCAN_FIT_YIELD_S sleeps between yaw slices. Blocks on ProbeRenderer with a timeout; on timeout it skips the check and logs it.
ProbeRenderer thread: owns its RemoteAPIClient (never shares the runner's sim). Builds the collection from original_wall_handles with addItemToCollection(handle_single) per wall; that covers live discoveries, since _obs_* are never added. Explicit-handling sensor at world level; setObjectMatrix + handleVisionSensor + getVisionSensorDepth per render. On a stale-handle exception it rebuilds; invalidate() comes from _build_maze; stop() removes the sensor and collection (shutdown only).
MjpegCamera: persistent. If WiFi load from two MJPEG streams shows up as telemetry latency (measure at B2), add an on-demand connect at session start without joining on stop.
11. Failure modes and fallbacks
Failure	Detection	Behaviour
No calibration or size mismatch	load/validate	camera disabled; today's pipeline
Stream down / no fresh frames	view timeout	view skipped; all lost → refused → Recenter → sweep fallback
Servo POST lost	no servo_telemetry ack	proceed flagged; residual/nuisance gate usually rejects the view
Kart bumped during capture	odometry delta	abort, refused
Segmentation poor (lighting, AE)	quality/separability, inlier fraction	view dropped / refused
Unmapped object in view	truncated loss, inlier gate	robust, or refused
Discovered obstacle in view	O(β) mask	columns vetoed (never evidence)
Room beyond the maze edge (tcc.maze has no perimeter)	E(β)	one-sided UNKNOWN interval
Pose lost > window / aliasing	edge, ambiguity gates	refused → sweep fallback
Lateral/yaw trade-off	data-only marginal std	lateral None → sweep fallback
Calibration drifted (camera knocked)	start-of-run fix:	δ
CenterMove blocked	Leg-style arrival check	skip verify → Recenter
Coppelia unavailable or slow	probe timeout	check skipped (logged); estimator unaffected
Worker exception	try/finally	refused with reason; pan restored
Operator moves the camera mid-fix	—	nudge/look_forward ignored while active
12. Verification
Unit tests (cd maze_solver && python -m unittest discover -s tests, no Coppelia):

test_reanchor.py:
Round trip: world_pose(rel) == corrected.
Translation-only reanchor equals shift_anchor.
odom_target stays the exact inverse after reanchor.
A yaw reanchor rotates future motion about the current point.
test_floor_visibility.py: slab caster vs a brute-force fine march; exits; inset; foot inside a box.
test_contact_model.py: analytic intervals vs reference_renderer + extract_contact_profile on Corridor, DeadEnd, pillar and corner cases at tilt 30° and pans 0/±45. Edge columns and corner occlusion must agree within 0.5 px. This specifically pins the non-vertical-column-plane case.
test_contact_profile.py: HIT/FREE/NEAR, min run, ignore band, sub-pixel crossing, median of frames.
test_floor_segmenter.py: GMM separation on synthetic Lab patches; adapt under a gain/white-balance shift.
test_camera_fit.py (stubs, as in test_sweep):
Corridor → lateral and yaw recovered, longitudinal None.
DeadEnd → longitudinal observable.
OpenField → refused.
Front wall only → lateral None despite the yaw prior.
Truth outside the window → refused.
20% outlier columns, ±1.5° pan nuisance on one view, a discovered box, out-of-maze clutter → no bias.
Synthetic data comes from the reference renderer with perturbed truth (pan ±1°, tilt ±0.3°, inset ±5 mm), never from contact_model.
test_camera_fit_tcc_maze.py:
For each cell on the solved path, truth = centre + U(±6 cm) + yaw U(±4°), with 1 px noise and dropouts.
Assert, from measured numbers in the codebase's style: lateral median < 0.5 cm, p90 < 1.0 cm, 0 gross (> 3 cm) among accepted, yaw median < 0.4°.
Record the yield (cells with lateral observable) against the sweep's 7/17.
test_camera_fix.py:
Fake clock, CameraMirror, stream and link.
Order and monotonic approach with preposition; stale frames rejected; kart-moved abort; view timeout; pan restored with tilt held.
The worker result passes through, and poll() stays under ~1 ms.
Frames are synthetic: reference labels colourised, noise, JPEG q90.
test_center_move.py: back-off then goto; target recomputed after the back-off from the live anchor; blocked cell behind falls back to a direct goto; blocked outcome.
test_arrival_policy.py: decision matrix (pass × move policy × next turn × goal × observability). test_arrival_log.py: header, NaN handling.
Coppelia sim-vs-sim (tools/camera_sim_check.py):

(a) --conventions: single cube at a known offset; render vs analytic projection gives axis_map, flip_v, perspective-angle dimension and principal point. Assert < 1 px.
(b) --degrade: tcc.maze, random truth vs belief.
"Real" image = probe RGB with a walls+floor collection, CAIXA_COR boxes, clutter cuboids outside the maze, per-view pan/tilt jitter.
Then re-distort with the calibrated dist, noise, gain/white-balance shift, JPEG q90, random occluder blobs.
Full pipeline, then error statistics per degradation level for both forward models.
(c) Timing: renders/s at 400×300 and 200×150, and main-loop frame-time hitch while rendering.
Bench:

B1 intrinsics → B2 servo (σ_η, settle, latency) → B3 tilt → B4 extrinsics/colour/ignore band at 3 placements.
B5 static accuracy: taped offsets (lat ±2/±4 cm, long ±3 cm, yaw ±3°) at 5+ cells, using the camfix dry-run. Compare predicted std with actual error to tune COLUMN_CORRELATION and the gates. Target: lateral p90 < 1.5 cm, yaw < 0.7°.
B6 closed loop: tcc.maze in --wait_to_step, ruler ground truth via medir, compared with the Phase 0 baseline. Targets: max arrival error < 4 cm, no wall contact over N runs, per-cell time below the sweep's.
B7 robustness: lights on/off, a person or chair near the maze, a discovered box, border cells, and a glossy-floor reflection check (a reflection would bias contacts nearer).
13. Phases and exit criteria
Phase 0 (groundwork, no camera): reanchor + tests, ArrivalLog with baseline metrics from Recenter and sweep, latest_stamped, ServoHead, constants scaffold. Exit: baseline CSV from one real run.
Phase 1 (calibration + overlay):
camera_link: calibration types, PinholeView, GMM, segmenter, contact_profile, ProbeRenderer.
Coppelia/helpers: original_wall_handles, create_probe_sensor.
runner: wall_world, floor_visibility, view_geometry, contact_model, reference_renderer.
Tools: camera_calibrate.py, get_cam.py --overlay, camera_sim_check --conventions.
Tests for the model and profile.
Exit: intrinsics < 0.5 px; σ_η, settle and latency measured; separability go; overlay RMS < 1.5 px at known poses; analytic vs Coppelia < 0.5 px median.
Phase 2 (estimator): camera_fit + judge, the stub and tcc tests, the --degrade script, camera_calibrate.py check, B5. Exit: sim targets met; worker tail < 0.5 s; B5 targets.
Phase 3 (runner): CameraFix, CenterMove, arrival_policy, wiring (_begin_arrival, _poll_camera_fix, _poll_center_move, _poll_recenter stash and fallback, _finalize_arrival), HUD, nudge lock, start-of-run fix, CLI --camera-calib / --no-camera-fix / camfix / medir, debug dumps. B6–B7. Exit: B6 targets.
Phase 4 (optional): camera fix on legs blocked without the front guard firing (belief only); obstacle discovery from the camera; Coppelia in-loop mode; fake camera kart server for end-to-end runs.
Critical Files for Implementation
C:\Users\Korvo\scripts\embeded\rasp\orquestrator\maze_solver\runner\live_maze_runner.py
C:\Users\Korvo\scripts\embeded\rasp\orquestrator\shadow\sim_mirror.py
C:\Users\Korvo\scripts\embeded\rasp\orquestrator\maze_solver\runner\constants.py
C:\Users\Korvo\scripts\embeded\rasp\orquestrator\camera_link\camera_mirror.py (with mjpeg_camera.py)
C:\Users\Korvo\scripts\embeded\rasp\orquestrator\get_cam.py