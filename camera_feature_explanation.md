# Camera pose fix — what the `with-camera` branch implements, and the math behind it

Branch `with-camera`: the camera work (commit `400ed10`, "camera added") merged
with `simple-features` up to `1e5761b` (closest obstacle reference, RL planner,
legs of up to two cells, steady-kart obstacle confirmation). Everything here was
validated in simulation only; it has **not yet run on the real kart** (see §15).

---

## 1. The problem it solves

At every cell the runner drives an absolute `goto` to the cell centre. That
`goto` closes its loop on **odometry**, so it cannot see two errors:

* **Heading error from turns.** If the kart turns 2° more than commanded,
  odometry records the commanded heading, and the belief and the target agree
  while the real kart drives off at an angle. A 0.30 m leg then puts the kart
  $0.30\sin(2°) \approx 1$ cm sideways, and this adds up leg after leg.
* **Lateral offset.** It sits across the beam of the only exteroceptive sensor
  (the forward sonar), so the 1-D `Recenter` / `range_fix` cannot observe it.

Before this branch, the lateral axis was measured by the **angular sweep**: the
kart rotated through about 10 headings and fitted the sonar ranges. It is slow,
every rotation adds heading error, and the HC-SR04's ~15° cone smears the very
profile being fitted (see the closing block of `runner/constants.py`).

`simple-features` added a second sonar corrector for that axis, the **closest
obstacle reference** (`CLOSEST_OBSTACLE_AS_REFERENCE`): turn ±90° to face the
nearest side wall, read it, drive the difference, re-anchor. On the real
tcc.maze it holds for about three full 12–15 cell runs before the error left
over starts to matter.

The camera measures **lateral, yaw and (when visible) longitudinal** error from
images, turning the *camera* (pan servo) instead of the kart. It does **not**
replace the sonar correctors; it works with them, at a different scale (§9.2):

* the sonar correctors make the **big** corrections, on their own schedule,
  physically moving the kart;
* the camera runs **last** at every boundary and makes the **small** ones:
  what the sonar leaves under its thresholds and 1 cm quantisation, and the
  heading error that no sonar corrector observes.

---

## 2. What was added (file map)

| Area | File | Role |
|---|---|---|
| Pose algebra | `shadow/sim_mirror.py` | `reanchor()` (heading-aware anchor correction), `relative()` |
| Camera I/O | `camera_link/mjpeg_camera.py` | `latest_stamped()`: frame + count + decode time |
| | `camera_link/servo_head.py` | `ServoHead`: pan/tilt CMD_SERVO over HTTP without a simulator |
| Camera model | `camera_link/camera_calibration.py` | Intrinsics, ideal pinhole, per-view pan/tilt mount, JSON load/save |
| | `camera_link/pinhole_view.py` | Projection, back-projection, floor intersection, horizon |
| Image → measurement | `camera_link/gaussian_mixture.py` | Numpy EM Gaussian mixture (OpenCV 5 has no `cv2.ml`) |
| | `camera_link/floor_segmenter.py` | Self-supervised wall/floor colour model per frame |
| | `camera_link/contact_profile.py` | Floor-contact profile per column (HIT/FREE/NEAR) |
| Maze model | `maze_solver/runner/wall_world.py` | Maze as wall boxes + maze bounds + discovered obstacles |
| | `maze_solver/runner/floor_visibility.py` | Vectorised 2-D slab ray caster, exit distance |
| | `maze_solver/runner/view_geometry.py` | Per-view precomputation (samples' range/bearing from the camera foot) |
| | `maze_solver/runner/contact_model.py` | **Analytic forward model**: predicted contact interval per column |
| | `maze_solver/runner/reference_renderer.py` | Independent per-pixel 3-D ray caster (tests, colour-model labels) |
| Estimation | `maze_solver/runner/camera_fit.py` | Coarse grid + robust Levenberg–Marquardt, uncertainty, gates |
| | `maze_solver/runner/camera_setup.py` | Hardware defaults, calibration loading, view geometries |
| Session / control | `maze_solver/runner/camera_fix.py` | Non-blocking `CameraFix` session (servo, freshness, worker thread) |
| | `maze_solver/runner/center_move.py` | `CenterMove`: back off, then `goto` the cell centre |
| | `maze_solver/runner/arrival_policy.py` | Small vs big: what to apply, what to escalate, whether to move |
| | `maze_solver/runner/arrival_log.py` | Per-arrival CSV + run summary |
| Runner wiring | `maze_solver/runner/live_maze_runner.py` | Boundary chain Recenter → [side reference / sweep] → camera, HUD, `camfix`, tilt hold |
| | `runner/build_parser.py`, `runner/main.py` | `--camera-calib`, `--no-camera-fix`, `camfix` command |
| | `runner/constants.py` | New "Camera pose fix" section (§13) |
| Tests | `maze_solver/tests/test_reanchor.py` (4), `test_camera_model.py` (15), `test_camera_fit.py` (6), `test_camera_fix.py` (19), `test_camera_closed_loop.py` (1), `camera_fixtures.py` | 45 tests |
| Tools | `camera_calibrate.py` | Bench tool: intrinsics, check, extrinsics (§11) |
| | `get_cam.py`, `test_get_cam.py` | Side-by-side real/sim viewer; fake kart server |
| Sonar correctors (from `simple-features`) | `maze_solver/runner/closest_obstacle.py` | Closest obstacle reference, the big lateral corrector the camera works with |

`camera_impl.md` (the original plan) is kept next to this file;
`maze_solver/logs/` (CSVs, debug PNGs) is git-ignored.

---

## 3. Frames and pose algebra

**Planar poses** are $p=(x,y,\theta)$, with $\theta$ counterclockwise from $+X$.
Composition and inverse (`shadow/compose.py`, `shadow/inverse.py`):

$$
a\oplus b=\big(a_x+\cos a_\theta\,b_x-\sin a_\theta\,b_y,\;
a_y+\sin a_\theta\,b_x+\cos a_\theta\,b_y,\;a_\theta+b_\theta\big)
$$
$$
a^{-1}=\big(-(\cos a_\theta\,a_x+\sin a_\theta\,a_y),\;
-(-\sin a_\theta\,a_x+\cos a_\theta\,a_y),\;-a_\theta\big)
$$

**The belief** (`SimMirror.world_pose`) is the anchor $A$ composed with the motion
since the first odometry sample $o_{\text{ref}}$:

$$
W(o)=A\oplus r(o),\qquad r(o)=o_{\text{ref}}^{-1}\oplus o
$$

and the `goto` target in odometry is its exact inverse,
$o^\star=o_{\text{ref}}\oplus(A^{-1}\oplus w)$ (`odom_target`).

### 3.1 Re-anchoring with heading (`SimMirror.reanchor`)

`shift_anchor` could only translate $A$. The camera also measures **yaw**, so the
new operation sets the anchor such that the pose at the capture sample becomes
the corrected pose $P$:

$$
A'=P\oplus r_c^{-1},\qquad r_c=r(o_{\text{cap}})
$$

* **Exact at capture:** $W'(o_{\text{cap}})=A'\oplus r_c=P\oplus r_c^{-1}\oplus r_c=P$.
* **Future motion is re-laid along the corrected heading:**
  $W'(o)=P\oplus(r_c^{-1}\oplus r(o))=P\oplus(o_{\text{cap}}^{-1}\oplus o)$,
  i.e. the motion since capture is applied from the corrected pose. This is the
  persistent form of "the gyro's yaw was off by $\psi$".
* **Translation-only case:** if $P=W(o_{\text{cap}})+(d_x,d_y)$, then $A'=A+(d_x,d_y)$,
  identical to `shift_anchor`.
* **`odom_target` stays the exact inverse**, so the next leg's `goto` aims from
  the corrected frame.

All four properties are pinned by `tests/test_reanchor.py`.

### 3.2 The correction's parameterisation

The fit estimates $\delta=(a,b,\psi)$ in the frame of the heading $\theta_0$
frozen at capture: $a$ along it, $b$ to its left, $\psi$ yaw.

$$
P(\delta)=\big(x_0+c\,a-s\,b,\;\; y_0+s\,a+c\,b,\;\; \theta_0+\psi\big),\qquad c=\cos\theta_0,\;s=\sin\theta_0
$$

The runner applies exactly this $P$ through `reanchor` (§9).

---

## 4. Camera model

### 4.1 Intrinsics: Raspberry Pi Camera rev 1.3 (OV5647, 3.6 mm)

`proc_camera.py` streams 400×300. The kart's `rpicam-hello --list-cameras` and
the libcamera log confirm that the sensor mode used is **640×480**, which reads a
2560×1920 window (offset 16,0) of the 2592×1944 sensor. With a 1.4 µm pixel
pitch and a 3.6 mm lens:

$$
f=\frac{F}{N_{\text{crop}}\,p/W}=\frac{3.6\text{ mm}}{2560\cdot 1.4\,\mu\text{m}/400}=401.8\text{ px}
$$

That gives a field of view of 52.9° horizontal, 41.0° vertical and 63.8°
diagonal. The module is sold as "60°", which fits neither axis: 60° horizontal
would mean $f=346$ px and 60° diagonal $f=433$ px. That is why
`camera_calibrate.py intrinsics` should replace the default.

Everything downstream works in an **ideal pinhole** with square pixels and a
centred principal point:

$$
K_{\text{ideal}}=\begin{bmatrix}f&0&c_x\\0&f&c_y\\0&0&1\end{bmatrix},\quad f=\tfrac{f_x+f_y}{2},\quad (c_x,c_y)=\left(\tfrac{W-1}{2},\tfrac{H-1}{2}\right)
$$

If calibration finds distortion or an off-centre principal point, frames are
remapped into this pinhole (`cv2.initUndistortRectifyMap`), and pixels with no
source behind them are marked invalid.

### 4.2 Extrinsics: the pan/tilt head

* **Centre frame:** origin on the floor under the odometry centre; $x$ forward,
  $y$ left, $z$ up.
* **Camera frame:** OpenCV convention; $x$ right, $y$ down, $z$ along the
  optical axis.

For a level camera looking forward, the camera axes expressed in the centre
frame are

$$
R_0=\begin{bmatrix}0&0&1\\-1&0&0\\0&-1&0\end{bmatrix}
$$

and for a view with pan $\gamma$, pitch-down $\phi$ and roll $\rho$:

$$
R=R_z(\gamma)\,R_y(\phi)\,R_0\,R_z(\rho),\qquad
R_y(\phi)=\begin{bmatrix}\cos\phi&0&\sin\phi\\0&1&0\\-\sin\phi&0&\cos\phi\end{bmatrix}
$$

(a positive $\phi$ tilts the forward axis down). The angles are
$\gamma=$ view pan + pan bias, and $\phi=$ fix pitch + tilt bias.

**The lens moves when the head pans.** The pan axis sits $\ell$ behind the lens,
so with the lens at $L=(0.13,\,0,\,0.10)$ m at home pan,

$$
C(\gamma)=(L-\ell\,\hat x)+R_z(\gamma)\,\ell\,\hat x,\qquad \ell=0.025\text{ m}
$$

At 45° this moves the lens $\ell\sin 45°\approx 1.8$ cm sideways, which is the
size of the error being measured, so it is modelled rather than ignored.

**Servo units:** `Server/servo.py` drives $500+(\text{angle}+10)/0.09\ \mu s$,
i.e. 11.1 µs per unit, which is one degree on a 2000 µs/180° servo. The fix tilt
of 12 % means $\phi=(37-12)\cdot1°=25°$ down.

### 4.3 Projection, back-projection, floor

$$
X_c=R^\top(P-C),\qquad u=c_x+f\frac{X_c}{Z_c},\quad v=c_y+f\frac{Y_c}{Z_c}
$$

A pixel's ray in the centre frame is $d=R\,[(u-c_x)/f,\ (v-c_y)/f,\ 1]^\top$,
and it hits the floor ($z=0$) at

$$
Q=C+t\,d,\qquad t=-\frac{C_z}{d_z}\quad(d_z<0)
$$

**Horizon row.** With no roll, $v_h=c_y-f\tan\phi=149.5-401.8\tan25°\approx-38$,
which is above the image, so every row looks at the floor.

**Coverage and resolution** at $h=0.10$ m and $\phi=25°$:

* The bottom row sees the floor 0.10 m ahead of the lens.
* For a floor point at range $r$, $v=c_y+f\tan(\arctan(h/r)-\phi)$, so
  $$
  \left|\frac{dv}{dr}\right|=f\,\sec^2(\cdot)\,\frac{h}{r^2+h^2}\approx
  \begin{cases}4.1\ \text{px/cm} & r=0.3\text{ m}\\ 0.55\ \text{px/cm} & r=0.9\text{ m}\end{cases}
  $$

This is why the tilt is held down and the range is cut at 0.90 m.

---

## 5. The measurement: floor-contact profile

For each sampled column (every 4 px), scan **up** from the top of the
kart-body band ($v_{\text{near}}=H-1-20$). The pixels look at floor points
further and further away, until the first one that is not floor. That row is
where the nearest wall in that column **stands on the floor**.

**Why this feature:**

* It depends only on wall **footprints**, which is exactly what a `.maze` file
  describes. It does not depend on box height, texture, labels or tape.
* The real image and the model are compared through the same 1-D quantity.

**Kinds of column:**

| Kind | Meaning | Allowed contact interval (rows grow downward) |
|---|---|---|
| HIT | contact found at row $v$ | $[v,v]$ |
| FREE | floor up to the range limit $v_{\text{far},j}$ | $(-\infty,\,v_{\text{far},j}]$ |
| NEAR | wall already at $v_{\text{near}}$ (contact hidden by the body band) | $[v_{\text{near}},\,+\infty)$ |
| INVALID | no image behind it | skipped |

**Extraction** (`extract_contact_profile`):

* A contact needs a run of $m=3$ consecutive wall pixels, so an isolated speck
  cannot end the column early.
* The boundary between wall pixel $r$ and floor pixel $r+1$ is at $r+\tfrac12$.
  It is refined to the zero of the log-likelihood ratio (§6),
  $$
  v^\star=r+1+\frac{\lambda_{r+1}}{\lambda_r-\lambda_{r+1}},\qquad \lambda_{r+1}<0<\lambda_r
  $$
* Three fresh frames per view are combined by **majority vote on the kind** and
  the **median of the HIT rows** (`median_profile`), so one blurred frame cannot
  move a column.

---

## 6. Wall vs floor: a self-supervised colour model

There is no fixed colour threshold and no colour calibration step. At each view,
the map is rendered from the **believed** pose (`reference_renderer`, half
resolution), giving predicted labels {floor, wall, unknown}. The prediction is
wrong only in a band around the predicted boundaries, so both regions are
**eroded by $2m+1$ px with $m=24$** before they seed the model. The margin has
to cover a heading error, not only translation: about 4 px/cm near the kart, but
$f\tan\psi$ sideways, which is 21 px at 3°. With $m=12$, sized for
translation alone, vertical wall edges leaked into the wrong class in the closed
loop.

**Self-training.** The seed is then refined twice: the pixels the model itself
classifies with $|\lambda|>2$ (inside the region the map calls maze) become the
next training set. The map decides *where* to look; the colours decide *what*
each pixel is. The accuracy gate below is checked against the seed, the only
labels that do not come from the model being judged.

* **Features:** CIE Lab with lightness down-weighted,
  $x=(L^*/3,\ a^*,\ b^*)$. Shadows and exposure mostly move $L^*$; the
  cardboard-vs-floor difference lives in $a^*,b^*$.
* **Model:** a Gaussian mixture per class ($k=2$), fitted by EM. The
  responsibilities are
  $$
  \gamma_{ij}=\frac{\pi_j\,\mathcal N(x_i;\mu_j,\Sigma_j)}{\sum_l\pi_l\,\mathcal N(x_i;\mu_l,\Sigma_l)}
  $$
  and the M-step updates are
  $$
  \pi_j=\tfrac1n\textstyle\sum_i\gamma_{ij},\quad
  \mu_j=\frac{\sum_i\gamma_{ij}x_i}{\sum_i\gamma_{ij}},\quad
  \Sigma_j=\frac{\sum_i\gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum_i\gamma_{ij}}+I
  $$
  (the $+I$ keeps a component that collapsed onto a flat colour usable).
* **Classification:**
  $\lambda(x)=\log p_{\text{wall}}(x)-\log p_{\text{floor}}(x)$, clipped
  **symmetrically** to $\pm4$, then a 5×5 Gaussian blur, threshold at 0,
  opening 3×3 and closing 5×5. The clip matters. Clipped wide (±50), a
  confident class next to a less confident one averages to a zero crossing
  pushed into the weaker side, and the boundary moved 1.4–2.2 px depending on
  how sure each class happened to be. Clipped at ±4 it moves about 1 px at
  worst.
* **Refusal:** if the **balanced accuracy** on the training pixels is below
  0.85, the view is dropped: walls and floor are too alike to segment.

This also absorbs auto-exposure and white-balance changes between views, because
each view trains on itself.

---

## 7. The forward model (analytic, exact)

### 7.1 The visibility lemma

The boxes are $H_b=0.30$ m tall and the lens is $h=0.10$ m high. Let $F$ be the
camera **foot**, the floor point under the lens. **A floor point $Q$ is visible
if and only if the 2-D segment $F\to Q$ crosses no box footprint.**

*Proof.* The sight line from $(F,h)$ to $(Q,0)$ is $\big(F+s(Q-F),\ h(1-s)\big)$
for $s\in[0,1]$. Its height is below $h<H_b$. If its ground shadow enters a
footprint at some $s$, the 3-D point is inside that box, so $Q$ is hidden. If the
shadow crosses no footprint, no box, being a vertical prism over its footprint,
contains any point of the line. ∎

So the column's contact is **the first sample along its floor trace that is not
visible from $F$**. Image columns are *not* vertical planes once the camera is
pitched: their floor traces fan out from the nadir point behind $F$. So each
sample has its own bearing from $F$, and the model keeps them per sample.

### 7.2 Precomputation per view (`view_geometry.py`)

For column $u_j$ and rows $v_{jk}=v_{\text{near}}-2k$, back-project to $Q_{jk}$
and store

$$
r_{jk}=\|Q_{jk}-F\|,\qquad \varphi_{jk}=\operatorname{atan2}(Q_{jk,y}-F_y,\ Q_{jk,x}-F_x)
$$

* A sample is valid while $r_{jk}\le 0.90$ m and it is more than 2 px below the
  horizon. Validity is a contiguous prefix, and $v_{\text{far},j}$ is the last
  valid row.
* Bearings are binned at 0.15°, which is 0.8 mm at 0.3 m.

Both $r$ and $\varphi$ are **invariant under translation** of the kart, and a
rotation only adds the heading to $\varphi$.

### 7.3 Per candidate pose $(x,y,\theta)$ (`contact_model.py`)

Foot in the world: $F_w=(x,y)+R(\theta)F$. Along every bearing $\beta=\theta+\varphi$:

* **First wall entry** $R(\beta)$, by the **slab method** per box
  $[x_0,x_1]\times[y_0,y_1]$:
  $$
  t_{\text{in}}=\max\!\big(\min(t_{x_0},t_{x_1}),\min(t_{y_0},t_{y_1})\big),\quad
  t_{\text{out}}=\min\!\big(\max(t_{x_0},t_{x_1}),\max(t_{y_0},t_{y_1})\big)
  $$
  with $t_{x_i}=(x_i-F_{w,x})/\cos\beta$ and similarly for $y$. The ray hits the
  box iff $t_{\text{out}}\ge\max(t_{\text{in}},0)$, and $R=\min_{\text{boxes}}\max(t_{\text{in}},0)$.
* **Exit** $E(\beta)$ from the maze rectangle. tcc.maze has no perimeter walls,
  so beyond the edge the map knows nothing.

Then, per column:

* **Hidden samples:** $r_{jk}\ge R(\beta_{jk})$; the first one is at index $k_b$.
* **Outside samples:** $r_{jk}\ge E(\beta_{jk})$; the first one is at index $k_e$.
* **HIT** when $k_b\le k_e$ and $k_b>0$. The row is linearly interpolated on
  $g_k=r_k-R_k$:
  $$
  v^\star=v_{k-1}+t\,(v_k-v_{k-1}),\qquad t=\frac{g_{k-1}}{g_{k-1}-g_k}
  $$
* **NEAR** when $k_b=0$. **FREE** when no sample is hidden or outside.
* **UNKNOWN** when the floor leaves the maze first. The interval is
  $(-\infty,v_{\text{exit}}]$: anything at or beyond the edge is allowed.

All candidate translations of one heading run as one vectorised batch. Raycasts
are done per bearing bin (121 poses × ~440 bins × ~15 boxes) and gathered per
sample, about 25 ms per batch of 121 poses.

**Discovered obstacles** are never evidence, the same trust boundary as
`range_fix`. Columns whose prediction changes when the obstacles, inflated by
5 cm, are added are **vetoed** for every candidate (`vetoed_columns`).

### 7.4 Residual: the gap between intervals

The measured interval $[m_\ell,m_h]$ and the predicted one $[p_\ell,p_h]$ give

$$
g=\max\!\big(0,\ m_\ell-p_h,\ p_\ell-m_h\big)
$$

which is 0 when they overlap. One rule covers HIT/FREE/NEAR/UNKNOWN in every
combination, and it is **continuous in the pose**. A measured "nothing up to the
range limit" against a predicted contact just short of it costs exactly the few
rows between them. The least-squares step uses the signed version
($+(m_\ell-p_h)$ or $-(p_\ell-m_h)$).

---

## 8. The estimator (`camera_fit.py`)

### 8.1 Cost

With pixel noise $\sigma=1.5$ px and normalised residual $e=g/\sigma$, the
**truncated Huber** loss ($k=2$, $T=6$) is

$$
\rho(e)=\begin{cases}\tfrac12 e^2 & |e|\le k\\ k\left(|e|-\tfrac k2\right) & k<|e|\le T\\ k\left(T-\tfrac k2\right)=10 & |e|>T\end{cases}
$$

The truncation makes a person or a bag in view a *bounded* outlier, not a lever.

Neighbouring columns see the same edge, so the data cost is divided by a
correlation factor $c=3$:

$$
J_{\text{data}}=\frac1c\sum_{v}\sum_{j}\rho(e_{vj}),\qquad
J=J_{\text{data}}+\tfrac12\left(\frac{\psi}{\sigma_\psi}\right)^2+\tfrac12\,\frac{a^2+b^2}{\sigma_t^2}+\tfrac12\sum_{v\in\text{side}}\left(\frac{\eta_v}{\sigma_\eta}\right)^2
$$

with $\sigma_\psi=3°$, $\sigma_t=0.08$ m, $\sigma_\eta=1°$ and $|\eta_v|\le2°$.

* **Pan nuisance $\eta_v$.** Each side view's own pan error is a free parameter,
  so servo repeatability is not read as kart yaw. View $v$ is predicted at
  heading $\theta_0+\psi+\eta_v$. It is modelled as a rotation about the kart
  centre rather than the pan pivot, a difference of at most
  $0.105\text{ m}\cdot2°\approx4$ mm. The forward view has no nuisance: it
  defines yaw.
* **The yaw prior is honest (3°), not tight.** Against a single wall seen
  face-on at distance $D$, lateral and yaw trade off along $b\approx-D\psi$. A
  tight prior would turn real yaw error into a confident wrong lateral.

### 8.2 Search: coarse grid, then robust Levenberg–Marquardt

**1. Coarse grid.** $a,b\in\{-0.10,\dots,0.10\}$ in 2 cm steps (11×11),
$\psi\in\{-8°,\dots,8°\}$ in 1° steps (17), with each view's nuisance minimised
out:

$$
J_{\text{coarse}}(a,b,\psi)=\sum_v\min_{\eta}\Big[C_v(a,b,\psi+\eta)+\tfrac12(\eta/\sigma_\eta)^2\Big]+\text{priors}
$$

It is scored with a **widened** $\sigma_c=4\sigma=6$ px, a graduated
non-convexity step. Near a side wall a contact moves about 4 px/cm, so a 2 cm
grid step moves it about 8 px. At the real $\sigma$ every candidate but the exact
one would sit on the truncated plateau, and the grid would step over the basin.
Measured: at scale 1 the coarse stage missed a 3 cm / 2° offset outright. Each
view's table is computed as soon as that view arrives, overlapping the servo
moves.

**2. Starts.** The strict local minima of the coarse table (26-neighbourhood),
up to 3, dropping any with $J>2J_{\text{best}}+10$. Refining only the global
coarse minimum converged on a neighbouring wrong basin (lateral one step off,
compensated by yaw and pan).

**3. Levenberg–Marquardt per start** over $x=(a,b,\psi,\eta_1,\dots)$:

* **IRLS weights:** $w=1$ for $|e|\le k$, $w=k/|e|$ for $k<|e|\le T$, $w=0$
  beyond.
* **Normal equations:**
  $$
  \big(H+\lambda\,\mathrm{diag}\,H\big)\Delta=-g,\quad
  H=\tfrac1c J^\top W J+J_p^\top J_p,\quad g=\tfrac1c J^\top W e+J_p^\top r_p
  $$
  A step is accepted if the robust cost drops ($\lambda/10$); otherwise
  $\lambda\times10$, up to 6 tries.
* **Annealing:** $\sigma$ goes 6 → 3 → 1.5 px, with 12 iterations per level.
* **Jacobian:** forward differences with $h=10^{-3}$ (m or rad). A column whose
  residual **jumps** by more than 2.5 px in one step is not a slope: its contact
  crossed a box corner onto another wall, and its derivative is set to 0. Taken
  as a slope it swamped the normal equations; at tcc.maze (0,3) the LM stalled
  and returned a 1.2 cm lateral error, which became 2 mm once fixed.

A pure grid was abandoned here. At the real $\sigma$ the basin is a narrow
**diagonal valley** (translation along a view's line of sight trades against
rotation), and a 5 mm / 0.25° grid only lands in it by luck. Measured: the grid
kept a point costing twice the truth's.

**4.** The best start by final $J$ wins. The others are kept for the ambiguity
test.

### 8.3 Uncertainty and observability

At the solution, the Gauss–Newton information of the **data** (nuisance priors
included, kart priors excluded) is

$$
H_{\text{data}}=\tfrac1c J^\top W J+J_\eta^\top J_\eta,\qquad
s^2=\max\!\left(1,\ \frac{2J_{\text{data}}}{n_{\text{eff}}-3}\right),\quad n_{\text{eff}}=n/c
$$

$$
\Sigma=s^2\,H_{\text{data}}^{-1}\quad(\text{eigenvalues floored at } s^2,\ \text{i.e. std}\le 1\text{ m}/1\text{ rad})
$$

The **marginal** standard deviations of $a,b,\psi$ are $\sqrt{\Sigma_{ii}}$ of the
full inverse. The nuisances are thereby integrated out (Schur complement). Being
marginal rather than per-axis, this catches diagonal valleys like the lateral–yaw
trade-off, which a one-axis rise test misses.

**Systematic error per view.** $s^2H^{-1}$ treats columns as independent noise,
but a whole view's rows can be off together (segmentation, JPEG's
half-resolution chroma, tilt repeatability). A bias of $\beta$ px on every row of
view $v$ moves the Gauss–Newton optimum by

$$
s_v=H^{-1}\,\tfrac1c J^\top W\,\mathbf 1_v\,\frac{\beta}{\sigma},\qquad
\Sigma=s^2H^{-1}+\sum_v s_v s_v^\top,\quad \beta=0.7\text{ px}
$$

so a geometry where one view's offset can be traded for lateral or yaw reports
a large std, and the observability gate below drops that axis.

An axis is reported only if observable: $\sigma_b\le 8$ mm, $\sigma_a\le 10$ mm,
$\sigma_\psi\le 0.6°$. Otherwise it is `None`. In an open corridor the
longitudinal axis comes out `None`, as it should: the side walls look identical
along it.

### 8.4 Gates (`_judge`), in order

1. A view needs ≥ 12 HIT columns to be used, and all views together ≥ 30.
2. If neither lateral nor yaw is observable → refuse.
3. Inlier fraction ($|e|<T$) ≥ 0.6 and inlier RMS gap ≤ 2 px; otherwise refuse
   ("modelo nao explica a imagem").
4. An observable axis sitting on the window edge ($|a|,|b|>8$ cm or $|\psi|>7°$)
   → refuse (lost pose).
5. **Ambiguity:** another start that converged at least 3 cm away in $b$ must
   satisfy $(J_{\text{rival}}-J_{\text{best}})/s^2\ge4.5$; otherwise lateral is
   dropped.
6. Plausibility: $|b|>10$ cm → refuse; $|a|>10$ cm → longitudinal dropped;
   $|\psi|>7°$ → refuse.
7. **Confirmed:** every reported component is under its dead-band (4 mm /
   0.25°) or under twice its own std, whichever is wider. Nothing is applied.
8. **Improvement:** $J(\delta^\star)\le0.8\,J(0)$, both with nuisances
   re-minimised on a 0.25° lattice; otherwise refuse ("sem ganho").

Outcome: `corrigido`, `confirmado` or `recusado`, with the reason in Portuguese.

---

## 9. Runner integration

### 9.1 The `CameraFix` session (`camera_fix.py`)

```
settle_kart -> preposition -> [aim -> wait_servo -> settle_servo -> capture] x views
            -> processing -> done | refused
```

1. **Settle the kart.** Wait 0.3 s with `goal_active` false. Then freeze the
   believed pose and the odometry sample: `capture_rel` is what `reanchor`
   receives later.
2. **Check the map.** Views whose predicted contacts at the believed pose number
   fewer than 12 are skipped.
3. **Preposition.** Pan to 6 % below the lowest view (clamped to 0), then visit
   right (5 %) → forward (50 %) → left (95 %) in increasing order. Every view is
   reached moving the same way, so servo backlash lands it in the same place.
4. **Per view:**
   * Command the pan and tilt.
   * Wait for the kart to report the servo angles (`servo_telemetry`), up to
     1 s. No telemetry means no wait.
   * Settle 0.5 s.
   * Accept a frame only if it is new **and** was decoded at least 0.2 s after
     the settle: $t_{\text{recv}}\ge t_{\text{settled}}+0.2$ s. A new count
     alone is not enough, because the first frame after a move was taken during
     it.
   * Three frames per view, 2.5 s timeout.
5. **Process.** Each bundle goes to the session's **worker thread** (remap,
   segmentation, profile, coarse table). The main thread only issues commands
   and checks timers, so the 60 FPS loop never blocks.
6. **Abort and restore.** The session aborts if odometry moves more than 5 mm or
   0.5° during capture. The head always ends at the home pan, at the fix tilt.
7. **Debug PNGs** go to `maze_solver/logs/camera/`: the frame, walls tinted,
   measured contacts (green HIT / yellow FREE / magenta NEAR), the model at the
   believed pose (red) and at the corrected pose (cyan).

### 9.2 The boundary chain: big corrections, then small ones (`live_maze_runner.py`)

```
leg arrived ──> Recenter                         (sonar, longitudinal, physical)
            ──> closest obstacle reference | sweep, when due
                                                 (sonar, lateral, physical; its own
                                                  schedule: SWEEP_EVERY_N_LEGS cells,
                                                  NEXT_BEST_CLOSEST_AS_REFERENCE)
            ──> CameraFix("medicao")             (always last, small corrections)
                 ├─ translation small  -> reanchor lateral + longitudinal
                 ├─ translation big    -> escalate: the big corrector runs NOW
                 │                         └─ cannot run here -> camera clamped to 3 cm
                 ├─ yaw                -> reanchor, clamped to ±3° (camera only)
                 └─ move? (off by default) -> CenterMove -> CameraFix("verificacao")
            ──> log the boundary (CSV) ──> next leg
```

The sonar correctors' schedule and behaviour are exactly `simple-features'`.
The camera only adds a step at the end. After `ok`, the camera alone runs once
as `"inicio"` to correct the operator's placement.

**Why the camera runs last.** Everything the sonar correctors do at a boundary
involves turning or driving the kart: Recenter's moves, the reference's ±90°
turns out and back. Looking last means the camera measures the pose the next leg
will actually start from, including the heading error those turns just added.
It also means Recenter has already driven the longitudinal axis against the
sonar, so the camera's longitudinal is a residual and can be applied directly.

**Decision table** (`arrival_policy.plan_after_camera`), only for `corrigido`:

| Component | Inside the small range | Beyond it |
|---|---|---|
| lateral, longitudinal ($\le 3$ cm) | applied | **escalated**: not applied; the closest obstacle reference / sweep is made due and started at this same boundary. If it cannot run here (no usable side wall, or a better one scheduled ahead), the camera applies its reading **clamped** to ±3 cm |
| yaw ($\le 3°$) | applied | applied, **clamped** to ±3° (never escalated) |

**Why the two kinds of axis differ.** Translation is shared with the sonar
correctors. A big translation is their job: they move the kart physically and
are proven on the real maze. And a big reading from one camera fix is either a
real large error or a wrong fit, and clamping bounds the damage of the latter.

Yaw has nobody else: no sonar corrector observes heading. In the closed loop, a
yaw error the camera refused to apply kept growing until even the fit's
plausibility gate refused the (accurate) reading. From then on nothing could
correct it. Clamped, a large heading error is worked down over successive
boundaries.

The 3 cm bound sits where the sonar corrector stops acting: the closest obstacle
reference only moves the kart above 3 cm (`CLOSEST_OBSTACLE_MINIMAL_DELTA_CM`)
and only re-anchors above 2 cm (`RANGE_FIX_MIN_CORRECTION_M`). 3° is about one
turn's worth of error.

**Centring move** (`CAMERA_FIX_MOVE_POLICY`, default `"never"`). A small
lateral offset is removed by the next leg's absolute `goto`, and large ones are
the side reference's, which moves the kart itself. The move is still available
(`"always"` / `"before_turns"`): it applies to a small fit whose corrected pose
sits at least 3 cm off the cell's centre line, measured as
$e_{\text{lat}}=(P(\delta)-c_{\text{cell}})\cdot n$ with
$n=(-\sin\theta_0,\cos\theta_0)$. Backing off $d=0.12$ m first turns
`goto`'s turn1 into

$$
\arctan\frac{e_{\text{lat}}}{d}\approx14°\quad\text{for } e_{\text{lat}}=3\text{ cm}
$$

instead of ~90°. The back-off needs the cell behind,
$(r+\mathrm{round}(\sin\theta),\ c-\mathrm{round}(\cos\theta))$, to be free;
otherwise a plain `goto` is used.

**Other runner changes:**

* The tilt is held at the fix tilt for the whole run; `C` (look forward) returns
  there.
* Arrow-key pan/tilt is ignored while a fix runs.
* The HUD has its own `camera: …` line, next to the sonar correctors' line.
* `camfix` command: one measurement at the current pose, nothing applied.
* The camera state has class-level defaults, so runners built without
  `__init__` (the `simple-features` tests) run with the camera off.
* Flags: `--camera-calib PATH`, `--no-camera-fix` (alongside `--rl`).

### 9.3 The boundary log (`arrival_log.py`)

Right after a leg, the belief *is* the `goto` target, so
$\lVert W-c_{\text{cell}}\rVert\approx0$ by construction and measures nothing.
Two numbers in the CSV do measure:

* `recenter_first_cm`: the sonar's longitudinal error **on arrival**, before
  any correction at this boundary;
* `err_*_cm`: the camera's estimate of the offset **left after the sonar
  correctors**, in the cell's axis frame:

$$
(e_{\parallel},e_\perp)=\text{rot}_{-\theta_{\text{axis}}}\big(P(\hat\delta)_{xy}-c_{\text{cell}}\big)
$$

That second one is the residual the small corrections exist for. Each row also
records the camera's outcome and reason, the stds, views and hits, what was
applied, whether it escalated, which big corrector ran and the shift it applied,
and Recenter's first/last error. A run summary (mean/p90/max residual,
refusals, escalations, big corrections) is printed at the goal and on exit.

---

## 10. Calibration file (`camera_link/camera_calib.json`)

It is optional. Missing, the runner uses the hardware defaults (§4) and says so.
Any field present overrides the default, and a stream size mismatch is refused:

```json
{ "version": 1,
  "width": 400, "height": 300, "fx": 401.8, "fy": 401.8, "cx": 199.5, "cy": 149.5,
  "dist": [k1, k2, p1, p2, k3],
  "lens_xyz": [0.13, 0.0, 0.10], "pan_arm_m": 0.025,
  "pitch_deg": 25.0, "tilt_pct": 12, "pan_home_pct": 50, "deg_per_pct": 1.0,
  "pan_left_is_higher": true,
  "views": [ {"name": "direita", "pan_deg": -45, "pan_bias_deg": 0, "tilt_bias_deg": 0,
              "roll_deg": 0, "ignore_bottom_px": 20, "nuisance": true}, ... ],
  "box_height_m": 0.30, "wall_inset_m": 0.0 }
```

---

## 11. Bench tool (`camera_calibrate.py`)

It talks to the kart directly (MJPEG + CMD_SERVO over HTTP), with no CoppeliaSim
and no runner. `-i` accepts an IP or `pikart.local`, and it fails within 3 s with
a clear message if the kart is unreachable.

* **`intrinsics`** — Zhang's method (`cv2.calibrateCamera`, $k_3$ fixed) from
  10 or more chessboard views. `--board` counts **inner** corners; a board with
  8×8 squares is 7×7, and the tool auto-detects which one is in view.
* **`check --cell r c --heading deg`** — one full camera fix with the kart placed
  by hand on that cell centre. It should read close to zero, and it writes the
  debug PNGs.
* **`extrinsics`** — same placement. For each view it grid-searches
  $\Delta\phi,\Delta\gamma\in[-6°,6°]$ in 0.5° steps, minimising the same robust
  cost at $\delta=0$. The mean $\Delta\phi$ goes into `pitch_deg`, the remainder
  into each view's `tilt_bias_deg`, and $\Delta\gamma$ into `pan_bias_deg`.

---

## 12. Validation (all synthetic; measured during development)

| Check | Setup | Result |
|---|---|---|
| Analytic model vs per-pixel 3-D ray caster | Corridor / pillars, 5 off-centre poses, all 3 views | Column kinds identical (except within one sample of the range limit); median row difference ≤ 0.5 px, p90 < 1 px |
| Fit over tcc.maze's solved path | 17 cells; truth ±4 cm long, ±5 cm lat, ±3° yaw; ±1° servo pan error; 0.7 px row noise | Lateral in **14/17** cells (sweep: 7/17); median error **1.1 mm**, max **2.1 mm**; ~1.1 s per fit |
| Closed loop (real runner chain, simulated kart) | Every real turn 3% larger than commanded, unseen by odometry; path (2,0)→(1,6); `LEG_MAX_CELLS = 2`, next-best scheduling on | Worst lateral offset at leg start / final distance to goal: **no lateral corrector 5.7 / 9.0 cm**, **closest obstacle only 2.7 / 5.6 cm**, **closest obstacle + camera 1.5 / 3.2 cm**. The reference ran twice in both of the last two runs; the camera corrected the belief at 4 of its 7 boundaries (2 refused: nothing in view) |
| Camera estimates in that loop vs truth | 7 boundaries, frames segmented from synthetic images | Lateral within 1 mm and yaw within 0.05° at most boundaries; one confidently wrong fit (lateral 1.6 cm, yaw 1.6° off), see §15 |
| Bench tool over real HTTP (fake kart server) | 3 cm lateral offset; 2° tilt error | Read as 2.8–2.9 cm; `extrinsics` recovered **+2.0°** |
| Lost pose | Truth 11–13 cm off (outside the ±10 cm window) | Always refused |

Synthetic frames are rendered flat-colour scenes (cardboard vs floor colour,
gradient, noise, JPEG q90). Real lighting, texture and lens behaviour are still
untested.

**Tests:** `cd maze_solver && python -m unittest discover -s tests` (229 tests).
The 45 camera tests pass. The 14 failures are exactly the ones `simple-features`
fails on its own at `1e5761b`. They come from its retuned sonar constants (sweep
span/step, `SCAN_FIT_MAX_SHIFT_M`, `RANGE_FIX_MAX_CORRECTION_M`, the 0.10 m
sensor offset): `test_sweep_tcc_maze` ×12, `test_range_fix` "huge residual" and
`test_recenter` "trust ceiling".

---

## 13. Key constants (`runner/constants.py`, "Camera pose fix")

| Constant | Value | Why |
|---|---|---|
| `CAMERA_FIX_TILT_PCT` | 12 (25° down) | Near contacts in view, best resolution (§4.3) |
| `CAMERA_FIX_VIEWS` | right −45°, forward 0°, left +45° | Side views see the cell's own side walls; forward defines yaw |
| `CAMERA_FIX_MAX_FLOOR_RANGE_M` | 0.90 | Beyond it a contact moves < 0.5 px/cm |
| `CAMERA_FIX_PIXEL_SIGMA_PX` / `HUBER_K` / `TRUNCATE_K` | 1.5 / 2 / 6 | Loss shape (§8.1) |
| `CAMERA_FIX_COLUMN_CORRELATION` | 3 | Effective independent columns |
| `CAMERA_FIX_COARSE_SIGMA_SCALE` | 4 | Basin width at 2 cm grid steps |
| `CAMERA_FIX_YAW_PRIOR_DEG` | 3 | Honest vs 1–5°/turn |
| `CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG` / `HALF_DEG` | 1 / 2 | Servo repeatability |
| `CAMERA_FIX_MAX_STD_LAT_M` / `LONG_M` / `YAW_DEG` | 8 mm / 10 mm / 0.6° | Observability |
| `CAMERA_FIX_SMALL_LATERAL_M` / `LONG_M` / `YAW_DEG` | 3 cm / 3 cm / 3° | The camera's range; beyond it translation escalates, yaw is clamped (§9.2) |
| `CAMERA_FIX_ESCALATES` | True | A big translation starts the side reference / sweep at the same boundary |
| `CAMERA_FIX_VIEW_BIAS_PX` | 0.7 | Systematic row error per view, propagated into the std (§8.3) |
| `CAMERA_FIX_SEARCH_HALF_YAW_DEG` / `MAX_YAW_DEG` | 8° / 7° | A heading past the small range must still be measurable to be worked down |
| `CAMERA_FIX_CONFIRM_M` / `CONFIRM_YAW_DEG` | 4 mm / 0.25° | Dead-band (or 2σ, whichever is wider) |
| `CAMERA_SEG_LABEL_MARGIN_PX` / `REFINE_ITERS` | 24 / 2 | Seed margin for ~3° of yaw; self-training rounds (§6) |
| `CAMERA_FIX_LATERAL_MOVE_M` | 0.03 | Smallest offset worth driving out |
| `CAMERA_FIX_MOVE_POLICY` / `MOVE_MODE` | `never` / `backoff` | §9.2 |
| `CAMERA_FIX_SERVO_SETTLE_S` / `FRAME_LATENCY_S` | 0.5 / 0.2 s | Freshness; re-measure on the bench |

---

## 14. How to run it

**On the kart, through the runner.** The camera fix is on by default. Choose a
calibration file or turn it off with:

```
--camera-calib <path>
--no-camera-fix
```

While the run is live, type `camfix` to take one measurement at the current
pose without applying anything.

**On the bench, before the first real run** (from the orquestrator folder):

1. Measure the lens:
   ```
   python camera_calibrate.py -i <kart IP> intrinsics --board 7x7 --square 0.033
   ```
2. Place the kart on a cell centre and run one fix there:
   ```
   python camera_calibrate.py -i <kart IP> check --maze tcc.maze --cell 2 0 --heading 90
   ```
   Look at the PNGs in `maze_solver/logs/camera/`.
3. Fit the pan and tilt offsets at the same placement:
   ```
   python camera_calibrate.py -i <kart IP> extrinsics --maze tcc.maze --cell 2 0 --heading 90
   ```
4. Repeat `check` in another cell.

---

## 15. Limitations and open items

* **Not yet run on hardware.** The camera currently reports `Camera frontend has
  timed out`: the sensor is detected over I2C but no frames arrive on CSI-2,
  which points to the flat cable. The pan/tilt flexes it on every move.
* **Defaults that are assumptions:**
  * lens at 13 cm forward / 10 cm up (read from "(13,10) cm");
  * pan arm 2.5 cm (a guess);
  * "lower tilt % = down" and 1 % = 1°;
  * the 3.6 mm focal length, rather than the advertised 60°.

  The bench order in §14 covers all of them, starting with the overlay PNGs.
* **No CoppeliaSim render check.** The plan's post-fit comparison against the
  Coppelia camera was not implemented; Coppelia wasn't available to validate its
  camera conventions. The analytic model is the forward model, checked against
  the independent ray caster instead.
* **Blind cells.** At dead ends and turn cells the lens stops about 2 cm from the
  front wall, and every view is NEAR. The fix refuses there (3 of 17 cells on
  tcc.maze: (1,0), (4,8), (5,9)); the sonar correctors still run there on their
  own schedule.
* **Confidently wrong fits still happen.** In the closed loop, one fix with two
  views (forward + one side, the front wall about 0.6 m away) read lateral
  +0.2 cm / yaw −4.4° against a truth of −1.4 cm / −2.7°. The data really did
  fit the wrong pose better, and its reported std stayed small even with the
  per-view bias term. What limits the damage is the cooperation design: yaw was
  clamped to −3° (close to the truth), and translation errors that large would
  have been escalated. The root cause is not found yet.
* **Pan nuisance is approximate.** It is modelled as a rotation about the kart
  centre, not the pan pivot: at most about 4 mm at 2°.
* **Centring move.** It is off by default (`"never"`). The originally chosen
  behaviour, centring after every fix over 3 cm, is `"always"`. With the camera
  limited to 3 cm it rarely triggers anyway.
