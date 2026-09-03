import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests as rq

class Env:
    SSID         = 'F_SANTOS_2_4'
    IPV4_ADDRESS = '192.168.0.238'
    PORT         = '8080'
    ROUTE        = 'command'

class TestConfig:
    PWM_RANGE  = [1000, 1500, 2000, 2500, 3000, 3500, 4095]
    TIME       = 2  # s
    RETRIES    = 3
    RUNS       = 3
    OUTPUT_DIR = Path(__file__).parent

def get_kart_server_url() -> str:
    return f"http://{Env.IPV4_ADDRESS}:{Env.PORT}/{Env.ROUTE}"

def motors_cmd_fmt(m1: int, m2: int, m3: int, m4: int) -> str:
    return f"CMD_MOTOR#{m1}#{m2}#{m3}#{m4}"

def servo_cmd_fmt(channel:int,angle:float) -> str:
    return f"CMD_SERVO#{channel}#{angle}"

def set_servos_angle(channel:int,angle:float) -> rq.Response:
    if type(channel) is not int:
        raise TypeError(f"Channel Must Be Int, got {type(channel).__name__}")
    if type(angle) is not float:
        raise TypeError(f"Angle Must Be a float/int, got {type(channel).__name__}")
    msg = servo_cmd_fmt(channel,angle)
    return rq.post(get_kart_server_url(), data=msg, timeout=1.0)

def set_motors_pwm(m1: int, m2: int, m3: int, m4: int) -> rq.Response:
    for x in (m1, m2, m3, m4):
        if type(x) is not int:
            raise TypeError(f"Motor PWM must be int, got {type(x).__name__}")
    msg = motors_cmd_fmt(m1, m2, m3, m4)
    return rq.post(get_kart_server_url(), data=msg, timeout=1.0)

def send_pwm_with_retries(m1: int, m2: int, m3: int, m4: int, retries: int) -> bool:
    for _ in range(retries):
        try:
            if set_motors_pwm(m1, m2, m3, m4).ok:
                return True
        except rq.RequestException:
            continue
    return False

def stop_motors() -> None:
    # best-effort: tenta várias vezes; se falhar, alerta para intervenção manual
    if not send_pwm_with_retries(0, 0, 0, 0, retries=5):
        print("[ERROR] FALHA AO PARAR MOTORES — VERIFIQUE O CARRO MANUALMENTE")

def prompt_distance() -> float | None:
    while True:
        raw = input("Distancia percorrida em metros (vazio = descartar amostra): ").strip()
        if not raw:
            return None
        try:
            return float(raw.replace(',', '.'))
        except ValueError:
            print("Valor invalido, tente novamente.")

def save_csv(path: Path, rows: list[tuple]) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pwm", "run", "distance_m", "time_s", "velocity_mps"])
        w.writerows(rows)

def fit_model(rows: list[tuple]) -> tuple[float, float] | None:
    # V(PWM) = K * (PWM - Threshold)  =>  v = a*PWM + b, K = a, Threshold = -b/a
    if len(rows) < 2:
        return None
    pwm = np.array([r[0] for r in rows], dtype=float)
    v   = np.array([r[4] for r in rows], dtype=float)
    a, b = np.polyfit(pwm, v, 1)
    if a == 0:
        return None
    return float(a), float(-b / a)

def set_all_turn(pwm:int,ccw=True):
    if ccw:
        set_motors_pwm(-pwm,-pwm,pwm,pwm)
    else:
        set_motors_pwm(pwm,pwm,-pwm,-pwm)
        
def make_turn_pwm(pwm,dt,ccw=True):
    set_all_turn(pwm,ccw)
    time.sleep(dt)
    set_all_turn(0,ccw)

# Linear N Steps Decay
MIN_PWM_DUTY = 1000

def make_turn_pwm_decayed(pwm,dt,N=10,ccw=True):
    step = dt / N
    for i in range(N):
        duty = max(MIN_PWM_DUTY,int(pwm*(1.0 - i/N)))
        set_all_turn(duty,ccw)
        time.sleep(step)
    set_motors_pwm(0,0,0,0)


# ---------------------------------------------------------------------------
# Open-loop (no-gyro) in-place turn profiles that minimise slip.
#
# A skid-steer turn always scrubs the wheels sideways, but the body angle for a
# given PWM x time is dominated by how the wheels break/regain traction at the
# START (slamming full PWM spins them without turning the body) and the STOP
# (the driver's 0 = active brake causes an end skid). Softening both makes the
# angle-per-run REPEATABLE, so you can tune one profile to land ~90 deg and it
# stays put on that surface. Tune the main knob (usually `pwm` or `dt`), measure
# by hand, repeat.
# ---------------------------------------------------------------------------
def _lerp(a, b, frac):
    """Linear interpolation a->b for frac in [0, 1]."""
    return a + (b - a) * frac


def make_turn_pwm_trapezoid(pwm, dt, ramp_frac=0.3, min_pwm=MIN_PWM_DUTY,
                            steps=15, ccw=True):
    """Ramp UP min_pwm->pwm, cruise, ramp DOWN pwm->min_pwm, then stop.

    Soft on BOTH ends: the ramp-up avoids the traction-break spin and the
    ramp-down avoids the end skid, so this is the most repeatable profile ->
    easiest to land a consistent 90 deg. `ramp_frac` (0..0.5) is the fraction of
    `dt` spent ramping on each end. Main knob: `pwm` (or `dt`) for total angle.
    Duty never drops below `min_pwm`, so the wheels never stall mid-turn.
    """
    ramp_frac = max(0.0, min(0.5, ramp_frac))
    steps = max(1, steps)
    step = dt / steps
    t_up = ramp_frac * dt
    t_dn = (1.0 - ramp_frac) * dt
    for i in range(steps):
        t = i * step
        if t_up > 0.0 and t < t_up:                    # ramp up
            duty = _lerp(min_pwm, pwm, t / t_up)
        elif (dt - t_dn) > 0.0 and t > t_dn:           # ramp down
            duty = _lerp(pwm, min_pwm, (t - t_dn) / (dt - t_dn))
        else:                                          # cruise
            duty = pwm
        set_all_turn(int(duty), ccw)
        time.sleep(step)
    set_motors_pwm(0, 0, 0, 0)


def make_turn_pwm_softstart(pwm, dt, ramp_s=0.15, min_pwm=MIN_PWM_DUTY,
                            steps=12, ccw=True):
    """Ramp UP min_pwm->pwm over `ramp_s`, hold `pwm` for the rest of `dt`, stop.

    Cheapest anti-slip: softens only the START (the initial break-traction spin
    is usually the biggest source of run-to-run variation). Use when the end
    skid isn't the problem. Main knob: `pwm` or `dt`.
    """
    ramp_s = max(0.0, min(ramp_s, dt))
    n_up = max(1, int(steps * (ramp_s / dt))) if dt > 0.0 else 0
    for i in range(n_up):                              # ramp up
        set_all_turn(int(_lerp(min_pwm, pwm, (i + 1) / n_up)), ccw)
        time.sleep(ramp_s / n_up)
    set_all_turn(int(pwm), ccw)                        # cruise
    time.sleep(max(0.0, dt - ramp_s))
    set_motors_pwm(0, 0, 0, 0)


def make_turn_pwm_pulsed(pwm, on_s=0.08, off_s=0.06, n_pulses=8, ccw=True):
    """Rotate in discrete PULSES: drive `on_s`, brake `off_s`, repeated.

    Between pulses the wheels stop and regain STATIC grip, so each pulse steps
    the body around with far less continuous sideways scrub than one long spin.
    Best on low-grip / dusty / smooth floors where a continuous turn just
    slides. Main knob: `n_pulses` (coarse) or `on_s` (fine) for total angle.
    """
    for _ in range(max(1, n_pulses)):
        set_all_turn(int(pwm), ccw)
        time.sleep(on_s)
        set_motors_pwm(0, 0, 0, 0)                     # brake -> regain traction
        time.sleep(off_s)


# ---------------------------------------------------------------------------
# Server-side scheduled turns (the `raw_turn_schedule` command).
#
# Unlike the make_turn_pwm_* helpers above -- which step the PWM over HTTP and
# so suffer per-step network jitter -- these send ONE command and the KART runs
# the whole profile locally (consistent timing). With no gyro, the server also
# fakes the odometry heading by `final_turn_angle` degrees so telemetry / the
# sim mirror reflect the turn.
#
# `final_turn_angle` is a RELATIVE delta (end = start + it); give it the sign of
# the turn (positive for the ccw=True direction, negative for cw). `wait` blocks
# for the profile's duration so back-to-back calls don't overlap -- the server
# ignores a turn started while another is still running.
# ---------------------------------------------------------------------------
def send_raw_turn_schedule(turn_fn: str, ccw: bool, pwm: int, min_pwm: int,
                           final_turn_angle: float, fn_params: dict,
                           retries: int = 3):
    payload = {
        "type": "raw_turn_schedule",
        "turn_fn": turn_fn,
        "ccw": bool(ccw),
        "pwm": int(pwm),
        "min_pwm": int(min_pwm),
        "final_turn_angle": float(final_turn_angle),
        "fn_params": fn_params or {},
    }
    body = json.dumps(payload)
    for _ in range(retries):
        try:
            r = rq.post(get_kart_server_url(), data=body, timeout=2.0)
            if r.ok:
                return r
        except rq.RequestException:
            continue
    print("[ERROR] raw_turn_schedule falhou (sem conexao com o kart)")
    return None


def _wait_profile(duration: float, wait: bool) -> None:
    if wait:
        time.sleep(duration + 0.3)                     # + margin for final brake


def schedule_turn_std(pwm, dt, final_turn_angle=90.0, ccw=True,
                      min_pwm=MIN_PWM_DUTY, steps=20, wait=True):
    """Constant PWM for `dt` (sub-stepped), run server-side."""
    send_raw_turn_schedule("std", ccw, pwm, min_pwm, final_turn_angle,
                           {"dt": dt, "steps": steps})
    _wait_profile(dt, wait)


def schedule_turn_decayed(pwm, dt, N=10, final_turn_angle=90.0, ccw=True,
                          min_pwm=MIN_PWM_DUTY, wait=True):
    """Linear ramp-down pwm->min over `dt` (N steps), run server-side."""
    send_raw_turn_schedule("decayed", ccw, pwm, min_pwm, final_turn_angle,
                           {"dt": dt, "N": N})
    _wait_profile(dt, wait)


def schedule_turn_trapezoid(pwm, dt, ramp_frac=0.3, steps=15,
                            final_turn_angle=90.0, ccw=True,
                            min_pwm=MIN_PWM_DUTY, wait=True):
    """Ramp up / cruise / ramp down over `dt`, run server-side (most repeatable)."""
    send_raw_turn_schedule("trapezoid", ccw, pwm, min_pwm, final_turn_angle,
                           {"dt": dt, "steps": steps, "ramp_frac": ramp_frac})
    _wait_profile(dt, wait)


def schedule_turn_pulsed(pwm, on_s=0.08, off_s=0.06, n_pulses=8,
                         final_turn_angle=90.0, ccw=True,
                         min_pwm=MIN_PWM_DUTY, wait=True):
    """PWM bursts with brake gaps, run server-side (best on low-grip floors)."""
    send_raw_turn_schedule("pulsed", ccw, pwm, min_pwm, final_turn_angle,
                           {"on_s": on_s, "off_s": off_s, "n_pulses": n_pulses})
    _wait_profile(n_pulses * (on_s + off_s), wait)


if __name__ == "__main__":

    # Tune ONE of these to land ~90 deg on the CURRENT floor (measure by hand),
    # then keep its parameters for that surface:
    #make_turn_pwm_softstart(2000, 0.5) # soft start only
    
    # HOUSE FLOOR - 90 DEGREES
    
    # dt_pulse = 0.1112
    # make_turn_pwm_pulsed(1750, on_s=dt_pulse,off_s=dt_pulse, n_pulses=10)  # stepped, low-grip floors
    
    # dt_pulse = 0.08
    # make_turn_pwm_pulsed(2000, on_s=dt_pulse,off_s=dt_pulse, n_pulses=10)  # stepped, low-grip floors
        
    # END HOUSE FLOOR - 90 DEGREES
    
    # make_turn_pwm_decayed(1800, 0.5, 10)             # ramp-down only (old)
    # make_turn_pwm_trapezoid(pwm=1300, dt=0.3)        # soft start+stop (best)

    # --- Server-side (raw_turn_schedule): no per-step HTTP jitter, fakes the
    #     odometry heading by final_turn_angle. Same profile as above, cleaner
    #     timing. Watch pose.theta in telemetry sweep to +90:
    # schedule_turn_pulsed(2000, on_s=0.08, off_s=0.08, n_pulses=10, final_turn_angle=90)
    # schedule_turn_trapezoid(pwm=1300, dt=0.3, final_turn_angle=90)
    
    # HOUSE FLOOR - + 90 DEGREES
    
    # dt_pulse = 0.1
    # print(schedule_turn_pulsed(2800, on_s=dt_pulse, off_s=dt_pulse, n_pulses=5, final_turn_angle=90, ccw=True))
    
    # MUITO PROMISSOR INCLUSIVE
    schedule_turn_trapezoid(pwm=2000, dt=0.65, 
                             ramp_frac=0.5,
                             final_turn_angle=90)
        
    
    exit()
    
    print("INICIANDO CALIBRACAO: POSICIONE O CARRO NA REFERENCIA\n")
    measurements: list[tuple] = []

    try:
        for pwm in TestConfig.PWM_RANGE:
            for run in range(TestConfig.RUNS):
                print(f"\nPWM: {pwm}, RUN: {run + 1}/{TestConfig.RUNS}")
                input("Pressione Enter para iniciar...")
                try:
                    if not send_pwm_with_retries(pwm, pwm, pwm, pwm, TestConfig.RETRIES):
                        print("[ERROR] CONEXAO COM KART PERDIDA — pulando amostra")
                        continue
                    time.sleep(TestConfig.TIME)
                finally:
                    stop_motors()

                distance = prompt_distance()
                if distance is None:
                    continue
                velocity = distance / TestConfig.TIME
                measurements.append((pwm, run, distance, TestConfig.TIME, velocity))
                print("POSICIONE O CARRO NA REFERENCIA\n")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    
    finally:
        stop_motors()

        if not measurements:
            print("\nNenhuma medida coletada.")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = TestConfig.OUTPUT_DIR / f"calibration_{ts}.csv"
            save_csv(out, measurements)
            print(f"\nMedidas salvas em {out}")

            fit = fit_model(measurements)
            if fit is None:
                print("Dados insuficientes para regressao.")
            else:
                K, threshold = fit
                print(f"\nModelo: V(PWM) = {K:.6g} * (PWM - {threshold:.2f})")
                print(f"  K         = {K:.6g}")
                print(f"  Threshold = {threshold:.2f}")


#Modelo: V(PWM) = 0.000381073 * (PWM - -529.69)
#  K         = 0.000381073
#  Threshold = -529.69