import time
import math

class TiredWheel:
    diameter          = 0.065   # meters
    radius            = diameter / 2
    colinear_distance = 0.12    # meters entre motores colineares (mesmo eixo)
    paralel_distance  = 0.151   # meters entre eixos (track / distancia entre lados)


class DiffDrive:
    """Cinematica diferencial. Rodas 1,3 = esquerda ; 2,4 = direita."""
    def __init__(self, sim, base, joints, wheel: TiredWheel):
        self.sim    = sim
        self.base   = base
        self.left   = [joints[0], joints[2]]
        self.right  = [joints[1], joints[3]]
        self.r      = wheel.radius
        self.track  = wheel.paralel_distance

        m0 = sim.getObjectMatrix(base, sim.handle_world)
        yaw0 = math.atan2(m0[4], m0[0])

        c, s = math.cos(-yaw0), math.sin(-yaw0)
        Rz_inv = [c, -s, 0.0, 0.0,
                  s,  c, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.base_matrix = sim.multiplyMatrices(Rz_inv, m0)

        # centro geometrico do robo, em coords do mundo (no instante inicial)
        bb_min_x = sim.getObjectFloatParam(base, sim.objfloatparam_objbbox_min_x)
        bb_max_x = sim.getObjectFloatParam(base, sim.objfloatparam_objbbox_max_x)
        bb_min_y = sim.getObjectFloatParam(base, sim.objfloatparam_objbbox_min_y)
        bb_max_y = sim.getObjectFloatParam(base, sim.objfloatparam_objbbox_max_y)
        # offset do centro em relacao a origem do frame (no frame local da base)
        self.cx_local = (bb_min_x + bb_max_x) / 2.0
        self.cy_local = (bb_min_y + bb_max_y) / 2.0

        self.x = 0.0
        self.y = 0.0
        self.z = m0[11]
        self.theta = yaw0

        self.wheel_angle = sim.getJointPosition(self.left[0])
        self.set_pose()

    def set_pose(self):
        self.sim.setObjectPosition(self.base, self.sim.handle_world,
                                   [self.x, self.y, self.z])
        c, s = math.cos(self.theta), math.sin(self.theta)
        Rz = [c, -s, 0.0,  0.0,
              s,  c, 0.0,  0.0,
              0.0, 0.0, 1.0, 0.0]
        m = self.sim.multiplyMatrices(Rz, self.base_matrix)
        m[3]  = self.x
        m[7]  = self.y
        m[11] = self.z
        self.sim.setObjectMatrix(self.base, self.sim.handle_world, m)

    def spin_wheels(self, v_left, v_right, dt):
        da_l = (v_left  / self.r) * dt
        da_r = (v_right / self.r) * dt
        for j in self.left:
            a = self.sim.getJointPosition(j) + da_l
            self.sim.setJointPosition(j, a % (2*math.pi))
        for j in self.right:
            a = self.sim.getJointPosition(j) + da_r
            self.sim.setJointPosition(j, a % (2*math.pi))

    def step(self, v_left, v_right, dt):
        v = (v_right + v_left) / 2.0
        w = (v_right - v_left) / self.track

        # posicao ATUAL do centro geometrico (origem + offset rotacionado)
        c, s = math.cos(self.theta), math.sin(self.theta)
        cx = self.x + (c * self.cx_local - s * self.cy_local)
        cy = self.y + (s * self.cx_local + c * self.cy_local)

        # integra o movimento NO centro
        self.theta += w * dt
        self.theta = (self.theta + math.pi) % (2*math.pi) - math.pi
        cx += v * math.cos(self.theta) * dt
        cy += v * math.sin(self.theta) * dt

        # volta para a origem do frame a partir do novo centro/heading
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x = cx - (c * self.cx_local - s * self.cy_local)
        self.y = cy - (s * self.cx_local + c * self.cy_local)

        self.set_pose()
        self.spin_wheels(v_left, v_right, dt)

    # passo unico para controle em tempo real (nao bloqueante)
    def drive(self, v_left, v_right, dt):
        self.step(v_left, v_right, dt)

    # --- comandos de alto nivel (bloqueantes) ---
    def drive_straight(self, distance, speed=0.15, dt=0.05):
        v = speed if distance >= 0 else -speed
        t_total = abs(distance) / speed
        t = 0.0
        while t < t_total:
            self.step(v, v, dt)
            time.sleep(dt)
            t += dt

    def turn_in_place(self, angle_deg, speed=0.12, dt=0.05):
        ang = math.radians(angle_deg)
        w = (speed*2) / self.track
        t_total = abs(ang) / w
        sign = 1 if ang >= 0 else -1
        v_r =  speed * sign
        v_l = -speed * sign
        t = 0.0
        while t < t_total:
            self.step(v_l, v_r, dt)
            time.sleep(dt)
            t += dt

    def drive_arc(self, radius, angle_deg, speed=0.15, dt=0.05):
        ang = math.radians(angle_deg)
        w = (speed / radius) if radius != 0 else 0.0
        v_left  = w * (radius - self.track/2)
        v_right = w * (radius + self.track/2)
        t_total = abs(ang) / abs(w) if w else 0.0
        t = 0.0
        while t < t_total:
            self.step(v_left, v_right, dt)
            time.sleep(dt)
            t += dt