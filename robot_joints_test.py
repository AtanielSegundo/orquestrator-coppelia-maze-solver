from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import math
import time
import Coppelia.helpers as cop

rgb_to_norm = lambda _tuple : tuple(map(lambda e: e/ 255,_tuple))
CAIXA_COR = rgb_to_norm((178, 131, 97))

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
        # gira visualmente as rodas conforme velocidade linear de cada lado
        self.wheel_angle += 0  # mantido por compat
        da_l = (v_left  / self.r) * dt
        da_r = (v_right / self.r) * dt
        for j in self.left:
            a = self.sim.getJointPosition(j) + da_l
            self.sim.setJointPosition(j, a % (2*math.pi))
        for j in self.right:
            a = self.sim.getJointPosition(j) + da_r
            self.sim.setJointPosition(j, a % (2*math.pi))

    def step(self, v_left, v_right, dt):
        # modelo diferencial padrao
        v = (v_right + v_left) / 2.0
        w = (v_right - v_left) / self.track
        self.theta += w * dt
        self.theta = (self.theta + math.pi) % (2*math.pi) - math.pi
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.set_pose()
        self.spin_wheels(v_left, v_right, dt)

    # --- comandos de alto nivel (bloqueantes) ---
    def drive_straight(self, distance, speed=0.15, dt=0.05):
        """distance em metros; negativo = re."""
        v = speed if distance >= 0 else -speed
        t_total = abs(distance) / speed
        t = 0.0
        while t < t_total:
            self.step(v, v, dt)
            time.sleep(dt)
            t += dt

    def turn_in_place(self, angle_deg, speed=0.12, dt=0.05):
        """gira parado; angulo positivo = anti-horario."""
        ang = math.radians(angle_deg)
        w = (speed*2) / self.track          # |v_left|=|v_right|=speed
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
        """arco de raio 'radius' (centro a esquerda se radius>0)."""
        ang = math.radians(angle_deg)
        # v = w*R ; difenca de rodas define w
        w = (speed / radius) if radius != 0 else 0.0
        v_left  = w * (radius - self.track/2)
        v_right = w * (radius + self.track/2)
        t_total = abs(ang) / abs(w) if w else 0.0
        t = 0.0
        while t < t_total:
            self.step(v_left, v_right, dt)
            time.sleep(dt)
            t += dt

def clamp_to_arena(robot, half=0.95):
    """mantem o robo dentro de (2x2) m centrado na origem."""
    robot.x = max(-half, min(half, robot.x))
    robot.y = max(-half, min(half, robot.y))
    robot.set_pose()

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim: cop.CoppeliaSimulation = client.require('sim')

    cop.change_floor(sim, (2.0, 2.0), thickness=0.25)

    kart_base     = sim.getObject("/kart_base")
    motor_numbers = 4
    motors_joints = [sim.getObject(f"/motor_{id}/joint")
                     for id in range(1, motor_numbers+1)]

    for joint in motors_joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 0)

    robot = DiffDrive(sim, kart_base, motors_joints, TiredWheel())

    SPEED = 0.5

    try:

        while True:
            robot.drive_straight(1.0,SPEED)
            time.sleep(0.3)
            robot.drive_straight(-1.0,SPEED)
            time.sleep(0.3)
            robot.turn_in_place(360,SPEED)
            time.sleep(0.3)
            robot.turn_in_place(-360,SPEED)

            print(f"pose: x={robot.x:.2f} y={robot.y:.2f} theta={math.degrees(robot.theta):.0f}°")

    except KeyboardInterrupt:
        
        print("parado")