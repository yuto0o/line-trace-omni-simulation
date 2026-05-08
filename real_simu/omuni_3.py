import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 100.0
WHEEL_DISTANCE = DISK_RADIUS  # 中心からオムニホイールまでの距離(L)
LINE_WIDTH = 20.0

MAX_SPEED = 100.0
MIN_SPEED = 40.0

# 減速用のPIDゲイン（絶対ズレ量に対するPID）
SPEED_KP = 0.03
SPEED_KI = 0.0
SPEED_KD = 0.01

NUM_SENSORS = 6
DT = 0.1
COURSE_TYPE = 6
SPIN_OMEGA = 2.0

# 追従制御用のPIDゲイン
KP = 0.7
KD = 0.0
KI = 0.12
I_THRESHOLD = LINE_WIDTH * 0

HEADING_BLEND = 0.249
LOST_THRESHOLD_STEPS = int((math.pi / 1.0) / (SPIN_OMEGA * DT))
FORCE_LEFT_SPEED = math.radians(100.0)

# 外乱（ノイズ）の強さ (標準偏差)
NOISE_STD = 20

# オムニホイールの配置角度 (前方をY軸正とした場合、60度、180度、300度によく配置される)
WHEEL_ANGLES = [math.pi / 3, math.pi, 5 * math.pi / 3]

# 運動学の変換行列作成
# H: (vx, vy, omega) -> (v1, v2, v3)
H_matrix = np.array([[-np.sin(a), np.cos(a), WHEEL_DISTANCE] for a in WHEEL_ANGLES])
# H_inv: (v1, v2, v3) -> (vx, vy, omega)
H_inv_matrix = np.linalg.pinv(H_matrix)


# ==========================================
# 2. ロボットの脳みそ
# ==========================================
class OmnibotController:
    def __init__(self):
        self.local_heading = np.array([0.0, 1.0])
        self.latest_front_local = np.array([0.0, DISK_RADIUS])
        self.latest_back_local = np.array([0.0, -DISK_RADIUS])

        # 操舵用の変数
        self.prev_err = 0.0
        self.err_integral = 0.0

        # 減速用の変数（絶対値のPID用）
        self.prev_abs_err = 0.0
        self.abs_err_integral = 0.0

        self.front_lost_steps = 0
        self.sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

    def update(self, active_sensor_indices, dt):
        # 1. 記憶の逆回転
        rot_angle = -SPIN_OMEGA * dt
        cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)

        def rotate_vec(v):
            return np.array([v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a])

        self.local_heading = rotate_vec(self.local_heading)
        self.latest_front_local = rotate_vec(self.latest_front_local)
        self.latest_back_local = rotate_vec(self.latest_back_local)

        # 2. センサー情報の処理
        active_local_positions = []
        front_updated = False

        for idx in active_sensor_indices:
            ang = self.sensor_angles[idx]
            s_pos = np.array([DISK_RADIUS * math.cos(ang), DISK_RADIUS * math.sin(ang)])
            active_local_positions.append(s_pos)

            if np.dot(s_pos, self.local_heading) > 0:
                self.latest_front_local = s_pos.copy()
                front_updated = True
            else:
                self.latest_back_local = s_pos.copy()

        if front_updated:
            self.front_lost_steps = 0
        else:
            self.front_lost_steps += 1

        # 3. 進行方向の更新
        is_force_left = False
        target_heading = self.latest_front_local - self.latest_back_local
        if np.linalg.norm(target_heading) > 1e-3:
            target_heading = target_heading / np.linalg.norm(target_heading)

        if self.front_lost_steps >= LOST_THRESHOLD_STEPS:
            f_rot = FORCE_LEFT_SPEED * dt
            cos_f, sin_f = math.cos(f_rot), math.sin(f_rot)
            new_x = self.local_heading[0] * cos_f - self.local_heading[1] * sin_f
            new_y = self.local_heading[0] * sin_f + self.local_heading[1] * cos_f
            self.local_heading = np.array([new_x, new_y])
            self.local_heading /= np.linalg.norm(self.local_heading)
            is_force_left = True
        else:
            self.local_heading = (
                1.0 - HEADING_BLEND
            ) * self.local_heading + HEADING_BLEND * target_heading
            self.local_heading /= np.linalg.norm(self.local_heading)

        # 4. 横ズレに対するPID制御
        lateral_dir = np.array([-self.local_heading[1], self.local_heading[0]])

        err_val = self.prev_err
        if len(active_local_positions) > 0:
            line_center = np.mean(active_local_positions, axis=0)
            err_val = np.dot(line_center, lateral_dir)

        # 操舵用のI制御
        excess_err = 0.0
        if err_val > I_THRESHOLD:
            excess_err = err_val - I_THRESHOLD
        elif err_val < -I_THRESHOLD:
            excess_err = err_val + I_THRESHOLD

        if excess_err != 0.0:
            self.err_integral += excess_err * dt
        else:
            self.err_integral = 0.0

        d_err_val = (err_val - self.prev_err) / dt
        self.prev_err = err_val
        correction_speed = (KP * err_val) + (KI * self.err_integral) + (KD * d_err_val)

        # ==========================================
        # 【NEW】5. 減速のPID制御
        # ==========================================
        abs_err = abs(err_val)
        self.abs_err_integral += abs_err * dt
        d_abs_err = (abs_err - self.prev_abs_err) / dt
        self.prev_abs_err = abs_err

        # ズレの絶対値に対するPIDで「減速量」を算出
        speed_reduction = (
            (SPEED_KP * abs_err)
            + (SPEED_KI * self.abs_err_integral)
            + (SPEED_KD * d_abs_err)
        )

        # 最高速から減速量を引く
        dynamic_speed = MAX_SPEED - speed_reduction
        dynamic_speed = max(MIN_SPEED, min(MAX_SPEED, dynamic_speed))  # 上下限クリップ

        v_forward = self.local_heading * dynamic_speed
        v_lateral = lateral_dir * correction_speed

        # 目標となるローカル速度
        cmd_v_local = v_forward + v_lateral

        return cmd_v_local, SPIN_OMEGA, is_force_left, dynamic_speed


# ==========================================
# 3. コース生成
# ==========================================
def generate_course(type_id):
    points = []
    if type_id == 6:
        for y in range(0, 300, 5):
            points.append([0, y])
        for y in range(300, 600, 5):
            points.append([(y - 300) * (200.0 / 300.0), y])
        for y in range(600, 800, 5):
            points.append([200 - (y - 600) * (300.0 / 200.0), y])
        for y in range(800, 1000, 5):
            points.append([-100 + (y - 800) * (400.0 / 200.0), y])
        for y in range(1000, 1200, 5):
            points.append([300 - (y - 1000) * (200.0 / 200.0), y])
        for theta in np.linspace(math.pi, 0, 100):
            if theta != math.pi:
                points.append(
                    [400 + 300 * math.cos(theta), 1200 + 300 * math.sin(theta)]
                )
        for y in range(1200, -200, -5):
            points.append([700 + 150 * math.sin((1200 - y) / 120.0), y])
    return np.array(points)


# ==========================================
# 4. シミュレータ本体 (神の視点)
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)
    true_robot_pos = np.array([path[0][0], path[0][1]])
    true_robot_theta = 0.0

    robot = OmnibotController()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for step in range(3000):
        ax.cla()
        ax.set_aspect("equal")
        ax.plot(
            path[:, 0],
            path[:, 1],
            "k-",
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            alpha=0.5,
        )

        # センサー判定
        active_sensor_indices = []
        for i, ang in enumerate(robot.sensor_angles):
            g_ang = ang + true_robot_theta
            s_pos = true_robot_pos + np.array(
                [DISK_RADIUS * math.cos(g_ang), DISK_RADIUS * math.sin(g_ang)]
            )
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensor_indices.append(i)
                ax.plot(s_pos[0], s_pos[1], "ro", markersize=6)
            else:
                ax.plot(s_pos[0], s_pos[1], "bo", markersize=4)

        # 1. コントローラからの目標値取得
        cmd_v_local, cmd_omega, is_force_left, dynamic_speed = robot.update(
            active_sensor_indices, DT
        )

        # ==========================================
        # 【NEW】2. 逆運動学 (目標速度 -> モーター速度)
        # ==========================================
        cmd_vector = np.array([cmd_v_local[0], cmd_v_local[1], cmd_omega])
        cmd_motors = H_matrix @ cmd_vector

        # ==========================================
        # 【NEW】3. 外乱の印加 (モーターごとのバラツキ)
        # ==========================================
        # 各モーターに独立した正規分布ノイズを加える
        actual_motors = cmd_motors + np.random.normal(0, NOISE_STD, 3)

        # ==========================================
        # 【NEW】4. 順運動学 (実際のモーター速度 -> 実際の機体速度)
        # ==========================================
        actual_vector = H_inv_matrix @ actual_motors
        actual_v_local = actual_vector[0:2]
        actual_omega = actual_vector[2]

        # ==========================================
        # 5. 絶対座標系の更新
        # ==========================================
        true_robot_theta += actual_omega * DT

        v_world_x = actual_v_local[0] * math.cos(true_robot_theta) - actual_v_local[
            1
        ] * math.sin(true_robot_theta)
        v_world_y = actual_v_local[0] * math.sin(true_robot_theta) + actual_v_local[
            1
        ] * math.cos(true_robot_theta)

        true_robot_pos += np.array([v_world_x, v_world_y]) * DT

        # --- 描画関連 ---
        # 本体円
        circle = patches.Circle(
            true_robot_pos, DISK_RADIUS, fill=False, edgecolor="green", linewidth=2
        )
        ax.add_patch(circle)

        # 【NEW】オムニホイールの描画
        for i, a in enumerate(WHEEL_ANGLES):
            wheel_global_angle = true_robot_theta + a
            wx = true_robot_pos[0] + WHEEL_DISTANCE * math.cos(wheel_global_angle)
            wy = true_robot_pos[1] + WHEEL_DISTANCE * math.sin(wheel_global_angle)

            # 車輪の向きは取り付け角度に対して直角（tangent）
            tangent_angle = wheel_global_angle + math.pi / 2
            dx = 15 * math.cos(tangent_angle)
            dy = 15 * math.sin(tangent_angle)
            ax.plot([wx - dx, wx + dx], [wy - dy, wy + dy], "k-", linewidth=5)

            # 各モーターの実際の速度をテキスト表示
            ax.text(
                wx + 15,
                wy + 15,
                f"v{i + 1}:{actual_motors[i]:.0f}",
                color="purple",
                fontsize=8,
            )

        # 認識している進行方向（オレンジ矢印）
        believed_heading_world_x = robot.local_heading[0] * math.cos(
            true_robot_theta
        ) - robot.local_heading[1] * math.sin(true_robot_theta)
        believed_heading_world_y = robot.local_heading[0] * math.sin(
            true_robot_theta
        ) + robot.local_heading[1] * math.cos(true_robot_theta)
        ax.arrow(
            true_robot_pos[0],
            true_robot_pos[1],
            believed_heading_world_x * 40,
            believed_heading_world_y * 40,
            head_width=5,
            head_length=8,
            fc="orange",
            ec="orange",
        )

        if is_force_left:
            ax.text(
                true_robot_pos[0] - 50,
                true_robot_pos[1] + 120,
                "FORCE LEFT!",
                color="red",
                fontweight="bold",
            )

        ax.set_xlim(true_robot_pos[0] - 200, true_robot_pos[0] + 200)
        ax.set_ylim(true_robot_pos[1] - 200, true_robot_pos[1] + 200)

        # ステータス表示
        ax.set_title(
            f"Omni-Kinematics & Disturbance (Step: {step})\n"
            f"Target Spd: {dynamic_speed:.1f} | Actual Spd: {np.linalg.norm(actual_v_local):.1f}"
        )

        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
