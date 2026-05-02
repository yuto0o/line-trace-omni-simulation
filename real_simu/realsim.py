import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 100.0
LINE_WIDTH = 20.0
# BASE_SPEED = 50.0  # スピードはそのまま、追従性を確認
MAX_SPEED = 100.0  # 直線時の最高速
MIN_SPEED = 40.0  # カーブ時の安全な最低速
SPEED_DECAY = 0.7  # ズレに対する減速の強さ（係数）
NUM_SENSORS = 6
DT = 0.1

COURSE_TYPE = 6
SPIN_OMEGA = 1.0

# PID制御・ハイブリッド制御用のゲイン
KP = 0.7
KD = 0.0
KI = 0.12
I_THRESHOLD = LINE_WIDTH * 0  # [NEW] 積分を開始する閾値（ラインの太さと同じ）

HEADING_BLEND = 0.249
LOST_THRESHOLD_STEPS = int((math.pi / 2.0) / (SPIN_OMEGA * DT))
FORCE_LEFT_SPEED = math.radians(120.0)


# ==========================================
# 2. ロボットの脳みそ (ローカル視点のみ)
# ==========================================
class OmnibotController:
    def __init__(self):
        self.local_heading = np.array([0.0, 1.0])
        self.latest_front_local = np.array([0.0, DISK_RADIUS])
        self.latest_back_local = np.array([0.0, -DISK_RADIUS])

        self.prev_err = 0.0
        self.err_integral = 0.0  # [NEW] 積分値の保存用
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
            # 迷子時の強制左旋回
            f_rot = FORCE_LEFT_SPEED * dt
            cos_f, sin_f = math.cos(f_rot), math.sin(f_rot)
            new_x = self.local_heading[0] * cos_f - self.local_heading[1] * sin_f
            new_y = self.local_heading[0] * sin_f + self.local_heading[1] * cos_f
            self.local_heading = np.array([new_x, new_y])
            self.local_heading /= np.linalg.norm(self.local_heading)
            is_force_left = True
        else:
            # 【修正1】神の視点コードと同様に、常に最新の記憶に向かって滑らかにブレンドし続ける
            self.local_heading = (
                1.0 - HEADING_BLEND
            ) * self.local_heading + HEADING_BLEND * target_heading
            self.local_heading /= np.linalg.norm(self.local_heading)

        # 4. 横ズレに対するPID制御
        lateral_dir = np.array([-self.local_heading[1], self.local_heading[0]])

        # 【修正2】センサーが外れた瞬間に 0 にせず、前回のズレを維持する（猛ブレーキ防止）
        err_val = self.prev_err
        if len(active_local_positions) > 0:
            line_center = np.mean(active_local_positions, axis=0)
            err_val = np.dot(line_center, lateral_dir)

        # 閾値付きI制御
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

        # P, I, D すべての合算で補正スピードを決定
        # ※以前と同じ Kp=0.6, Kd=0.1, Ki=0.0 で試してみてください
        correction_speed = (KP * err_val) + (KI * self.err_integral) + (KD * d_err_val)

        # 5. ローカル速度の算出
        # v_forward = self.local_heading * BASE_SPEED
        # --- 【変更前】 ---
        # v_forward = self.local_heading * BASE_SPEED

        # --- 【変更後】 ---
        # ズレの絶対値が大きいほど前進速度を落とす（MIN_SPEEDで底打ち）
        dynamic_speed = MAX_SPEED - (SPEED_DECAY * abs(err_val))
        dynamic_speed = max(MIN_SPEED, dynamic_speed)

        v_forward = self.local_heading * dynamic_speed
        # ------------------
        v_lateral = lateral_dir * correction_speed
        v_local = v_forward + v_lateral

        return v_local, SPIN_OMEGA, is_force_left, excess_err, self.err_integral


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
    fig, ax = plt.subplots(figsize=(6, 8))

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

        # 制御の更新 (I成分のステータスも受け取る)
        cmd_v_local, cmd_omega, is_force_left, excess_err, err_integral = robot.update(
            active_sensor_indices, DT
        )

        # 物理演算
        actual_omega = cmd_omega * 1.0
        actual_v_local = cmd_v_local * 1.0

        true_robot_theta += actual_omega * DT

        v_world_x = actual_v_local[0] * math.cos(true_robot_theta) - actual_v_local[
            1
        ] * math.sin(true_robot_theta)
        v_world_y = actual_v_local[0] * math.sin(true_robot_theta) + actual_v_local[
            1
        ] * math.cos(true_robot_theta)

        true_robot_pos += np.array([v_world_x, v_world_y]) * DT

        # 描画
        circle = patches.Circle(
            true_robot_pos, DISK_RADIUS, fill=False, edgecolor="green", linewidth=3
        )
        ax.add_patch(circle)

        believed_heading_world_x = robot.local_heading[0] * math.cos(
            true_robot_theta
        ) - robot.local_heading[1] * math.sin(true_robot_theta)
        believed_heading_world_y = robot.local_heading[0] * math.sin(
            true_robot_theta
        ) + robot.local_heading[1] * math.cos(true_robot_theta)

        ax.arrow(
            true_robot_pos[0],
            true_robot_pos[1],
            believed_heading_world_x * 20,
            believed_heading_world_y * 20,
            head_width=3,
            head_length=5,
            fc="orange",
            ec="orange",
        )

        if is_force_left:
            ax.text(
                true_robot_pos[0] - 50,
                true_robot_pos[1] + 100,
                "FORCE LEFT!",
                color="red",
                fontsize=12,
                fontweight="bold",
            )

        # I制御が発動中なら青字で知らせる
        if excess_err != 0.0:
            ax.text(
                true_robot_pos[0] + 60,
                true_robot_pos[1],
                f"I-Control: {err_integral:.1f}",
                color="blue",
                fontsize=10,
            )

        ax.set_xlim(true_robot_pos[0] - 150, true_robot_pos[0] + 150)
        ax.set_ylim(true_robot_pos[1] - 150, true_robot_pos[1] + 150)

        ax.set_title(f"PID Control (Step: {step})\nKp={KP}, Ki={KI}, Kd={KD}")
        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
