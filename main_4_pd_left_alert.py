import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 80.0
LINE_WIDTH = 20.0
BASE_SPEED = 60.0
NUM_SENSORS = 6
DT = 0.1

COURSE_TYPE = 5
SPIN_OMEGA = 1.0  # 機体の自転角速度 (rad/s)

# PD制御用のゲイン
KP = 0.6
KD = 0.1
HEADING_BLEND = 0.249

# [NEW] 左急カーブ対策用のパラメータ
# 90度回転するのにかかるステップ数
LOST_THRESHOLD_STEPS = int((math.pi / 2.0) / (SPIN_OMEGA * DT))
# 強制的に左に曲げる角速度 (rad/s)
FORCE_LEFT_SPEED = math.radians(60.0)


# ==========================================
# 2. コース生成
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
    elif type_id == 5:
        # 正方形 (一辺800に拡大)
        for y in range(0, 800, 5):
            points.append([0, y])
        for x in range(5, 800, 5):
            points.append([x, 800])
        for y in range(795, -1, -5):
            points.append([800, y])
        for x in range(795, -1, -5):
            points.append([x, 0])
    return np.array(points)


# ==========================================
# 3. シミュレーション本体
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)
    robot_pos = np.array([path[0][0], path[0][1]])

    current_heading = np.array([1.0, 0.0])
    target_heading = np.array([1.0, 0.0])  # ベクトル凍結用に保持
    robot_theta = 0.0

    latest_front_pos = robot_pos + current_heading * DISK_RADIUS
    latest_back_pos = robot_pos - current_heading * DISK_RADIUS

    prev_err_val = 0.0
    front_lost_steps = 0  # [NEW] 前の点を見失っているステップ数

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

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

        robot_theta += SPIN_OMEGA * DT

        # センサー座標計算
        sensor_global_pos = []
        for angle in sensor_angles:
            g_ang = angle + robot_theta
            sensor_global_pos.append(
                np.array(
                    [
                        robot_pos[0] + DISK_RADIUS * math.cos(g_ang),
                        robot_pos[1] + DISK_RADIUS * math.sin(g_ang),
                    ]
                )
            )
        sensor_global_pos = np.array(sensor_global_pos)

        # センサー判定と前後点の更新
        active_sensors = []
        front_updated = False  # [NEW] 今のステップで前方が更新されたかフラグ

        for s_pos in sensor_global_pos:
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)
                ax.plot(s_pos[0], s_pos[1], "ro", markersize=6)

                local_vec = s_pos - robot_pos
                if np.dot(local_vec, current_heading) > 0:
                    latest_front_pos = s_pos.copy()
                    front_updated = True
                else:
                    latest_back_pos = s_pos.copy()
            else:
                ax.plot(s_pos[0], s_pos[1], "bo", markersize=4)

        ax.plot(latest_front_pos[0], latest_front_pos[1], "y*", markersize=10)
        ax.plot(latest_back_pos[0], latest_back_pos[1], "y*", markersize=10)

        # [NEW] フロントロストのカウント
        if front_updated:
            front_lost_steps = 0
        else:
            front_lost_steps += 1
        # ------------------------------------------------
        # 1. 進行方向の更新 (凍結 ＆ 強制左旋回ロジック)
        # ------------------------------------------------
        if front_lost_steps == 0:
            # 正常時：前方が更新された時だけ目標ベクトル(target_heading)を新しく作り直す
            raw_heading = latest_front_pos - latest_back_pos
            if np.linalg.norm(raw_heading) > 1e-3:
                target_heading = raw_heading / np.linalg.norm(raw_heading)

        # 異常時：90度分見失ったら強制左旋回
        if front_lost_steps >= LOST_THRESHOLD_STEPS:
            rot_angle = FORCE_LEFT_SPEED * DT
            cos_a = math.cos(rot_angle)
            sin_a = math.sin(rot_angle)

            new_x = current_heading[0] * cos_a - current_heading[1] * sin_a
            new_y = current_heading[0] * sin_a + current_heading[1] * cos_a

            current_heading = np.array([new_x, new_y])
            current_heading = current_heading / np.linalg.norm(current_heading)

            ax.text(
                robot_pos[0] - 50,
                robot_pos[1] + 100,
                "FORCE LEFT!",
                color="red",
                fontsize=12,
                fontweight="bold",
            )

        else:
            # 【修正ポイント】強制旋回中でなければ、毎ステップ必ずブレンド処理を行う！
            # （フロントを見失っている間は、凍結された古いtarget_headingが使われる）
            current_heading = (
                1.0 - HEADING_BLEND
            ) * current_heading + HEADING_BLEND * target_heading
            current_heading = current_heading / np.linalg.norm(current_heading)
        # ------------------------------------------------
        # 2. 横ズレに対するPD制御 (疑似ステアリング)
        # ------------------------------------------------
        err_val = 0.0
        lateral_dir = np.array([-current_heading[1], current_heading[0]])

        if len(active_sensors) > 0:
            line_center = np.mean(active_sensors, axis=0)
            raw_err_vec = line_center - robot_pos
            err_val = np.dot(raw_err_vec, lateral_dir)

        d_err_val = (err_val - prev_err_val) / DT
        prev_err_val = err_val

        correction_speed = (KP * err_val) + (KD * d_err_val)

        velocity_forward = current_heading * BASE_SPEED
        velocity_lateral = lateral_dir * correction_speed
        velocity = velocity_forward + velocity_lateral

        robot_pos = robot_pos + velocity * DT

        # ------------------------------------------------
        # [描画] 機体と各種ベクトル
        # ------------------------------------------------
        circle = patches.Circle(
            robot_pos, DISK_RADIUS, fill=False, edgecolor="green", linewidth=3
        )
        ax.add_patch(circle)

        front_x = robot_pos[0] + DISK_RADIUS * math.cos(robot_theta)
        front_y = robot_pos[1] + DISK_RADIUS * math.sin(robot_theta)
        ax.plot(
            [robot_pos[0], front_x], [robot_pos[1], front_y], color="blue", linewidth=1
        )

        ax.arrow(
            robot_pos[0],
            robot_pos[1],
            current_heading[0] * 20,
            current_heading[1] * 20,
            head_width=3,
            head_length=5,
            fc="orange",
            ec="orange",
        )

        if abs(correction_speed) > 1.0:
            ax.arrow(
                robot_pos[0],
                robot_pos[1],
                velocity_lateral[0] * 0.5,
                velocity_lateral[1] * 0.5,
                head_width=2,
                head_length=3,
                fc="green",
                ec="green",
            )

        ax.set_xlim(robot_pos[0] - 150, robot_pos[0] + 150)
        ax.set_ylim(robot_pos[1] - 150, robot_pos[1] + 150)

        status_text = f"Spin + PD Control (Step: {step})\nKp={KP}, Kd={KD}\n"
        status_text += f"Lost Steps: {front_lost_steps}/{LOST_THRESHOLD_STEPS}"
        ax.set_title(status_text)

        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
