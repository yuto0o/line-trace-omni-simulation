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

COURSE_TYPE = 6
SPIN_OMEGA = 1.5  # 機体の自転角速度 (rad/s)

# [NEW] PD制御用のゲイン
KP = 1.1  # P(比例)ゲイン: ズレに比例して引き戻す力
KD = 0.1  # D(微分)ゲイン: ズレの変化(勢い)を抑える、または予測してブレーキをかける力
HEADING_BLEND = 0.249  # 前後の点から求めた進行方向を、現在の進行方向に混ぜる割合


# ==========================================
# 2. コース（ライン）の生成関数
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
# 3. シミュレーション本体
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)
    robot_pos = np.array([path[0][0], path[0][1]])

    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    # 前後の記憶ポイント
    latest_front_pos = robot_pos + current_heading * DISK_RADIUS
    latest_back_pos = robot_pos - current_heading * DISK_RADIUS

    # D制御のための過去エラー値
    prev_err_val = 0.0

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

        # 常に自転
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
        for s_pos in sensor_global_pos:
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)
                ax.plot(s_pos[0], s_pos[1], "ro", markersize=6)

                local_vec = s_pos - robot_pos
                if np.dot(local_vec, current_heading) > 0:
                    latest_front_pos = s_pos.copy()
                else:
                    latest_back_pos = s_pos.copy()
            else:
                ax.plot(s_pos[0], s_pos[1], "bo", markersize=4)

        ax.plot(latest_front_pos[0], latest_front_pos[1], "y*", markersize=10)
        ax.plot(latest_back_pos[0], latest_back_pos[1], "y*", markersize=10)

        # ------------------------------------------------
        # 1. 進行方向の更新 (前後のポイントを結ぶベクトル)
        # ------------------------------------------------
        target_heading = latest_front_pos - latest_back_pos
        if np.linalg.norm(target_heading) > 1e-3:
            target_heading = target_heading / np.linalg.norm(target_heading)
            current_heading = (
                1.0 - HEADING_BLEND
            ) * current_heading + HEADING_BLEND * target_heading
            current_heading = current_heading / np.linalg.norm(current_heading)

        # ------------------------------------------------
        # 2. 横ズレに対するPD制御 (疑似ステアリング)
        # ------------------------------------------------
        err_val = 0.0
        # 進行方向に対して左向きを正とする横ベクトル
        lateral_dir = np.array([-current_heading[1], current_heading[0]])

        if len(active_sensors) > 0:
            line_center = np.mean(active_sensors, axis=0)
            raw_err_vec = line_center - robot_pos
            # 中心からどれだけ横にズレているか（スカラー量）
            err_val = np.dot(raw_err_vec, lateral_dir)

        # D制御: エラーの変化量
        d_err_val = (err_val - prev_err_val) / DT
        prev_err_val = err_val

        # PD制御の出力を計算
        correction_speed = (KP * err_val) + (KD * d_err_val)

        # ベースの直進速度と、PD制御による横スライド速度を合成
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

        # 進行方向ベクトル (オレンジ)
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

        # PD制御による横修正ベクトル (緑)
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
        ax.set_title(f"Spin + PD Control (Step: {step})\nKp={KP}, Kd={KD}")
        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
