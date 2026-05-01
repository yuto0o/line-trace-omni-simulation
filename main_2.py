import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 40.0
LINE_WIDTH = 20.0
BASE_SPEED = 40.0
NUM_SENSORS = 6
DT = 0.1

COURSE_TYPE = 6

# スピンと追従用の基本パラメータ
SPIN_OMEGA = 3.0  # 機体が常に自転する角速度 (rad/s)
BLEND_RATIO = 0.3  # 予測した接線ベクトルを現在の進行方向に混ぜる割合
KP_LATERAL = 0.1  # ライン中心への弱い引き戻し

# [NEW] 遅れ補償（インベタ走行）用のパラメータ
ANGLE_THRESHOLD = (
    30.0  # 前・中・後の3点が作る進行方向の偏角がこれを超えたら急カーブ判定
)
SHRINK_RATIO = 0.70  # 急カーブ時に仮定する「少し小さい円」の半径の割合 (例: 70%にする)
SHRINK_GAIN = 0.05  # 小さい円に向かって機体を引き込む求心力の強さ


SPIN_OMEGA = 7.0871
BLEND_RATIO = 0.2455
KP_LATERAL = 3.9385
ANGLE_THRESHOLD = 11.5908
SHRINK_RATIO = 0.8012
SHRINK_GAIN = 0.0249
DISK_RADIUS = 28.9740


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
# 3. [NEW] 遅れを先読みした円弧ベクトルの計算
# ==========================================
def get_target_vector_with_shortcut(p_back, p_center, p_front):
    B = p_back - p_center
    F = p_front - p_center

    # --- ① 進行方向の「偏角（曲がり具合）」を計算 ---
    # 後方から中心へのベクトルと、中心から前方へのベクトルのなす角
    v_in = p_center - p_back
    v_out = p_front - p_center
    norm_in = np.linalg.norm(v_in)
    norm_out = np.linalg.norm(v_out)

    angle_deg = 0.0
    if norm_in > 1e-3 and norm_out > 1e-3:
        cos_theta = np.dot(v_in, v_out) / (norm_in * norm_out)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_theta))

    # --- ② 3点を通る外接円の計算 ---
    xb, yb = B[0], B[1]
    xf, yf = F[0], F[1]
    denom = 2 * (xb * yf - xf * yb)

    # 直線の場合は前を向くだけ
    if abs(denom) < 1e-3:
        vec = F - B
        return vec / np.linalg.norm(vec), None, None, None, False

    cx = (xf * (xb**2 + yb**2) - xb * (xf**2 + yf**2)) / denom
    cy = (yb * (xf**2 + yf**2) - yf * (xb**2 + yb**2)) / denom
    center_local = np.array([cx, cy])
    original_radius = np.linalg.norm(center_local)
    center_global = center_local + p_center

    # 通常の接線ベクトルを求める
    r_vec = -center_local
    tan1 = np.array([-r_vec[1], r_vec[0]])
    tan2 = np.array([r_vec[1], -r_vec[0]])
    tangent = tan1 if np.dot(tan1, F) > 0 else tan2
    tangent = tangent / np.linalg.norm(tangent)

    # --- ③ 角度が30度を超えた場合の「仮想の小さい円」へのショートカット制御 ---
    is_sharp_curve = angle_deg > ANGLE_THRESHOLD
    target_radius = original_radius
    target_vector = tangent

    if is_sharp_curve:
        # 半径を意図的に小さく見積もる
        target_radius = original_radius * SHRINK_RATIO

        # 機体をその「小さい円」に向かって引き込むための補正ベクトルを計算
        # (現在の外側の位置から、中心に向かって少し引っ張る)
        vec_to_center = center_global - p_center
        dir_to_center = vec_to_center / np.linalg.norm(vec_to_center)
        radius_error = original_radius - target_radius  # 外側にいるので正の値

        # 接線方向に「内側への求心力」を足し合わせる
        target_vector = tangent + dir_to_center * (radius_error * SHRINK_GAIN)
        target_vector = target_vector / np.linalg.norm(target_vector)

    return target_vector, center_global, original_radius, target_radius, is_sharp_curve


# ==========================================
# 4. シミュレーション本体
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)
    robot_pos = np.array([path[0][0], path[0][1]])

    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    latest_front_pos = robot_pos + current_heading * DISK_RADIUS
    latest_back_pos = robot_pos - current_heading * DISK_RADIUS

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 8))

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

        # センサー座標計算と反応チェック
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
        # [制御] 小さい円の仮定を取り入れたベクトル計算
        # ------------------------------------------------
        target_vector, circle_center, orig_rad, target_rad, is_sharp = (
            get_target_vector_with_shortcut(
                latest_back_pos, robot_pos, latest_front_pos
            )
        )

        current_heading = (
            1.0 - BLEND_RATIO
        ) * current_heading + BLEND_RATIO * target_vector
        current_heading = current_heading / np.linalg.norm(current_heading)

        lateral_error = np.array([0.0, 0.0])
        if len(active_sensors) > 0:
            line_center = np.mean(active_sensors, axis=0)
            raw_err = line_center - robot_pos
            lateral_error = raw_err - (
                np.dot(raw_err, current_heading) * current_heading
            )

        velocity = (current_heading * BASE_SPEED) + (lateral_error * KP_LATERAL)
        robot_pos = robot_pos + velocity * DT

        # ------------------------------------------------
        # [描画] 機体と予測円の可視化
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

        # ★ 急カーブ判定時と通常時で円の描画を変える
        if circle_center is not None and orig_rad < 1000:
            if is_sharp:
                # 急カーブ: 元の円を薄く描き、目標とする小さい円を強調して描く
                orig_circle = patches.Circle(
                    circle_center,
                    orig_rad,
                    fill=False,
                    edgecolor="gray",
                    linestyle=":",
                    alpha=0.5,
                )
                target_circle = patches.Circle(
                    circle_center,
                    target_rad,
                    fill=False,
                    edgecolor="magenta",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.add_patch(orig_circle)
                ax.add_patch(target_circle)
                ax.set_title(
                    f"SHARP CURVE! 縮小円でインベタ走行中",
                    color="magenta",
                    fontweight="bold",
                )
            else:
                # 通常カーブ
                pred_circle = patches.Circle(
                    circle_center,
                    orig_rad,
                    fill=False,
                    edgecolor="cyan",
                    linestyle="--",
                    alpha=0.5,
                )
                ax.add_patch(pred_circle)
                ax.set_title(f"Spin & Arc Predict (Step: {step})", color="black")

        ax.set_xlim(robot_pos[0] - 150, robot_pos[0] + 150)
        ax.set_ylim(robot_pos[1] - 150, robot_pos[1] + 150)
        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
