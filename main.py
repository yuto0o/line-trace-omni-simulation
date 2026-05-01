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

# [NEW] スピンと追従用のパラメータ
SPIN_OMEGA = 3.0  # 機体が常に自転する角速度 (rad/s)
BLEND_RATIO = 0.2  # 予測した接線ベクトルを現在の進行方向に混ぜる割合(滑らかさ)
KP_LATERAL = 0.1  # ライン中心への弱い引き戻し(念のため)


SPIN_OMEGA = 3.1146
BLEND_RATIO = 0.2984
KP_LATERAL = 1.1511
BASE_SPEED = 94.7316
DISK_RADIUS = 65.1464

# ==========================================
# 2. コース（ライン）の生成関数
# ==========================================


def generate_course(type_id):
    points = []
    if type_id == 1:
        # 直線
        for y in range(0, 2000, 5):
            points.append([y * 0.2, y])
    elif type_id == 2:
        # S字カーブ (振幅と周期を拡大)
        for y in range(0, 2000, 5):
            points.append([200 * math.sin(y / 200.0), y])
    elif type_id == 3:
        # 円形コース (半径を300に拡大)
        for theta in np.linspace(0, 2 * math.pi, 200):
            points.append([300 * math.cos(theta), 300 * math.sin(theta) + 300])
    elif type_id == 4:
        # 直角（クランク）- 大型化
        for y in range(0, 800, 5):
            points.append([0, y])
        for x in range(5, 800, 5):
            points.append([x, 800])
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
    elif type_id == 6:
        # 【ボスステージ 改】 機体サイズ(半径8cm)に合わせた超特大コース

        # ① 直線 (スタート 〜 Y=300)
        for y in range(0, 300, 5):
            points.append([0, y])

        # ② 斜め (Y=300 〜 600)
        for y in range(300, 600, 5):
            x = (y - 300) * (200.0 / 300.0)  # Xは0から200へ
            points.append([x, y])

        # ③ 特大ギザギザ (Y方向の間隔を常に200mm以上確保)
        # 右(200)から左(-100)へ (Y: 600 -> 800)
        for y in range(600, 800, 5):
            x = 200 - (y - 600) * (300.0 / 200.0)
            points.append([x, y])

        # 左(-100)から右(300)へ (Y: 800 -> 1000)
        for y in range(800, 1000, 5):
            x = -100 + (y - 800) * (400.0 / 200.0)
            points.append([x, y])

        # 右(300)から中央(100)へ (Y: 1000 -> 1200)
        for y in range(1000, 1200, 5):
            x = 300 - (y - 1000) * (200.0 / 200.0)
            points.append([x, y])

        # ④ 特大U字カーブ (右回り)
        # 中心(400, 1200)、半径300(直径600)。開始(100) -> 終了(700)
        # 往路と復路の間隔が600mm空くので絶対に干渉しない
        for theta in np.linspace(math.pi, 0, 100):
            if theta == math.pi:
                continue
            points.append([400 + 300 * math.cos(theta), 1200 + 300 * math.sin(theta)])

        # ⑤ 特大S字カーブ (帰路)
        # 往路(X=300付近)に干渉しないよう、X=700をベースにうねりながら降りてくる
        for y in range(1200, -200, -5):
            x = 700 + 150 * math.sin((1200 - y) / 120.0)
            points.append([x, y])

    return np.array(points)


# ==========================================
# 3. 3点から円の中心を求める関数
# ==========================================
def get_circle_tangent(p_back, p_center, p_front):
    # 機体中心を原点とした相対座標にする
    B = p_back - p_center
    F = p_front - p_center

    xb, yb = B[0], B[1]
    xf, yf = F[0], F[1]

    denom = 2 * (xb * yf - xf * yb)

    # ほぼ直線の場合は、後方から前方へのベクトルをそのまま返す
    if abs(denom) < 1e-3:
        vec = F - B
        return vec / np.linalg.norm(vec), None, None

    # 外接円の中心 (相対座標)
    cx = (xf * (xb**2 + yb**2) - xb * (xf**2 + yf**2)) / denom
    cy = (yb * (xf**2 + yf**2) - yf * (xb**2 + yb**2)) / denom
    center_local = np.array([cx, cy])
    radius = np.linalg.norm(center_local)
    center_global = center_local + p_center

    # 接線ベクトルの計算 (円の中心から機体へのベクトルの直交ベクトル)
    # [-y, x] または [y, -x] の2パターンある
    r_vec = -center_local  # 円中心から原点(機体)へのベクトル
    tan1 = np.array([-r_vec[1], r_vec[0]])
    tan2 = np.array([r_vec[1], -r_vec[0]])

    # 前方の点(F)に向かう方の接線ベクトルを選ぶ
    if np.dot(tan1, F) > 0:
        tangent = tan1
    else:
        tangent = tan2

    tangent = tangent / np.linalg.norm(tangent)
    return tangent, center_global, radius


# ==========================================
# 4. シミュレーション本体
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)
    robot_pos = np.array([path[0][0], path[0][1]])

    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    # [NEW] 前後のライン位置を常に記憶する変数
    # 初期値は進行方向の少し前後に置いておく
    latest_front_pos = robot_pos + current_heading * DISK_RADIUS
    latest_back_pos = robot_pos - current_heading * DISK_RADIUS

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

        # 常に自転させる
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

        # センサー反応チェックと前後ポイントの更新
        active_sensors = []
        for s_pos in sensor_global_pos:
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)
                ax.plot(s_pos[0], s_pos[1], "ro", markersize=6)

                # 進行方向に対して前か後ろかを判定して記憶を上書き
                local_vec = s_pos - robot_pos
                if np.dot(local_vec, current_heading) > 0:
                    latest_front_pos = s_pos.copy()
                else:
                    latest_back_pos = s_pos.copy()
            else:
                ax.plot(s_pos[0], s_pos[1], "bo", markersize=4)

        # 記憶している前後ポイントを描画（黄緑の星）
        ax.plot(latest_front_pos[0], latest_front_pos[1], "y*", markersize=10)
        ax.plot(latest_back_pos[0], latest_back_pos[1], "y*", markersize=10)

        # ------------------------------------------------
        # [制御] 3点から円弧を予測して進む
        # ------------------------------------------------
        target_vector, circle_center, circle_radius = get_circle_tangent(
            latest_back_pos, robot_pos, latest_front_pos
        )

        # 予測された接線方向に、現在の進行方向を滑らかに近づける（ローパスフィルタ）
        current_heading = (
            1.0 - BLEND_RATIO
        ) * current_heading + BLEND_RATIO * target_vector
        current_heading = current_heading / np.linalg.norm(current_heading)

        # 少しだけ横ずれ修正（機体がライン中心を通るための補助）
        lateral_error = np.array([0.0, 0.0])
        if len(active_sensors) > 0:
            line_center = np.mean(active_sensors, axis=0)
            raw_err = line_center - robot_pos
            lateral_error = raw_err - (
                np.dot(raw_err, current_heading) * current_heading
            )

        # 速度の決定と移動
        velocity = (current_heading * BASE_SPEED) + (lateral_error * KP_LATERAL)
        robot_pos = robot_pos + velocity * DT

        # ------------------------------------------------
        # [描画] 機体と予測円
        # ------------------------------------------------
        circle = patches.Circle(
            robot_pos, DISK_RADIUS, fill=False, edgecolor="green", linewidth=3
        )
        ax.add_patch(circle)

        # 自転していることを示す線
        front_x = robot_pos[0] + DISK_RADIUS * math.cos(robot_theta)
        front_y = robot_pos[1] + DISK_RADIUS * math.sin(robot_theta)
        ax.plot(
            [robot_pos[0], front_x], [robot_pos[1], front_y], color="blue", linewidth=1
        )

        # 進行方向ベクトル
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

        # 予測した円弧を描画
        if (
            circle_center is not None and circle_radius < 1000
        ):  # 半径がデカすぎる(直線)場合は描画しない
            pred_circle = patches.Circle(
                circle_center,
                circle_radius,
                fill=False,
                edgecolor="cyan",
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(pred_circle)

        ax.set_xlim(robot_pos[0] - 150, robot_pos[0] + 150)
        ax.set_ylim(robot_pos[1] - 150, robot_pos[1] + 150)
        ax.set_title(f"Spin & Arc Predict Tracking (Step: {step})")
        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
