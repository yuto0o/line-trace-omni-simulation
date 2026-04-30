import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 80.0  # 円盤（機体）の半径 [mm]
LINE_WIDTH = 20.0  # ラインの太さ [mm]
BASE_SPEED = 40.0  # ロボットの基本進行速度
NUM_SENSORS = 6  # センサーの数
DT = 0.1  # 1ループの進む時間（タイムステップ）

# 制御ゲイン
KP_TRANS = 0.5  # ライン中心への引き戻し力（Pゲイン）

# 状態保持・復帰用パラメータ
MAX_LOST_STEPS = (
    200  # ラインを見失った後、推測で進み続ける最大ステップ数 (30 * 0.1秒 = 3秒)
)

COURSE_TYPE = 3  # (1: 直線, 2: S字カーブ, 3: 円形コース, 4 : 直角)


# ==========================================
# 2. コース（ライン）の生成関数
# ==========================================
def generate_course(type_id):
    points = []
    if type_id == 1:
        for y in range(0, 1000, 5):
            points.append([y * 0.2, y])
    elif type_id == 2:
        for y in range(0, 1000, 5):
            x = 100 * math.sin(y / 100.0)
            points.append([x, y])
    elif type_id == 3:
        for theta in np.linspace(0, 2 * math.pi, 200):
            points.append([200 * math.cos(theta), 200 * math.sin(theta) + 200])
    elif type_id == 4:
        # 直角コース（クランク）
        # まずY方向に進み、(0, 500)で直角に右に曲がる
        for y in range(0, 500, 5):
            points.append([0, y])
        for x in range(5, 500, 5):  # 5から始めて角の点の重複を防ぐ
            points.append([x, 500])
    return np.array(points)


# ==========================================
# 3. シミュレーション本体
# ==========================================
def main():
    path = generate_course(COURSE_TYPE)

    robot_pos = np.array([path[0][0], path[0][1]])
    robot_theta = 0.0

    # 状態保持用の変数
    current_heading = np.array([0.0, 1.0])
    last_valid_heading = np.array([0.0, 1.0])  # 最後に記憶した確かな進行方向
    lost_counter = 0  # 見失っている時間をカウント

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)
    sensor_local_pos = np.array(
        [[DISK_RADIUS * math.cos(a), DISK_RADIUS * math.sin(a)] for a in sensor_angles]
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))

    for step in range(1000):
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

        sensor_global_pos = robot_pos + sensor_local_pos

        active_sensors = []
        for i, s_pos in enumerate(sensor_global_pos):
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)
                ax.plot(s_pos[0], s_pos[1], "ro", markersize=6)
            else:
                ax.plot(s_pos[0], s_pos[1], "bo", markersize=4)
        # ------------------------------------------------
        # [制御アルゴリズム (状態遷移付き)]
        # ------------------------------------------------
        robot_color = "green"  # デフォルトは正常(緑)

        if len(active_sensors) > 0:
            # --- 正常状態 (TRACKING) ---
            lost_counter = 0
            ax.set_title(f"TRACKING: 正常トレース中 (Step: {step})", color="green")

            active_sensors = np.array(active_sensors)

            # ① センサーを進行方向に対する「前」と「後ろ」に分類する
            front_sensors = []
            back_sensors = []
            for s_pos in active_sensors:
                local_vec = s_pos - robot_pos
                dot_product = np.dot(local_vec, current_heading)
                if dot_product > 0:
                    front_sensors.append(s_pos)
                else:
                    back_sensors.append(s_pos)

            # ② 真の進行方向ベクトルの推測
            # if len(front_sensors) > 0 and len(back_sensors) > 0:
            #     f_mean = np.mean(front_sensors, axis=0)
            #     b_mean = np.mean(back_sensors, axis=0)
            #     line_vector = f_mean - b_mean
            #     line_vector = line_vector / np.linalg.norm(line_vector)

            #     current_heading = 0.8 * current_heading + 0.2 * line_vector
            #     current_heading = current_heading / np.linalg.norm(current_heading)
            # ② 真の進行方向ベクトルの推測（1点のみの反応も考慮した重み付け合成）
            line_vector = np.copy(current_heading)
            blend_weight = 0.0  # 状態に応じた重み（ゲイン）

            if len(front_sensors) > 0 and len(back_sensors) > 0:
                # 状態A：前後両方が反応（最も信頼度が高い完全なベクトル）
                f_mean = np.mean(front_sensors, axis=0)
                b_mean = np.mean(back_sensors, axis=0)
                line_vector = f_mean - b_mean
                blend_weight = 0.3  # 強めに今の向きを更新する

            elif len(front_sensors) > 0:
                # 状態B：前だけ反応（未来のカーブを予測する Pure Pursuit）
                f_mean = np.mean(front_sensors, axis=0)
                # 機体中心から前センサーへのベクトルを「これから行くべき道」とみなす
                line_vector = f_mean - robot_pos
                blend_weight = 0.1  # あくまで予測なので少し弱めにブレンドする

            elif len(back_sensors) > 0:
                # 状態C：後ろだけ反応（過去の軌跡から直線を推測する）
                b_mean = np.mean(back_sensors, axis=0)
                # 後ろセンサーから機体中心へのベクトルを「いままで来た道」とみなす
                line_vector = robot_pos - b_mean
                blend_weight = 0.05  # 不確実性が高いのでさらに弱め

            # ベクトルの正規化（長さを1にする）と、現在方向への合成
            if blend_weight > 0:
                line_vector = line_vector / np.linalg.norm(line_vector)
                # 現在の向きに、新しい推測ベクトルを滑らかに足し合わせる
                current_heading = (
                    1.0 - blend_weight
                ) * current_heading + blend_weight * line_vector
                current_heading = current_heading / np.linalg.norm(current_heading)

            # ③ ズレ（エラー）の計算と「直交射影」
            line_center = np.mean(active_sensors, axis=0)
            raw_error = line_center - robot_pos
            dot_err = np.dot(raw_error, current_heading)
            lateral_error = raw_error - (dot_err * current_heading)

            # ④ 速度の決定
            velocity = (current_heading * BASE_SPEED) + (lateral_error * KP_TRANS)
            robot_pos = robot_pos + velocity * DT
            last_valid_heading = current_heading.copy()

        else:
            # --- ライン見失い状態 (LOST) ---
            # ここが消えていたため、初期位置から動けなかった
            lost_counter += 1

            if lost_counter < MAX_LOST_STEPS:
                # 探索モード：記憶を頼りに少し進んでみる
                robot_color = "orange"
                ax.set_title(
                    f"SEARCHING: 記憶を頼りに推測航法中... ({lost_counter}/{MAX_LOST_STEPS})",
                    color="orange",
                )

                # 最後に記憶した方向へ、基本速度だけで進む
                velocity = last_valid_heading * BASE_SPEED
                robot_pos = robot_pos + velocity * DT

                current_heading = last_valid_heading.copy()
            else:
                # タイムアウト：完全に復帰不可
                robot_color = "red"
                ax.set_title("CRITICAL LOST: コースアウトしました (停止)", color="red")
                circle = patches.Circle(
                    robot_pos,
                    DISK_RADIUS,
                    fill=False,
                    edgecolor=robot_color,
                    linewidth=2,
                )
                ax.add_patch(circle)
                ax.arrow(
                    robot_pos[0],
                    robot_pos[1],
                    current_heading[0] * 15,
                    current_heading[1] * 15,
                    head_width=3,
                    head_length=5,
                    fc="red",
                    ec="red",
                )
                plt.pause(2.0)
                break

        # ------------------------------------------------
        # [機体の描画]
        # ------------------------------------------------
        circle = patches.Circle(
            robot_pos, DISK_RADIUS, fill=False, edgecolor=robot_color, linewidth=3
        )
        ax.add_patch(circle)
        ax.arrow(
            robot_pos[0],
            robot_pos[1],
            current_heading[0] * 15,
            current_heading[1] * 15,
            head_width=3,
            head_length=5,
            fc="orange",
            ec="orange",
        )

        ax.set_xlim(robot_pos[0] - 100, robot_pos[0] + 100)
        ax.set_ylim(robot_pos[1] - 100, robot_pos[1] + 100)
        plt.grid(True)

        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
