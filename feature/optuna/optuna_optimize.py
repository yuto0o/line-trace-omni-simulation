import math

import numpy as np
import optuna

# ==========================================
# 1. 固定パラメータ設定
# ==========================================
LINE_WIDTH = 20.0
NUM_SENSORS = 6
DT = 0.1
COURSE_TYPE = 6
MAX_SIM_STEPS = 3000  # 最大シミュレーションステップ数
MAX_LOST_STEPS = 30  # これ以上連続でロストしたらリタイアとする


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
# 3. 3点から円の中心を求める関数
# ==========================================
def get_circle_tangent(p_back, p_center, p_front):
    B = p_back - p_center
    F = p_front - p_center

    xb, yb = B[0], B[1]
    xf, yf = F[0], F[1]

    denom = 2 * (xb * yf - xf * yb)

    if abs(denom) < 1e-3:
        vec = F - B
        return vec / np.linalg.norm(vec)

    cx = (xf * (xb**2 + yb**2) - xb * (xf**2 + yf**2)) / denom
    cy = (yb * (xf**2 + yf**2) - yf * (xb**2 + yb**2)) / denom
    center_local = np.array([cx, cy])

    r_vec = -center_local
    tan1 = np.array([-r_vec[1], r_vec[0]])
    tan2 = np.array([r_vec[1], -r_vec[0]])

    if np.dot(tan1, F) > 0:
        tangent = tan1
    else:
        tangent = tan2

    return tangent / np.linalg.norm(tangent)


# ==========================================
# 4. 描画なしのシミュレーション実行関数 (評価値を返す)
# ==========================================
def run_simulation(spin_omega, blend_ratio, kp_lateral, base_speed, disk_radius, path):
    robot_pos = np.array([path[0][0], path[0][1]])
    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    latest_front_pos = robot_pos + current_heading * disk_radius
    latest_back_pos = robot_pos - current_heading * disk_radius

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

    # 評価用の変数
    total_lateral_error = 0.0
    max_path_index = 0
    lost_counter = 0
    survived_steps = 0

    for step in range(MAX_SIM_STEPS):
        robot_theta += spin_omega * DT

        # センサー座標計算
        sensor_global_pos = []
        for angle in sensor_angles:
            g_ang = angle + robot_theta
            sensor_global_pos.append(
                np.array(
                    [
                        robot_pos[0] + disk_radius * math.cos(g_ang),
                        robot_pos[1] + disk_radius * math.sin(g_ang),
                    ]
                )
            )
        sensor_global_pos = np.array(sensor_global_pos)

        # センサー反応チェック
        active_sensors = []
        for s_pos in sensor_global_pos:
            distances_to_path = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances_to_path) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)
                local_vec = s_pos - robot_pos
                if np.dot(local_vec, current_heading) > 0:
                    latest_front_pos = s_pos.copy()
                else:
                    latest_back_pos = s_pos.copy()

        # 進捗（パスのどこまで到達したか）を計算
        dist_to_path_nodes = np.linalg.norm(path - robot_pos, axis=1)
        closest_path_idx = np.argmin(dist_to_path_nodes)
        if closest_path_idx > max_path_index:
            max_path_index = closest_path_idx

        lateral_error = np.array([0.0, 0.0])

        if len(active_sensors) > 0:
            lost_counter = 0
            survived_steps += 1

            target_vector = get_circle_tangent(
                latest_back_pos, robot_pos, latest_front_pos
            )
            current_heading = (
                1.0 - blend_ratio
            ) * current_heading + blend_ratio * target_vector
            current_heading = current_heading / np.linalg.norm(current_heading)

            line_center = np.mean(active_sensors, axis=0)
            raw_err = line_center - robot_pos
            lateral_error = raw_err - (
                np.dot(raw_err, current_heading) * current_heading
            )

            # 誤差を蓄積
            total_lateral_error += np.linalg.norm(lateral_error)

        else:
            lost_counter += 1
            if lost_counter > MAX_LOST_STEPS:
                break  # コースアウトで終了

        # ゴール判定（コースの末尾に到達したか）
        if max_path_index >= len(path) - 5:
            break

        # 速度の決定と移動
        velocity = (current_heading * base_speed) + (lateral_error * kp_lateral)
        robot_pos = robot_pos + velocity * DT

    # --- 評価スコアの算出 ---
    # Optunaはスコアが小さいほど「良い」と判断する（minimize）
    # 1. 完走できなかったら重いペナルティ (進んだインデックスをマイナス評価)
    # 2. 誤差の平均値を加算（小さいほど良い）
    # 3. かかったステップ数を加算（速くゴールするほど良い）

    path_progress_score = -(max_path_index * 1000)
    avg_error = total_lateral_error / max(1, survived_steps)

    score = path_progress_score + (avg_error * 10) + survived_steps
    return score


# ==========================================
# 5. Optunaの目的関数
# ==========================================
def objective(trial):
    # ① 探索するパラメータの範囲を定義
    spin_omega = trial.suggest_float("SPIN_OMEGA", 0.5, 10.0)
    blend_ratio = trial.suggest_float("BLEND_RATIO", 0.01, 0.99)
    kp_lateral = trial.suggest_float("KP_LATERAL", 0.0, 2.0)
    base_speed = trial.suggest_float("BASE_SPEED", 20.0, 100.0)
    disk_radius = trial.suggest_float("DISK_RADIUS", 20.0, 80.0)

    # 毎回コースを生成（使い回してもOKですが安全のため）
    path = generate_course(COURSE_TYPE)

    # ② シミュレーションを実行してスコアを取得
    score = run_simulation(
        spin_omega, blend_ratio, kp_lateral, base_speed, disk_radius, path
    )

    return score


# ==========================================
# 6. メイン処理（最適化の実行）
# ==========================================
if __name__ == "__main__":
    print("最適化を開始します...")

    # TPEサンプラーを使用して最適化の学習器を作成 (minimize: スコアを最小化する)
    study = optuna.create_study(direction="minimize")

    # トライアル数（シミュレーション実行回数）を指定して最適化実行
    # ※100〜300回程度回すと良いパラメータが見つかりやすいです
    study.optimize(objective, n_trials=150)

    print("\n=== 最適化完了 ===")
    print("ベストスコア: ", study.best_value)
    print("ベストパラメータ: ")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
