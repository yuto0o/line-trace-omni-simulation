import math

import numpy as np
import optuna

# ==========================================
# 1. 固定パラメータ設定
# ==========================================
LINE_WIDTH = 20.0
NUM_SENSORS = 6
DT = 0.1
BASE_SPEED = 40.0
COURSE_TYPE = 6
MAX_SIM_STEPS = 3000
MAX_LOST_STEPS = 30


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
# 3. 遅れを先読みした円弧ベクトルの計算
# ==========================================
def get_target_vector_with_shortcut(
    p_back, p_center, p_front, angle_threshold, shrink_ratio, shrink_gain
):
    B = p_back - p_center
    F = p_front - p_center

    v_in = p_center - p_back
    v_out = p_front - p_center
    norm_in = np.linalg.norm(v_in)
    norm_out = np.linalg.norm(v_out)

    angle_deg = 0.0
    if norm_in > 1e-3 and norm_out > 1e-3:
        cos_theta = np.dot(v_in, v_out) / (norm_in * norm_out)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_theta))

    xb, yb = B[0], B[1]
    xf, yf = F[0], F[1]
    denom = 2 * (xb * yf - xf * yb)

    if abs(denom) < 1e-3:
        vec = F - B
        return vec / np.linalg.norm(vec)

    cx = (xf * (xb**2 + yb**2) - xb * (xf**2 + yf**2)) / denom
    cy = (yb * (xf**2 + yf**2) - yf * (xb**2 + yb**2)) / denom
    center_local = np.array([cx, cy])
    original_radius = np.linalg.norm(center_local)
    center_global = center_local + p_center

    r_vec = -center_local
    tan1 = np.array([-r_vec[1], r_vec[0]])
    tan2 = np.array([r_vec[1], -r_vec[0]])
    tangent = tan1 if np.dot(tan1, F) > 0 else tan2
    tangent = tangent / np.linalg.norm(tangent)

    is_sharp_curve = angle_deg > angle_threshold
    target_vector = tangent

    if is_sharp_curve:
        target_radius = original_radius * shrink_ratio
        vec_to_center = center_global - p_center
        dir_to_center = vec_to_center / np.linalg.norm(vec_to_center)
        radius_error = original_radius - target_radius

        target_vector = tangent + dir_to_center * (radius_error * shrink_gain)
        target_vector = target_vector / np.linalg.norm(target_vector)

    return target_vector


# ==========================================
# 4. 描画なしのシミュレーション実行関数
# ==========================================
def run_simulation(
    spin_omega,
    blend_ratio,
    kp_lateral,
    angle_threshold,
    shrink_ratio,
    shrink_gain,
    disk_radius,
    path,
):
    robot_pos = np.array([path[0][0], path[0][1]])
    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    latest_front_pos = robot_pos + current_heading * disk_radius
    latest_back_pos = robot_pos - current_heading * disk_radius

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

    total_lateral_error = 0.0
    max_path_index = 0
    lost_counter = 0
    survived_steps = 0
    reverse_counter = 0  # 逆走検知用カウンター
    is_reversed = False

    for step in range(MAX_SIM_STEPS):
        robot_theta += spin_omega * DT

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

        # 進捗の計算
        dist_to_path_nodes = np.linalg.norm(path - robot_pos, axis=1)
        closest_path_idx = np.argmin(dist_to_path_nodes)
        if closest_path_idx > max_path_index:
            max_path_index = closest_path_idx

        # --- [NEW] 逆走検知ロジック ---
        # コースが本来進むべき方向（ローカルベクトル）を取得
        next_idx = min(closest_path_idx + 2, len(path) - 1)
        path_forward_vec = path[next_idx] - path[closest_path_idx]
        if np.linalg.norm(path_forward_vec) > 1e-3:
            path_forward_vec = path_forward_vec / np.linalg.norm(path_forward_vec)
            # 現在の進行方向とコースの進行方向の内積をとる（マイナスなら逆を向いている）
            if np.dot(current_heading, path_forward_vec) < -0.3:
                reverse_counter += 1
            else:
                reverse_counter = 0

        # 10ステップ（1秒相当）以上逆走を続けたら強制リタイア
        if reverse_counter > 10:
            is_reversed = True
            break

        lateral_error = np.array([0.0, 0.0])

        if len(active_sensors) > 0:
            lost_counter = 0
            survived_steps += 1

            target_vector = get_target_vector_with_shortcut(
                latest_back_pos,
                robot_pos,
                latest_front_pos,
                angle_threshold,
                shrink_ratio,
                shrink_gain,
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

            total_lateral_error += np.linalg.norm(lateral_error)

        else:
            lost_counter += 1
            if lost_counter > MAX_LOST_STEPS:
                break

        if max_path_index >= len(path) - 5:
            break

        velocity = (current_heading * BASE_SPEED) + (lateral_error * kp_lateral)
        robot_pos = robot_pos + velocity * DT

    # --- [NEW] スコア計算（追従重視・逆走厳罰） ---
    # 1. 完走できなかったペナルティ (残りの距離 × 10000)
    progress_penalty = (len(path) - max_path_index) * 10000

    # 2. 逆走した場合は問答無用で超特大ペナルティ
    reverse_penalty = 500000 if is_reversed else 0

    # 3. 追従誤差のペナルティ (誤差の平均 × 1000倍 にして極めて重視する)
    avg_error = total_lateral_error / max(1, survived_steps)
    error_penalty = avg_error * 1000

    # スコアは小さいほど優秀（Minimize）
    score = progress_penalty + reverse_penalty + error_penalty + survived_steps
    return score


# ==========================================
# 5. Optunaの目的関数
# ==========================================
def objective(trial):
    spin_omega = trial.suggest_float("SPIN_OMEGA", 0.5, 10.0)
    blend_ratio = trial.suggest_float("BLEND_RATIO", 0.01, 0.99)
    kp_lateral = trial.suggest_float(
        "KP_LATERAL", 0.0, 5.0
    )  # 横ズレ修正の幅を少し広げる

    angle_threshold = trial.suggest_float("ANGLE_THRESHOLD", 10.0, 60.0)
    shrink_ratio = trial.suggest_float("SHRINK_RATIO", 0.3, 0.95)
    shrink_gain = trial.suggest_float("SHRINK_GAIN", 0.01, 0.5)

    disk_radius = trial.suggest_float("DISK_RADIUS", 20.0, 80.0)

    path = generate_course(COURSE_TYPE)

    score = run_simulation(
        spin_omega,
        blend_ratio,
        kp_lateral,
        angle_threshold,
        shrink_ratio,
        shrink_gain,
        disk_radius,
        path,
    )

    return score


# ==========================================
# 6. メイン処理（最適化の実行）
# ==========================================
if __name__ == "__main__":
    print("最適化を開始します（追従精度重視・逆走防止モード）...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    print("\n=== 最適化完了 ===")
    print("ベストスコア: ", study.best_value)
    print("ベストパラメータ: ")
    for key, value in study.best_params.items():
        print(f"  {key} = {value:.4f}")
