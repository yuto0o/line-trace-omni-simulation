import math

import numpy as np
import optuna

# ログがうるさくならないようにOptunaの標準出力をWarningのみに制限
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    return np.array(points)


# ==========================================
# 3. 描画なしのPD制御シミュレーション
# ==========================================
def run_simulation(spin_omega, kp, kd, heading_blend, disk_radius, path):
    robot_pos = np.array([path[0][0], path[0][1]])
    current_heading = np.array([0.0, 1.0])
    robot_theta = 0.0

    latest_front_pos = robot_pos + current_heading * disk_radius
    latest_back_pos = robot_pos - current_heading * disk_radius

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

    prev_err_val = 0.0
    total_lateral_error = 0.0
    max_path_index = 0
    lost_counter = 0
    survived_steps = 0
    reverse_counter = 0
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

        # 進捗計算と逆走検知
        dist_to_path_nodes = np.linalg.norm(path - robot_pos, axis=1)
        closest_path_idx = np.argmin(dist_to_path_nodes)
        if closest_path_idx > max_path_index:
            max_path_index = closest_path_idx

        next_idx = min(closest_path_idx + 2, len(path) - 1)
        path_forward_vec = path[next_idx] - path[closest_path_idx]
        if np.linalg.norm(path_forward_vec) > 1e-3:
            path_forward_vec = path_forward_vec / np.linalg.norm(path_forward_vec)
            if np.dot(current_heading, path_forward_vec) < -0.3:
                reverse_counter += 1
            else:
                reverse_counter = 0

        if reverse_counter > 10:
            is_reversed = True
            break

        # 進行方向の更新
        target_heading = latest_front_pos - latest_back_pos
        if np.linalg.norm(target_heading) > 1e-3:
            target_heading = target_heading / np.linalg.norm(target_heading)
            current_heading = (
                1.0 - heading_blend
            ) * current_heading + heading_blend * target_heading
            current_heading = current_heading / np.linalg.norm(current_heading)

        lateral_dir = np.array([-current_heading[1], current_heading[0]])
        err_val = 0.0

        if len(active_sensors) > 0:
            lost_counter = 0
            survived_steps += 1

            line_center = np.mean(active_sensors, axis=0)
            raw_err_vec = line_center - robot_pos
            err_val = np.dot(raw_err_vec, lateral_dir)
            total_lateral_error += abs(err_val)
        else:
            lost_counter += 1
            if lost_counter > MAX_LOST_STEPS:
                break

        if max_path_index >= len(path) - 5:
            break

        # PD制御計算
        d_err_val = (err_val - prev_err_val) / DT
        prev_err_val = err_val

        correction_speed = (kp * err_val) + (kd * d_err_val)

        velocity_forward = current_heading * BASE_SPEED
        velocity_lateral = lateral_dir * correction_speed
        velocity = velocity_forward + velocity_lateral

        robot_pos = robot_pos + velocity * DT

    # スコア計算（逆走厳罰・追従精度重視）
    progress_penalty = (len(path) - max_path_index) * 10000
    reverse_penalty = 500000 if is_reversed else 0
    avg_error = total_lateral_error / max(1, survived_steps)
    error_penalty = avg_error * 1000

    score = progress_penalty + reverse_penalty + error_penalty + survived_steps
    return score


# ==========================================
# 4. Optuna目的関数 (DISK_RADIUS 固定版)
# ==========================================
def objective_fixed(trial):
    spin_omega = trial.suggest_float("SPIN_OMEGA", 0.5, 10.0)
    kp = trial.suggest_float("KP", 0.1, 5.0)
    kd = trial.suggest_float("KD", 0.0, 5.0)
    heading_blend = trial.suggest_float("HEADING_BLEND", 0.01, 0.99)
    disk_radius = 80.0  # 固定

    path = generate_course(COURSE_TYPE)
    return run_simulation(spin_omega, kp, kd, heading_blend, disk_radius, path)


# ==========================================
# 5. Optuna目的関数 (DISK_RADIUS 変動版)
# ==========================================
def objective_variable(trial):
    spin_omega = trial.suggest_float("SPIN_OMEGA", 0.5, 10.0)
    kp = trial.suggest_float("KP", 0.1, 5.0)
    kd = trial.suggest_float("KD", 0.0, 5.0)
    heading_blend = trial.suggest_float("HEADING_BLEND", 0.01, 0.99)
    disk_radius = trial.suggest_float("DISK_RADIUS", 20.0, 100.0)  # 変動

    path = generate_course(COURSE_TYPE)
    return run_simulation(spin_omega, kp, kd, heading_blend, disk_radius, path)


# ==========================================
# 6. メイン処理
# ==========================================
if __name__ == "__main__":
    n_trials = 200

    print(f"--- [1/2] DISK_RADIUS=80.0 固定での最適化を開始 ({n_trials} trials) ---")
    study_fixed = optuna.create_study(direction="minimize")
    study_fixed.optimize(objective_fixed, n_trials=n_trials)

    print(f"--- [2/2] DISK_RADIUS 変動での最適化を開始 ({n_trials} trials) ---")
    study_variable = optuna.create_study(direction="minimize")
    study_variable.optimize(objective_variable, n_trials=n_trials)

    # 最終結果の比較出力
    print("\n=============================================")
    print("                 最適化結果                  ")
    print("=============================================")

    print("\n【パターンA】DISK_RADIUS = 80 固定")
    print(f"ベストスコア: {study_fixed.best_value:.2f}")
    for key, value in study_fixed.best_params.items():
        print(f"  {key} = {value:.4f}")
    print("  DISK_RADIUS = 80.0000 (Fixed)")

    print("\n【パターンB】DISK_RADIUS も最適化")
    print(f"ベストスコア: {study_variable.best_value:.2f}")
    for key, value in study_variable.best_params.items():
        print(f"  {key} = {value:.4f}")
    print("=============================================")
