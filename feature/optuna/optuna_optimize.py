import math

import numpy as np
import optuna

# ==========================================
# 1. 固定パラメータ設定
# ==========================================
DISK_RADIUS = 40.0
LINE_WIDTH = 20.0
BASE_SPEED = 40.0
NUM_SENSORS = 6
DT = 0.1
MAX_LOST_STEPS = 200
MAX_SIM_STEPS = 1500  # 1回のシミュレーションの最大ステップ数


# ==========================================
# 2. コース生成関数 (変更なし)
# ==========================================
def generate_course(type_id):
    points = []
    if type_id == 4:
        for y in range(0, 800, 5):
            points.append([0, y])
        for x in range(5, 800, 5):
            points.append([x, 800])
    elif type_id == 5:
        for y in range(0, 800, 5):
            points.append([0, y])
        for x in range(5, 800, 5):
            points.append([x, 800])
        for y in range(795, -1, -5):
            points.append([800, y])
        for x in range(795, -1, -5):
            points.append([x, 0])
    elif type_id == 6:
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
            if theta == math.pi:
                continue
            points.append([400 + 300 * math.cos(theta), 1200 + 300 * math.sin(theta)])
        for y in range(1200, -200, -5):
            points.append([700 + 150 * math.sin((1200 - y) / 120.0), y])
    return np.array(points)


# ==========================================
# 3. ヘッドレスシミュレーション（描画なし・評価スコアを返す）
# ==========================================
def run_headless_simulation(
    course_type, kp_trans, kd_heading, blend_both, blend_front, blend_back
):
    path = generate_course(course_type)
    robot_pos = np.array([path[0][0], path[0][1]])
    current_heading = np.array([0.0, 1.0])
    last_valid_heading = np.array([0.0, 1.0])
    previous_line_vector = np.array([0.0, 1.0])
    lost_counter = 0

    sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)
    sensor_local_pos = np.array(
        [[DISK_RADIUS * math.cos(a), DISK_RADIUS * math.sin(a)] for a in sensor_angles]
    )

    total_error = 0.0  # これが少ないほど優秀（ふらついていない）

    for step in range(MAX_SIM_STEPS):
        sensor_global_pos = robot_pos + sensor_local_pos
        active_sensors = []
        for s_pos in sensor_global_pos:
            distances = np.linalg.norm(path - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_sensors.append(s_pos)

        if len(active_sensors) > 0:
            lost_counter = 0
            active_sensors = np.array(active_sensors)
            front_sensors, back_sensors = [], []
            for s_pos in active_sensors:
                if np.dot(s_pos - robot_pos, current_heading) > 0:
                    front_sensors.append(s_pos)
                else:
                    back_sensors.append(s_pos)

            line_vector = np.copy(current_heading)
            blend_weight = 0.0

            if len(front_sensors) > 0 and len(back_sensors) > 0:
                line_vector = np.mean(front_sensors, axis=0) - np.mean(
                    back_sensors, axis=0
                )
                blend_weight = blend_both
            elif len(front_sensors) > 0:
                line_vector = np.mean(front_sensors, axis=0) - robot_pos
                blend_weight = blend_front
            elif len(back_sensors) > 0:
                line_vector = robot_pos - np.mean(back_sensors, axis=0)
                blend_weight = blend_back

            if blend_weight > 0:
                line_vector = line_vector / np.linalg.norm(line_vector)
                diff_vector = line_vector - previous_line_vector
                current_heading = (
                    (1.0 - blend_weight) * current_heading
                    + blend_weight * line_vector
                    + kd_heading * diff_vector
                )
                current_heading = current_heading / np.linalg.norm(current_heading)
                previous_line_vector = line_vector.copy()

            line_center = np.mean(active_sensors, axis=0)
            raw_error = line_center - robot_pos
            lateral_error = raw_error - (
                np.dot(raw_error, current_heading) * current_heading
            )

            # 【評価】横ズレの大きさをペナルティとして加算
            total_error += np.linalg.norm(lateral_error)

            velocity = (current_heading * BASE_SPEED) + (lateral_error * kp_trans)
            robot_pos = robot_pos + velocity * DT
            last_valid_heading = current_heading.copy()

        else:
            lost_counter += 1
            if lost_counter < MAX_LOST_STEPS:
                robot_pos = robot_pos + (last_valid_heading * BASE_SPEED) * DT
                current_heading = last_valid_heading.copy()
            else:
                # 【評価】完全にコースアウトした場合は、残りのステップ数に応じた超巨大ペナルティ
                return total_error + (MAX_SIM_STEPS - step) * 5000.0

    return total_error


# ==========================================
# 4. Optunaの目的関数（スコア計算）
# ==========================================
def objective(trial):
    # ① Optunaに探索させるパラメータの範囲を定義
    kp_trans = trial.suggest_float("kp_trans", 0.05, 1.0)
    kd_heading = trial.suggest_float("kd_heading", 0.0, 2.0)
    blend_both = trial.suggest_float("blend_both", 0.1, 0.8)
    blend_front = trial.suggest_float("blend_front", 0.05, 0.5)
    blend_back = trial.suggest_float("blend_back", 0.0, 0.3)

    total_penalty = 0.0
    # ② 難関コース（直角、正方形、ボスステージ）を走らせる
    test_courses = [4, 5, 6]

    for c_type in test_courses:
        penalty = run_headless_simulation(
            course_type=c_type,
            kp_trans=kp_trans,
            kd_heading=kd_heading,
            blend_both=blend_both,
            blend_front=blend_front,
            blend_back=blend_back,
        )
        total_penalty += penalty

    # ペナルティ（横ズレ＋コースアウト）の合計を返す。Optunaはこれを「最小化」しようとする。
    return total_penalty


# ==========================================
# 5. 最適化の実行
# ==========================================
if __name__ == "__main__":
    print("最適化を開始します... (数分かかる場合があります)")

    # ペナルティを最小化(minimize)する方向で学習
    study = optuna.create_study(direction="minimize")

    # 500パターンの組み合わせを自動テスト
    study.optimize(objective, n_trials=500)

    print("\n==================================")
    print("🎉 最適化完了！最強のパラメータが決定しました 🎉")
    print("==================================")
    print(f"最小ペナルティスコア: {study.best_value}")
    print("【適用すべきパラメータ】")
    for key, value in study.best_params.items():
        print(f"  {key.upper()} = {value:.4f}")
