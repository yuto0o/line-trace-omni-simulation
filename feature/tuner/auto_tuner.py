import math
import time

import numpy as np

# ==========================================
# 1. パラメータ設定
# ==========================================
DISK_RADIUS = 80.0
LINE_WIDTH = 20.0
BASE_SPEED = 40.0
NUM_SENSORS = 6
DT = 0.1
SPIN_OMEGA = 1.0

# 制御の前提パラメータ
HEADING_BLEND = 0.249
LOST_THRESHOLD_STEPS = int((math.pi / 2.0) / (SPIN_OMEGA * DT))
FORCE_LEFT_SPEED = math.radians(60.0)
I_THRESHOLD = LINE_WIDTH


# ==========================================
# 2. 自動評価用コースの生成 (直角2つ + カーブ)
# ==========================================
def generate_eval_course():
    points = []
    # スタートから直線 (北へ)
    for y in range(0, 300, 5):
        points.append([0.0, float(y)])
    # 90度 右ターン (東へ)
    for x in range(0, 300, 5):
        points.append([float(x), 300.0])
    # 90度 左ターン (北へ)
    for y in range(300, 600, 5):
        points.append([300.0, float(y)])
    # 緩やかな左カーブ (円弧)
    for theta in np.linspace(0, math.pi / 2, 60):
        points.append(
            [300.0 - 200.0 * (1 - math.cos(theta)), 600.0 + 200.0 * math.sin(theta)]
        )
    return np.array(points)


# ==========================================
# 3. 描画なしの軽量版ロボットクラス (滑らか修正版)
# ==========================================
class HeadlessOmnibot:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.local_heading = np.array([0.0, 1.0])
        self.latest_front_local = np.array([0.0, DISK_RADIUS])
        self.latest_back_local = np.array([0.0, -DISK_RADIUS])

        self.prev_err = 0.0
        self.err_integral = 0.0
        self.front_lost_steps = 0
        self.sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)

        # オートチューナーの「ガクガク評価」用
        self.prev_correction_speed = 0.0

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
            # 【修正点】常に滑らかにブレンド
            self.local_heading = (
                1.0 - HEADING_BLEND
            ) * self.local_heading + HEADING_BLEND * target_heading
            self.local_heading /= np.linalg.norm(self.local_heading)

        # 4. 横ズレに対するPID制御
        lateral_dir = np.array([-self.local_heading[1], self.local_heading[0]])

        # 【修正点】センサーが外れても前回のズレを維持 (微分キック防止)
        err_val = self.prev_err
        if len(active_local_positions) > 0:
            line_center = np.mean(active_local_positions, axis=0)
            err_val = np.dot(line_center, lateral_dir)

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

        correction_speed = (
            (self.kp * err_val) + (self.ki * self.err_integral) + (self.kd * d_err_val)
        )

        v_forward = self.local_heading * BASE_SPEED
        v_lateral = lateral_dir * correction_speed
        v_local = v_forward + v_lateral

        return v_local, SPIN_OMEGA, is_force_left, err_val, correction_speed


# ==========================================
# 4. 評価関数 (1回の走行テスト)
# ==========================================
def evaluate_gains(kp, ki, kd, course):
    robot = HeadlessOmnibot(kp, ki, kd)
    true_robot_pos = np.array([course[0][0], course[0][1]])
    true_robot_theta = 0.0

    total_cost = 0.0
    max_steps = 1500
    goal_pos = course[-1]

    for step in range(max_steps):
        # センサー判定
        active_indices = []
        for i, ang in enumerate(robot.sensor_angles):
            g_ang = ang + true_robot_theta
            s_pos = true_robot_pos + np.array(
                [DISK_RADIUS * math.cos(g_ang), DISK_RADIUS * math.sin(g_ang)]
            )
            distances = np.linalg.norm(course - s_pos, axis=1)
            if np.min(distances) < (LINE_WIDTH / 2.0):
                active_indices.append(i)

        # ロボット思考
        v_local, omega, is_force_left, current_err, correction_speed = robot.update(
            active_indices, DT
        )

        # 【重要】コスト計算：ズレの絶対値 ＋ ガクガク度合い（速度の急激な変化）へのペナルティ
        jerk_penalty = abs(correction_speed - robot.prev_correction_speed)
        robot.prev_correction_speed = correction_speed

        # ガクガク度合いを0.5倍の重みでスコアに悪影響として加算する
        total_cost += abs(current_err) + (jerk_penalty * 0.5)

        # 物理更新
        true_robot_theta += omega * DT
        v_world_x = v_local[0] * math.cos(true_robot_theta) - v_local[1] * math.sin(
            true_robot_theta
        )
        v_world_y = v_local[0] * math.sin(true_robot_theta) + v_local[1] * math.cos(
            true_robot_theta
        )
        true_robot_pos += np.array([v_world_x, v_world_y]) * DT

        # 失敗判定: ロストした（強制左旋回が発動した）ら失格
        if is_force_left:
            return float("inf")

        # 成功判定: ゴールに十分近づいたらクリア
        if np.linalg.norm(true_robot_pos - goal_pos) < 60.0:
            break

    return total_cost


# ==========================================
# 5. 自動探索ロジック (メイン処理)
# ==========================================
def main():
    course = generate_eval_course()
    print("=== PIDパラメータ自動探索を開始します ===")
    print(f"コース: 直角ターン2箇所 + カーブ (ベース速度: {BASE_SPEED})\n")

    start_time = time.time()

    best_kp = 0.0
    best_kd = 0.0
    best_ki = 0.0

    # ------------------------------------------------
    # Phase 1: Pゲインの探索 (D=0, I=0)
    # ------------------------------------------------
    print("[Phase 1] Pゲイン(Kp)の最適値を探しています...")
    min_cost = float("inf")
    for p in np.arange(0.1, 2.0, 0.1):
        cost = evaluate_gains(p, 0.0, 0.0, course)
        if cost < min_cost:
            min_cost = cost
            best_kp = p
    print(f" -> Phase 1 完了: 暫定 Kp = {best_kp:.2f} (スコア: {min_cost:.1f})\n")

    # ------------------------------------------------
    # Phase 2: Dゲインの探索 (I=0)
    # ------------------------------------------------
    print("[Phase 2] Dゲイン(Kd)によるブレーキ調整を探しています...")
    min_cost = float("inf")
    for d in np.arange(0.0, 0.6, 0.02):
        cost = evaluate_gains(best_kp, 0.0, d, course)
        if cost < min_cost:
            min_cost = cost
            best_kd = d
    print(f" -> Phase 2 完了: 暫定 Kd = {best_kd:.2f} (スコア: {min_cost:.1f})\n")

    # ------------------------------------------------
    # Phase 3: Iゲインの探索
    # ------------------------------------------------
    print("[Phase 3] Iゲイン(Ki)による微調整を探しています...")
    min_cost = float("inf")
    for i in np.arange(0.0, 0.5, 0.02):
        cost = evaluate_gains(best_kp, i, best_kd, course)
        if cost < min_cost:
            min_cost = cost
            best_ki = i
    print(f" -> Phase 3 完了: 最終 Ki = {best_ki:.2f} (スコア: {min_cost:.1f})\n")

    elapsed_time = time.time() - start_time
    print("============================================")
    print("🎉 探索完了！")
    print(f"所要時間: {elapsed_time:.1f} 秒")
    print("おすすめのパラメータ設定:")
    print(f"  KP = {best_kp:.2f}")
    print(f"  KD = {best_kd:.2f}")
    print(f"  KI = {best_ki:.2f}")
    print("============================================")


if __name__ == "__main__":
    main()
