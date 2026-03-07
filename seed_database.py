import sqlite3
import random
import uuid
import hashlib  # 用于生成简单的密码哈希

# ============================================================
# 数据填充脚本 (seed.py)
# 用于生成模拟用户和反馈数据，填充到数据库中。
# 目的：
#   1. 创建足够数量的用户（600个），每个用户有一条反馈记录。
#   2. 反馈数据包含不同模式（OFFICE/REST/ENTERTAINMENT），
#      并引入一些合理的随机性和规律，使数据看起来真实。
#   3. 特意注入一些脏数据（超出正常范围的数值），用于测试数据清洗逻辑。
#   4. 清空旧数据，重新填充。
# ============================================================

DB_PATH = "spine.db"
NUM_USERS = 600  # 生成用户数量，足够每个模式都有一定数据量
MODES = ["OFFICE", "REST", "ENTERTAINMENT"]  # 三种核心模式


def get_password_hash(password: str) -> str:
    """
    简单的密码哈希函数，使用 SHA256 加密（与 security.py 中一致）。
    用于生成模拟用户的密码哈希，实际注册时使用相同的哈希算法。
    """
    return hashlib.sha256(password.encode()).hexdigest()


def seed():
    print("🌱 开始向数据库注入模拟数据 (v1.5 - 全场景版)...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 清空旧数据 (保留表结构)
        cursor.execute("DELETE FROM users")
        cursor.execute("DELETE FROM feedbacks")
        print("🗑️ 已清空旧数据...")

        users = []   # 存储用户数据元组的列表
        feedbacks = []  # 存储反馈数据元组的列表

        for i in range(NUM_USERS):
            # --- 生成用户数据 ---
            user_id = str(uuid.uuid4())                     # 唯一用户 ID
            username = f"user_{i + 1000}"                   # 用户名，避免重复
            password_hash = get_password_hash("123456")     # 所有用户默认密码 123456

            # 身高：正态分布，均值170，标准差8，再裁剪到150-195之间
            height = int(random.normalvariate(170, 8))
            height = max(150, min(195, height))

            # 体重：正态分布，均值70，标准差15，裁剪到40-120之间，保留一位小数
            weight = round(random.normalvariate(70, 15), 1)
            weight = max(40, min(120, weight))

            users.append((user_id, username, password_hash, height, weight))

            # --- 为每个用户生成一条反馈数据（随机分配一种模式）---
            mode = random.choice(MODES)

            # 1. 高度偏好：不同模式对应不同的基础因子
            #    OFFICE: 0.27左右，REST: 0.25左右，ENTERTAINMENT: 0.26左右
            if mode == "OFFICE":
                height_factor = random.uniform(0.26, 0.28)
            elif mode == "REST":
                height_factor = random.uniform(0.24, 0.26)
            else:  # ENTERTAINMENT
                height_factor = random.uniform(0.25, 0.27)

            # 计算最终高度：身高(cm) * 因子 * 10（因为因子对应的是身高倍数关系，mm单位）
            final_height = int(height * height_factor * 10)
            final_height = max(350, min(600, final_height))  # 钳位到座椅物理范围

            # 2. 角度偏好：不同模式的基础角度不同
            if mode == "OFFICE":
                base_angle = 95
            elif mode == "REST":
                base_angle = 115
            else:
                base_angle = 105

            # 引入影响因素：身高高的人倾向于更大角度，体重大的人也倾向更大角度
            height_effect = (height - 170) / 5   # 每高出5cm，角度增加1度
            weight_effect = (weight - 70) / 10   # 每重10kg，角度增加1度
            noise = random.uniform(-2, 2)        # 随机波动 ±2 度

            final_angle = int(base_angle + height_effect + weight_effect + noise)
            final_angle = max(90, min(135, final_angle))  # 钳位到安全范围

            feedbacks.append((
                user_id,
                height,
                weight,
                final_height,
                final_angle,
                mode
            ))

        # === 特意注入脏数据，用于测试数据清洗逻辑 ===
        # 脏数据1: 角度过低 (80度)，应被清洗掉
        feedbacks.append((str(uuid.uuid4()), 175, 70, 450, 80, "OFFICE"))
        # 脏数据2: 角度过高 (150度)，应被清洗掉
        feedbacks.append((str(uuid.uuid4()), 175, 70, 450, 150, "REST"))

        # --- 批量插入用户数据 ---
        cursor.executemany(
            "INSERT INTO users (user_id, username, hashed_password, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?)",
            users
        )

        # --- 批量插入反馈数据 ---
        cursor.executemany(
            """
            INSERT INTO feedbacks (user_id, height_cm, weight_kg, final_height_mm, final_angle_deg, current_mode)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            feedbacks
        )

        conn.commit()
        print(f"✅ 成功生成 {len(users)} 个用户和 {len(feedbacks)} 条全场景反馈数据。")

    except sqlite3.Error as e:
        print(f"❌ 数据库操作失败: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    seed()