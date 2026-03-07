import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

# ============================================================
# AI 模型训练脚本
# 从 feedbacks 表中读取用户反馈数据，为 OFFICE、REST、ENTERTAINMENT
# 三种场景分别训练线性回归模型（多输出：座高和角度）。
# 训练好的模型保存为 model_office.pkl、model_rest.pkl、model_entertainment.pkl，
# 供 ergonomics.py 中的推荐引擎加载使用。
# ============================================================

# 数据库文件路径
DB_PATH = "spine.db"

# 需要训练的核心模式列表
MODES_TO_TRAIN = ["OFFICE", "REST", "ENTERTAINMENT"]


def train_single_model(target_mode: str, all_data: pd.DataFrame):
    """
    为单个模式训练并保存一个模型。

    参数:
        target_mode: 要训练的模式名称，如 "OFFICE"
        all_data: 包含所有反馈数据的 DataFrame，必须包含字段：
            current_mode, height_cm, weight_kg, final_height_mm, final_angle_deg
    """
    print(f"\n{'=' * 20} 模式: {target_mode} {'=' * 20}")

    # --- 1. 按模式筛选数据 ---
    df_mode = all_data[all_data['current_mode'] == target_mode]
    print(f"📦 模式 '{target_mode}' 原始数据: {len(df_mode)} 条")
    if df_mode.empty:
        print("🤷 数据为空，跳过此模式。")
        return

    # --- 2. 数据清洗：剔除不合理的数据 ---
    # 人体测量值和座椅物理范围限制
    df_clean = df_mode[
        (df_mode['height_cm'] >= 100) & (df_mode['height_cm'] <= 230) &
        (df_mode['weight_kg'] >= 30) & (df_mode['weight_kg'] <= 150) &
        (df_mode['final_height_mm'] >= 350) & (df_mode['final_height_mm'] <= 600) &
        (df_mode['final_angle_deg'] >= 90) & (df_mode['final_angle_deg'] <= 135)
    ].copy()

    # 额外的合理性检查：座高与身高的比例应在合理范围内
    # 座高(mm) = 身高(cm) * 10 * 比例因子。正常比例大约 0.23 ~ 0.31
    df_clean['ratio'] = df_clean['final_height_mm'] / (df_clean['height_cm'] * 10)
    df_clean = df_clean[(df_clean['ratio'] >= 0.23) & (df_clean['ratio'] <= 0.31)]

    print(f"🧹 清洗后保留: {len(df_clean)} 条")

    # 如果清洗后数据太少，无法训练出可靠的模型
    if len(df_clean) < 10:
        print("⚠️ 有效数据不足 (<10)，无法训练此模式的模型。")
        return

    # --- 3. 准备训练集和测试集 ---
    # 特征：身高(cm)、体重(kg)
    X = df_clean[['height_cm', 'weight_kg']]
    # 目标值：最终座高(mm)、最终角度(°)
    y = df_clean[['final_height_mm', 'final_angle_deg']]

    # 划分训练集和测试集，80% 训练，20% 测试，固定随机种子确保可重现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. 训练模型 ---
    # 使用 MultiOutputRegressor 包装 LinearRegression，支持多输出（两个目标值）
    # 内部会对每个目标分别训练一个线性回归模型
    multi_target_model = MultiOutputRegressor(estimator=LinearRegression())
    multi_target_model.fit(X_train, y_train)

    # --- 5. 评估模型 ---
    predictions = multi_target_model.predict(X_test)
    # 分别计算座高和角度的 R² 分数（决定系数），衡量模型拟合优度
    r2_height = r2_score(y_test['final_height_mm'], predictions[:, 0])
    r2_angle = r2_score(y_test['final_angle_deg'], predictions[:, 1])

    print(f"📊 模型评分 (R2): 高度={r2_height:.2f}, 角度={r2_angle:.2f}")

    # --- 6. 保存模型 ---
    # 模型文件名根据模式小写命名，如 model_office.pkl
    model_file = f"model_{target_mode.lower()}.pkl"
    joblib.dump(multi_target_model, model_file)
    print(f"💾 模型已保存至: {model_file}")


def main():
    """
    主函数：连接数据库，读取所有反馈数据，遍历 MODES_TO_TRAIN 进行训练。
    """
    print("🤖 开始分场景模型批量训练...")
    if not os.path.exists(DB_PATH):
        print("❌ 错误: 数据库文件不存在。")
        return

    # 从数据库读取所有反馈数据
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT current_mode, height_cm, weight_kg, final_height_mm, final_angle_deg FROM feedbacks"
    all_feedback_data = pd.read_sql_query(query, conn)
    conn.close()

    # 获取数据库中实际存在的模式（用于提示）
    available_modes = all_feedback_data['current_mode'].unique()
    print(f"🔍 发现数据库中存在以下模式: {available_modes}")

    # 对每个需要训练的模式进行训练（如果数据存在）
    for mode in MODES_TO_TRAIN:
        if mode in available_modes:
            train_single_model(mode, all_feedback_data)
        else:
            print(f"\n{'=' * 20} 模式: {mode} {'=' * 20}")
            print(f"🤷 数据库中无 '{mode}' 模式的数据，跳过。")

    print("\n✅ 所有模式训练完成！")


if __name__ == "__main__":
    main()