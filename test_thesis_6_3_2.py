import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 设置中文字体，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # Mac 用户请改为 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False


def run_thesis_test():
    print("开始执行论文 6.3.2 小节测试...")

    # 1. 从数据库读取真实生成的 OFFICE 模式数据
    try:
        conn = sqlite3.connect("spine.db")
        df = pd.read_sql_query("SELECT * FROM feedbacks WHERE current_mode = 'OFFICE'", conn)
        conn.close()
    except Exception as e:
        print(f"❌ 数据库读取失败，请先运行你的 seed 脚本生成数据。错误: {e}")
        return

    # 2. 完全复用你后端的清洗逻辑
    df['ratio'] = df['final_height_mm'] / (df['height_cm'] * 10)
    df_clean = df[(df['ratio'] >= 0.23) & (df['ratio'] <= 0.31)].copy()
    print(f"📦 读取到 OFFICE 模式数据 {len(df)} 条，清洗后剩余 {len(df_clean)} 条。")

    X = df_clean[['height_cm', 'weight_kg']]
    y = df_clean[['final_height_mm', 'final_angle_deg']]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 3. 传统规则预测 (采用你代码中的固定比例)
    # 你的逻辑：高度 = 身高 * 0.27 * 10，角度 = 95
    # ==========================================
    y_pred_trad_height = X_test['height_cm'] * 0.27 * 10
    y_pred_trad_angle = np.full(shape=len(X_test), fill_value=95)

    mae_trad_height = mean_absolute_error(y_test['final_height_mm'], y_pred_trad_height)
    mae_trad_angle = mean_absolute_error(y_test['final_angle_deg'], y_pred_trad_angle)

    # ==========================================
    # 4. AI 模型预测 (复用你后端的 MultiOutputRegressor)
    # ==========================================
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae_ai_height = mean_absolute_error(y_test['final_height_mm'], predictions[:, 0])
    mae_ai_angle = mean_absolute_error(y_test['final_angle_deg'], predictions[:, 1])

    print(f"\n--- 📏 座高预测 MAE (mm) ---")
    print(f"传统规则: {mae_trad_height:.2f} mm")
    print(f"AI 模型:  {mae_ai_height:.2f} mm")
    print(f"\n--- 📐 角度预测 MAE (°) ---")
    print(f"传统规则: {mae_trad_angle:.2f}°")
    print(f"AI 模型:  {mae_ai_angle:.2f}°")

    # ==========================================
    # 5. 画图表保存
    # ==========================================
    labels = ['座高误差 (mm)', '角度误差 (°)']
    trad_maes = [mae_trad_height, mae_trad_angle]
    ai_maes = [mae_ai_height, mae_ai_angle]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, trad_maes, width, label='传统规则 (身高×0.27 / 固定95°)', color='#d62728', alpha=0.8)
    rects2 = ax.bar(x + width / 2, ai_maes, width, label='AI 多输出回归模型', color='#1f77b4', alpha=0.8)

    ax.set_ylabel('平均绝对误差 (MAE)')
    ax.set_title('传统规则与 AI 模型个性化推荐误差对比 (OFFICE场景)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    plt.tight_layout()
    plt.savefig('figure6-2_mae_comparison.png', dpi=300)
    print("\n✅ 已生成图表: figure6-2_mae_comparison.png")


if __name__ == "__main__":
    run_thesis_test()
