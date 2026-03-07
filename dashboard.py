import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import plotly.express as px

#streamlit run dashboard.py

# === 配置页面 ===
st.set_page_config(page_title="智脊助手 AI 数据看板", layout="wide")

st.title("🧠 智脊助手 v1.5 - 全场景数据监控中心")

# === 1. 连接数据库 ===
DB_PATH = "spine.db"


def get_data(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return pd.DataFrame()


# === 2. 侧边栏控制 ===
st.sidebar.header("🕹️ 控制面板")
selected_mode = st.sidebar.selectbox(
    "选择分析场景",
    ["OFFICE", "REST", "ENTERTAINMENT"],
    index=0
)

# === 3. 核心指标 (KPI) ===
col1, col2, col3, col4 = st.columns(4)

df_users = get_data("SELECT * FROM users")
# 🔥 核心修改：只筛选当前模式的反馈数据进行分析
df_feedback = get_data(f"SELECT * FROM feedbacks WHERE current_mode = '{selected_mode}'")

with col1:
    st.metric(label="总注册用户", value=len(df_users))
with col2:
    st.metric(label=f"{selected_mode} 样本数", value=len(df_feedback))
with col3:
    avg_height = df_users['height_cm'].mean() if not df_users.empty else 0
    st.metric(label="用户平均身高", value=f"{avg_height:.1f} cm")
with col4:
    avg_weight = df_users['weight_kg'].mean() if not df_users.empty else 0
    st.metric(label="用户平均体重", value=f"{avg_weight:.1f} kg")

st.markdown("---")

# === 4. AI 效能分析 (修复 Bug) ===
st.subheader(f"🤖 AI 效能分析: {selected_mode} 模式")

model_path = f"model_{selected_mode.lower()}.pkl"

if not df_feedback.empty and os.path.exists(model_path):
    try:
        # 1. 加载对应模式的模型
        model = joblib.load(model_path)
        analysis_df = df_feedback.copy()

        # 2. 准备输入特征 (🔥 修复: 必须包含 height 和 weight)
        X_input = analysis_df[['height_cm', 'weight_kg']].values

        # 3. AI 预测 (输出包含 [height, angle])
        predictions = model.predict(X_input)
        analysis_df['AI_Height'] = predictions[:, 0]
        analysis_df['AI_Angle'] = predictions[:, 1]

        # 4. 传统规则计算 (用于对比)
        if selected_mode == "OFFICE":
            analysis_df['Rule_Height'] = analysis_df['height_cm'] * 0.27 * 10
            analysis_df['Rule_Angle'] = 95
        elif selected_mode == "REST":
            analysis_df['Rule_Height'] = analysis_df['height_cm'] * 0.25 * 10
            analysis_df['Rule_Angle'] = 115
        else:  # ENTERTAINMENT
            analysis_df['Rule_Height'] = analysis_df['height_cm'] * 0.26 * 10
            analysis_df['Rule_Angle'] = 105

        # 5. 计算误差 (MAE)
        analysis_df['Rule_Err_H'] = abs(analysis_df['final_height_mm'] - analysis_df['Rule_Height'])
        analysis_df['AI_Err_H'] = abs(analysis_df['final_height_mm'] - analysis_df['AI_Height'])

        analysis_df['Rule_Err_A'] = abs(analysis_df['final_angle_deg'] - analysis_df['Rule_Angle'])
        analysis_df['AI_Err_A'] = abs(analysis_df['final_angle_deg'] - analysis_df['AI_Angle'])

        # === 可视化展示 ===
        tab1, tab2 = st.tabs(["📏 座高预测效能", "📐 角度预测效能"])

        with tab1:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("##### 平均误差 (mm)")
                err_data = pd.DataFrame({
                    '方法': ['传统公式', 'AI 模型'],
                    '误差': [analysis_df['Rule_Err_H'].mean(), analysis_df['AI_Err_H'].mean()]
                })
                st.bar_chart(err_data.set_index('方法'), color="#2196F3")

                imp = analysis_df['Rule_Err_H'].mean() - analysis_df['AI_Err_H'].mean()
                if imp > 0:
                    st.success(f"AI 在座高上更准 **{imp:.1f} mm**")

            with c2:
                st.markdown("##### 真实值 vs 预测值分布")
                fig = px.scatter(analysis_df, x='height_cm', y='final_height_mm',
                                 color='weight_kg',
                                 title="散点: 真实偏好 | 线: AI 预测趋势",
                                 labels={'final_height_mm': '座高 (mm)', 'height_cm': '身高 (cm)'})

                # 绘制 AI 趋势线 (取平均体重)
                mean_weight = analysis_df['weight_kg'].mean()
                x_range = np.linspace(analysis_df['height_cm'].min(), analysis_df['height_cm'].max(), 50)
                # 构造预测输入: 变化的 height + 固定的 mean_weight
                X_trend = np.column_stack((x_range, np.full_like(x_range, mean_weight)))
                y_trend = model.predict(X_trend)

                fig.add_scatter(x=x_range, y=y_trend[:, 0], mode='lines', name=f'AI 趋势 (体重{int(mean_weight)}kg)',
                                line=dict(color='red', width=3))
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("##### 平均误差 (度)")
                err_data_a = pd.DataFrame({
                    '方法': ['固定角度', 'AI 模型'],
                    '误差': [analysis_df['Rule_Err_A'].mean(), analysis_df['AI_Err_A'].mean()]
                })
                st.bar_chart(err_data_a.set_index('方法'), color="#FF9800")

                imp_a = analysis_df['Rule_Err_A'].mean() - analysis_df['AI_Err_A'].mean()
                if imp_a > 0:
                    st.success(f"AI 在角度上更准 **{imp_a:.1f}°**")

            with c2:
                st.markdown("##### 体重对角度的影响")
                fig2 = px.scatter(analysis_df, x='weight_kg', y='final_angle_deg',
                                  trendline="ols",
                                  title="体重越重，倾向角度越大？",
                                  labels={'final_angle_deg': '偏好角度 (°)', 'weight_kg': '体重 (kg)'})
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"分析过程中发生错误: {e}")
        st.code(str(e))  # 打印详细堆栈
else:
    if df_feedback.empty:
        st.warning(f"⚠️ {selected_mode} 模式暂无反馈数据，无法分析。")
    else:
        st.error(f"❌ 模型文件 {model_path} 缺失，请先运行 train_model.py。")

st.markdown("---")

# === 5. 用户画像 (保持不变) ===
st.subheader("👥 用户画像分析")
c1, c2 = st.columns(2)
with c1:
    if not df_users.empty:
        st.markdown("**身高分布**")
        fig, ax = plt.subplots()
        df_users['height_cm'].hist(bins=15, ax=ax, color='skyblue', edgecolor='black', grid=False)
        ax.set_xlabel("Height (cm)")
        st.pyplot(fig)
with c2:
    if not df_users.empty:
        st.markdown("**体重分布**")
        fig, ax = plt.subplots()
        df_users['weight_kg'].hist(bins=15, ax=ax, color='lightgreen', edgecolor='black', grid=False)
        ax.set_xlabel("Weight (kg)")
        st.pyplot(fig)
