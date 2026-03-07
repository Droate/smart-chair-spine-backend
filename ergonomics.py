import os
import joblib
import numpy as np
from models import UserProfile, ErgoRecommendation, SingleModeRecommendation, MultiSceneRecommendation

# ============================================================
# 核心推荐引擎模块
# 负责加载 AI 模型，并根据用户身体数据和当前场景计算推荐值。
# 实现了 AI 预测优先、规则回退的混合推荐策略。
# ============================================================

# 模型缓存字典，键为模式小写字符串（如 "office"），值为加载的模型对象或 None（加载失败）
_loaded_models = {}


def load_model(mode: str, force_reload: bool = False):
    """
    动态加载指定模式的 AI 模型，并进行缓存。
    如果模型已缓存且不强制刷新，直接返回缓存。
    如果模型文件不存在或加载失败，缓存 None 并返回 None。

    :param mode: 模式名称，例如 "OFFICE"（大小写不敏感）
    :param force_reload: 是否强制从磁盘重新加载，用于热更新（当后台训练完成后调用）
    :return: 加载的 scikit-learn 模型对象，或 None（如果模型不可用）
    """
    mode_key = mode.lower()  # 统一小写作为缓存键，避免大小写混乱

    # 如果已缓存且不强制刷新，直接返回缓存
    if not force_reload and mode_key in _loaded_models:
        return _loaded_models[mode_key]

    model_file = f"model_{mode_key}.pkl"  # 构造文件名，例如 "model_office.pkl"
    if os.path.exists(model_file):
        try:
            model = joblib.load(model_file)  # 使用 joblib 加载模型（支持 scikit-learn 对象）
            if force_reload:
                print(f"🔄 模型 '{model_file}' 已热重载！")
            else:
                print(f"✅ AI 模型 '{model_file}' 加载成功！")
            _loaded_models[mode_key] = model  # 更新缓存
            return model
        except Exception as e:
            print(f"⚠️ 模型 '{model_file}' 加载失败: {e}")
            _loaded_models[mode_key] = None  # 缓存失败状态，避免重复尝试加载（除非强制重载）
            return None
    else:
        # 模型文件不存在，缓存 None 并返回 None
        _loaded_models[mode_key] = None
        return None


def calculate_settings(profile: UserProfile) -> ErgoRecommendation:
    """
    智能决策引擎：根据用户的身体档案和当前模式，计算推荐的高度和角度。
    策略：优先使用 AI 模型预测，如果 AI 不可用，则回退到基于规则的固定公式。

    :param profile: 用户身体档案，包含身高、体重、当前模式等
    :return: ErgoRecommendation 对象，包含推荐值及理由
    """
    current_mode = profile.current_mode.upper()  # 确保模式为大写，便于比较
    height = profile.height_cm
    weight = profile.weight_kg
    recommended_height_mm = 0
    recommended_angle_deg = 0
    reason_str = ""

    # --- 分支 A: 动态加载并执行 AI 预测 ---
    model = load_model(current_mode)  # 尝试加载对应模式的模型

    if model:
        try:
            # 构造输入特征：身高、体重，形状为 (1, 2)
            X_input = np.array([[height, weight]])
            # 模型预测输出两个值：座高和角度
            prediction = model.predict(X_input)
            recommended_height_mm = int(prediction[0][0])  # 第一列是座高
            recommended_angle_deg = int(prediction[0][1])  # 第二列是角度
            reason_str = f"✨ AI 为「{current_mode}」模式个性化推荐"
            print(f"🧠 AI 预测 ({current_mode}): 座高={recommended_height_mm}mm, 角度={recommended_angle_deg}°")
        except Exception as e:
            # 预测过程中发生任何异常（如模型格式不对），打印日志并回退到规则
            print(f"- 模式 '{current_mode}' 预测出错，回退至专属规则: {e}")
            model = None  # 标记为无 AI，以便进入回退分支

    # --- 分支 B: 专属规则计算 (Fallback) ---
    if not model:
        reason_str = f"📏 基于「{current_mode}」模式的预设规则"
        if current_mode == "OFFICE":
            # 办公模式：座高 = 身高 * 0.27 * 10，角度基础 95°，体重大于85kg时角度+5°
            recommended_height_mm = int(height * 0.27 * 10)
            recommended_angle_deg = 95 if weight < 85 else 100
        elif current_mode == "REST":
            # 休息模式：座高稍低，角度稍大
            recommended_height_mm = int(height * 0.25 * 10)
            recommended_angle_deg = 115 if weight < 85 else 120
        elif current_mode == "ENTERTAINMENT":
            # 娱乐模式：座高适中，角度中等
            recommended_height_mm = int(height * 0.26 * 10)
            recommended_angle_deg = 105 if weight < 85 else 110
        else:
            # 未知模式，默认使用办公模式规则（安全起见）
            recommended_height_mm = int(height * 0.27 * 10)
            recommended_angle_deg = 100
        print(f"- 规则计算 ({current_mode}): 座高={recommended_height_mm}mm, 角度={recommended_angle_deg}°")

    # --- 通用后处理 (安全限制) ---
    # 确保推荐值在物理安全范围内：高度 350-600mm，角度 90-135°
    clamped_height = max(350, min(600, recommended_height_mm))
    clamped_angle = max(90, min(135, recommended_angle_deg))

    return ErgoRecommendation(
        recommended_height_mm=clamped_height,
        recommended_angle_deg=clamped_angle,
        reason=reason_str
    )


def calculate_all_modes(profile: UserProfile) -> MultiSceneRecommendation:
    """
    一次性计算 OFFICE, REST, ENTERTAINMENT 三种模式的推荐值。
    为每种模式创建一个新的 UserProfile 副本（仅修改 current_mode），
    然后调用 calculate_settings，最后将结果打包成 MultiSceneRecommendation。

    :param profile: 用户身体档案（包含身高体重等基本信息，但不包含模式）
    :return: 包含三条推荐结果的多场景推荐对象
    """
    target_modes = ["OFFICE", "REST", "ENTERTAINMENT"]
    results = []

    for mode in target_modes:
        # 为当前模式创建一个新的 UserProfile 对象，避免修改原对象（尽管 UserProfile 是不可变的，但显式复制是良好实践）
        temp_profile = UserProfile(
            user_id=profile.user_id,
            height_cm=profile.height_cm,
            weight_kg=profile.weight_kg,
            current_mode=mode,  # 关键：设置当前模式
            upper_body_ratio=profile.upper_body_ratio,
            thigh_length_cm=profile.thigh_length_cm
        )
        rec = calculate_settings(temp_profile)  # 计算单个模式的推荐
        # 转换为列表项模型
        results.append(SingleModeRecommendation(
            mode=mode,
            recommended_height_mm=rec.recommended_height_mm,
            recommended_angle_deg=rec.recommended_angle_deg,
            reason=rec.reason
        ))

    return MultiSceneRecommendation(recommendations=results)