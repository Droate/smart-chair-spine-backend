from pydantic import BaseModel, Field
from typing import Optional, List

# ------------------------------------------------------------
# 认证相关模型（用于注册和登录）
# ------------------------------------------------------------

class UserCreate(BaseModel):
    """
    用户注册请求模型。
    当新用户通过 App 注册时，需要提供以下信息。
    """
    username: str                # 用户名，必填
    password: str                # 密码，必填
    height_cm: int = 170         # 身高（厘米），默认为 170
    weight_kg: float = 65.0      # 体重（千克），默认为 65.0


class Token(BaseModel):
    """
    认证令牌响应模型。
    登录/注册成功后，后端返回 JWT 令牌给客户端。
    """
    access_token: str            # JWT 访问令牌
    token_type: str              # 令牌类型，固定为 "bearer"


# ------------------------------------------------------------
# 用户身体档案（用于请求推荐）
# ------------------------------------------------------------

class UserProfile(BaseModel):
    """
    用户身体档案请求模型。
    当 App 请求 AI 推荐时，需要提供当前用户的身体数据和场景模式。
    注意：user_id 是可选的，因为登录后后端会从令牌中解析用户身份。
    """
    user_id: Optional[str] = Field(
        None,
        description="用户ID (登录模式下可不传)"
    )
    height_cm: int = Field(
        ...,
        ge=100,
        le=250,
        description="身高 (cm) —— 必须在 100~250 之间"
    )
    weight_kg: float = Field(
        ...,
        ge=30,
        le=200,
        description="体重 (kg) —— 必须在 30~200 之间"
    )
    # 当前使用场景，App 必须传递，例如 "OFFICE", "REST", "ENTERTAINMENT"
    current_mode: str = Field(
        ...,
        description="当前模式 (e.g., OFFICE, REST)"
    )
    # 可选的高级人体测量数据（用于未来更精确的推荐）
    upper_body_ratio: Optional[float] = Field(
        0.48,
        description="上半身比例系数，默认 0.48"
    )
    thigh_length_cm: Optional[float] = Field(
        None,
        description="大腿长 (cm)，默认不提供"
    )


# ------------------------------------------------------------
# 推荐配置（后端计算后返回的结果）
# ------------------------------------------------------------

class ErgoRecommendation(BaseModel):
    """
    单项推荐模型（内部使用或向前兼容）。
    包含推荐的高度、角度和理由。
    """
    recommended_height_mm: int   # 推荐座高（毫米）
    recommended_angle_deg: int   # 推荐椅背角度（度）
    reason: str                  # 推荐理由，例如 "AI 模型预测" 或 "基于规则"


class SingleModeRecommendation(BaseModel):
    """
    单个模式的推荐详情。
    用于列表项，包含模式名称及对应的推荐值。
    """
    mode: str                                # 模式名称，如 "OFFICE"
    recommended_height_mm: int               # 该模式下的推荐座高（毫米）
    recommended_angle_deg: int               # 该模式下的推荐角度（度）
    reason: str                               # 推荐理由


class MultiSceneRecommendation(BaseModel):
    """
    多场景总推荐响应模型。
    API 最终返回给客户端的数据格式，包含三种模式的推荐列表。
    """
    recommendations: List[SingleModeRecommendation]   # 推荐列表，通常为三条（办公、休息、娱乐）


# ------------------------------------------------------------
# 用户反馈数据（用于 AI 训练和模型优化）
# ------------------------------------------------------------

class UserFeedbackUpload(BaseModel):
    """
    用户反馈上传请求模型。
    当用户手动调整座椅或保存预设时，App 将反馈数据发送给后端，用于后续模型训练。
    """
    user_id: Optional[str] = None       # 用户ID，可选（后端会从令牌中获取）
    height_cm: int                       # 反馈时的身高（厘米）
    weight_kg: float                      # 反馈时的体重（千克）
    final_height_mm: int                  # 用户最终确定的高度（毫米）
    final_angle_deg: int                  # 用户最终确定的角度（度）
    problem_area: str                     # 问题区域，例如 "腰部支撑"、"坐垫高度" 等
    current_mode: str                      # 反馈时的使用场景，如 "OFFICE"