from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base

# ------------------------------------------------------------
# SQLAlchemy 模型：定义数据库中的表结构
# 每个类对应一张表，每个类属性对应表中的一列。
# 这些模型用于操作数据库（增删改查），与 Pydantic 模型（API 数据校验）是分离的。
# ------------------------------------------------------------

# === 用户表 ===
class DBUser(Base):
    """
    用户表，存储注册用户的基本信息和身体数据。
    对应数据库中的 'users' 表。
    """
    __tablename__ = "users"   # 指定数据库中的表名

    # 用户唯一标识，主键。由服务器生成 UUID 字符串，例如 "550e8400-e29b-41d4-a716-446655440000"
    user_id = Column(String, primary_key=True, index=True)
    # primary_key=True: 主键，唯一标识一条记录
    # index=True: 为该列创建数据库索引，加速基于 user_id 的查询

    # 账户信息（用于认证）
    username = Column(String, unique=True, index=True)  # 用户名，必须唯一，用于登录
    hashed_password = Column(String)                    # 密码哈希值（使用 security.py 中的 SHA256 哈希）

    # 用户创建时间，自动设置为当前时间（数据库服务器时间）
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # DateTime(timezone=True): 存储带时区的时间戳
    # server_default=func.now(): 插入记录时，由数据库自动填充当前时间，而不是由 Python 生成

    # 身体数据（用于 AI 推荐）
    height_cm = Column(Integer)           # 身高（厘米），整数
    weight_kg = Column(Float)             # 体重（千克），浮点数
    upper_body_ratio = Column(Float, default=0.48)   # 上半身比例，默认 0.48（可选，存储用户自定义值）
    thigh_length_cm = Column(Float, nullable=True)   # 大腿长（厘米），允许为空


# === 反馈数据表 (AI 训练集) ===
class DBFeedback(Base):
    """
    用户反馈表，存储每次用户手动调节座椅后的最终状态。
    这些数据将用于训练 AI 模型，让模型学习用户偏好。
    对应数据库中的 'feedbacks' 表。
    """
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # 自增主键，每条反馈的唯一 ID

    user_id = Column(String, index=True)  # 提交反馈的用户 ID（来自 DBUser.user_id），用于关联用户，但不设外键约束（简化）

    # 反馈提交时间，自动填充
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # 反馈时的身体特征（可能与用户当前档案不同，例如用户临时修改了身高体重，所以单独记录）
    height_cm = Column(Integer)           # 身高（厘米）
    weight_kg = Column(Float)             # 体重（千克）

    # 用户的真实偏好（即最终确定的座椅状态），作为机器学习中的“标签”（label）
    final_height_mm = Column(Integer)     # 最终座高（毫米）
    final_angle_deg = Column(Integer)     # 最终椅背角度（度）

    problem_area = Column(String)          # 问题区域，例如 "腰部支撑"、"坐垫高度" 等（来自用户反馈）
    # 当前使用场景，例如 "OFFICE", "REST", "ENTERTAINMENT"（用于分场景训练模型）
    current_mode = Column(String)