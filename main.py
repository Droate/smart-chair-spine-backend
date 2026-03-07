import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import uuid
import socket
from zeroconf import ServiceInfo, Zeroconf
from contextlib import asynccontextmanager
import asyncio
import pandas as pd
import sqlite3
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, List

# 引入项目内部模块
from models import UserProfile, ErgoRecommendation, UserFeedbackUpload, MultiSceneRecommendation, UserCreate, Token
from ergonomics import calculate_settings, calculate_all_modes, load_model
from database import engine, Base, get_db
import sql_models
import security
from train_model import train_single_model, DB_PATH

# ------------------------------------------------------------
# 初始化数据库
# ------------------------------------------------------------
# 根据 SQLAlchemy 模型创建所有表（如果不存在）
sql_models.Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------
# DeepSeek AI 配置
# ------------------------------------------------------------
# 生产环境中应将 API Key 存储在环境变量中，避免硬编码
DEEPSEEK_API_KEY = "sk-1a64fb1450e54625b7365dd9364ad710"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 初始化 OpenAI 客户端（兼容 DeepSeek API）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ------------------------------------------------------------
# 全局变量（用于 mDNS 服务发现）
# ------------------------------------------------------------
zeroconf = None
service_info = None


# ------------------------------------------------------------
# 获取本机局域网 IP
# ------------------------------------------------------------
def get_local_ip():
    """
    通过连接外部 DNS 服务器（8.8.8.8）获取本机局域网 IP 地址。
    这种方法能可靠地返回当前使用的网卡 IP（如 192.168.x.x）。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 如果失败，返回本地回环地址
        return "127.0.0.1"


# ------------------------------------------------------------
# 生命周期管理（mDNS 广播）
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器，在应用启动时执行 mDNS 服务注册，
    在应用关闭时执行注销。这使得 Android App 可以通过网络服务发现自动找到后端地址。
    """
    global zeroconf, service_info
    try:
        local_ip = get_local_ip()
        print(f"🌍 本机局域网 IP: {local_ip}")

        # 初始化 Zeroconf 实例
        zeroconf = Zeroconf()

        # 定义服务类型和名称，必须与 Android 端 NetworkModule 中配置的一致
        service_type = "_spine-api._tcp.local."
        service_name = "Spine Assistant API._spine-api._tcp.local."

        # 创建服务信息对象
        service_info = ServiceInfo(
            service_type,
            service_name,
            addresses=[socket.inet_aton(local_ip)],  # IP 地址转换为字节格式
            port=8000,                                # 后端服务端口
            properties={"version": "1.6", "path": "/"},
            server=f"spine-server.local.",
        )
        print(f"📡 正在注册 mDNS 服务: {service_name}")
        # 将服务注册到网络，由于 zeroconf 注册是同步阻塞的，使用 to_thread 避免阻塞事件循环
        # 简单来说就是讲注册服务的过程 放到线程当中去 避免阻塞
        await asyncio.to_thread(zeroconf.register_service, service_info)
        print("✅ 服务注册成功，App 可自动发现。")
    except Exception as e:
        print(f"❌ mDNS 服务注册失败: {repr(e)}")

    yield  # 应用运行期间保持

    # 关闭时注销服务
    if zeroconf and service_info:
        print("🔕 正在注销 mDNS 服务...")
        try:
            await asyncio.to_thread(zeroconf.unregister_service, service_info)
            zeroconf.close()
            print("✅ 服务已注销。")
        except Exception as e:
            print(f"⚠️ 注销服务时出错: {e}")


# ------------------------------------------------------------
# 创建 FastAPI 应用实例，指定生命周期管理器
# ------------------------------------------------------------
app = FastAPI(title="Spine Assistant Cloud API", version="1.6", lifespan=lifespan)

# OAuth2 密码流认证方案，用于 Swagger UI 自动添加认证按钮
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ------------------------------------------------------------
# 内部数据模型（用于特定 API）
# ------------------------------------------------------------
class HealthReportRequest(BaseModel):
    """
    健康报告分析请求模型，对应 /api/v1/report/analysis 接口的请求体。
    """
    total_hours: float          # 今日总坐姿时长（小时）
    sedentary_count: int        # 今日久坐报警次数
    posture_score: int          # 综合坐姿健康分（0-100）
    mode_distribution: Dict[str, float]  # 各场景模式时长占比（如 {"OFFICE":0.6, "REST":0.3}）
    weekly_trend: List[float]   # 过去 7 天每日坐姿时长（小时），顺序从 6 天前到当天


# ------------------------------------------------------------
# 依赖项：获取当前登录用户
# ------------------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    从 JWT 令牌中解析用户名，并从数据库查询用户对象。
    如果令牌无效或用户不存在，抛出 401 异常。
    该依赖项用于保护需要登录的 API。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 解码 JWT 令牌
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username: str = payload.get("sub")  # subject 字段存储用户名
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # 查询数据库
    user = db.query(sql_models.DBUser).filter(sql_models.DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# ------------------------------------------------------------
# 后台任务：触发模型重新训练
# ------------------------------------------------------------
def trigger_retrain_task(target_mode: str):
    """
    在后台线程中执行模型重新训练任务。
    当用户提交反馈后，此函数会被添加为后台任务，避免阻塞 API 响应。
    """
    print(f"🔄 [后台任务] 开始为模式 '{target_mode}' 重新训练模型...")
    try:
        # 连接 SQLite 数据库，读取所有反馈数据
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT current_mode, height_cm, weight_kg, final_height_mm, final_angle_deg FROM feedbacks"
        all_data = pd.read_sql_query(query, conn)
        conn.close()

        # 调用训练函数，只针对指定模式训练
        train_single_model(target_mode, all_data)

        # 强制重新加载模型到内存缓存，使新模型立即生效
        load_model(target_mode, force_reload=True)

        print(f"✅ [后台任务] '{target_mode}' 模型更新完毕！")
    except Exception as e:
        print(f"❌ [后台任务] 训练失败: {e}")


# ------------------------------------------------------------
# API 端点定义
# ------------------------------------------------------------

@app.get("/")
def read_root():
    """健康检查接口，返回服务状态信息。"""
    return {"status": "online", "db": "sqlite", "auth": "enabled", "ai": "deepseek"}


@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    用户注册接口。
    - 检查用户名是否已存在
    - 生成新用户 ID (UUID)
    - 密码哈希存储
    - 返回 JWT 令牌，方便注册后直接登录
    """
    # 检查用户名是否已被注册
    db_user = db.query(sql_models.DBUser).filter(sql_models.DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # 生成新用户 ID
    new_user_id = str(uuid.uuid4())
    # 对密码进行哈希（使用 security.py 中的简单哈希）
    hashed_pwd = security.get_password_hash(user.password)

    # 创建数据库用户对象
    db_user = sql_models.DBUser(
        user_id=new_user_id,
        username=user.username,
        hashed_password=hashed_pwd,
        height_cm=user.height_cm,
        weight_kg=user.weight_kg
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)  # 获取数据库生成的字段（如 created_at）

    # 生成访问令牌
    access_token = security.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    用户登录接口，兼容 OAuth2 密码模式。
    接收表单格式的用户名密码，验证后返回 JWT 令牌。
    """
    # 根据用户名查询用户
    user = db.query(sql_models.DBUser).filter(sql_models.DBUser.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 生成访问令牌
    access_token = security.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/v1/recommend", response_model=MultiSceneRecommendation)
def get_recommendation(
        profile: UserProfile,
        current_user: sql_models.DBUser = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    获取全场景推荐（办公、休息、娱乐）。
    - 需要登录
    - 将请求中的身高体重更新到用户档案（便于下次使用）
    - 调用 ergonomics.calculate_all_modes 计算三种模式的推荐值
    """
    # 更新用户的当前身高体重（记录最近一次使用的数据）
    current_user.height_cm = profile.height_cm
    current_user.weight_kg = profile.weight_kg
    db.commit()

    print(f"🚀 用户 [{current_user.username}] 请求全场景推荐")

    # 计算推荐值（内部会使用 AI 模型或规则）
    multi_rec = calculate_all_modes(profile)
    return multi_rec


@app.post("/api/v1/feedback")
def upload_feedback(
        feedback: UserFeedbackUpload,
        background_tasks: BackgroundTasks,
        current_user: sql_models.DBUser = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    用户反馈上传接口。
    - 将反馈数据存储到数据库（用于后续模型训练）
    - 添加后台任务重新训练对应模式的 AI 模型
    """
    # 创建反馈记录
    db_feedback = sql_models.DBFeedback(
        user_id=current_user.user_id,
        height_cm=feedback.height_cm,
        weight_kg=feedback.weight_kg,
        final_height_mm=feedback.final_height_mm,
        final_angle_deg=feedback.final_angle_deg,
        problem_area=feedback.problem_area,
        current_mode=feedback.current_mode
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)

    print(f"✅ 用户 [{current_user.username}] 反馈已入库")

    # 如果反馈属于预定义的三种模式之一，则触发后台重新训练
    if feedback.current_mode.upper() in ["OFFICE", "REST", "ENTERTAINMENT"]:
        background_tasks.add_task(trigger_retrain_task, feedback.current_mode.upper())

    return {"status": "success", "id": db_feedback.id, "training_triggered": True}


@app.post("/api/v1/report/analysis")
def analyze_report(
        data: HealthReportRequest,
        current_user: sql_models.DBUser = Depends(get_current_user)
):
    """
    调用 DeepSeek AI 生成健康报告分析建议。
    接收用户今日数据和历史趋势，构造 prompt 调用 AI，返回纯文本建议。
    """
    print(f"🤖 正在调用 DeepSeek 分析用户 [{current_user.username}] 的健康数据...")

    # 构造 Prompt（包含严格的格式和逻辑指导）
    prompt = f"""
你是一位“智脊助手”智能人体工学椅专属的AI健康顾问。请根据用户的今日数据，给出专业的点评和座椅调节建议。

【强制输出格式要求】：
1. 绝对禁止使用任何 Markdown 符号（如 *、#、-、>、` 等），只能输出纯文本。
2. 字数必须严格控制在 80 到 110 字之间。

【用户今日数据】：
- 今日入座总时长: {data.total_hours}小时
- 触发不良坐姿报警: {data.sedentary_count}次
- 坐姿综合评分: {data.posture_score}分 (满分100)
- 过去7天时长趋势: {data.weekly_trend}

【回复逻辑指导】：
1. 若数据恶化（时长久、报警多）：语气要严厉警告。必须建议用户“开启座椅休息模式”，并要求其起身拉伸。
2. 若数据改善（分数高、报警少）：语气要热情鼓励。建议用户“保持当前座椅高度与直立靠背角度”。
3. 若数据极少（如小于1小时）：简单问候即可。建议后续长时间办公时随时注意椅背贴合度。

请直接输出你的点评与建议：
"""

    try:
        # 调用 DeepSeek API（使用 OpenAI 兼容接口）
        response = client.chat.completions.create(
            model="deepseek-chat",          # 模型名称，具体取决于 API Key 权限
            messages=[
                {"role": "system", "content": "你是一个严格遵守格式指令的智能座椅AI助手。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        advice = response.choices[0].message.content.strip()
        print(f"✨ DeepSeek 响应: {advice}")
        return {"advice": advice}
    except Exception as e:
        print(f"❌ DeepSeek API 调用失败: {e}")
        # 降级处理：返回静态建议
        return {"advice": "AI 暂时无法连接，建议您每45分钟起身活动，保持健康坐姿。"}


# ------------------------------------------------------------
# 直接运行入口（仅开发调试用）
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)