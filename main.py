import os
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
from typing import Dict, List
from pydantic import BaseModel, Field

# 引入项目内部模块
from models import UserProfile, ErgoRecommendation, UserFeedbackUpload, MultiSceneRecommendation, UserCreate, Token
from ergonomics import calculate_settings, calculate_all_modes, load_model
from database import engine, Base, get_db
import sql_models
import security
from train_model import train_single_model, DB_PATH

# === 引入 LangChain 依赖 ===
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# === 引入 RAG (云端 Embedding + 本地 Chroma) 依赖 ===
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------
# 初始化数据库
# ------------------------------------------------------------
sql_models.Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------
# 1. DeepSeek AI 核心大脑配置
# ------------------------------------------------------------
DEEPSEEK_API_KEY = "sk-1a64fb1450e54625b7365dd9364ad710"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
    max_tokens=256
)

# ------------------------------------------------------------
# 2. RAG 知识库初始化 (DashScope + Chroma)
# ------------------------------------------------------------
print("📚 正在连接阿里百炼云端，初始化医学知识库...")

# 使用阿里百炼云端 Embedding（极速，无需本地下载）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key="sk-0e255a019c6d41a2abdf4b65d1e01de8"
)

chroma_db_dir = "./chroma_knowledge_db"
if not os.path.exists(chroma_db_dir):
    print("⏳ 初次运行，正在将医学指南写入数据库...")
    # 读取 knowledge.txt
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    docs = loader.load()

    # 文本切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=['\n\n', '\n', ' ', '']
    )
    splits = splitter.split_documents(docs)

    # 存入 Chroma 数据库
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=chroma_db_dir,
        collection_name="spine_medical_knowledge"
    )
    print("✅ 知识库云端向量化并保存完毕！")
else:
    # 以后运行秒开，直接从本地加载 Chroma
    db = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embeddings,
        collection_name="spine_medical_knowledge"
    )
    print("✅ 本地 Chroma 知识库加载成功！")

# 创建检索器（每次搜索最相关的 2 段文档）
retriever = db.as_retriever(search_kwargs={"k": 2})


# ------------------------------------------------------------
# 3. LangChain 业务管线构建 (Schema & Prompt)
# ------------------------------------------------------------

# --- RAG 报告管线 ---
class HealthAnalysisOutput(BaseModel):
    advice: str = Field(description="100字以内的健康评估和物理调节建议，必须是纯文本，绝对禁止使用任何Markdown格式。")
    score_evaluation: str = Field(description="用一句话(10个字以内)总结今天的表现，例如：'表现完美'、'需注意久坐'")


report_parser = PydanticOutputParser(pydantic_object=HealthAnalysisOutput)
report_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位严谨的脊柱健康康复专家。
在给出评价和建议时，请【务必优先参考】以下检索到的权威医学知识：
-----------------------
{context}
-----------------------
\n{format_instructions}"""),
    ("user", """请根据用户的今日坐姿数据给出专业的健康建议：
    - 今日总坐姿时长: {total_hours} 小时
    - 触发久坐报警次数: {sedentary_count} 次
    - 综合坐姿健康分: {posture_score} 分
    - 模式使用分布: {mode_distribution}
    - 过去7天坐姿时长趋势: {weekly_trend}
    """)
]).partial(format_instructions=report_parser.get_format_instructions())

report_chain = report_prompt | llm | report_parser


# --- 语音控制管线 ---
# 定义单条对话记录
class ChatMessage(BaseModel):
    role: str = Field(description="角色，'user' 或 'ai'")
    content: str = Field(description="对话内容")

class ChairAction(BaseModel):
    #  新增了一个指令：SAVE_PRESET
    command: str = Field(description="必须是: ADJUST_HEIGHT, ADJUST_ANGLE, APPLY_PRESET, ALERT_VIBRATION, SAVE_PRESET")
    parameters: dict = Field(description="参数字典。如 {'height': 480}, {'presetName': 'REST'}", default_factory=dict)

class ChatControlResponse(BaseModel):
    reply: str = Field(description="给用户的自然语言回复，语气亲切")
    actions: List[ChairAction] = Field(description="需要执行的物理指令", default_factory=list)

class ChatControlRequest(BaseModel):
    user_input: str
    current_height: int = 450
    current_angle: int = 95
    current_mode: str = Field(default="NONE", description="当前处于什么模式，如 OFFICE, REST, ENTERTAINMENT, CUSTOM 等")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="最近的历史对话记录")


# ============================================================
# 升级版：具备主动交互与安全拦截的 Agent Prompt
# ============================================================
control_parser = PydanticOutputParser(pydantic_object=ChatControlResponse)

control_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个名为“智脊助手”的顶级智能座椅管家。你不只是执行命令，还会主动关心用户的健康。

【当前座椅状态】
- 模式: {current_mode}
- 高度: {current_height} mm
- 角度: {current_angle} 度

【你可以使用的底层指令】
1. APPLY_PRESET: {{"presetName": "OFFICE"|"REST"|"ENTERTAINMENT"}}
2. ADJUST_HEIGHT: {{"height": 整数}} (范围350-600)
3. ADJUST_ANGLE: {{"angle": 整数}} (范围90-135)
4. ALERT_VIBRATION: {{}} (震动提醒)
5. SAVE_PRESET: {{"presetName": "{current_mode}"}} (将当前状态永久保存到当前模式中)

【 核心交互法则（必须严格遵守）】
法则一（状态偏离与保存询问）：
如果用户要求调节高度或角度，且 `current_mode` 不是 "NONE"。你应该输出 ADJUST 指令执行动作，并在 `reply` 中反问用户：“我已经为您调整了。请问只是临时调一下，还是需要保存为 [{current_mode}] 的默认设置？”

法则二（执行保存）：
如果用户在历史对话中被问过是否保存，且当前回答了“保存/是的/确定”，你必须输出 SAVE_PRESET 指令，并在 `reply` 告知“已为您永久保存该模式偏好”。

法则三（模糊意图的“悬丝诊脉”）：
如果用户只说“腰疼/脖子酸/累了”，没有明确指令。请【不要】输出任何动作指令！在 `reply` 中利用医学知识给出2个调节选项让用户选（例如：稍微后仰还是开启休息模式？）。

法则四（安全拦截）：
如果用户要求在办公(OFFICE)状态下将角度调到 120 度以上，请拦截该请求。在 `reply` 警告其危险性，并最多只为其调节到 105 度。

【历史对话上下文】
{history_str}
-------------------------
\n{format_instructions}"""),
    ("user", "{user_input}")
]).partial(format_instructions=control_parser.get_format_instructions())

control_chain = control_prompt | llm | control_parser

# ------------------------------------------------------------
# mDNS 服务发现与网络配置
# ------------------------------------------------------------
zeroconf = None
service_info = None


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global zeroconf, service_info
    try:
        local_ip = get_local_ip()
        print(f"🌍 本机局域网 IP: {local_ip}")
        zeroconf = Zeroconf()
        service_type = "_spine-api._tcp.local."
        service_name = "Spine Assistant API._spine-api._tcp.local."
        service_info = ServiceInfo(
            service_type, service_name, addresses=[socket.inet_aton(local_ip)],
            port=8000, properties={"version": "1.6", "path": "/"}, server=f"spine-server.local.",
        )
        await asyncio.to_thread(zeroconf.register_service, service_info)
        print("✅ 服务注册成功，App 可自动发现。")
    except Exception as e:
        print(f"❌ mDNS 失败: {repr(e)}")

    yield

    if zeroconf and service_info:
        try:
            await asyncio.to_thread(zeroconf.unregister_service, service_info)
            zeroconf.close()
            print("✅ 服务已注销。")
        except Exception as e:
            pass


app = FastAPI(title="Spine Assistant Cloud API", version="1.6", lifespan=lifespan)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ------------------------------------------------------------
# 路由模型定义
# ------------------------------------------------------------
class HealthReportRequest(BaseModel):
    total_hours: float
    sedentary_count: int
    posture_score: int
    mode_distribution: Dict[str, float]
    weekly_trend: List[float]


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                          detail="Could not validate credentials",
                                          headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(sql_models.DBUser).filter(sql_models.DBUser.username == username).first()
    if user is None: raise credentials_exception
    return user


def trigger_retrain_task(target_mode: str):
    print(f"🔄 [后台任务] '{target_mode}' 重新训练...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT current_mode, height_cm, weight_kg, final_height_mm, final_angle_deg FROM feedbacks"
        all_data = pd.read_sql_query(query, conn)
        conn.close()
        train_single_model(target_mode, all_data)
        load_model(target_mode, force_reload=True)
    except Exception as e:
        print(f"❌ 训练失败: {e}")


# ------------------------------------------------------------
# 基础 API 端点
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "db": "sqlite", "auth": "enabled", "ai": "langchain_deepseek_chroma"}


@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(sql_models.DBUser).filter(sql_models.DBUser.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user_id = str(uuid.uuid4())
    hashed_pwd = security.get_password_hash(user.password)
    db_user = sql_models.DBUser(user_id=new_user_id, username=user.username, hashed_password=hashed_pwd,
                                height_cm=user.height_cm, weight_kg=user.weight_kg)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token = security.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(sql_models.DBUser).filter(sql_models.DBUser.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": security.create_access_token(data={"sub": user.username}), "token_type": "bearer"}


@app.post("/api/v1/recommend", response_model=MultiSceneRecommendation)
def get_recommendation(profile: UserProfile, current_user: sql_models.DBUser = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    current_user.height_cm = profile.height_cm
    current_user.weight_kg = profile.weight_kg
    db.commit()
    return calculate_all_modes(profile)


@app.post("/api/v1/feedback")
def upload_feedback(feedback: UserFeedbackUpload, background_tasks: BackgroundTasks,
                    current_user: sql_models.DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    db_feedback = sql_models.DBFeedback(user_id=current_user.user_id, height_cm=feedback.height_cm,
                                        weight_kg=feedback.weight_kg, final_height_mm=feedback.final_height_mm,
                                        final_angle_deg=feedback.final_angle_deg, problem_area=feedback.problem_area,
                                        current_mode=feedback.current_mode)
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    if feedback.current_mode.upper() in ["OFFICE", "REST", "ENTERTAINMENT"]:
        background_tasks.add_task(trigger_retrain_task, feedback.current_mode.upper())
    return {"status": "success", "id": db_feedback.id, "training_triggered": True}


# ------------------------------------------------------------
# LangChain 核心 API 端点
# ------------------------------------------------------------

@app.post("/api/v1/report/analysis")
def analyze_report(data: HealthReportRequest, current_user: sql_models.DBUser = Depends(get_current_user)):
    print(f"🤖 [LangChain RAG] 分析用户 [{current_user.username}] ...")
    try:
        query = f"针对久坐时长{data.total_hours}小时和腰椎压力的医学建议"
        retrieved_docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"🔍 [RAG 命中知识]: {context_text}")

        result: HealthAnalysisOutput = report_chain.invoke({
            "context": context_text,
            "total_hours": data.total_hours,
            "sedentary_count": data.sedentary_count,
            "posture_score": data.posture_score,
            "mode_distribution": data.mode_distribution,
            "weekly_trend": data.weekly_trend
        })
        return {"advice": result.advice}
    except Exception as e:
        print(f"❌ LangChain RAG 调用失败: {e}")
        return {"advice": "AI 暂时无法连接，建议您每45分钟起身活动，保持健康坐姿。"}


# === 升级版：自然语言座椅控制接口 ===
@app.post("/api/v1/chair/chat", response_model=ChatControlResponse)
def chat_control_chair(request: ChatControlRequest, current_user: sql_models.DBUser = Depends(get_current_user)):
    print(f"🎙️ 收到语音指令: '{request.user_input}' (当前模式: {request.current_mode})")

    try:
        # 1. 将历史记录格式化为易读的字符串
        history_str = ""
        if request.chat_history:
            for msg in request.chat_history:
                role_name = "用户" if msg.role == "user" else "管家(你)"
                history_str += f"{role_name}: {msg.content}\n"
        else:
            history_str = "无历史对话"

        # 2. 注入所有状态到 LangChain
        result: ChatControlResponse = control_chain.invoke({
            "user_input": request.user_input,
            "current_height": request.current_height,
            "current_angle": request.current_angle,
            "current_mode": request.current_mode,
            "history_str": history_str  # 👈 注入历史记录
        })
        print(f"🤖 Agent 决策结果: {result.model_dump()}")
        return result
    except Exception as e:
        print(f"❌ 语音控制解析失败: {e}")
        return ChatControlResponse(reply="抱歉，我刚刚在思考别的，请再试一次。", actions=[])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
