#  基于大模型与云边端协同的智能座椅健康管家 (AI 云端)

> **提示**：本项目为《智能座椅健康管家》的 **云端 AI 推理与大模型微服务部分**。
> **配套的 Android 端侧感知与 UI 源码请移步至**：https://github.com/Droate/smart-chair-spine-frontend

## 项目简介
本项目是《智能座椅健康管家》的后端与 AI 核心。主要提供基于大模型 Agent 的自然语言意图解析、基于 RAG 架构的健康报告生成，以及无状态并发隔离的 API 服务。

## 核心技术栈
* **后端框架**: Python, FastAPI
* **大语言模型**: LangChain, DeepSeek LLM
* **RAG 架构**: ChromaDB (向量数据库), 阿里百炼 DashScope (Embedding 模型)
* **机器学习**: Scikit-learn (多输出线性回归模型)

## 核心亮点
* **Agentic Workflow**：基于 LangChain (LCEL) 将自然语言意图精准降维为底层物理设备动作序列，打通“零触控”驱动闭环。
* **权威医学 RAG**：对《世卫组织久坐指南》进行切片与向量化检索，有效消除大模型生成健康建议时的“幻觉”。
* **无状态安全架构**：针对多用户并发越权隐患，采用彻底的 Stateless 设计，配合端侧实现绝对的会话隔离与数据安全。

```mermaid
graph TD
    %% 样式定义
    classDef client fill:#E1F5FE,stroke:#0288D1,stroke-width:2px;
    classDef server fill:#FFF3E0,stroke:#F57C00,stroke-width:2px;
    classDef ai fill:#E8F5E9,stroke:#388E3C,stroke-width:2px;
    classDef hardware fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px;

    %% --- 客户端 (Android 端) ---
    subgraph Client ["📱 边缘计算与状态托管端 (Android)"]
        direction TB
        UI["Jetpack Compose 3D 可视化 UI"]
        
        subgraph Perception ["端侧视觉感知"]
            Camera["CameraX 视频流采集\n背压防 OOM"]
            MediaPipe["MediaPipe 面部网格\n提取头部三维欧拉角"]
            Filter["EMA 滤波 & 防抖状态机\n准确率 96.9%"]
            Camera --> MediaPipe --> Filter
        end

        subgraph Context ["端侧上下文托管"]
            StateFlow["StateFlow 状态管理"]
            RoomDB[("Room 本地数据库\n存储多轮对话记忆")]
        end

        subgraph Comms ["硬件通信模块"]
            BLE["动态权限管理\n经典蓝牙/BLE 协议"]
            Delay["防拥塞时序控制"]
            BLE --> Delay
        end

        UI <--> Context
        Filter --> Context
    end

    %% --- 云端服务 (FastAPI) ---
    subgraph Server ["☁️ 云端微服务后端 (FastAPI)"]
        direction TB
        API["FastAPI 网关\nStateless 无状态设计"]
        mDNS["mDNS 局域网服务发现"]

        subgraph AgentEngine ["Agentic Workflow 执行引擎"]
            LCEL["LangChain LCEL 管道"]
            Pydantic["Pydantic 结构化输出\n降维解析 JSON 动作"]
        end

        subgraph KnowledgeBase ["垂直医疗知识库 RAG"]
            DashScope["DashScope Embedding 模型"]
            ChromaDB[("Chroma 本地向量数据库\n检索久坐指南")]
        end

        subgraph Models ["核心大模型"]
            DeepSeek(("DeepSeek LLM"))
            Sklearn["Scikit-learn 多输出回归\n个性化参数推荐"]
        end

        API --> AgentEngine
        AgentEngine <--> DeepSeek
        AgentEngine <--> KnowledgeBase
        API <--> Sklearn
    end

    %% --- 物理执行层 ---
    subgraph HardwareLayer ["🪑 物理执行层"]
        Simulator["数字孪生硬件模拟器"]
    end

    %% --- 跨边界交互链路 ---
    
    %% 1. HTTP 请求
    Context -- "1. HTTP POST\n自然语言意图+物理状态+多轮对话记忆" --> API
    
    %% 2. HTTP 响应
    API -- "2. HTTP Response\n动作序列 JSON + RAG 诊断报告" --> Context

    %% 3. 蓝牙驱动
    Context -- "3. 解析动作序列" --> Comms
    Comms -- "4. RFCOMM 蓝牙字节流\n串行驱动电机" --> Simulator

    %% 应用样式
    class Client client;
    class Server server;
    class AgentEngine,KnowledgeBase,Models ai;
    class HardwareLayer hardware;
```
