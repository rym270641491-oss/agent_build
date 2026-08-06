# -*- coding: utf-8 -*-
"""全局配置模块。

集中管理 Agent 运行所需的全部配置项：
- 大模型（DeepSeek-V4-Flash）相关参数
- ReAct 推理循环的步数与超时限制
- 向量库（Chroma）与经验文档目录
- Embedding 的接入方式（API / 本地离线）
- 可选的验证 SQL 执行能力（默认关闭，避免误连生产库）
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：config 目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Agent 配置类，支持通过环境变量 / .env 文件覆盖默认值。"""

    model_config = SettingsConfigDict(
        # 优先读取本项目文件夹内的 .env；不存在时回退到仓库根目录的 .env
        # （根目录 .env 同时被学习示例脚本使用，因此保留在那里共享）
        env_file=(
            str(PROJECT_ROOT / ".env"),
            str(PROJECT_ROOT.parent / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略 .env 中与本类无关的变量
    )

    # ---------------- 大模型配置（DeepSeek，OpenAI 兼容接口） ----------------
    # DeepSeek 模型名，可按需改为 deepseek-chat 等
    MODEL_NAME: str = "deepseek-v4-flash"
    # API 密钥：从环境变量 DEEPSEEK_API_KEY 或 .env 读取
    DEEPSEEK_API_KEY: str = ""
    # OpenAI 兼容接口地址
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    # 推理任务使用低温，减少发散
    LLM_TEMPERATURE: float = 0.1
    # 单次回复的最大 token 数
    LLM_MAX_TOKENS: int = 2048
    # 单次 LLM 请求的超时时间（秒）。
    # 实测 DeepSeek 单次响应 13 秒 ~ 超过 30 秒，30 秒会偶发掐断，
    # 默认放宽到 60 秒保证循环稳定；整个循环的总预算由 MAX_TOTAL_TIME_SECONDS 控制。
    LLM_REQUEST_TIMEOUT: float = 60.0

    # ---------------- ReAct 推理循环限制 ----------------
    # 最多推理步数（每步最多 MAX_HYPOTHESES_PER_STEP 个假设）
    MAX_STEPS: int = 8
    # 整个排查过程的总超时（秒），超时后强制输出“信息不足”。
    # 实测 DeepSeek 单次请求约 13~28 秒，30 秒总预算只够 1~2 步，
    # 因此放宽到 180 秒，保证 8 步循环可完整执行（可按需配置）。
    MAX_TOTAL_TIME_SECONDS: float = 180.0
    # 每步最多输出的假设数
    MAX_HYPOTHESES_PER_STEP: int = 3

    # ---------------- 经验文档与向量检索 ----------------
    # 经验文档目录（Agent 启动时读取其下所有 Markdown）
    DOCS_DIR: Path = PROJECT_ROOT / "data" / "user_experience"
    # Chroma 持久化目录
    VECTOR_DIR: Path = PROJECT_ROOT / "data" / "vector_store"
    # 启动时是否重建索引（文档量小，默认重建保证文档更新即时生效）
    RECREATE_INDEX_ON_START: bool = True
    # 每次检索返回的文档块数量
    RETRIEVE_TOP_K: int = 4
    # 融合检索权重：表名关键词占比 / 向量相似度占比（两者之和建议为 1）
    RETRIEVE_KEYWORD_WEIGHT: float = 0.4
    RETRIEVE_VECTOR_WEIGHT: float = 0.6

    # ---------------- Embedding 配置 ----------------
    # auto : 配置了接口密钥就用 API 嵌入，否则自动降级为本地哈希嵌入（离线可用）
    # api  : 强制使用 OpenAI 兼容嵌入接口（需配置 EMBEDDING_API_KEY 或 DEEPSEEK_API_KEY）
    # local: 强制使用本地哈希嵌入，无需网络
    EMBEDDING_MODE: str = "auto"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.deepseek.com/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # 本地哈希嵌入的向量维度
    LOCAL_EMBEDDING_DIM: int = 256

    # ---------------- 可选：验证 SQL 执行（默认关闭） ----------------
    # 开启后 run_tools 节点会尝试连接数据库执行验证 SQL（需额外安装 sqlalchemy）
    SQL_EXECUTION_ENABLED: bool = False
    # 数据库连接串，例如 sqlite:///./data/check.db
    DATA_DB_URL: str = ""

    # ---------------- 其他 ----------------
    LOG_LEVEL: str = "INFO"


# 模块级单例：保证整个进程共享同一份配置
settings = Settings()
