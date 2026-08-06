# -*- coding: utf-8 -*-
"""LangGraph 状态定义。

AgentState 是整张图流转共享的状态字典：
- 对话消息（跨轮累积，供聊天模式使用）
- 输入信息与首轮检索结果
- 逐步产生的假设与工具观察结果
- 最终报告
"""
from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage

from models.schemas import DataMismatchInput, DataQualityReport, Hypothesis


class AgentState(TypedDict, total=False):
    """LangGraph 的共享状态。"""

    # ---- 对话 ----
    # 对话消息（human/ai），使用 operator.add 累加，跨轮保留
    messages: Annotated[List[BaseMessage], operator.add]
    # 本轮用户的原始消息：可能是排查输入 JSON，也可能是追问文本
    pending_input: str

    # ---- 输入与检索 ----
    input_data: DataMismatchInput      # 原始排查输入（源表/目标表/数据量/时间窗口）
    retrieved_cases: List[str]         # 首轮按表名检索到的经验案例（文本）
    started_at: float                  # 开始时间戳（用于总超时判断）

    # ---- ReAct 推理过程 ----
    # 每轮排查开始时由 ingest 节点整体重置，因此使用普通替换通道
    hypotheses: List[Hypothesis]       # 已提出的假设（本轮）
    observations: List[str]            # 与假设一一对应的工具观察结果（本轮）
    steps_used: int                    # 已使用步数
    next_action: str                   # 路由标记：run_tools / conclude / insufficient
    error: Optional[str]               # 运行期错误信息
    llm_notes: Optional[str]           # 模型输出的补充说明（JSON 字符串，信息不足时使用）

    # ---- 最终报告 ----
    report: Optional[DataQualityReport]  # 最终结构化报告
