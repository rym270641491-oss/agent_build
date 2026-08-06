# -*- coding: utf-8 -*-
"""LangGraph ReAct 排查图。

节点：
- retrieve_cases : 启动时按“表名 + 时间窗口”检索经验案例
- agent          : 大模型推理，输出假设批或结论
- run_tools      : 执行 agent 指定的工具，把观察结果写回状态
- finalize       : 汇总报告（模型结论优先，“信息不足”有兜底逻辑）

循环控制：
- 步数达到 MAX_STEPS 或总时长超过 MAX_TOTAL_TIME_SECONDS 时强制收尾
- 每步最多 MAX_HYPOTHESES_PER_STEP 个假设（超出的会被截断）
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.prompts import (
    CHAT_SYSTEM_PROMPT,
    EXTRACT_SYSTEM_PROMPT,
    build_system_message,
    build_user_message,
)
from agent.state import AgentState
from config.settings import Settings
from models.schemas import (
    CheckedItem,
    DataMismatchInput,
    DataQualityReport,
    Hypothesis,
    UncheckedItem,
    compute_severity,
)
from tools.retriever import ExperienceRetriever
from tools.flink_checklist import generate_flink_checklist
from tools.sql_generator import generate_validation_sql

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """对话一轮的结果：要么是结构化报告，要么是自然语言回答。"""

    reply: str = ""
    report: Optional["DataQualityReport"] = None

# 兜底检查清单：信息不足时，据此列出尚未覆盖的未检查项
DEFAULT_CHECKLIST = [
    ("源表分区完整性", "partition_missing"),
    ("目标表分区完整性", "partition_missing"),
    ("增量同步任务执行状态", "incremental_job_failed"),
    ("去重口径一致性", "dedup_difference"),
    ("时间窗口/时区边界一致性", "time_window_difference"),
    ("字段映射与类型转换", "field_mapping_difference"),
    ("空值/非法值过滤一致性", "null_filter_difference"),
    ("同步延迟", "sync_lag"),
    ("源表重复数据", "duplicate_in_source"),
    ("抽样/限量逻辑", "sampling_difference"),
    # ---- Flink 作业侧检查项（Flink UI 可观测） ----
    ("Flink Checkpoint 状态", "checkpoint_failure"),
    ("背压/资源使用情况", "backpressure"),
    ("Exactly-Once/重复消费配置", "exactly_once_duplication"),
    ("维表 JOIN 行数变化", "dimension_join_amplification"),
    ("水位线/迟到数据处理", "watermark_stall"),
]


def build_llm(settings: Settings):
    """构建 DeepSeek 的 OpenAI 兼容客户端。

    不同版本的 langchain-openai 对超时参数命名略有差异
    （timeout / request_timeout），这里做兼容处理。
    """
    from langchain_openai import ChatOpenAI

    kwargs: Dict[str, Any] = {
        "model": settings.MODEL_NAME,
        "api_key": settings.DEEPSEEK_API_KEY,
        "base_url": settings.DEEPSEEK_BASE_URL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "max_retries": 1,
    }
    try:
        return ChatOpenAI(timeout=settings.LLM_REQUEST_TIMEOUT, **kwargs)
    except TypeError:
        # 旧版本使用 request_timeout 参数名
        return ChatOpenAI(request_timeout=settings.LLM_REQUEST_TIMEOUT, **kwargs)


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON 对象（兼容 ```json 代码块等格式）。"""
    text = text.strip()
    # 去掉 ```json ... ``` 代码块包裹
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 直接从第一个 { 截取到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("输出中未找到 JSON：{}".format(text[:200]))
    return json.loads(text[start : end + 1])


def parse_hypotheses(
    raw_hypotheses: List[Any],
    step: int,
    base_id: int,
    settings: Settings,
) -> List[Hypothesis]:
    """解析模型输出的一批假设，最多保留 MAX_HYPOTHESES_PER_STEP 个。"""
    result: List[Hypothesis] = []
    for idx, item in enumerate(
        raw_hypotheses[: settings.MAX_HYPOTHESES_PER_STEP], start=1
    ):
        if not isinstance(item, dict):
            continue
        result.append(
            Hypothesis(
                # 全局递增编号，避免不同步之间编号重复
                id=base_id + idx,
                description=str(item.get("description", "")),
                hypothesis_type=str(item.get("hypothesis_type", "custom")),
                tool_name=str(item.get("tool_name", "search_experience")),
                tool_query=str(item.get("tool_query", "")),
                step=step,
                status="proposed",
            )
        )
    return result


def parse_report(raw_report: Dict[str, Any]) -> DataQualityReport:
    """用 Pydantic 校验模型输出的报告结构，字段缺失时用默认值兜底。"""
    return DataQualityReport.model_validate(raw_report)


def build_fallback_report(
    state: AgentState, settings: Settings, reason: str
) -> DataQualityReport:
    """不使用模型结论时，根据状态中的假设/观察结果生成“信息不足”报告。"""
    input_data: Optional[DataMismatchInput] = state.get("input_data")
    hypotheses: List[Hypothesis] = state.get("hypotheses", [])
    observations: List[str] = state.get("observations", [])
    checked: List[CheckedItem] = []
    checked_types: set = set()
    obs_iter = iter(observations)
    for h in hypotheses:
        obs = next(obs_iter, "")
        checked_types.add(h.hypothesis_type)
        checked.append(
            CheckedItem(
                item="假设：{}".format(h.description),
                result="已提出并通过工具收集证据" if obs else "已提出但无工具结果",
                evidence=obs[:300],
            )
        )
    # 至少记录“按表名检索历史案例”这一已检查项，保证报告信息完整
    retrieved = state.get("retrieved_cases", [])
    if not checked and retrieved:
        checked.append(
            CheckedItem(
                item="按表名检索历史经验案例",
                result="已检索到相关案例 {} 篇".format(len(retrieved)),
                evidence=retrieved[0][:300],
            )
        )
    # 未检查项 = 兜底清单中尚未覆盖的类型
    unchecked: List[UncheckedItem] = []
    for name, htype in DEFAULT_CHECKLIST:
        if htype not in checked_types:
            unchecked.append(UncheckedItem(item=name, reason=reason))
    return DataQualityReport(
        status="insufficient_info",
        severity=(
            compute_severity(input_data.source_count, input_data.target_count)
            if input_data is not None
            else "unknown"
        ),
        summary="{}；已用时 {} 步，未能确定唯一根因，建议人工补充检查。".format(
            reason, state.get("steps_used", 0)
        ),
        root_causes=[],
        checked_items=checked,
        unchecked_items=unchecked,
        steps_used=state.get("steps_used", 0),
        total_steps=settings.MAX_STEPS,
    )


class DataQualityAgent:
    """数据质量排查 Agent：封装检索器、大模型与 LangGraph ReAct 图。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_llm: bool = True,
        build_index: bool = True,
    ):
        self.settings = settings or Settings()
        self.use_llm = use_llm
        # 启动时读取 data/user_experience/ 并建立向量索引
        self.retriever = ExperienceRetriever(self.settings, build_index=build_index)
        # 构建大模型客户端（--no-llm 模式下为 None）
        self.llm = build_llm(self.settings) if use_llm else None
        self.graph = self._build_graph()

    # ---------------- 图构建 ----------------
    def _build_graph(self) -> Any:
        """构建并编译 LangGraph 状态图。"""
        builder = StateGraph(AgentState)
        builder.add_node("ingest", self._ingest_node)
        builder.add_node("retrieve_cases", self._retrieve_cases_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("run_tools", self._run_tools_node)
        builder.add_node("finalize", self._finalize_node)
        # 对话模式下每轮都从 ingest 进入：解析消息并重置本轮排查状态
        builder.add_edge(START, "ingest")
        builder.add_edge("ingest", "retrieve_cases")
        builder.add_edge("retrieve_cases", "agent")
        # 根据 agent 的输出路由：去执行工具 / 直接收尾
        builder.add_conditional_edges(
            "agent",
            lambda state: state.get("next_action", "run_tools"),
            {
                "run_tools": "run_tools",
                "conclude": "finalize",
                "insufficient": "finalize",
            },
        )
        builder.add_edge("run_tools", "agent")
        builder.add_edge("finalize", END)
        # InMemorySaver：内存检查点，用于单次运行状态管理
        return builder.compile(checkpointer=MemorySaver())

    # ---------------- 节点实现 ----------------
    def _ingest_node(self, state: AgentState) -> Dict[str, Any]:
        """每轮对话入口：解析用户消息、重置本轮排查状态。"""
        text = (state.get("pending_input") or "").strip()
        # 用户消息是 JSON 时视为新的排查输入；否则沿用上一轮的输入（追问场景）
        new_input = state.get("input_data")
        if text.startswith("{"):
            try:
                new_input = DataMismatchInput.model_validate(json.loads(text))
            except Exception:
                pass  # JSON 解析失败则沿用旧输入，由模型在推理中提示用户
        return {
            "input_data": new_input,
            "messages": [HumanMessage(content=text or "（空消息）")],
            # 每轮重置排查过程，避免上一轮的假设/观察残留
            "retrieved_cases": [],
            "hypotheses": [],
            "observations": [],
            "steps_used": 0,
            "next_action": "run_tools",
            "started_at": time.monotonic(),
            "error": None,
            "llm_notes": None,
            "report": None,
        }

    def _retrieve_cases_node(self, state: AgentState) -> Dict[str, Any]:
        """首轮节点：按表名 + 时间窗口检索相关经验案例。"""
        input_data: Optional[DataMismatchInput] = state.get("input_data")
        if input_data is None:
            # 尚未提供排查输入：跳过检索，直接交给 agent 提示用户
            return {"retrieved_cases": []}
        query = "{} {} 数据量不一致 排查 {} {}".format(
            input_data.source_table,
            input_data.target_table,
            input_data.time_window_start,
            input_data.time_window_end,
        )
        cases = self.retriever.format_search_results(query)
        return {"retrieved_cases": [cases], "started_at": time.monotonic()}

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """ReAct 推理节点：调用大模型，输出假设批或结论。"""
        steps_used = state.get("steps_used", 0)
        started_at = state.get("started_at", time.monotonic())
        # 1) 超步数/超时检查：直接进入“信息不足”分支
        if steps_used >= self.settings.MAX_STEPS:
            return {
                "next_action": "insufficient",
                "error": "已达到最大步数 {}".format(self.settings.MAX_STEPS),
            }
        elapsed = time.monotonic() - started_at
        if elapsed > self.settings.MAX_TOTAL_TIME_SECONDS:
            return {
                "next_action": "insufficient",
                "error": "总耗时超过 {} 秒".format(self.settings.MAX_TOTAL_TIME_SECONDS),
            }
        # 没有排查输入：直接提示用户提供 JSON，不调用模型
        input_data = state.get("input_data")
        if input_data is None:
            return {
                "next_action": "insufficient",
                "steps_used": steps_used + 1,
                "error": "尚未提供排查输入：请发送包含源表/目标表/数据量/时间窗口的 JSON，或输入 /demo",
            }
        # 2) 无模型模式（离线冒烟测试）：直接按信息不足处理
        if not self.use_llm or self.llm is None:
            return {
                "next_action": "insufficient",
                "steps_used": steps_used + 1,
                "error": "未启用大模型（--no-llm 模式）",
            }

        messages = [
            build_system_message(self.settings),
            build_user_message(
                input_data,
                state.get("retrieved_cases", []),
                state.get("hypotheses", []),
                state.get("observations", []),
                steps_used,
                self.settings,
                chat_messages=state.get("messages", []),
            ),
        ]
        new_steps = steps_used + 1
        try:
            response = self.llm.invoke(messages)
            decision = extract_json(response.content)
            action = decision.get("action", "insufficient")

            if action == "hypotheses":
                # 3) 提出假设：最多 3 个，编号接续已有假设
                raw = decision.get("hypotheses", [])
                hyps = parse_hypotheses(
                    raw,
                    new_steps,
                    len(state.get("hypotheses", [])),
                    self.settings,
                )
                if not hyps:
                    return {
                        "next_action": "insufficient",
                        "steps_used": new_steps,
                        "error": "模型未输出有效假设",
                        "llm_notes": json.dumps(decision, ensure_ascii=False),
                    }
                return {
                    # hypotheses 现在是普通通道：返回“已有 + 新增”的完整列表
                    "hypotheses": state.get("hypotheses", []) + hyps,
                    "steps_used": new_steps,
                    "next_action": "run_tools",
                }

            if action == "conclude":
                # 4) 得出结论：解析并校验报告结构
                try:
                    report = parse_report(decision.get("report", {}))
                    report.steps_used = new_steps
                    report.total_steps = self.settings.MAX_STEPS
                    # 模型没给严重等级时，用数据量差异兜底计算
                    if report.severity == "unknown":
                        report.severity = compute_severity(
                            input_data.source_count,
                            input_data.target_count,
                        )
                    return {
                        "next_action": "conclude",
                        "steps_used": new_steps,
                        "report": report,
                    }
                except Exception as exc:
                    logger.warning("结论 JSON 解析失败：%s", exc)
                    return {
                        "next_action": "insufficient",
                        "steps_used": new_steps,
                        "error": "结论解析失败：{}".format(exc),
                        "llm_notes": json.dumps(decision, ensure_ascii=False),
                    }

            # 5) 模型判定信息不足（或未知动作，按信息不足处理）
            return {
                "next_action": "insufficient",
                "steps_used": new_steps,
                "error": "模型判定信息不足",
                "llm_notes": json.dumps(decision, ensure_ascii=False),
            }
        except Exception as exc:
            # 网络/超时/解析异常：兜底为信息不足，不中断整个流程
            logger.exception("调用大模型失败：%s", exc)
            return {
                "next_action": "insufficient",
                "steps_used": new_steps,
                "error": "调用大模型失败：{}".format(exc),
            }

    def _run_tools_node(self, state: AgentState) -> Dict[str, Any]:
        """执行最新一批假设对应的工具，把观察结果追加回状态。"""
        hypotheses: List[Hypothesis] = state.get("hypotheses", [])
        observations: List[str] = state.get("observations", [])
        # 本轮新增的假设 = 假设总数 - 已有观察数（两者顺序一致）
        batch = hypotheses[len(observations) :]
        input_data: DataMismatchInput = state["input_data"]
        new_observations = [
            self._execute_tool(hypothesis, input_data) for hypothesis in batch
        ]
        # observations 现在是普通通道：返回“已有 + 新增”的完整列表
        return {"observations": observations + new_observations}

    def _execute_tool(self, hypothesis: Hypothesis, input_data: DataMismatchInput) -> str:
        """执行单个假设指定的工具，返回文本形式的观察结果。"""
        tool = hypothesis.tool_name
        query = hypothesis.tool_query
        try:
            if tool == "search_experience":
                # 检索经验文档：查询词优先用工具参数，其次用假设描述
                search_query = query or hypothesis.description
                return self.retriever.format_search_results(search_query)
            if tool == "generate_validation_sql":
                sql = generate_validation_sql(
                    input_data, hypothesis.hypothesis_type, hypothesis.description
                )
                return "生成的验证 SQL：\n{}".format(sql)
            if tool == "generate_flink_checklist":
                checklist = generate_flink_checklist(
                    input_data, hypothesis.hypothesis_type, hypothesis.description
                )
                return checklist
            if tool == "list_experience_docs":
                return "经验文档清单：\n{}".format(
                    "\n".join(self.retriever.list_docs())
                )
            return "未知工具 {}，已跳过。".format(tool)
        except Exception as exc:
            return "工具执行失败：{}".format(exc)

    def _finalize_node(self, state: AgentState) -> Dict[str, Any]:
        """汇总最终报告：模型结论优先，否则用兜底报告。"""
        report = state.get("report")
        if report is not None:
            return {
                "report": report,
                # 把报告作为助手消息写入对话历史，供后续追问使用
                "messages": [
                    AIMessage(content=report.model_dump_json(indent=2, ensure_ascii=False))
                ],
            }
        # 信息不足：先生成兜底报告，再尝试用模型给出的检查清单补全
        report = build_fallback_report(
            state, self.settings, state.get("error") or "信息不足"
        )
        llm_notes = state.get("llm_notes")
        if llm_notes:
            try:
                notes = json.loads(llm_notes)
                if notes.get("checked_items"):
                    report.checked_items = [
                        CheckedItem.model_validate(x)
                        for x in notes["checked_items"]
                        if isinstance(x, dict)
                    ]
                if notes.get("unchecked_items"):
                    report.unchecked_items = [
                        UncheckedItem.model_validate(x)
                        for x in notes["unchecked_items"]
                        if isinstance(x, dict)
                    ]
                if notes.get("summary"):
                    report.summary = "{}（信息不足）".format(notes["summary"])
            except Exception as exc:
                logger.warning("解析模型补充信息失败：%s", exc)
        return {
            "report": report,
            "messages": [
                AIMessage(content=report.model_dump_json(indent=2, ensure_ascii=False))
            ],
        }

    # ---------------- 对外入口 ----------------
    def run(self, input_data: DataMismatchInput) -> DataQualityReport:
        """一次性执行完整排查（兼容原有用法），返回结构化报告。"""
        # 把输入序列化为 JSON 文本，走与聊天一致的 ingest 入口
        return self.chat_turn(
            input_data.model_dump_json(ensure_ascii=False),
            "dq-{}".format(uuid.uuid4().hex[:12]),
        )

    def chat_turn(self, text: str, thread_id: str) -> DataQualityReport:
        """对话模式下的一轮交互：发送一条消息并返回本轮报告。

        仅用于排查类消息（JSON 输入）；纯追问请使用 ask()。
        同一个 thread_id 的多轮调用会通过 MemorySaver 保留对话历史
        （messages 通道）。
        """
        result = self.graph.invoke(
            {"pending_input": text},
            config={
                # 固定 thread_id：跨轮复用同一会话状态
                "configurable": {"thread_id": thread_id},
                # 预留足够大的递归上限，避免 8 步循环被默认限制打断
                "recursion_limit": 64,
            },
        )
        report = result.get("report")
        if report is None:
            # 极端兜底：任何节点都未产出报告时，按信息不足处理
            reason = result.get("error") or "图执行未产出报告"
            if result.get("input_data") is None:
                # 连排查输入都没有时，直接构造一条提示性报告
                report = DataQualityReport(
                    status="insufficient_info",
                    severity="unknown",
                    summary="{}；请先提供排查输入 JSON。".format(reason),
                    steps_used=result.get("steps_used", 0),
                    total_steps=self.settings.MAX_STEPS,
                )
            else:
                report = build_fallback_report(result, self.settings, reason)
        return report

    def ask(self, text: str, thread_id: str) -> str:
        """自然语言回答：结合知识库检索结果与对话历史，不重新排查。"""
        if not self.use_llm or self.llm is None:
            return "（未启用大模型，无法回答；可先提供排查输入 JSON）"
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        history: List = (snapshot.values or {}).get("messages", [])
        # 检索相关经验文档，作为回答依据（自然语言也能得到有根据的答复）
        retrieval = self.retriever.format_search_results(text, top_k=3)
        content_parts = [
            "## 检索到的经验文档（仅供参考，引用数值须与原文一致）",
            retrieval,
        ]
        if history:
            content_parts.append("")
            content_parts.append("## 对话历史（之前的问答）")
            for msg in history[-6:]:
                role = "用户" if msg.type == "human" else "助手"
                raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                content_parts.append("- {}：{}".format(role, raw[:600]))
        content_parts.append("")
        content_parts.append("## 用户问题")
        content_parts.append(text)
        messages = [
            SystemMessage(content=CHAT_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(content_parts)),
        ]
        try:
            response = self.llm.invoke(messages)
            reply = str(response.content).strip()
            if not reply:
                reply = "（模型未返回有效回答）"
            # 把回答写回对话历史，供后续追问使用
            self.graph.update_state(config, {"messages": [AIMessage(content=reply)]})
            return reply
        except Exception as exc:
            return "回答失败：{}".format(exc)

    def has_history(self, thread_id: str) -> bool:
        """判断该会话是否已有对话历史。"""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return bool((snapshot.values or {}).get("messages"))

    def try_extract_input(self, text: str) -> Optional[DataMismatchInput]:
        """尝试从自然语言中提取排查输入；提取不到返回 None。"""
        if not self.use_llm or self.llm is None:
            return None
        # 启发式门槛：文本里没有类表名 token 或数字时，不浪费一次模型调用
        if not re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_.]+|\d{2,}", text):
            return None
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
                    HumanMessage(content=text),
                ]
            )
            data = extract_json(response.content)
            candidate = DataMismatchInput.model_validate(data)
            # 至少要有源表与目标表，才认为可以自动转入排查
            if candidate.source_table and candidate.target_table:
                return candidate
        except Exception as exc:
            logger.warning("自然语言提取排查输入失败：%s", exc)
        return None

    def respond(self, text: str, thread_id: str) -> ChatResult:
        """统一对话入口：
        - JSON 输入 -> 完整 ReAct 排查（返回报告）
        - 自然语言且能提取出表名/数据量 -> 自动转入排查
        - 其他自然语言（问题模式/追问） -> 基于知识库直接回答
        """
        if text.strip().startswith("{"):
            return ChatResult(report=self.chat_turn(text, thread_id))
        # 首条自然语言消息：尝试提取排查输入；已有历史的追问不再重复提取
        if not self.has_history(thread_id):
            extracted = self.try_extract_input(text)
            if extracted is not None:
                return ChatResult(
                    report=self.chat_turn(
                        extracted.model_dump_json(ensure_ascii=False), thread_id
                    )
                )
        return ChatResult(reply=self.ask(text, thread_id))
