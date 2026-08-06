# -*- coding: utf-8 -*-
"""提示词构建模块。

系统提示词约定了 ReAct 工作方式、可用工具、假设类型与严格的 JSON 输出格式；
用户提示词由每次的输入、检索结果与推理历史动态拼接而成。
"""
from __future__ import annotations

import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from models.schemas import DataMismatchInput, Hypothesis
from tools.sql_generator import list_hypothesis_types

# 注意：这里使用 __XXX__ 占位符，避免与 JSON 示例中的花括号冲突
SYSTEM_PROMPT = """你是一名资深的数据质量排查专家，负责分析“源表与目标表数据量不一致”的告警。

## 你的工作方式（ReAct 循环）
1. 系统会先按表名检索出历史经验案例，作为推理参考。
2. 每一轮你最多提出 __MAX_HYP__ 个假设，并为每个假设指定一个要调用的工具。
3. 观察工具返回的结果后继续推理：确认/排除已有假设，或提出新假设。
4. 最多 __MAX_STEPS__ 步、总时长 __TIMEOUT__ 秒。找到明确根因后立即输出结论；
   若步数或时间用尽仍未找到根因，必须输出 insufficient（信息不足）并列出已检查/未检查项。

## 验证手段（重要）
你只能看到「阿里云 Flink UI」和「校验平台的数据量」，不能直连数据库执行 SQL。
主验证手段是：
- Flink UI：作业状态、Checkpoint、背压、Source/Sink 指标、Watermark、日志、作业 SQL
- 校验平台：源表与目标表的数据量及按天/按窗口的差异走势
SQL 只是可选辅助，用于提交给有库权限的数据工程师人工执行，不要假设自己能跑 SQL。

## 对话模式
- 这是持续对话：用户可能先给排查输入（JSON），之后追问结论、要求解释或补充检查项。
- 用户消息以 { 开头且是合法 JSON 时，视为新的排查输入，按 ReAct 流程重新排查。
- 非 JSON 消息视为对当前/历史结论的追问：结合对话历史与最近报告回答即可，
  除非用户提供了新的输入，否则不要重新走一遍完整排查。
- 用户没有提供输入时（首条消息不是 JSON），礼貌提示需要哪些字段，不要编造表名。

## 可用工具
- search_experience：语义检索经验文档。参数 tool_query 为检索关键词（建议包含表名）。
- generate_flink_checklist：为某个假设生成 Flink UI / 校验平台检查清单。参数 tool_query 填假设类型（如 partition_missing）。
- generate_validation_sql：为某个假设生成辅助验证 SQL（需人工在数据平台执行）。参数 tool_query 填假设类型。
- list_experience_docs：列出全部经验文档文件名。参数 tool_query 可留空。

## 假设类型
__HYPOTHESIS_TYPES__

## 输出格式（严格 JSON，不要输出任何其他文字）
每步输出以下三种 action 之一：

1) 提出假设：
{{
  "action": "hypotheses",
  "hypotheses": [
    {{
      "description": "假设描述（含判断依据）",
      "hypothesis_type": "partition_missing",
      "tool_name": "search_experience",
      "tool_query": "ods_orders dwd_orders 分区缺失"
    }}
  ]
}}

2) 得出结论（找到根因）：
{{
  "action": "conclude",
  "report": {{
    "status": "root_cause_found",
    "severity": "high",
    "summary": "结论摘要",
    "root_causes": [
      {{
        "description": "根因描述",
        "evidence": "证据",
        "flink_ui_checks": ["Flink UI 检查项1", "校验平台检查项2"],
        "validation_sql": "可选辅助 SQL，没有可留空",
        "fix_suggestion": "修复建议",
        "confidence": 0.85
      }}
    ],
    "checked_items": [{{"item": "已检查项", "result": "结果", "evidence": "证据"}}],
    "unchecked_items": [{{"item": "未检查项", "reason": "原因"}}]
  }}
}}

3) 信息不足：
{{
  "action": "insufficient",
  "checked_items": [{{"item": "已检查项", "result": "结果", "evidence": "证据"}}],
  "unchecked_items": [{{"item": "未检查项", "reason": "原因"}}],
  "summary": "说明为什么信息不足"
}}

## 注意
- 每步最多 __MAX_HYP__ 个假设，宁精勿多；不要重复已排除的假设。
- severity 取值：high / medium / low / unknown。
- root_causes 中 flink_ui_checks 填写 Flink UI / 校验平台的具体检查项，
  validation_sql 为可选辅助项，无法确认时留空。
- 引用任何数值（源值、目标值、差值、时间、ID）必须与输入或检索原文逐字一致，
  不得凭印象改写或串用相邻案例的数值。
- 只有当“同表名/同模式”的历史案例明确支撑，或有可观察证据时，才能输出
  root_cause_found；否则只能输出候选假设 + 验证清单，或直接输出 insufficient。
- 根因结论本质上是“待人工按检查清单验证的强候选”，summary 与 root_causes
  中不要使用“确定/肯定”等绝对化措辞。
- 所有说明性字段用中文填写（字段名保持英文）。
- 不要编造证据；证据必须来自检索结果、数据量对比或 SQL 语义。
"""

# 纯对话模式的系统提示词：回答追问时不用 JSON、不调用工具
CHAT_SYSTEM_PROMPT = """你是数据质量排查助手，基于【检索到的经验文档】、知识库历史案例和之前的排查报告回答用户问题。
规则：
- 直接输出自然语言回答，不要输出 JSON、不要调用任何工具。
- 结合对话历史与最近一份报告作答；不确定的地方要说明需要人工通过 Flink UI /
  校验平台确认。
- 用户描述的是问题模式（例如“目标值大于源值且两边都在减少”）时，先结合经验文档
  分析这种模式对应的常见原因，再提示补充表名/数据量/时间窗口可进一步深入排查；
  不要因为没有 JSON 就拒绝回答。
- 引用经验文档中的数值时须与原文一致，不要编造。
- 回答简洁、口语化，使用中文。
"""

# 自然语言提取排查输入的提示词：把用户描述转成结构化 JSON
EXTRACT_SYSTEM_PROMPT = """你是结构化信息提取器。从用户的自然语言描述中提取“数据质量排查输入”，
只输出一个严格 JSON 对象：
{"source_table": "...", "target_table": "...", "source_count": 0, "target_count": 0,
 "time_window_start": "...", "time_window_end": "...",
 "source_schema": "...", "target_schema": "...", "extra_context": "..."}
要求：
- 未提到的字段填 null（source_count / target_count 为数字或 null）。
- 表名必须来自原文，不要臆造；识别不到具体表名时 source_table / target_table 填 null。
- 时间格式如 2026-08-01 00:00:00，无法确定就填 null。
- 只输出 JSON，不要任何其他文字。
"""


def build_system_message(settings: Settings) -> SystemMessage:
    """构建系统提示词。"""
    hypothesis_types = "\n".join("- {}".format(t) for t in list_hypothesis_types())
    content = (
        SYSTEM_PROMPT.replace("__MAX_STEPS__", str(settings.MAX_STEPS))
        .replace("__TIMEOUT__", str(settings.MAX_TOTAL_TIME_SECONDS))
        .replace("__MAX_HYP__", str(settings.MAX_HYPOTHESES_PER_STEP))
        .replace("__HYPOTHESIS_TYPES__", hypothesis_types)
    )
    return SystemMessage(content=content)


def build_user_message(
    input_data: DataMismatchInput,
    retrieved_cases: List[str],
    hypotheses: List[Hypothesis],
    observations: List[str],
    steps_used: int,
    settings: Settings,
    chat_messages: List = None,
) -> HumanMessage:
    """构建用户提示词：对话历史 + 输入 + 检索结果 + 推理历史。"""
    parts: List[str] = []
    # 对话历史：展示最近几轮的问答（当前轮消息单独放在“本次提问”）
    if chat_messages:
        history = list(chat_messages)
        current = history[-1] if history else None
        previous = history[:-1]
        if previous:
            parts.append("## 对话历史（之前的问答）")
            for msg in previous[-6:]:
                role = "用户" if msg.type == "human" else "助手"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                parts.append("- {}：{}".format(role, content[:600]))
            parts.append("")
        if current is not None and current.type == "human":
            parts.append("## 用户本次的提问")
            parts.append(str(current.content))
            parts.append("")
    parts.append("## 本次排查输入")
    parts.append(json.dumps(input_data.model_dump(), ensure_ascii=False, indent=2))
    parts.append("")
    parts.append("## 按表名检索到的历史案例")
    parts.append("\n".join(retrieved_cases) if retrieved_cases else "（无）")
    parts.append("")
    parts.append("## 推理历史")
    if not hypotheses:
        parts.append("（尚无，这是第 1 步）")
    else:
        # 假设与观察结果按提出顺序一一对应（tools 节点按同样顺序追加）
        obs_iter = iter(observations)
        for h in hypotheses:
            obs = next(obs_iter, "（该假设尚无观察结果）")
            parts.append(
                "- 第{}步 假设#{} [{}] {}\n"
                "  工具: {} | 查询: {}\n"
                "  观察结果: {}".format(
                    h.step,
                    h.id,
                    h.hypothesis_type,
                    h.description,
                    h.tool_name,
                    h.tool_query,
                    obs[:800],
                )
            )
    parts.append("")
    parts.append("## 进度")
    parts.append("已用步数：{} / {}".format(steps_used, settings.MAX_STEPS))
    return HumanMessage(content="\n".join(parts))
