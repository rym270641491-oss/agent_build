# -*- coding: utf-8 -*-
"""ReAct 循环测试（使用假模型，无需网络与 API 密钥）。

验证：
1. “提出假设 -> 工具执行 -> 模型得出结论”的完整链路
2. 模型一直提假设时，8 步后自动输出“信息不足”

运行方式：
    .venv/bin/python tests/react_loop_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 保证从项目根目录导入包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from agent.graph import DataQualityAgent  # noqa: E402
from config.settings import Settings  # noqa: E402
from models.schemas import DataMismatchInput  # noqa: E402


def make_input() -> DataMismatchInput:
    """构造一个与示例经验文档呼应的测试输入。"""
    return DataMismatchInput(
        source_table="ods_orders",
        target_table="dwd_orders",
        source_count=1285430,
        target_count=1209876,
        time_window_start="2026-08-01 00:00:00",
        time_window_end="2026-08-04 00:00:00",
        source_schema="ods",
        target_schema="dwd",
    )


def test_conclude_path() -> None:
    """模型先提 2 个假设，工具执行后再给出结论。"""
    # 第 1 步：提出 3 个假设（生成 SQL、生成 Flink 检查清单、检索经验）
    step1 = json.dumps(
        {
            "action": "hypotheses",
            "hypotheses": [
                {
                    "description": "目标表 08-02 分区缺失",
                    "hypothesis_type": "partition_missing",
                    "tool_name": "generate_validation_sql",
                    "tool_query": "partition_missing",
                },
                {
                    "description": "生成 Flink UI 检查清单验证分区缺失",
                    "hypothesis_type": "partition_missing",
                    "tool_name": "generate_flink_checklist",
                    "tool_query": "partition_missing",
                },
                {
                    "description": "检索分区缺失相关历史案例",
                    "hypothesis_type": "partition_missing",
                    "tool_name": "search_experience",
                    "tool_query": "ods_orders dwd_orders 分区缺失",
                },
            ],
        },
        ensure_ascii=False,
    )
    # 第 2 步：根据观察结果给出结论
    step2 = json.dumps(
        {
            "action": "conclude",
            "report": {
                "status": "root_cause_found",
                "severity": "high",
                "summary": "目标表 08-02 分区缺失，已生成验证 SQL",
                "root_causes": [
                    {
                        "description": "目标表分区缺失",
                        "evidence": "检索到的经验案例与分区对比 SQL",
                        "flink_ui_checks": [
                            "Flink UI 查看 Sink 端 records-sent 走势",
                            "校验平台按天对比确认缺口日期",
                        ],
                        "validation_sql": "SELECT dt, COUNT(*) FROM dwd.dwd_orders GROUP BY dt;",
                        "fix_suggestion": "补跑失败任务并增加分区完整性校验",
                        "confidence": 0.9,
                    }
                ],
                "checked_items": [
                    {"item": "分区完整性", "result": "已检查", "evidence": "SQL"}
                ],
                "unchecked_items": [
                    {"item": "去重口径", "reason": "已找到根因，无需继续"}
                ],
            },
        },
        ensure_ascii=False,
    )

    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    # 用假模型替换真实 LLM，返回脚本化的两步回答
    agent.llm = FakeListChatModel(responses=[step1, step2])
    report = agent.run(make_input())

    assert report.status == "root_cause_found"
    assert len(report.root_causes) == 1
    assert report.root_causes[0].fix_suggestion, "根因应包含修复建议"
    assert report.root_causes[0].flink_ui_checks, "根因应包含 Flink UI 检查项"
    assert report.steps_used == 2
    print(
        "结论链路测试通过 ✔  steps: {}  severity: {}".format(
            report.steps_used, report.severity
        )
    )


def test_step_limit() -> None:
    """模型一直提假设，8 步后应强制输出“信息不足”。"""
    hypothesis_step = json.dumps(
        {
            "action": "hypotheses",
            "hypotheses": [
                {
                    "description": "继续排查的假设",
                    "hypothesis_type": "custom",
                    "tool_name": "search_experience",
                    "tool_query": "test",
                }
            ],
        },
        ensure_ascii=False,
    )
    # 第 1~8 步都用同一个“提假设”回答，第 9 步触发步数上限
    responses = [hypothesis_step] * 8

    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    agent.llm = FakeListChatModel(responses=responses)
    report = agent.run(make_input())

    assert report.status == "insufficient_info"
    assert report.steps_used == 8, "应在第 8 步停止"
    assert report.checked_items, "应有已检查项"
    assert report.unchecked_items, "应有未检查项"
    print("8 步限制测试通过 ✔  steps: {}".format(report.steps_used))


def test_chat_multiturn() -> None:
    """对话模式：同一 thread_id 多轮消息，对话历史跨轮保留。"""
    input_json = json.dumps(make_input().model_dump(), ensure_ascii=False)
    # 第 1 轮：提出 1 个假设 -> 工具执行 -> 给出结论
    step1 = json.dumps(
        {
            "action": "hypotheses",
            "hypotheses": [
                {
                    "description": "目标表 08-02 分区缺失",
                    "hypothesis_type": "partition_missing",
                    "tool_name": "generate_flink_checklist",
                    "tool_query": "partition_missing",
                }
            ],
        },
        ensure_ascii=False,
    )
    step2 = json.dumps(
        {
            "action": "conclude",
            "report": {
                "status": "root_cause_found",
                "severity": "high",
                "summary": "目标表分区缺失",
                "root_causes": [
                    {
                        "description": "目标表分区缺失",
                        "evidence": "检索案例",
                        "fix_suggestion": "补跑任务",
                        "confidence": 0.8,
                    }
                ],
                "checked_items": [],
                "unchecked_items": [],
            },
        },
        ensure_ascii=False,
    )
    # 第 2 轮：用户追问，模型直接回答（结论路径）
    step3 = json.dumps(
        {
            "action": "conclude",
            "report": {
                "status": "root_cause_found",
                "severity": "medium",
                "summary": "回答追问：依据是检索到的历史案例与数据量缺口",
                "root_causes": [],
                "checked_items": [{"item": "追问回答", "result": "已说明"}],
                "unchecked_items": [],
            },
        },
        ensure_ascii=False,
    )

    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    agent.llm = FakeListChatModel(responses=[step1, step2, step3])
    thread_id = "chat-test-1"

    report1 = agent.chat_turn(input_json, thread_id)
    assert report1.status == "root_cause_found"
    # 第二轮追问使用同一 thread_id，应保留对话历史
    report2 = agent.chat_turn("为什么怀疑分区缺失？", thread_id)
    assert report2.status == "root_cause_found"

    # 对话历史应为 2 条 human + 2 条 ai
    snapshot = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = snapshot.values.get("messages", [])
    assert len(messages) == 4, "对话历史应跨轮保留，实际 {} 条".format(len(messages))
    print("多轮对话测试通过 ✔  messages: {}".format(len(messages)))


def test_chat_followup_answer() -> None:
    """追问路径：非 JSON 消息直接返回自然语言回答并写入对话历史。"""
    from langchain_core.messages import AIMessage, HumanMessage

    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    agent.llm = FakeListChatModel(
        responses=["根据历史案例，该表曾出现目标值为 0 且 Flink 任务停止的情况。"]
    )
    thread_id = "chat-test-2"
    config = {"configurable": {"thread_id": thread_id}}
    # 预置一轮排查历史（1 条 human + 1 条 ai）
    agent.graph.update_state(
        config,
        {"messages": [HumanMessage(content="排查输入"), AIMessage(content="排查报告")]},
    )
    reply = agent.ask("为什么怀疑任务停止？", thread_id)
    assert "任务停止" in reply
    # 历史 2 条 + 追问回答 1 条
    snapshot = agent.graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    assert len(messages) == 3, "追问回答应写入历史，实际 {} 条".format(len(messages))
    print("追问回答测试通过 ✔  reply: {}".format(reply[:40]))


def test_respond_natural_language_pattern() -> None:
    """自然语言描述问题模式（无表名/数字）：应直接基于知识库回答，不要求 JSON。"""
    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    agent.llm = FakeListChatModel(
        responses=[
            "这种“目标值大于源值且两边都在减少”的模式，通常是源表在删数据、"
            "目标表还在同步中；经验文档里对应案例建议着重看目标值小于源值的情况。"
        ]
    )
    thread_id = "chat-test-3"
    result = agent.respond(
        "我遇到一个新问题，目标表比源表数据量大，但是两个表数据量都在不断减少，问题可能是什么",
        thread_id,
    )
    assert result.report is None
    assert "删数据" in result.reply
    print("自然语言模式回答测试通过 ✔  reply: {}".format(result.reply[:40]))


def test_respond_extract_and_investigate() -> None:
    """自然语言含表名与数据量：应自动提取并转入完整排查。"""
    extract_json = json.dumps(
        {
            "source_table": "ods_orders",
            "target_table": "dwd_orders",
            "source_count": 1285430,
            "target_count": 1209876,
            "time_window_start": "2026-08-01 00:00:00",
            "time_window_end": "2026-08-04 00:00:00",
            "source_schema": None,
            "target_schema": None,
            "extra_context": None,
        },
        ensure_ascii=False,
    )
    step1 = json.dumps(
        {
            "action": "hypotheses",
            "hypotheses": [
                {
                    "description": "目标表分区缺失",
                    "hypothesis_type": "partition_missing",
                    "tool_name": "generate_flink_checklist",
                    "tool_query": "partition_missing",
                }
            ],
        },
        ensure_ascii=False,
    )
    step2 = json.dumps(
        {
            "action": "conclude",
            "report": {
                "status": "root_cause_found",
                "severity": "medium",
                "summary": "目标表分区缺失",
                "root_causes": [],
                "checked_items": [],
                "unchecked_items": [],
            },
        },
        ensure_ascii=False,
    )
    agent = DataQualityAgent(settings=Settings(), use_llm=True)
    agent.llm = FakeListChatModel(responses=[extract_json, step1, step2])
    thread_id = "chat-test-4"
    result = agent.respond(
        "帮我排查 ods_orders 和 dwd_orders，源表 128 万，目标表 121 万，窗口 8 月 1 日到 8 月 4 日",
        thread_id,
    )
    assert result.report is not None
    assert result.report.status == "root_cause_found"
    print(
        "自然语言自动提取排查测试通过 ✔  status: {}".format(result.report.status)
    )


if __name__ == "__main__":
    test_conclude_path()
    test_step_limit()
    test_chat_multiturn()
    test_chat_followup_answer()
    test_respond_natural_language_pattern()
    test_respond_extract_and_investigate()
    print("\nReAct 循环测试全部通过 ✔")
