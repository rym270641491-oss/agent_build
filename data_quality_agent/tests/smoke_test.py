# -*- coding: utf-8 -*-
"""离线冒烟测试（无需调用大模型）。

验证：
1. 启动时能读取经验文档并建立 Chroma 索引
2. 按表名检索能命中相关案例
3. 不调用大模型时，ReAct 图能正常跑完并输出“信息不足”兜底报告

运行方式：
    .venv/bin/python tests/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证从项目根目录导入包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import DataQualityAgent  # noqa: E402
from config.settings import Settings  # noqa: E402
from models.schemas import DataMismatchInput  # noqa: E402


def main() -> None:
    """执行冒烟测试并打印关键结果。"""
    settings = Settings()
    # use_llm=False：不调用大模型，仅验证检索 + 图流程
    agent = DataQualityAgent(settings=settings, use_llm=False)

    # 1) 检索测试：用表名检索，应命中“分区缺失”与“时间窗口”相关案例
    print("== 检索测试 ==")
    result = agent.retriever.format_search_results("ods_orders dwd_orders 数据量不一致")
    print(result[:800])
    print()

    # 2) 图流程测试：构造输入，跑一遍完整图
    print("== 图流程测试（无模型模式） ==")
    input_data = DataMismatchInput(
        source_table="ods_orders",
        target_table="dwd_orders",
        source_count=1285430,
        target_count=1209876,
        time_window_start="2026-08-01 00:00:00",
        time_window_end="2026-08-04 00:00:00",
        source_schema="ods",
        target_schema="dwd",
    )
    report = agent.run(input_data)
    print(report.model_dump_json(indent=2, ensure_ascii=False)[:1200])

    # 3) 校验关键约束：报告必须包含已检查项与未检查项
    assert report.status == "insufficient_info"
    assert report.checked_items, "兜底报告应包含已检查项"
    assert report.unchecked_items, "兜底报告应包含未检查项"

    # 4) Flink 检查清单工具测试：假设类型应能生成 UI 检查项
    from tools.flink_checklist import generate_flink_checklist

    print("\n== Flink 检查清单测试 ==")
    checklist = generate_flink_checklist(input_data, "partition_missing", "分区缺失")
    print(checklist[:300])
    assert "Flink UI" in checklist and "校验平台" in checklist
    assert "查看位置" in checklist
    print("\n冒烟测试通过 ✔")


if __name__ == "__main__":
    main()
