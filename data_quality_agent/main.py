# -*- coding: utf-8 -*-
"""数据质量排查 Agent 命令行入口。

用法：
    python main.py input.json                 # 从文件读取输入 JSON
    python main.py -                          # 从 stdin 读取输入 JSON
    python main.py input.json -o report.json  # 报告写入文件
    python main.py --demo                     # 使用内置示例输入跑一遍
    python main.py --no-llm                   # 不调用大模型（离线冒烟测试）
    python main.py --list-docs                # 只列出经验文档
    python main.py --chat                     # 启动终端对话（可持续聊天）
    python main.py --serve --port 8000        # 启动 Web 聊天服务
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from agent.graph import DataQualityAgent
from config.settings import settings
from models.schemas import DataMismatchInput, DataQualityReport

# 内置演示输入：与示例经验文档中的表名呼应
DEMO_INPUT = {
    "source_table": "ods_orders",
    "target_table": "dwd_orders",
    "source_count": 1285430,
    "target_count": 1209876,
    "time_window_start": "2026-08-01 00:00:00",
    "time_window_end": "2026-08-04 00:00:00",
    "source_schema": "ods",
    "target_schema": "dwd",
    "extra_context": "每日凌晨 1 点收到校验平台告警",
}


def setup_logging(level: str) -> None:
    """初始化日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_input(raw: str) -> DataMismatchInput:
    """解析输入 JSON 字符串为数据模型，失败时直接报错退出。"""
    try:
        data = json.loads(raw)
        return DataMismatchInput.model_validate(data)
    except Exception as exc:
        raise SystemExit("输入 JSON 解析失败：{}".format(exc))


def load_input(path: str) -> DataMismatchInput:
    """从文件（或 stdin）读取输入 JSON。"""
    if path == "-":
        return parse_input(sys.stdin.read())
    p = Path(path)
    if not p.exists():
        raise SystemExit("输入文件不存在：{}".format(p))
    return parse_input(p.read_text(encoding="utf-8"))


def output_report(report: DataQualityReport, out_path: Optional[str]) -> None:
    """打印报告；若指定了输出路径则同时写入文件。"""
    text = report.model_dump_json(indent=2, ensure_ascii=False)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print("报告已写入：{}".format(out_path))
    else:
        print(text)


def format_report_for_chat(report: DataQualityReport) -> str:
    """把报告格式化成适合对话展示的简洁文本。"""
    lines = [
        "状态：{} | 严重度：{} | 步数：{}/{}".format(
            report.status, report.severity, report.steps_used, report.total_steps
        ),
        "摘要：{}".format(report.summary),
    ]
    for i, cause in enumerate(report.root_causes, start=1):
        lines.append("")
        lines.append("根因{}：{}".format(i, cause.description))
        lines.append("证据：{}".format(cause.evidence[:200]))
        if cause.flink_ui_checks:
            lines.append("检查项：{}".format("；".join(cause.flink_ui_checks)))
        if cause.fix_suggestion:
            lines.append("修复建议：{}".format(cause.fix_suggestion))
    lines.append("")
    lines.append(
        "已检查 {} 项，未检查 {} 项".format(
            len(report.checked_items), len(report.unchecked_items)
        )
    )
    return "\n".join(lines)


def run_chat(agent: DataQualityAgent, settings) -> None:
    """终端对话循环：启动后可持续提问，直到输入 /quit。"""
    thread_id = "chat-{}".format(uuid.uuid4().hex[:12])
    last_report: Optional[DataQualityReport] = None
    print("=" * 56)
    print("数据质量排查 Agent 对话模式")
    print("命令：/demo 内置示例  /new 开启新会话  /json 查看上轮完整报告  /help 帮助  /quit 退出")
    print("=" * 56)
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "q"):
            print("再见！")
            break
        if text == "/help":
            print(
                "粘贴排查输入 JSON（含 source_table/target_table/source_count/"
                "target_count/time_window_start/time_window_end）即可开始排查；"
                "之后可以直接追问结论或要求解释。"
            )
            continue
        if text == "/new":
            thread_id = "chat-{}".format(uuid.uuid4().hex[:12])
            last_report = None
            print("已开启新会话（清除上下文）。")
            continue
        if text == "/json":
            if last_report is not None:
                print(last_report.model_dump_json(indent=2, ensure_ascii=False))
            else:
                print("还没有排查报告。")
            continue
        if text == "/demo":
            text = json.dumps(DEMO_INPUT, ensure_ascii=False)
        logging.info("本轮消息：%s", text[:120])
        # 统一入口：JSON -> 排查报告；自然语言 -> 自动提取或知识库回答
        result = agent.respond(text, thread_id)
        if result.report is not None:
            last_report = result.report
            print("Agent>")
            print(format_report_for_chat(result.report))
        else:
            print("Agent> {}".format(result.reply))


def run_server(agent: DataQualityAgent, host: str, port: int) -> None:
    """启动 Web 聊天服务（FastAPI + 内置页面）。"""
    import uvicorn

    from api import create_app

    print("Web 聊天服务已启动：http://{}:{}/".format(host, port))
    uvicorn.run(create_app(agent), host=host, port=port, log_level="info")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="数据质量排查 Agent")
    parser.add_argument(
        "input_file", nargs="?", help="输入 JSON 文件路径（- 表示从 stdin 读取）"
    )
    parser.add_argument("-o", "--output", help="报告输出文件路径")
    parser.add_argument(
        "--demo", action="store_true", help="使用内置示例输入运行"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调用大模型，仅走检索与兜底流程（离线冒烟测试）",
    )
    parser.add_argument(
        "--list-docs", action="store_true", help="只列出经验文档并退出"
    )
    parser.add_argument(
        "--chat", action="store_true", help="启动终端对话模式（可持续聊天）"
    )
    parser.add_argument(
        "--serve", action="store_true", help="启动 Web 聊天服务"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务监听端口")
    parser.add_argument(
        "--log-level", default=settings.LOG_LEVEL, help="日志级别（默认取配置）"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # 构建 Agent：启动时自动读取 data/user_experience/ 并建立向量索引
    agent = DataQualityAgent(use_llm=not args.no_llm)

    if args.chat:
        run_chat(agent, settings)
        return
    if args.serve:
        run_server(agent, args.host, args.port)
        return

    if args.list_docs:
        print("经验文档清单：")
        for doc in agent.retriever.list_docs():
            print(" -", doc)
        return

    if args.demo:
        input_data = DataMismatchInput.model_validate(DEMO_INPUT)
    elif args.input_file:
        input_data = load_input(args.input_file)
    else:
        parser.print_help()
        raise SystemExit("请提供输入 JSON 文件，或使用 --demo")

    logging.info(
        "开始排查：%s -> %s（窗口 %s ~ %s）",
        input_data.source_table,
        input_data.target_table,
        input_data.time_window_start,
        input_data.time_window_end,
    )
    report = agent.run(input_data)
    output_report(report, args.output)
    logging.info(
        "排查结束：status=%s, severity=%s, steps=%s",
        report.status,
        report.severity,
        report.steps_used,
    )


if __name__ == "__main__":
    main()
