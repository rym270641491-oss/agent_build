"""
example_6.py：在代码里接入 jcodemunch MCP 服务器

流程: 启动 jcodemunch (stdio) → 拉取工具列表 → 转成 OpenAI 格式 →
      DeepSeek 多轮对话（模型可以随时调用 jcodemunch 的代码探索工具）

依赖: mcp（已随 jcodemunch-mcp 自动装进 .venv）、openai、python-dotenv
用法:
  python example_6.py --selftest          # 只测试 MCP 连接和工具列表，不调用 API
  python example_6.py "你的问题"          # 完整跑一次 Agent 对话（会调用 DeepSeek API）
"""
import argparse
import asyncio
import contextlib
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# jcodemunch 可执行文件（就是 .venv 里安装的入口脚本）
JCODEMUNCH_CMD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv", "bin", "jcodemunch-mcp",
)

# DeepSeek 对话模型（如需更换直接改这里）
LLM_MODEL = "deepseek-chat"

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY！请在项目根目录的 .env 文件中填写。"
    )

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ============================================================
# jcodemunch MCP 客户端封装
# ============================================================
class JCodeMunch:
    """连接 jcodemunch 服务器：拉取工具、执行工具"""

    def __init__(self, cwd: str | None = None):
        self.params = StdioServerParameters(
            command=JCODEMUNCH_CMD,
            args=[],
            cwd=cwd or os.getcwd(),
        )
        self.session = None
        self._stack = None

    async def __aenter__(self):
        self._stack = contextlib.AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(self.params)
        )
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._stack.__aexit__(*exc)

    async def list_tools(self):
        """返回 MCP 工具对象列表"""
        result = await self.session.list_tools()
        return result.tools

    @staticmethod
    def to_openai_tools(tools) -> list[dict]:
        """把 MCP 工具转成 OpenAI chat.completions 需要的 tools 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in tools
        ]

    async def call_tool(self, name: str, arguments: dict | None) -> str:
        """执行 MCP 工具，把结果整理成纯文本"""
        result = await self.session.call_tool(name, arguments=arguments or {})
        parts = []
        for item in result.content:
            if getattr(item, "type", None) == "text" and item.text:
                parts.append(item.text)
            elif getattr(item, "type", None) == "image":
                parts.append("[图片结果]")
        if not parts and result.structuredContent is not None:
            parts.append(
                json.dumps(result.structuredContent, ensure_ascii=False, indent=2)
            )
        text = "\n".join(parts) if parts else "(无返回内容)"
        return f"[工具执行错误] {text}" if result.isError else text


# ============================================================
# Agent 对话循环（和 example_t3.py 相同的结构）
# ============================================================
async def run_agent(user_message: str) -> str:
    """执行一次完整的 Agent 对话，模型可随时调用 jcodemunch 工具"""
    async with JCodeMunch() as jcm:
        mcp_tools = await jcm.list_tools()
        tools = jcm.to_openai_tools(mcp_tools)
        print(f"🔌 已连接 jcodemunch，可用工具 {len(tools)} 个")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个代码分析助手。你拥有 jcodemunch 提供的代码探索工具"
                    "（搜索符号、检索代码上下文、查看源码等）。"
                    "当问题涉及项目代码时，请先调用合适的工具获取信息，再回答。"
                ),
            },
            {"role": "user", "content": user_message},
        ]

        max_iterations = 6  # 限制最大迭代次数，防止无限循环
        for iteration in range(max_iterations):
            print(f"\n=== 第 {iteration + 1} 轮对话 ===")

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_message = choice.message

            # 情况1: 模型决定调用工具
            if choice.finish_reason == "tool_calls":
                print(f"🔧 模型决定调用 {len(assistant_message.tool_calls)} 个工具")
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments or "{}")
                    print(f"  调用: {func_name}({func_args})")

                    result_text = await jcm.call_tool(func_name, func_args)
                    print(f"  结果: {result_text[:200]}{'…' if len(result_text) > 200 else ''}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result_text),
                        }
                    )

                # 继续循环，让模型基于工具结果作答
            else:
                # 情况2: 模型直接回答
                print("✅ 模型完成回答")
                return assistant_message.content

    return "达到最大迭代次数，未能完成任务"


# ============================================================
# 自检：只连 MCP 服务器，不调用 API
# ============================================================
async def selftest() -> None:
    async with JCodeMunch() as jcm:
        tools = await jcm.list_tools()
        print(f"MCP 连接成功，共 {len(tools)} 个工具：")
        for t in tools[:30]:
            desc = (t.description or "").strip().replace("\n", " ")
            print(f"  - {t.name}: {desc[:70]}")
        if len(tools) > 30:
            print(f"  …（其余 {len(tools) - 30} 个略）")


def main() -> None:
    parser = argparse.ArgumentParser(description="在代码里接入 jcodemunch MCP")
    parser.add_argument("--selftest", action="store_true",
                        help="只测试 MCP 连接，不调用 API")
    parser.add_argument("question", nargs="?", default=None,
                        help="要问的问题（默认给一个示例问题）")
    args = parser.parse_args()

    if args.selftest:
        asyncio.run(selftest())
        return

    question = args.question or (
        "在这个项目里，example_t3.py 中定义了哪两个工具函数？它们分别做什么？"
    )
    result = asyncio.run(run_agent(question))
    print(f"\n最终回答:\n{result}")


if __name__ == "__main__":
    main()
