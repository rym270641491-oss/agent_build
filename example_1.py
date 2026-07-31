import os
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

response = client.chat.completions.create(
    model = "deepseek-chat",
    messages = [{"role": "user", "content" : "用JSON格式返回：姓名张三，年龄25，城市北京"}],
    max_tokens = 200
)

# --- 1. 基本信息 ---
print(f"模型: {response.model}")
print(f"id: {response.id}")
print(f"创建时间: {response.created}")

# --- 2. choices 列表 ---
# 通常只有一个 choice（除非设置 n > 1）
print(f"\nChoices 数量:{len(response.choices)}")
choice = response.choices[0]
print(f"结束原因: {choice.finish_reason}")
print(f"索引: {choice.index}")

# --- 3. 消息内容 ---
message = choice.message
print(f"\n角色: {message.role}")
print(f"内容\n:{message.content}")

# --- 4. Token 用量 ---
print(f"\n=== Token 用量 ===")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# --- 5. finish_reason 的含义 ---
"""
finish_reason 的可能值：
- "stop": 正常结束，模型完成了回答
- "length": 达到了 max_tokens 限制，回答被截断
- "content_filter": 内容被安全过滤器拦截
- "tool_calls": 模型决定调用工具（见 Function Calling 章节）
- "function_call": 同上（旧版 API）
"""

def safe_chat(client: OpenAI, messages: list, max_tokens: int = 500) -> dict:
    """安全的对话调用，处理各种结束情况"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
    )
    
    choice = response.choices[0]
    
    if choice.finish_reason == "stop":
        return {"status": "ok", "content": choice.message.content}
    elif choice.finish_reason == "length":
        return {
            "status": "truncated",
            "content": choice.message.content,
            "warning": f"回答被截断(max_tokens={max_tokens})，建议增大 max_tokens 或分段处理"
        }
    elif choice.finish_reason == "content_filter":
        return {
            "status": "filtered",
            "content": None,
            "warning": "内容被安全过滤器拦截，请检查输入内容"
        }
    elif choice.finish_reason == "tool_calls":
        return {
            "status": "tool_calls",
            "content": None,
            "tool_calls": choice.message.tool_calls,
        }
    else:
        return {"status": "unknown", "content": choice.message.content}