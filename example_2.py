import os 
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

#response = client.chat.completions.create(
#    model = "deepseek-chat",
#    messages = [{"role": "user", "content": "用500字介绍一下深度学习的基本概念和应用"}],
#    max_tokens = 500
#)

def stream_chat(prompt:str):
    """流式输出示例"""
    full_response = ""
    usage = None

    with client.chat.completions.stream(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream_options={"include_usage": True}  # 需要在流式中获取 usage
    ) as stream:
        for event in stream:
            # 累积增量内容
            if event.type == "content.delta":
                full_response += event.delta
            # 最后一个 chunk 事件带有 usage 统计
            if event.type == "chunk" and event.chunk.usage:
                usage = {
                    "prompt": event.chunk.usage.prompt_tokens,
                    "completion": event.chunk.usage.completion_tokens,
                    "total": event.chunk.usage.total_tokens
                }

    return usage, full_response


if __name__ == "__main__":
    prompt = "用500字介绍一下深度学习的基本概念和应用"
    usage, response_text = stream_chat(prompt)
    print("=" * 50)
    print("完整回复：")
    print(response_text)
    print("=" * 50)
    if usage:
        print(f"Token 消耗 — 输入: {usage['prompt']}, 输出: {usage['completion']}, 总计: {usage['total']}")