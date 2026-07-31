import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 自动读取同目录下的 .env 文件

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY！请在项目根目录的 .env 文件中填写，例如：\n"
        "DEEPSEEK_API_KEY=sk-你的key"
    )

client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com"
)

def stream_chat(prompt: str):
    full_response = ""

    with client.chat.completions.stream(
        model = "deepseek-chat",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 500,
        temperature = 0.6,
        top_p = 0.8,
        stream_options = {"include_usage": True}
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                # 流式输出：每收到一块内容就立即打印，不换行、强制刷新
                print(event.delta, end="", flush=True)
                full_response += event.delta

    print()  # 流结束后换行
    return full_response


if __name__ == "__main__":
    prompt = "你是一个轻小说创作者，请设计一个有意思的主题，并写出1000字的开头"
    response_text = stream_chat(prompt)
    print("=" * 30)
    print("完整回复已保存到变量中")

    