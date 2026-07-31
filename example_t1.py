import os
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url= "https://api.deepseek.com"
)

def stream_chat(prompt:str):
    full_response = ""
    usage = None

    with client.chat.completions.stream(
        model = "deepseek-chat",
        messages = [{"role": "user","content":prompt}],
        maxtokens = 500,
        temperature = 0.3,
        top_p = 0.8,
        stream_options = {"include_usage": True}
    ) as stream:
        for event in stream:
            if event.type == "conteng.delta"
    