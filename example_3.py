import json
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

@app.post("/chat")  # 路由装饰器
async def chat_stream(prompt: str):
    async def generate():
        with client.chat.completions.stream(
            model = "deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            stream_options = {"include_usage": True}
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    data = json.dumps({"content": event.delta}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
