import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import urllib.parse
import urllib.request

load_dotenv()  # 自动读取同目录下的 .env 文件


# ============================================================
# 步骤1: 定义工具的 Schema（告诉模型有哪些工具可用）
# ============================================================
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

tools=[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }
]

# ============================================================
# 步骤2: 定义工具的实际实现
# ============================================================
# WMO 天气代码 -> 中文描述
WEATHER_CODES = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}

def get_weather(city: str) -> str:
    """查询真实天气（Open-Meteo 免费 API，无需 Key）"""
    try:
        # 1. 中文城市名 -> 经纬度
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
            "name": city, "count": 1, "language": "zh", "format": "json"
        })
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))
        results = geo_data.get("results") or []
        if not results:
            return f"未找到{city}的天气信息"
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        # 2. 经纬度 -> 实时天气
        weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
        })
        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            weather_data = json.loads(resp.read().decode("utf-8"))
        current = weather_data["current"]
        desc = WEATHER_CODES.get(current["weather_code"], "未知天气")
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        return f"{city}: {desc}，{temp:g}°C，湿度{humidity}%"
    except Exception as e:
        return f"获取{city}的天气信息失败: {e}"

def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")

available_functions = {
    "get_weather": get_weather,
    "get_time": get_time
}

# ============================================================
# 步骤3: 实现完整的对话循环
# ============================================================
def run_agent(user_message: str) -> str:
    """执行一次完整的 Agent 对话（可能包含多轮工具调用）"""
    messages = [{"role": "system", "content": "你是一个天气助手，可以查询天气和时间，请用中文回答"},
                {"role": "user", "content": user_message}
                ]
    max_iterations = 5  # 限制最大迭代次数，防止无限循环
    for iteration in range(max_iterations):
        print(f"\n=== 第 {iteration + 1} 轮对话 ===")

        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = messages,
            tools = tools,
            tool_choice = "auto",
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # 情况1: 模型决定调用工具
        if choice.finish_reason == "tool_calls":
            print(f"🔧 模型决定调用 {len(assistant_message.tool_calls)} 个工具")
            
            # 把模型的工具调用请求加入消息历史
            messages.append(assistant_message)
            
            # 逐个执行工具调用
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"  调用: {func_name}({func_args})")
                
                # 执行工具
                func = available_functions.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"错误: 未知工具 {func_name}"
                
                print(f"  结果: {result}")
                
                # 将工具结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
            
            # 继续循环，让模型基于工具结果生成回答
            
        # 情况2: 模型直接回答（不再调用工具）
        else:
            print(f"✅ 模型完成回答")
            return assistant_message.content
    
    return "达到最大迭代次数，未能完成任务"

# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试1: 需要调用工具的查询
    result = run_agent("今天北京天气怎么样？上海呢？")
    print(f"\n最终回答:\n{result}")

    # 测试2: 不需要工具的查询
    result = run_agent("你好，请介绍一下你自己")
    print(f"\n最终回答:\n{result}")
