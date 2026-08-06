import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "判断以下新闻是什么类型的"},
    {"role": "user", "content": "新闻内容：据新华社报道，近日，国家统计局发布了最新的经济数据，显示我国经济保持稳定增长。专家表示，这一数据反映了我国经济结构的优化和产业升级的成效。"},
    {"role": "assistant", "content": "经济"},
    {"role": "user", "content" : "近日tf家族要在上海举办演唱会，吸引大量粉色前来观看"},
    {"role": "assistant", "content": "娱乐"},
    {"role": "user", "content" : "近日，某地发生了一起严重的交通事故，造成多人伤亡。警方正在调查事故原因，并呼吁公众注意交通安全。"},
    {"role": "assistant", "content": "社会"},
    #让agent自己回答下面这个问题
    {"role": "user", "content": "据美国新闻报道，其一所军工厂在四年里累计投入5亿美元用于炮弹产线的建设和升级，但累计产量却为0"},
]
response = client.chat.completions.create(
    model = "deepseek-v4-flash",
    messages = messages,
    temperature = 0
)
answer =  response.choices[0].message.content
print(answer)