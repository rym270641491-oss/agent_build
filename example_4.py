from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取同目录下的 .env 文件

api_key = os.getenv("DEEPSEEK_API_KEY")

def get_embedding(text: str,model: str = "text-embedding-3-small") -> list:
    """获取文本的向量表示"""
    text = text.replace("\n","")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding  

#测试
text1 = "数据库连接超时导致服务不可用"
text2 = "MySQL连接池耗尽，应用报错"
text3 = "今天天气很好适合出去散步"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度（-1到1，1表示完全相同）"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    print(f"text1 vs text2 相似度: {cosine_similarity(emb1, emb2):.4f}")  # 高（都是数据库问题）
    print(f"text1 vs text3 相似度: {cosine_similarity(emb1, emb3):.4f}")  # 低（话题不同）

# Embedding 模型对比
"""
text-embedding-3-small: 1536维, $0.02/1M tokens, 适合大多数场景
text-embedding-3-large: 3072维, $0.13/1M tokens, 精度更高
text-embedding-ada-002: 1536维, $0.10/1M tokens, 旧版（建议迁移）

中文场景推荐:
- bge-large-zh-v1.5 (BAAI): 1024维, 开源免费, 中文效果好
- m3e-base (Moka): 768维, 开源免费, 中文效果好
- text2vec-large-chinese: 1024维, 开源免费
"""