# ============================================================
# example_4.py：文本相似度计算（Embedding + 余弦相似度）
#
# 说明：DeepSeek API 目前不提供 embedding 接口（实测 /v1/embeddings 返回 404），
#       所以这里改用本地开源中文模型 bge-small-zh-v1.5：
#       无需 API Key、无需付费，首次运行会自动下载模型（约 100MB）。
# 依赖安装：pip install fastembed
# ============================================================

import numpy as np
from fastembed import TextEmbedding


# ============================================================
# 步骤1: 加载本地中文 Embedding 模型（只加载一次，全局复用）
# ============================================================
# 可选模型（按需替换 model_name）:
# - BAAI/bge-small-zh-v1.5 (512维，轻量快速，推荐)
# - BAAI/bge-base-zh-v1.5  (768维，精度更高，体积更大)
# - BAAI/bge-large-zh-v1.5 (1024维，精度最高，体积最大)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")


# ============================================================
# 步骤2: 定义"文本 -> 向量"的函数
# ============================================================
def get_embedding(text: str) -> list:
    """获取文本的向量表示（返回 512 维的浮点数列表）"""
    # 去掉换行符，避免换行干扰向量的生成
    text = text.replace("\n", " ")

    # embed() 接收一个文本列表，返回生成器；取第一条并转成 Python list
    return list(embedding_model.embed([text]))[0].tolist()


# ============================================================
# 步骤3: 定义余弦相似度计算函数
# ============================================================
def cosine_similarity(a: list, b: list) -> float:
    """计算两个向量的余弦相似度（范围 -1 ~ 1，越接近 1 表示越相似）"""
    a = np.array(a)
    b = np.array(b)
    # 余弦相似度 = 两向量点积 / (向量a的长度 × 向量b的长度)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# 步骤4: 测试（只有直接运行本文件时才执行）
# ============================================================
if __name__ == "__main__":
    text1 = "数据库连接超时导致服务不可用"
    text2 = "MySQL连接池耗尽，应用报错"
    text3 = "今天天气很好适合出去散步"

    # 把每段文本转成向量（首次运行会下载模型，需要稍等片刻）
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)

    print(f"向量维度: {len(emb1)}")
    # 期望：text1 和 text2 都是数据库问题 -> 相似度高
    print(f"text1 vs text2 相似度: {cosine_similarity(emb1, emb2):.4f}")
    # 期望：text1 和 text3 话题不同 -> 相似度低
    print(f"text1 vs text3 相似度: {cosine_similarity(emb1, emb3):.4f}")


# Embedding 模型对比
"""
text-embedding-3-small: 1536维, $0.02/1M tokens, 适合大多数场景（OpenAI API 模型）
text-embedding-3-large: 3072维, $0.13/1M tokens, 精度更高（OpenAI API 模型）

中文场景推荐（本地开源免费模型）:
- bge-large-zh-v1.5 (BAAI): 1024维, 中文效果好
- m3e-base (Moka): 768维
- text2vec-large-chinese: 1024维
"""
