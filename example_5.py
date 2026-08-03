"""
完整的 RAG (Retrieval-Augmented Generation) Pipeline 实现

流程: 文档加载 → 分块 → Embedding → 向量存储 → 检索 → LLM 生成
说明: LLM 使用 DeepSeek API；Embedding 使用本地开源模型 bge-small-zh-v1.5
      （DeepSeek 不提供 embedding 接口，本地模型免费且无需 Key）。
依赖: openai, chromadb, fastembed, python-dotenv
"""
import os
from typing import List, Dict

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()  # 自动读取同目录下的 .env 文件

# 读取并校验 DeepSeek API Key（为空时直接报错，提示更友好）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY！请在项目根目录的 .env 文件中填写，例如：\n"
        "DEEPSEEK_API_KEY=sk-你的key"
    )


# ============================================================
# 配置
# ============================================================
class RAGConfig:
    """RAG 系统配置"""
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地中文 Embedding 模型（512维）
    LLM_MODEL = "deepseek-v4-flash"             # DeepSeek 当前可用的对话模型（实测 API 返回的名称）
    CHUNK_SIZE = 500       # 分块大小（字符）
    CHUNK_OVERLAP = 50     # 相邻分块的重叠字符数（避免切断语义）
    TOP_K = 3              # 检索返回的文档数
    COLLECTION_NAME = "knowledge_base"          # Chroma 集合（表）名称


# ============================================================
# 自定义 Embedding 函数（Chroma 需要，负责把文本变成向量）
# ============================================================
class LocalEmbeddingFunction(EmbeddingFunction):
    """基于本地 fastembed 模型的向量生成函数（无需 API Key、无需联网）"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 模型只在首次构造时加载一次；fastembed 会缓存已下载的模型
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        """把一段或多段文本转换成向量列表（Chroma 会调用这个方法）"""
        # embed() 接收一个文本列表，返回 numpy 数组的生成器
        return list(self._model.embed(list(input)))


# ============================================================
# RAG 系统
# ============================================================
class RAGSystem:
    """完整的 RAG 问答系统"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        # 初始化 LLM 客户端（DeepSeek 兼容 OpenAI SDK，改 base_url 即可）
        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        # 初始化 Chroma 向量数据库（持久化到 ./rag_db 目录）
        self.chroma_client = chromadb.PersistentClient(path="./rag_db")

        # 使用本地 Embedding 模型（替代 OpenAIEmbeddingFunction，无需 API Key）
        self.embedding_fn = LocalEmbeddingFunction(
            model_name=self.config.EMBEDDING_MODEL,
        )

        # 创建集合；hnsw:space=cosine 表示用余弦距离检索，
        # 这样 score = 1 - distance 就正好是余弦相似度（-1 ~ 1）
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 文档处理 ----------

    def _split_text(self, text: str) -> List[str]:
        """将长文本分割为小块（尽量在句子边界处切，避免切断语义）"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.CHUNK_SIZE

            # 如果还没到文本末尾，尝试在段落/句号/换行等自然断点处分割
            if end < len(text):
                # 从后往前找断点（rfind），优先使用更长的分隔符（如段落换行）
                for sep in ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ']:
                    pos = text.rfind(sep, start, end)
                    # 断点至少要在块的后半段，否则块太短，不值得调整
                    if pos > start + self.config.CHUNK_SIZE // 2:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 下一块从"当前块结尾 - 重叠长度"开始，实现分块重叠
            start = end - self.config.CHUNK_OVERLAP

        return chunks

    def add_document(self, text: str, metadata: Dict = None) -> int:
        """添加文档到知识库（自动分块后写入 Chroma），返回分块数量"""
        chunks = self._split_text(text)

        if not chunks:
            return 0

        # 生成唯一 ID（基于当前库内已有的文档数量）
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]

        # 关键：每个分块都要有一份独立的 metadata 副本。
        # 若写成 [metadata] * n，所有分块会共享同一个字典，
        # 后面的循环会把所有块的 chunk_index 都改成同一个值。
        metadatas = [dict(metadata or {}) for _ in chunks]
        # 为每个块记录它在原文档中的位置信息
        for i, meta in enumerate(metadatas):
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"✅ 添加了 {len(chunks)} 个文本块到知识库")
        return len(chunks)

    def add_documents_batch(self, documents: List[Dict[str, str]]):
        """批量添加文档
        documents: [{"text": "...", "metadata": {...}}, ...]
        """
        total = 0
        for doc in documents:
            total += self.add_document(doc["text"], doc.get("metadata", {}))
        print(f"✅ 批量添加完成，共 {total} 个文本块")

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """检索与查询最相关的文档块"""
        top_k = top_k or self.config.TOP_K

        # 查询数量不能超过库内已有文档数，否则 Chroma 会报错
        top_k = min(top_k, self.collection.count())
        if top_k <= 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # 余弦距离 → 余弦相似度
            })

        return documents

    # ---------- 生成 ----------

    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """构建包含检索上下文的 Prompt，让 LLM 只能依据文档回答"""
        context_text = "\n\n---\n\n".join([
            f"[参考文档 {i+1}]\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""你是一个知识库问答助手。请基于以下参考文档回答用户问题。

## 规则
1. 只使用参考文档中的信息回答问题
2. 如果文档中没有相关信息，明确说"根据现有知识库，我无法回答这个问题"
3. 回答时引用具体的文档编号（如"根据参考文档1..."）
4. 如果信息不完整，说明缺少什么信息

## 参考文档
{context_text}

## 用户问题
{query}

## 回答
"""
        return prompt

    def ask(self, query: str, top_k: int = None) -> Dict:
        """执行一次完整的 RAG 问答：检索 → 拼 Prompt → 调用 LLM"""
        # 步骤1: 检索相关文档
        docs = self.retrieve(query, top_k)

        if not docs:
            return {
                "answer": "知识库中没有找到相关信息。",
                "sources": [],
            }

        # 步骤2: 构建 Prompt
        prompt = self._build_prompt(query, docs)

        # 步骤3: 调用 LLM 生成回答
        response = self.llm.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个基于知识库的问答助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            # 附带检索到的来源，方便追溯回答依据
            "sources": [
                {
                    "content": doc["content"][:200] + "...",
                    "score": round(doc["score"], 4),
                    "metadata": doc.get("metadata", {}),
                }
                for doc in docs
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

    # ---------- 管理 ----------

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.config.COLLECTION_NAME,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "llm_model": self.config.LLM_MODEL,
        }

    def clear(self):
        """清空知识库（删除集合后重新创建）"""
        self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print("✅ 知识库已清空")


# ============================================================
# 使用示例（只有直接运行本文件时才执行）
# ============================================================
if __name__ == "__main__":
    rag = RAGSystem()

    # 添加知识文档（自动分块 + 生成向量 + 存入 Chroma）
    rag.add_document(
        text="""
        数据库慢查询排查指南

        1. 识别慢查询
        当数据库响应时间超过2秒时，需要关注慢查询。
        在MySQL中，通过以下命令开启慢查询日志：
        SET GLOBAL slow_query_log = 'ON';
        SET GLOBAL long_query_time = 2;

        2. 使用EXPLAIN分析
        EXPLAIN SELECT * FROM orders WHERE user_id = 123;
        关注type列（应为ref/eq_ref/const，避免ALL）、rows列（扫描行数越少越好）、
        Extra列（Using filesort和Using temporary通常不好）。

        3. 常见优化方法
        - 添加合适的索引
        - 避免SELECT *，只查需要的列
        - 优化JOIN顺序
        - 使用连接池减少连接开销

        4. 紧急处理
        当数据库CPU飙升至90%以上时：
        1. 立即查看SHOW PROCESSLIST，找出正在执行的查询
        2. 如果发现长时间运行的查询，评估是否可以KILL
        3. 查看连接数是否接近max_connections上限
        """,
        metadata={"category": "数据库", "type": "排查手册"}
    )

    # 提问：检索相关文档 + LLM 生成回答
    result = rag.ask("数据库CPU突然飙到95%怎么办")

    print("=" * 50)
    print("回答:")
    print(result["answer"])
    print("\n参考来源:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] {s['content'][:100]}...")
    print(f"\nToken用量: {result['usage']}")
