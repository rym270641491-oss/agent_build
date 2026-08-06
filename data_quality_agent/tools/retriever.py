# -*- coding: utf-8 -*-
"""经验文档检索工具（RAG）。

职责：
1. 启动时读取 data/user_experience/ 下所有 Markdown 文档
2. 按标题切块后写入 Chroma 向量库（持久化到 data/vector_store）
3. 提供“表名关键词 + 向量相似度”的融合检索，保证表名命中优先
4. 提供两种 Embedding 实现：
   - API 嵌入（OpenAI 兼容接口）
   - 本地 N-gram 哈希嵌入（离线可用、零额外依赖）
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import Settings

# 英文表名 / 字段名 token 的正则（用于关键词打分）
_TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
# Markdown 标题行的正则
_HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.*)$")


class LocalNgramEmbeddings(Embeddings):
    """本地 N-gram 哈希嵌入（离线兜底方案）。

    思路：把文本切成字符二元组，用哈希函数映射到固定维度的稀疏向量，
    再做 L2 归一化。适合经验文档的小规模检索，无需网络与额外模型；
    效果虽不如语义嵌入，但能保证在缺少 Embedding API 时功能可用。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vectorize(self, text: str) -> List[float]:
        """把一段文本转成归一化的稀疏向量。"""
        vec = [0.0] * self.dim
        # 去掉空白并统一小写，增强泛化性
        cleaned = re.sub(r"\s+", "", text).lower()
        for i in range(len(cleaned) - 1):
            gram = cleaned[i : i + 2]  # 字符二元组
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        # L2 归一化，避免长文本得分虚高
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档。"""
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """向量化查询。"""
        return self._vectorize(text)


def _build_api_embeddings(settings: Settings, api_key: str) -> Embeddings:
    """构建 OpenAI 兼容的 Embedding 客户端。"""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=api_key,
        base_url=settings.EMBEDDING_BASE_URL,
        timeout=settings.LLM_REQUEST_TIMEOUT,
        # 关闭 tiktoken 长度检查与分词：避免联网下载词表，
        # 也兼容非 OpenAI 的嵌入服务
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def get_embeddings(settings: Settings) -> Embeddings:
    """按配置创建 Embedding 实例。

    - EMBEDDING_MODE=api   ：强制使用 OpenAI 兼容嵌入接口（需配置 EMBEDDING_API_KEY）
    - EMBEDDING_MODE=local ：强制本地哈希嵌入
    - EMBEDDING_MODE=auto  ：配置了 EMBEDDING_API_KEY 就用 API，否则降级为本地。
      注意：不要用 DEEPSEEK_API_KEY 充当嵌入密钥，DeepSeek 不提供嵌入接口。
    """
    mode = settings.EMBEDDING_MODE.lower()
    # 嵌入接口使用独立的 EMBEDDING_API_KEY，避免误用大模型密钥
    api_key = settings.EMBEDDING_API_KEY

    if mode == "local":
        return LocalNgramEmbeddings(dim=settings.LOCAL_EMBEDDING_DIM)
    if mode == "api":
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_MODE=api 但未配置 EMBEDDING_API_KEY"
            )
        return _build_api_embeddings(settings, api_key)
    # auto：配置了嵌入密钥才用 API，否则用本地哈希嵌入（离线可用）
    if api_key:
        try:
            return _build_api_embeddings(settings, api_key)
        except Exception:
            return LocalNgramEmbeddings(dim=settings.LOCAL_EMBEDDING_DIM)
    return LocalNgramEmbeddings(dim=settings.LOCAL_EMBEDDING_DIM)


def chunk_markdown(text: str, source: str, max_chars: int = 1500) -> List[Document]:
    """把一篇 Markdown 按标题切块。

    每个标题下的内容形成一个块；块超过 max_chars 且遇到空行时再切一刀，
    避免把一句话拦腰截断，保证检索到的内容语义完整。
    """
    lines = text.splitlines()
    chunks: List[str] = []
    current_title = "（文档开头）"
    buffer: List[str] = []

    def flush() -> None:
        """把当前缓冲区的内容追加为一个块。"""
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return
        first_line = content.splitlines()[0].strip()
        if first_line.startswith("#"):
            # 块本身以标题开头（如文档开头的 H1），不再重复拼接标题
            chunks.append(content)
        else:
            chunks.append("# {}\n{}".format(current_title, content))
        buffer = []

    for line in lines:
        m = _HEADING_PATTERN.match(line.strip())
        if m:
            # 遇到新标题：先把上一节收尾
            flush()
            current_title = m.group(2).strip()
            buffer = [line.strip()]
        else:
            buffer.append(line)
        # 块超长且当前行是空行时切一刀
        if len("\n".join(buffer)) > max_chars and buffer and not buffer[-1].strip():
            flush()
    flush()

    docs: List[Document] = []
    for chunk in chunks:
        # 提取块中出现的表名/字段名作为元数据，供关键词过滤使用
        tables = list(dict.fromkeys(_TOKEN_PATTERN.findall(chunk)))
        # Chroma 不接受空列表/None 元数据，因此仅在提取到表名时写入
        metadata: dict = {"source": source}
        if tables:
            metadata["tables"] = tables
        docs.append(
            Document(
                page_content=chunk,
                metadata=metadata,
            )
        )
    return docs


def load_experience_docs(docs_dir: Path) -> List[Document]:
    """读取目录下所有 Markdown 文档并切块。"""
    if not docs_dir.exists():
        raise FileNotFoundError("经验文档目录不存在: {}".format(docs_dir))
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            "经验文档目录为空: {}（请放入 Markdown 文档）".format(docs_dir)
        )
    docs: List[Document] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        docs.extend(chunk_markdown(text, source=str(md_file)))
    return docs


class ExperienceRetriever:
    """经验文档检索器。

    启动时构建/加载 Chroma 索引；检索时使用“关键词 + 向量”融合打分，
    其中关键词部分优先匹配表名，符合“先用表名检索相关案例”的要求。
    """

    COLLECTION_NAME = "experience_cases"

    def __init__(self, settings: Settings, build_index: bool = True):
        self.settings = settings
        self.docs_dir = settings.DOCS_DIR
        self.vector_dir = settings.VECTOR_DIR
        self.embeddings = get_embeddings(settings)
        self.top_k = settings.RETRIEVE_TOP_K
        self.docs: List[Document] = []
        self._vectorstore: Optional[Chroma] = None
        if build_index:
            self.build_index()

    def build_index(self) -> None:
        """读取经验文档并写入 Chroma（启动时调用）。

        文档量小，默认每次启动重建索引，保证文档增删改后即时生效；
        若后续文档量变大，可改为增量更新或只在 RECREATE_INDEX_ON_START
        为 True 时重建。
        """
        self.docs = load_experience_docs(self.docs_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.vector_dir))
        # 删除旧集合后重建，避免文档增删后残留脏数据
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass  # 集合不存在时忽略
        self._vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=self.embeddings,
            persist_directory=str(self.vector_dir),
            collection_name=self.COLLECTION_NAME,
            # 使用余弦距离，与 L2 归一化的向量更匹配
            collection_metadata={"hnsw:space": "cosine"},
        )

    def list_docs(self) -> List[str]:
        """返回经验文档文件名清单。"""
        return [str(p) for p in sorted(self.docs_dir.glob("*.md"))]

    def _keyword_score(self, query: str, doc: Document) -> float:
        """关键词打分：命中表名/字段名越多的文档块得分越高。"""
        query_tokens = set(_TOKEN_PATTERN.findall(query))
        if not query_tokens:
            return 0.0
        text = doc.page_content.lower()
        hits = sum(1 for token in query_tokens if token.lower() in text)
        return hits / len(query_tokens)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """融合检索：向量相似度 + 表名关键词得分，按总分降序返回。"""
        if self._vectorstore is None:
            raise RuntimeError("向量索引尚未构建")
        k = top_k or self.top_k
        # 1) 向量检索：多取一些候选，留给融合排序
        results = self._vectorstore.similarity_search_with_score(query, k=k * 3)
        # 2) 融合打分：score 越小距离越近，粗略转成相似度
        scored: List[Tuple[Document, float]] = []
        for doc, score in results:
            vector_sim = max(1.0 - score, 0.0)
            kw = self._keyword_score(query, doc)
            final = (
                self.settings.RETRIEVE_VECTOR_WEIGHT * vector_sim
                + self.settings.RETRIEVE_KEYWORD_WEIGHT * kw
            )
            scored.append((doc, round(final, 4)))
        # 3) 按融合分降序，取前 k 条
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def format_search_results(self, query: str, top_k: Optional[int] = None) -> str:
        """把检索结果格式化成适合喂给大模型的文本。"""
        results = self.search(query, top_k=top_k)
        if not results:
            return "（未检索到相关经验文档）"
        lines = ["【检索结果】"]
        for i, (doc, score) in enumerate(results, start=1):
            source_name = Path(doc.metadata.get("source", "未知")).name
            lines.append("[{}] 来源: {} | 相关度: {}".format(i, source_name, score))
            lines.append(doc.page_content[:600])
            lines.append("")
        return "\n".join(lines)
