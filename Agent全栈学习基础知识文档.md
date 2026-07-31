# Agent 全栈学习：从入门到精通到实战

> **文档定位**：面向零基础初学者，从 LLM 最底层概念开始，逐步深入到 Agent 全栈开发的每一个细节。  
> **文档特点**：每个概念都有详细解释 + 可运行的 Python 代码示例，不跳步骤，不省略细节。  
> **适合场景**：告警排查 Agent、数据质量分析 Agent、知识问答 Agent、工作流 Agent 等业务场景。

---

# 目录

- [第一部分：LLM 大模型基础（入门篇）](#第一部分llm-大模型基础入门篇)
  - [第1章：什么是大语言模型](#第1章什么是大语言模型)
  - [第2章：Token 与 Tokenization](#第2章token-与-tokenization)
  - [第3章：上下文窗口](#第3章上下文窗口)
  - [第4章：Temperature 与采样策略](#第4章temperature-与采样策略)
  - [第5章：调用你的第一个 LLM API](#第5章调用你的第一个-llm-api)
  - [第6章：理解 Chat Completion API 的返回结构](#第6章理解-chat-completion-api-的返回结构)
  - [第7章：流式输出](#第7章流式输出)
  - [第8章：Top-P 与 Top-K 采样](#第8章top-p-与-top-k-采样)
  - [第9章：模型选型指南](#第9章模型选型指南)
- [第二部分：Prompt Engineering 提示词工程（入门篇）](#第二部分prompt-engineering-提示词工程入门篇)
  - [第10章：Prompt 的本质](#第10章prompt-的本质)
  - [第11章：System Prompt 与 User Prompt](#第11章system-prompt-与-user-prompt)
  - [第12章：Few-Shot Prompting（少样本提示）](#第12章few-shot-prompting少样本提示)
  - [第13章：Chain-of-Thought（思维链）](#第13章chain-of-thought思维链)
  - [第14章：结构化输出控制](#第14章结构化输出控制)
  - [第15章：Prompt 模板化](#第15章prompt-模板化)
  - [第16章：Prompt 调试技巧与常见陷阱](#第16章prompt-调试技巧与常见陷阱)
- [第三部分：Function Calling 工具调用（进阶篇）](#第三部分function-calling-工具调用进阶篇)
  - [第17章：什么是 Function Calling](#第17章什么是-function-calling)
  - [第18章：定义工具函数](#第18章定义工具函数)
  - [第19章：完整的 Function Calling 流程](#第19章完整的-function-calling-流程)
  - [第20章：多工具并行调用](#第20章多工具并行调用)
  - [第21章：工具调用错误处理](#第21章工具调用错误处理)
  - [第22章：工具调用的最佳实践](#第22章工具调用的最佳实践)
- [第四部分：RAG 检索增强生成（进阶篇）](#第四部分rag-检索增强生成进阶篇)
  - [第23章：RAG 完整原理](#第23章rag-完整原理)
  - [第24章：文档分块策略](#第24章文档分块策略)
  - [第25章：Embedding 向量嵌入](#第25章embedding-向量嵌入)
  - [第26章：向量数据库选型与使用](#第26章向量数据库选型与使用)
  - [第27章：完整的 RAG Pipeline 实现](#第27章完整的-rag-pipeline-实现)
  - [第28章：检索质量优化](#第28章检索质量优化)
  - [第29章：HyDE 假设文档嵌入](#第29章hyde-假设文档嵌入)
  - [第30章：多路召回与融合排序](#第30章多路召回与融合排序)
- [第五部分：Agent 核心架构（精通篇）](#第五部分agent-核心架构精通篇)
  - [第31章：Agent 的完整定义与架构](#第31章agent-的完整定义与架构)
  - [第32章：ReAct Agent 完整实现](#第32章react-agent-完整实现)
  - [第33章：Plan-and-Execute Agent](#第33章plan-and-execute-agent)
  - [第34章：Agent 状态管理](#第34章agent-状态管理)
  - [第35章：Agent 记忆系统设计](#第35章agent-记忆系统设计)
  - [第36章：Agent 循环与终止条件](#第36章agent-循环与终止条件)
- [第六部分：Agent 框架实战（实战篇）](#第六部分agent-框架实战实战篇)
  - [第37章：LangChain Agent 实战](#第37章langchain-agent-实战)
  - [第38章：AutoGen 多 Agent 实战](#第38章autogen-多-agent-实战)
  - [第39章：CrewAI 多 Agent 协作](#第39章crewai-多-agent-协作)
  - [第40章：从零手写 Agent 框架](#第40章从零手写-agent-框架)
- [第七部分：多 Agent 系统（架构篇）](#第七部分多-agent-系统架构篇)
  - [第41章：多 Agent 架构模式](#第41章多-agent-架构模式)
  - [第42章：Agent 间通信协议](#第42章agent-间通信协议)
  - [第43章：Orchestrator 编排模式](#第43章orchestrator-编排模式)
- [第八部分：评估、监控与运维（工程篇）](#第八部分评估监控与运维工程篇)
  - [第44章：Agent 评估体系](#第44章agent-评估体系)
  - [第45章：LangSmith / LangFuse 追踪](#第45章langsmith--langfuse-追踪)
  - [第46章：成本分析与优化](#第46章成本分析与优化)
  - [第47章：安全与护栏](#第47章安全与护栏)
- [第九部分：生产部署（落地篇）](#第九部分生产部署落地篇)
  - [第48章：FastAPI 部署 Agent 服务](#第48章fastapi-部署-agent-服务)
  - [第49章：WebSocket 实时对话](#第49章websocket-实时对话)
  - [第50章：Docker 容器化部署](#第50章docker-容器化部署)
  - [第51章：可观测性搭建](#第51章可观测性搭建)
- [第十部分：综合实战项目（项目篇）](#第十部分综合实战项目项目篇)
  - [第52章：告警排查 Agent 完整项目](#第52章告警排查-agent-完整项目)
  - [第53章：数据质量检测 Agent](#第53章数据质量检测-agent)
  - [第54章：智能知识库问答系统](#第54章智能知识库问答系统)

---

# 第一部分：LLM 大模型基础（入门篇）

## 第1章：什么是大语言模型

### 1.1 直观理解

大语言模型（Large Language Model，简称 LLM）本质上是一个"超级预测下一个词的机器"。它通过阅读互联网上数十亿甚至数万亿的文字（书籍、网页、论文、代码等），学会了语言的规律、知识和推理能力。

你可以把 LLM 想象成一个非常博学的人——它读过几乎所有的公开书籍和文章，但它不是真的"理解"世界，而是学会了：当它看到一段文字的开头，预测接下来最合适的文字是什么。

### 1.2 核心原理（极简版）

LLM 的核心架构是 **Transformer**（由 Google 在 2017 年的论文《Attention Is All You Need》中提出）。Transformer 有两个核心机制：

1. **自注意力机制（Self-Attention）**：让模型在处理每个词时，能够"关注"到句子中所有其他词，理解它们之间的关系。比如处理"小明把苹果给了小红，她很开心"这句话时，模型需要知道"她"指的是"小红"还是"小明"。

2. **多层堆叠**：将多个 Transformer 层堆叠起来，每一层学到不同层次的语言特征——底层学语法，中层学语义，高层学推理。

### 1.3 训练过程

LLM 的训练通常分为三个阶段：

- **预训练（Pre-training）**：在海量文本上做"下一个词预测"，让模型学会语言的基本规律。这个阶段花费最大（GPT-4 级别的模型训练一次需要数千万甚至上亿美元）。

- **指令微调（Instruction Fine-tuning / SFT）**：用高质量的人工标注对话数据训练模型，让它学会"遵循指令"而不是单纯预测下一个词。

- **RLHF（Reinforcement Learning from Human Feedback）**：让人类对模型的不同回答进行排序/打分，然后用强化学习让模型学会生成人类偏好的回答。这就是为什么 ChatGPT 的回答读起来很"舒服"。

### 1.4 主流模型一览

| 模型 | 开发者 | 特点 | 适用场景 |
|------|--------|------|----------|
| GPT-4o / GPT-4o-mini | OpenAI | 综合能力强，多模态 | 通用场景，复杂推理 |
| Claude 3.5 / 4 | Anthropic | 安全性好，长上下文 | 长篇文档分析，代码生成 |
| DeepSeek-V3 / R1 | DeepSeek | 开源，性价比极高 | 中文场景，成本敏感场景 |
| Qwen 2.5 / 3 | 阿里 | 中文能力强，开源 | 中文业务场景 |
| Gemini 2.5 | Google | 多模态，超长上下文 | 多模态分析 |
| Llama 4 | Meta | 开源，社区活跃 | 私有化部署，定制化 |

---

## 第2章：Token 与 Tokenization

### 2.1 什么是 Token

**Token 是 LLM 处理文本的最小单位。** 模型不是逐字逐句地阅读文本，而是先把文本切分成 token，然后处理这些 token 序列。

- 1 个 token ≈ 0.75 个英文单词 ≈ 0.5 个中文字
- "Hello world" ≈ 2-3 个 token
- "你好世界" ≈ 4-6 个 token
- "ChatGPT" 可能被切分为 ["Chat", "G", "PT"] ≈ 3 个 token

### 2.2 为什么 Token 很重要

1. **计费单位**：大多数 LLM API 按 token 数量收费（输入 token + 输出 token）
2. **上下文窗口限制**：模型的上下文窗口大小以 token 数衡量（如 128K tokens）
3. **性能影响**：token 越多，模型处理越慢，成本越高

### 2.3 Tokenizer 的工作原理

Tokenizer（分词器）将文本转换为 token ID 序列。主流的分词算法是 **BPE（Byte Pair Encoding）**：

1. 从字符级别开始
2. 统计所有相邻字符对的频率
3. 合并最高频的字符对为新 token
4. 重复直到达到预设的词汇表大小

### 2.4 代码示例：使用 tiktoken 计算 Token

```python
# 安装：pip install tiktoken
import tiktoken

# 使用 GPT-4o 的 tokenizer
encoding = tiktoken.encoding_for_model("gpt-4o")

# 中英文混合文本
text = "你好，世界！Hello World! 大语言模型正在改变软件开发的方式。"

# 编码为 token ID 列表
tokens = encoding.encode(text)
print(f"Token 数量: {len(tokens)}")
print(f"Token IDs: {tokens}")

# 解码回文本
decoded = encoding.decode(tokens)
print(f"解码后: {decoded}")

# 查看每个 token 对应的文本
for token_id in tokens:
    token_bytes = encoding.decode_single_token_bytes(token_id)
    print(f"ID {token_id}: '{token_bytes.decode('utf-8', errors='replace')}'")

# 常见输出示例：
# Token 数量: 20
# Token IDs: [57668, 53901, 3922, 6447, ...]
# ID 57668: '你好'
# ID 53901: '，'
# ID 3922: '世界'
```

### 2.5 不同模型的 Token 差异

不同模型使用的 tokenizer 不同，同一个文本在不同模型下的 token 数也不同。例如中文在 GPT 系列通常比在 Claude 系列消耗更多 token。

---

## 第3章：上下文窗口

### 3.1 什么是上下文窗口

**上下文窗口（Context Window）** 是模型在一次处理中能"看到"的最大 token 数量。它决定了模型能记住多少对话历史或文档内容。

当前主流模型的上下文窗口：
- GPT-4o: 128K tokens（约 20 万个中文字）
- Claude 3.5 Sonnet: 200K tokens
- Gemini 2.5 Pro: 1M tokens
- DeepSeek-V3: 128K tokens
- Qwen 2.5: 128K tokens

### 3.2 上下文窗口的"注意力"问题

虽然理论上模型能看 128K tokens，但有一个著名的现象叫 **"Lost in the Middle"（迷失在中间）**：模型对文档开头和结尾的内容记得最好，对中间部分的内容容易忽略。

解决方案：
- 把最重要的信息放在开头（System Prompt）或结尾
- 对长文档进行分段处理
- 使用 RAG 只检索相关内容而非塞入全文

### 3.3 上下文管理策略

```python
# 上下文管理示例：控制发送给模型的对话历史
from typing import List, Dict

class ContextManager:
    """管理对话上下文，确保不超过 token 限制"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.messages: List[Dict] = []
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息，如果超出限制则移除最早的消息"""
        self.messages.append({"role": role, "content": content})
        self._trim()
    
    def _trim(self) -> None:
        """从最早的对话开始移除，直到 token 数在限制内"""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            # 始终保留 system prompt（第一条）和最后一条消息
            self.messages.pop(1)  # 移除第二条（最早的非 system 消息）
    
    def _estimate_tokens(self) -> int:
        """粗略估算 token 数（中文约 2 字符/token，英文约 4 字符/token）"""
        total_chars = sum(len(m["content"]) for m in self.messages)
        # 粗略估算：中文 1.5 字符/token，英文 4 字符/token，取平均值
        return total_chars // 2
    
    def get_messages(self) -> List[Dict]:
        return self.messages

# 使用示例
ctx = ContextManager(max_tokens=2000)
ctx.add_message("system", "你是一个数据分析助手")
ctx.add_message("user", "帮我分析这份销售数据...")
ctx.add_message("assistant", "好的，我来分析...")
ctx.add_message("user", "能不能详细说说趋势？")
print(f"当前对话轮数: {len(ctx.get_messages())}")
```

---

## 第4章：Temperature 与采样策略

### 4.1 什么是 Temperature

**Temperature（温度）** 控制模型输出的"随机性"和"创造性"。当你让模型生成下一个词时，模型会给出每个可能词的"概率分数"。Temperature 调整这些概率的分布。

- **Temperature = 0**：模型总是选择概率最高的词（确定性输出）。适合需要精确答案的场景。
- **Temperature = 0.2 - 0.5**：输出比较稳定，偶尔有小变化。适合代码生成、数据提取。
- **Temperature = 0.7 - 1.0**：输出有一定创造性。适合内容创作、头脑风暴。
- **Temperature > 1.0**：输出非常随机，可能产生无意义内容。

### 4.2 数学原理

给定每个 token 的原始分数（logits），temperature 的作用是：

```
scaled_logits = logits / temperature
probabilities = softmax(scaled_logits)
```

Temperature 越高，概率分布越平滑（随机）；越低，概率分布越尖锐（确定性）。

### 4.3 代码示例

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def generate_with_temperature(prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=200
    )
    return response.choices[0].message.content

# 同一个 prompt，不同 temperature 的对比
prompt = "写一首关于春天的五言绝句"

print("=== Temperature = 0.0 (确定性) ===")
print(generate_with_temperature(prompt, 0.0))

print("\n=== Temperature = 0.9 (创造性) ===")
print(generate_with_temperature(prompt, 0.9))

# 注意：即使 temperature=0，不同 API 调用也可能有细微差异，
# 因为后台计算中存在浮点数精度问题。对于严格确定性，
# 某些 API 提供了 seed 参数。
```

### 4.4 实际使用建议

| 场景 | 推荐 Temperature | 原因 |
|------|-----------------|------|
| 数据提取 / 结构化输出 | 0.0 - 0.1 | 要求精确 |
| 代码生成 | 0.0 - 0.2 | 需要确定性 |
| 事实问答 | 0.1 - 0.3 | 减少幻觉 |
| 翻译 | 0.2 - 0.4 | 需要一定灵活性 |
| 写作 / 创意 | 0.6 - 0.9 | 需要创造性 |
| 头脑风暴 | 0.8 - 1.0 | 追求多样性 |

---

## 第5章：调用你的第一个 LLM API

### 5.1 准备工作

在开始之前，你需要：
1. 获取一个 LLM 的 API Key（推荐从 DeepSeek 或 OpenAI 开始）
2. 安装 openai Python 库：`pip install openai`

### 5.2 调用 OpenAI API

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-your-api-key-here",  # 替换为你的 API Key
    # 如果使用 DeepSeek、通义千问等其他兼容 OpenAI 接口的服务：
    # base_url="https://api.deepseek.com/v1",
)

# 最简单的调用
response = client.chat.completions.create(
    model="gpt-4o-mini",  # 模型名
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "什么是机器学习？请用通俗的语言解释。"}
    ],
    max_tokens=500,        # 最大输出 token 数
    temperature=0.7,       # 随机性
)

# 提取回答内容
answer = response.choices[0].message.content
print(answer)
```

### 5.3 调用 DeepSeek API

DeepSeek 的 API 兼容 OpenAI 格式，只需要改 base_url：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-deepseek-key",
    base_url="https://api.deepseek.com/v1",
)

response = client.chat.completions.create(
    model="deepseek-chat",  # DeepSeek-V3
    messages=[
        {"role": "user", "content": "请用中文解释：什么是深度学习？"}
    ],
    max_tokens=500,
    temperature=0.7,
)

print(response.choices[0].message.content)
print(f"\n消耗 token: {response.usage.total_tokens}")
print(f"  - 输入: {response.usage.prompt_tokens}")
print(f"  - 输出: {response.usage.completion_tokens}")
```

### 5.4 调用通义千问 API（阿里云）

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-qwen-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen-plus",  # 或 qwen-max, qwen-turbo
    messages=[
        {"role": "user", "content": "介绍一下杭州西湖的历史"}
    ],
    max_tokens=500,
)

print(response.choices[0].message.content)
```

### 5.5 封装通用调用函数

```python
from openai import OpenAI
from typing import List, Dict, Optional

class LLMClient:
    """统一的 LLM 调用封装"""
    
    # 各平台的默认配置
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-plus",
        },
    }
    
    def __init__(self, provider: str = "deepseek", api_key: Optional[str] = None):
        import os
        config = self.PROVIDERS[provider]
        self.client = OpenAI(
            api_key=api_key or os.getenv(f"{provider.upper()}_API_KEY"),
            base_url=config["base_url"],
        )
        self.default_model = config["default_model"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """发送对话并返回文本回答"""
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> tuple[str, dict]:
        """发送对话并返回文本回答 + token 用量"""
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return text, usage

# 使用示例
# llm = LLMClient(provider="deepseek", api_key="sk-xxx")
# answer = llm.chat([{"role": "user", "content": "你好"}])
```

---

## 第6章：理解 Chat Completion API 的返回结构

### 6.1 完整的响应对象

当调用 `client.chat.completions.create()` 时，返回的是一个复杂的对象，而不仅仅是文本。理解这个结构对后续开发非常重要。

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用JSON格式返回：姓名张三，年龄25，城市北京"}],
    max_tokens=200,
)

# --- 1. 基本信息 ---
print(f"模型: {response.model}")
print(f"ID: {response.id}")
print(f"创建时间: {response.created}")

# --- 2. choices 列表 ---
# 通常只有一个 choice（除非设置 n > 1）
print(f"\nChoices 数量: {len(response.choices)}")
choice = response.choices[0]
print(f"结束原因: {choice.finish_reason}")  # "stop", "length", "content_filter"
print(f"索引: {choice.index}")

# --- 3. 消息内容 ---
message = choice.message
print(f"\n角色: {message.role}")
print(f"内容:\n{message.content}")

# --- 4. Token 用量 ---
print(f"\n=== Token 用量 ===")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# --- 5. finish_reason 的含义 ---
"""
finish_reason 的可能值：
- "stop": 正常结束，模型完成了回答
- "length": 达到了 max_tokens 限制，回答被截断
- "content_filter": 内容被安全过滤器拦截
- "tool_calls": 模型决定调用工具（见 Function Calling 章节）
- "function_call": 同上（旧版 API）
"""
```

### 6.2 finish_reason 在代码中的处理

```python
def safe_chat(client: OpenAI, messages: list, max_tokens: int = 500) -> dict:
    """安全的对话调用，处理各种结束情况"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
    )
    
    choice = response.choices[0]
    
    if choice.finish_reason == "stop":
        return {"status": "ok", "content": choice.message.content}
    elif choice.finish_reason == "length":
        return {
            "status": "truncated",
            "content": choice.message.content,
            "warning": f"回答被截断(max_tokens={max_tokens})，建议增大 max_tokens 或分段处理"
        }
    elif choice.finish_reason == "content_filter":
        return {
            "status": "filtered",
            "content": None,
            "warning": "内容被安全过滤器拦截，请检查输入内容"
        }
    elif choice.finish_reason == "tool_calls":
        return {
            "status": "tool_calls",
            "content": None,
            "tool_calls": choice.message.tool_calls,
        }
    else:
        return {"status": "unknown", "content": choice.message.content}
```

---

## 第7章：流式输出

### 7.1 为什么需要流式输出

默认的 API 调用是"同步"的：发送请求 → 等待 → 一次性收到完整回答。对于长回答，用户可能需要等待几秒甚至几十秒。

**流式输出（Streaming）** 让模型边生成边返回，像 ChatGPT 打字一样逐字显示。这大大提升了用户体验。

### 7.2 完整的流式输出实现

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def stream_chat(prompt: str):
    """流式输出示例"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,           # 关键参数：开启流式
        max_tokens=500,
    )
    
    full_response = ""
    
    for chunk in stream:
        # chunk 是增量数据，不是完整回答
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # 逐字打印
            full_response += content
    
    print()  # 换行
    return full_response

# stream_chat("请用 500 字介绍杭州西湖的历史和文化")
```

### 7.3 流式输出中的 Token 用量

流式输出中，usage 信息通常只在最后一个 chunk 中返回：

```python
def stream_with_usage(prompt: str):
    """流式输出并获取 token 用量"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},  # 需要在流式中获取 usage
    )
    
    full_response = ""
    usage = None
    
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
        if chunk.usage:
            usage = {
                "prompt": chunk.usage.prompt_tokens,
                "completion": chunk.usage.completion_tokens,
                "total": chunk.usage.total_tokens,
            }
    
    return full_response, usage
```

### 7.4 FastAPI 中的 SSE 流式输出

Server-Sent Events (SSE) 是实现流式输出的标准 Web 协议：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI(api_key="your-key")

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    """SSE 流式对话接口"""
    
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                # SSE 格式：data: {json}\n\n
                data = json.dumps({
                    "content": chunk.choices[0].delta.content
                }, ensure_ascii=False)
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )

# 运行：uvicorn main:app --reload
# 前端可以用 EventSource API 接收：
# const eventSource = new EventSource('/chat/stream?prompt=你好');
# eventSource.onmessage = (event) => { console.log(event.data); };
```

---

## 第8章：Top-P 与 Top-K 采样

### 8.1 Top-K 采样

**Top-K** 采样是指：模型在生成下一个 token 时，只从概率最高的 K 个候选中随机选择。

```
所有可能的下一个 token（按概率排序）：
"猫"  (概率 0.30)
"狗"  (概率 0.25)
"鸟"  (概率 0.15)
"鱼"  (概率 0.10)
...
(还有很多低概率 token)

Top-K=3 采样：只从 {"猫": 0.30, "狗": 0.25, "鸟": 0.15} 中按概率随机选
```

- K 越小，输出越确定性
- K 越大，输出越多样

### 8.2 Top-P（Nucleus Sampling）核采样

**Top-P** 采样是指：模型只从累积概率达到 P 的最小 token 集合中随机选择。

```
所有可能的下一个 token（按概率排序）：
"猫"  (概率 0.30)  → 累积 0.30
"狗"  (概率 0.25)  → 累积 0.55
"鸟"  (概率 0.15)  → 累积 0.70
"鱼"  (概率 0.10)  → 累积 0.80  ← 累积超过 P=0.8
"虫"  (概率 0.08)  → 累积 0.88
...

Top-P=0.9 采样：从 {"猫","狗","鸟","鱼","虫"} 中按概率随机选
Top-P=0.8 采样：从 {"猫","狗","鸟","鱼"} 中按概率随机选
```

Top-P 比 Top-K 更"智能"，因为它根据实际概率分布动态调整候选集大小。

### 8.3 实际使用

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一个创意故事的开头"}],
    temperature=0.8,  # 先调 temperature
    top_p=0.9,        # 再调 top_p
    max_tokens=300,
)

# 一般建议：
# - 先调整 temperature，再调整 top_p
# - 二者不建议同时设置得很极端
# - 大多数情况下只设置 temperature 就足够了
# - top_p 通常保持在 0.9-1.0
```

---

## 第9章：模型选型指南

### 9.1 选型维度

选择模型时需要考虑以下几个维度：

| 维度 | 说明 |
|------|------|
| **能力** | 推理、代码、多语言、多模态等 |
| **速度** | 首 token 延迟（TTFT）和生成速度（tokens/s） |
| **成本** | 每百万 token 的价格（输入/输出可能不同） |
| **上下文窗口** | 最大支持的 token 数 |
| **Function Calling** | 是否支持工具调用，准确率如何 |
| **可用性** | API 稳定性、限流策略 |
| **数据安全** | 数据是否出境、是否有私有化部署方案 |

### 9.2 不同场景的推荐

```python
# 场景 1: 中文业务场景，成本敏感
RECOMMENDED = {
    "model": "deepseek-chat",  # DeepSeek-V3
    "price_input": "1元/百万token",
    "price_output": "2元/百万token",
    "strength": "中文能力强，性价比极高",
}

# 场景 2: 复杂推理任务
RECOMMENDED = {
    "model": "gpt-4o",
    "price_input": "约$2.5/百万token",
    "strength": "综合推理能力最强",
}

# 场景 3: 长文档分析
RECOMMENDED = {
    "model": "claude-3-5-sonnet-20241022",
    "price_input": "约$3/百万token",
    "context": "200K tokens",
    "strength": "长上下文理解和代码生成",
}

# 场景 4: 私有化部署
RECOMMENDED = {
    "model": "qwen2.5-72b-instruct",  # 或 llama 系列
    "deployment": "vLLM / TGI / Ollama",
    "strength": "可控、数据不出域",
}

# 场景 5: 多模态
RECOMMENDED = {
    "model": "gpt-4o",  # 或 gemini-2.5-pro
    "strength": "图文理解 + 生成",
}
```

### 9.3 价格对比（参考，价格可能变动）

```python
# 2025年大致参考价格（每百万token）
PRICES = {
    "deepseek-chat":      {"input": 1.0,   "output": 2.0,   "currency": "CNY"},
    "deepseek-reasoner":  {"input": 4.0,   "output": 16.0,  "currency": "CNY"},
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60,  "currency": "USD"},
    "gpt-4o":             {"input": 2.50,  "output": 10.00, "currency": "USD"},
    "claude-3.5-sonnet":  {"input": 3.00,  "output": 15.00, "currency": "USD"},
    "qwen-plus":          {"input": 0.8,   "output": 2.0,   "currency": "CNY"},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """估算一次 API 调用的费用"""
    p = PRICES.get(model)
    if not p:
        return 0.0
    cost = (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]
    return round(cost, 4)

# 示例：用 deepseek-chat 发送 2000 token 输入，收到 1000 token 输出
cost = estimate_cost("deepseek-chat", 2000, 1000)
print(f"预估费用: ¥{cost}")  # 约 ¥0.004
```

---

# 第二部分：Prompt Engineering 提示词工程（入门篇）

## 第10章：Prompt 的本质

### 10.1 什么是 Prompt Engineering

**Prompt Engineering（提示词工程）** 是设计、优化和组织输入文本，以引导 LLM 产生期望输出的方法和实践。

简单说，就是"怎么跟模型说话"。它不像传统编程需要精确的语法，而更像"沟通的艺术"——但其中也有很多可复用的模式和技巧。

### 10.2 Prompt 的组成部分

一个好的 prompt 通常包含以下元素：

```
┌─────────────────────────────────────────┐
│ 1. 角色定义（Role）                      │
│    "你是一个资深的数据分析师..."          │
├─────────────────────────────────────────┤
│ 2. 任务描述（Task）                      │
│    "请分析以下销售数据并找出异常..."       │
├─────────────────────────────────────────┤
│ 3. 上下文信息（Context）                 │
│    "以下是2024年Q1的销售数据..."         │
├─────────────────────────────────────────┤
│ 4. 约束条件（Constraints）               │
│    "只分析利润率为负的产品..."           │
├─────────────────────────────────────────┤
│ 5. 输出格式（Output Format）             │
│    "以JSON格式输出，包含以下字段..."      │
├─────────────────────────────────────────┤
│ 6. 示例（Examples / Few-Shot）           │
│    "例如，当输入为X时，输出应为Y..."      │
├─────────────────────────────────────────┤
│ 7. 行为规则（Behavior Rules）            │
│    "如果不确定，请明确说'不确定'..."      │
└─────────────────────────────────────────┘
```

### 10.3 一个完整的 Prompt 示例

```python
prompt = """
# 角色
你是一个电商平台的数据质量检测专家，擅长发现数据中的异常模式。

# 任务
分析以下订单数据，找出可能的数据质量问题（如重复订单、金额异常、时间不合理等）。

# 数据
| 订单ID | 用户ID | 金额  | 下单时间            |
|--------|--------|-------|---------------------|
| ORD001 | U001   | 99.00 | 2024-01-15 10:30:00 |
| ORD002 | U002   | 150.00| 2024-01-15 10:31:00 |
| ORD003 | U001   | 99.00 | 2024-01-15 10:30:01 |  <-- 疑似重复
| ORD004 | U003   | -50.00| 2024-01-15 11:00:00 |  <-- 金额为负
| ORD005 | U004   | 999999| 2024-01-15 12:00:00 |  <-- 金额异常大
| ORD006 | U005   | 200.00| 2025-12-31 23:59:59 |  <-- 时间在未来？

# 约束
- 每条异常都需要给出严重程度（高/中/低）
- 给出可能的产生原因
- 给出修复建议

# 输出格式
{
  "总记录数": <number>,
  "异常记录": [
    {
      "订单ID": "<string>",
      "异常类型": "<string>",
      "严重程度": "<高/中/低>",
      "可能原因": "<string>",
      "修复建议": "<string>"
    }
  ],
  "数据质量评分": "<0-100>"
}
"""
```

---

## 第11章：System Prompt 与 User Prompt

### 11.1 对话消息的角色

Chat Completion API 中有三种消息角色（加上工具角色共四种）：

| 角色 | 含义 | 使用场景 |
|------|------|----------|
| `system` | 系统级指令，定义 AI 的行为边界和角色 | 设置 AI 的角色、规则、约束 |
| `user` | 用户发送的消息 | 提问、指令、数据输入 |
| `assistant` | AI 的历史回答 | 多轮对话中的上下文 |
| `tool` | 工具调用的返回结果 | Function Calling 中的工具返回值 |

### 11.2 System Prompt 的最佳实践

System Prompt 的优先级通常高于 User Prompt（模型更倾向于遵守 system prompt 中的规则）。

```python
SYSTEM_PROMPT = """
# 角色定义
你是一个运维告警排查助手，名叫"小维"。

# 核心规则（必须遵守）
1. 你只基于提供的证据进行分析，不编造不存在的数据
2. 如果不确定原因，明确说"当前证据不足，建议进一步排查：..."
3. 严禁执行以下操作：
   - 重启服务器、修改配置、执行命令
   - 暴露敏感信息（密码、密钥、内部IP）
   - 给出可能造成生产事故的建议
4. 所有金额、百分比、数量必须附上数据来源
5. 每次回答后，列出"已确认的事实"和"待验证的假设"

# 你的能力
- 分析告警日志和监控数据
- 查询数据库指标
- 给出排查思路和优先级建议

# 输出风格
- 专业但易于理解
- 先给结论，再给分析过程
- 使用 Markdown 格式组织信息
"""

# 使用
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "数据库连接池使用率达到95%，请帮我分析原因"},
]
```

### 11.3 System Prompt 的常见组成模块

```python
def build_system_prompt(
    role: str,
    domain: str,
    rules: list[str],
    capabilities: list[str],
    output_format: str,
) -> str:
    """构造结构化的 System Prompt"""
    
    parts = []
    
    parts.append(f"# 角色\n你是{role}，专注于{domain}领域。\n")
    
    parts.append("# 核心规则\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules)) + "\n")
    
    parts.append("# 你的能力\n" + "\n".join(f"- {c}" for c in capabilities) + "\n")
    
    parts.append(f"# 输出格式\n{output_format}\n")
    
    return "\n".join(parts)

# 使用示例
system_prompt = build_system_prompt(
    role="数据质量检测专家",
    domain="电商交易数据",
    rules=[
        "只基于数据说话，不臆测",
        "发现异常时必须附带数据证据",
        "不确定时明确标记为'待确认'",
    ],
    capabilities=[
        "检测重复数据",
        "检测数值异常",
        "检测时间异常",
        "检测逻辑矛盾",
    ],
    output_format="以JSON格式输出检测结果，包含：record_count, anomalies[], quality_score",
)
```

---

## 第12章：Few-Shot Prompting（少样本提示）

### 12.1 什么是 Few-Shot

**Few-Shot Prompting** 是在 prompt 中给出几个"输入→期望输出"的示例，帮助模型理解你想要的格式和风格。

```
Zero-Shot（零样本）:  只给任务，不给示例
One-Shot（单样本）:   给 1 个示例
Few-Shot（少样本）:   给 2-5 个示例
Many-Shot（多样本）:  给很多示例（受限于上下文窗口）
```

### 12.2 Few-Shot 实战示例

```python
def classify_alert_severity(alert_message: str) -> str:
    """用 Few-Shot Prompting 让模型分类告警严重程度"""
    
    prompt = f"""
你是一个告警分类助手。请根据告警信息判断严重程度，只输出：P0（紧急）、P1（严重）、P2（一般）、P3（提示）。

# 示例

告警: "核心数据库主库宕机，所有交易中断"
严重程度: P0

告警: "用户登录服务响应时间超过5秒，影响部分用户"
严重程度: P1

告警: "某台非核心服务器CPU使用率超过80%"
严重程度: P2

告警: "磁盘空间使用率达到70%，建议扩容"
严重程度: P3

# 现在请分类以下告警

告警: "{alert_message}"
严重程度:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 分类任务用低温度
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

# 测试
print(classify_alert_severity("Redis缓存集群全部不可用，所有前端页面白屏"))
# 预期输出: P0
```

### 12.3 Few-Shot 示例的选择技巧

1. **覆盖边界情况**：示例要包含正常情况、异常情况和边界情况
2. **多样性**：示例之间要有差异，避免过于相似
3. **一致性**：示例的输出格式必须完全一致
4. **难度递进**：从简单到复杂排列示例
5. **数量适度**：通常 2-3 个示例就足够，过多浪费 token

```python
# 好的 Few-Shot 示例设计
FEW_SHOT_EXAMPLES = """
示例1（简单情况）:
输入: "服务器CPU 100%"
输出: {"原因": "CPU过载", "建议": "检查进程列表，找出CPU占用最高的进程"}

示例2（复杂情况）:
输入: "订单服务响应慢，数据库连接数高，同时CPU正常"
输出: {"原因": "数据库连接池瓶颈，可能慢SQL导致连接堆积", "建议": "1. 检查慢查询日志 2. 检查连接池配置 3. 检查是否有未释放的连接"}

示例3（罕见情况）:
输入: "定时任务间歇性失败，日志显示OOM，但重启后恢复正常"
输出: {"原因": "内存泄漏导致定时任务OOM", "建议": "1. 分析heap dump 2. 检查是否有大对象未释放 3. 临时方案：增加内存或拆分任务"}
"""
```

---

## 第13章：Chain-of-Thought（思维链）

### 13.1 什么是思维链

**Chain-of-Thought（CoT，思维链）** 是一种让模型在给出最终答案之前先"展示推理过程"的技术。就像做数学题要写解题步骤一样。

### 13.2 两种 CoT 方式

**方式一：在 prompt 中要求模型逐步推理**

```python
# 不用 CoT（直接给答案）
prompt_no_cot = """
问题: 一个农夫有17只羊，除了9只之外都死了，还剩几只？
答案:"""

# 使用 CoT（要求逐步推理）
prompt_with_cot = """
问题: 一个农夫有17只羊，除了9只之外都死了，还剩几只？

请按以下步骤思考：
第一步：理解题意。"除了9只之外都死了"是什么意思？
第二步：计算。"除了9只之外"意味着有9只没死。
第三步：结论。剩下的就是这9只。

答案:"""

# CoT 通常能显著提高复杂推理的准确率！
```

**方式二：Few-Shot CoT（在示例中展示推理过程）**

```python
def analyze_performance_issue(metrics: str) -> str:
    """用 Few-Shot CoT 分析性能问题"""
    
    prompt = f"""
你是一个性能分析专家。对于性能问题，请按以下思路分析：
1. 先看哪个指标异常
2. 分析异常指标之间的关联
3. 追溯可能的根因
4. 给出排查优先级

# 示例

告警: "API响应时间从200ms升到5s，数据库CPU从30%升到90%，应用服务器CPU正常"

分析过程:
步骤1 - 识别异常指标:
- API响应时间: 异常（200ms → 5s，增长25倍）
- 数据库CPU: 异常（30% → 90%）
- 应用服务器CPU: 正常

步骤2 - 关联分析:
- 数据库CPU飙升 + 应用服务器CPU正常 → 瓶颈在数据库，不在应用层
- API响应时间飙升 + 数据库CPU飙升 → API响应慢根因很可能是数据库

步骤3 - 根因推测（按可能性排序）:
1. 慢SQL（最可能）：突然出现的慢查询拖垮数据库
2. 锁等待：大事务持有锁导致其他查询等待
3. 连接数暴增：突发流量导致连接池耗尽
4. 数据量变化：统计信息过期导致执行计划变差

步骤4 - 排查建议:
1. 立即: 查看慢查询日志，找出耗时最长的SQL
2. 立即: 查看数据库当前连接数和等待事件
3. 后续: 检查是否有新上线的代码或数据变更

---

现在请分析以下告警:
{metrics}

请按照上述格式逐步分析:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 分析任务用低温度
        max_tokens=1000,
    )
    return response.choices[0].message.content
```

### 13.3 CoT 的适用与不适用场景

**适用场景**：
- 数学推理、逻辑推理
- 多步骤问题分析
- 需要权衡多种因素的决策
- "为什么"类的问题

**不适用场景**：
- 简单的翻译或摘要
- 已知答案的事实性问题
- 创意写作（可能会让输出变得机械）
- 对延迟要求很高的场景（CoT 输出更长，耗时更多）

### 13.4 零样本 CoT（Zero-Shot CoT）

只需要加一句"Let's think step by step"（让我们一步步思考）：

```python
# 零样本 CoT：只需一个魔法短语
prompt = """
问题: 如果3个人3天喝3桶水，那么9个人9天喝几桶水？

让我们一步步思考。
"""

# 模型会自己推理：
# 3人3天喝3桶 → 3人1天喝1桶 → 1人1天喝1/3桶 → 9人9天喝 9×9×(1/3)=27桶
# 正确！如果不加"一步步思考"，模型可能直接错误地回答"9桶"
```

---

## 第14章：结构化输出控制

### 14.1 为什么需要结构化输出

在 Agent 开发中，你几乎总是需要让模型返回结构化的数据（JSON），而不是自由文本，因为：

- 程序需要解析模型的输出
- 需要从输出中提取特定字段
- 需要验证输出的正确性

### 14.2 让模型输出 JSON

```python
import json

def extract_alert_info(alert_text: str) -> dict:
    """从告警文本中提取结构化信息"""
    
    prompt = f"""
从以下告警信息中提取关键字段，以JSON格式返回。

# 告警信息
{alert_text}

# 输出要求
请严格按照以下JSON schema输出，不要输出任何其他内容：

{{
  "告警级别": "P0/P1/P2/P3",
  "告警来源": "服务名或系统名",
  "告警类型": "CPU/内存/磁盘/网络/应用/数据库/其他",
  "告警摘要": "一句话总结（不超过30字）",
  "影响范围": "描述影响的用户或业务",
  "发生时间": "ISO 8601格式，如果原文没有则填null",
  "涉及指标": [
    {{"指标名": "xxx", "当前值": "xxx", "阈值": "xxx"}}
  ]
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},  # OpenAI 的 JSON 模式
        max_tokens=500,
    )
    
    # 解析 JSON
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"原始输出: {response.choices[0].message.content}")
        return {"error": "JSON 解析失败"}

# 测试
alert = "【P1告警】订单服务order-service在2024-06-15 14:30:00发生数据库连接超时，当前连接数200/最大200，影响用户下单功能"
result = extract_alert_info(alert)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

### 14.3 JSON Mode vs 手动解析

OpenAI 的 `response_format={"type": "json_object"}` 能提高 JSON 输出的可靠性，但：

- 需要在 prompt 中明确提到 "JSON"
- 模型仍然可能产生不完整的 JSON
- 不是所有模型都支持 JSON Mode（只有 OpenAI 和部分兼容 API 支持）

### 14.4 健壮的 JSON 解析

```python
import json
import re

def robust_json_parse(text: str) -> dict:
    """健壮的 JSON 解析，处理各种格式问题"""
    
    # 尝试 1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试 2: 提取 ```json ... ``` 代码块中的内容
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试 3: 提取 { ... } 最外层大括号
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 尝试 4: 修复常见问题后重试
    # 移除尾部逗号
    fixed = re.sub(r',\s*}', '}', text)
    fixed = re.sub(r',\s*]', ']', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return {"error": "无法解析JSON", "raw_text": text[:500]}
```

### 14.5 使用 Pydantic 定义输出 Schema

在实际项目中，强烈建议使用 Pydantic 定义数据模型：

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class Severity(str, Enum):
    P0 = "P0"  # 紧急
    P1 = "P1"  # 严重
    P2 = "P2"  # 一般
    P3 = "P3"  # 提示

class AlertMetric(BaseModel):
    """告警指标"""
    指标名: str = Field(description="指标名称")
    当前值: str = Field(description="当前指标值")
    阈值: Optional[str] = Field(default=None, description="告警阈值")

class AlertInfo(BaseModel):
    """告警信息结构"""
    告警级别: Severity = Field(description="告警严重程度")
    告警来源: str = Field(description="产生告警的服务名或系统名")
    告警类型: str = Field(description="告警类型：CPU/内存/磁盘/网络/应用/数据库/其他")
    告警摘要: str = Field(description="一句话总结，不超过30字")
    影响范围: str = Field(description="影响的用户或业务范围")
    发生时间: Optional[str] = Field(default=None, description="ISO 8601 格式时间")
    涉及指标: List[AlertMetric] = Field(default_factory=list, description="相关的监控指标")

# 将 Pydantic schema 嵌入 prompt
schema_json = AlertInfo.model_json_schema()
print(json.dumps(schema_json, ensure_ascii=False, indent=2))
```

---

## 第15章：Prompt 模板化

### 15.1 为什么需要模板化

在实际项目中，你不会每次手写 prompt。模板化让你：

- 复用 prompt 结构
- 用变量动态填充
- 统一管理和版本控制 prompt
- 方便 A/B 测试不同的 prompt 版本

### 15.2 使用 Jinja2 模板

Jinja2 是 Python 最流行的模板引擎：

```python
# 安装：pip install jinja2
from jinja2 import Template

# 定义模板
ALERT_ANALYSIS_TEMPLATE = Template("""
# 角色
你是一个{{ domain }}领域的告警排查专家。

# 当前告警
告警级别: {{ severity }}
告警来源: {{ source }}
告警时间: {{ time }}
告警详情: {{ detail }}

{% if related_alerts %}
# 相关告警（可能有关联）
{% for alert in related_alerts %}
- {{ alert.time }}: {{ alert.source }} - {{ alert.summary }}
{% endfor %}
{% endif %}

# 请分析
1. 告警根因分析
2. 影响范围评估
3. 排查步骤（按优先级排列）
4. 是否需要升级处理

{% if context %}
# 补充上下文
{{ context }}
{% endif %}

请使用中文输出，格式为Markdown。
""")

# 使用模板
prompt = ALERT_ANALYSIS_TEMPLATE.render(
    domain="电商交易系统",
    severity="P1",
    source="order-service",
    time="2024-06-15 14:30:00",
    detail="数据库连接池耗尽，当前连接数200/200，大量订单创建失败",
    related_alerts=[
        {"time": "2024-06-15 14:25:00", "source": "db-monitor", "summary": "慢查询数量突增"},
        {"time": "2024-06-15 14:28:00", "source": "payment-service", "summary": "支付接口超时"},
    ],
    context="今早10:00发布了一个新的促销活动，流量是平时的3倍",
)

print(prompt)
```

### 15.3 结构化 Prompt 管理

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class PromptTemplate:
    """可管理的 Prompt 模板"""
    name: str
    version: str
    system_prompt: str
    user_prompt_template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    
    def render(self, **kwargs) -> tuple[str, str]:
        """渲染 system prompt 和 user prompt"""
        from jinja2 import Template
        system = Template(self.system_prompt).render(**kwargs)
        user = Template(self.user_prompt_template).render(**kwargs)
        return system, user
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "variables": self.variables,
            "description": self.description,
        }

# Prompt 模板库
class PromptLibrary:
    """管理所有 Prompt 模板"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
    
    def register(self, template: PromptTemplate):
        self.templates[template.name] = template
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        return list(self.templates.keys())

# 注册模板
library = PromptLibrary()

library.register(PromptTemplate(
    name="alert_analysis",
    version="1.0.0",
    description="告警分析模板",
    variables=["domain", "severity", "source", "time", "detail", "related_alerts", "context"],
    system_prompt="你是一个{{ domain }}领域的告警排查专家。请基于证据分析，不编造事实。",
    user_prompt_template="""
告警级别: {{ severity }}
告警来源: {{ source }}
告警时间: {{ time }}
告警详情: {{ detail }}

{% if related_alerts %}
相关告警:
{% for a in related_alerts %}
- {{ a.time }} {{ a.source }}: {{ a.summary }}
{% endfor %}
{% endif %}

{% if context %}
背景: {{ context }}
{% endif %}

请分析根因并给出排查建议。
"""
))

# 使用时
system, user = library.get("alert_analysis").render(
    domain="电商",
    severity="P1",
    source="order-service",
    time="2024-06-15 14:30:00",
    detail="数据库连接池耗尽",
    related_alerts=[],
    context="促销活动期间",
)
```

---

## 第16章：Prompt 调试技巧与常见陷阱

### 16.1 调试技巧

#### 技巧1: 系统化地调优 Prompt

```python
def prompt_debugger(
    system_prompt: str,
    user_prompt: str,
    test_cases: list[dict],
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    对同一套 prompt 运行多个测试用例，观察输出一致性
    """
    results = []
    for case in test_cases:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(**case["input"])},
            ],
            temperature=0.0,
        )
        results.append({
            "input": case["input"],
            "expected": case.get("expected"),
            "actual": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
        })
    return results

# 使用
test_cases = [
    {"input": {"question": "什么是微服务？"}, "expected": "应包含'独立部署'"},
    {"input": {"question": "什么是Docker？"}, "expected": "应包含'容器化'"},
]
results = prompt_debugger(
    system_prompt="你是技术专家，用简洁的语言回答问题，不超过100字。",
    user_prompt="{question}",
    test_cases=test_cases,
)
for r in results:
    print(f"输入: {r['input']}")
    print(f"期望: {r.get('expected', 'N/A')}")
    print(f"实际: {r['actual'][:200]}")
    print("---")
```

#### 技巧2: 对比不同 Prompt 版本

```python
def ab_test_prompts(prompt_a: str, prompt_b: str, test_questions: list[str]) -> list[dict]:
    """A/B 测试两个 prompt"""
    results = []
    for q in test_questions:
        # Prompt A
        ra = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_a + "\n" + q}],
            temperature=0.0,
        )
        # Prompt B
        rb = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_b + "\n" + q}],
            temperature=0.0,
        )
        results.append({
            "question": q,
            "prompt_a_result": ra.choices[0].message.content,
            "prompt_b_result": rb.choices[0].message.content,
            "a_tokens": ra.usage.total_tokens,
            "b_tokens": rb.usage.total_tokens,
        })
    return results
```

### 16.2 常见陷阱

#### 陷阱1: 指令冲突

```python
# ❌ 错误：System Prompt 和 User Prompt 冲突
system = "你只输出JSON格式的结果"
user = "请用自然语言详细解释..."

# ✅ 正确：确保指令一致
system = "你只输出JSON格式的结果"
user = "请分析以下数据并以JSON格式输出结果"
```

#### 陷阱2: 过度约束

```python
# ❌ 错误：约束太多，模型无所适从
prompt = """
你必须回答得详细、但又要简洁；
必须包含数据、但不能超过100字；
必须给出建议、但只能是3条；
...（更多矛盾指令）
"""

# ✅ 正确：清晰的约束
prompt = """
- 回答：简洁（不超过200字）
- 格式：先用一句话总结，再列出3个要点
- 风格：专业但不晦涩
"""
```

#### 陷阱3: 忽略模型的"懒惰"

```python
# ❌ 模型可能偷懒回答"是的"/"没问题"
user_prompt = "这个告警严重吗？"

# ✅ 要求具体分析
user_prompt = "请分析这个告警的严重程度（P0-P3），并说明你的判断依据和影响范围"
```

#### 陷阱4: 没有处理模型"拒绝回答"

```python
# 有时模型会因为安全策略拒绝回答
# 需要在代码中处理

response = client.chat.completions.create(...)
content = response.choices[0].message.content

if response.choices[0].finish_reason == "content_filter":
    print("⚠️ 内容被安全过滤器拦截")
elif "无法" in content and "回答" in content:
    print("⚠️ 模型拒绝回答，可能需要调整 prompt 措辞")
```

### 16.3 Prompt 版本管理建议

```python
# 建议：用 YAML 文件管理 prompt
"""
# prompts/alert_analysis_v2.yaml
name: alert_analysis
version: "2.0.0"
system_prompt: |
  你是一个{domain}领域的告警排查专家。
  
  ## 规则
  1. 只基于证据分析
  2. 不确定时标注"待确认"
  3. 不给出执行命令的建议
  
  ## 输出格式
  {output_format}

user_prompt_template: |
  ## 告警信息
  级别: {severity}
  来源: {source}
  时间: {time}
  详情: {detail}
  
  ## 分析要求
  {analysis_requirements}

variables:
  - domain
  - output_format
  - severity
  - source
  - time
  - detail
  - analysis_requirements
"""
```

---

# 第三部分：Function Calling 工具调用（进阶篇）

## 第17章：什么是 Function Calling

### 17.1 核心概念

**Function Calling（函数调用/工具调用）** 是 LLM 最关键的能力之一，它让模型不仅能"说"，还能"做"。

传统模式：
```
用户: "今天北京天气怎么样？"
模型: "抱歉，我没有实时数据，无法查询天气。"  ❌
```

Function Calling 模式：
```
用户: "今天北京天气怎么样？"
模型: [决定调用 get_weather(city="北京")]  ← 模型决定"我需要查天气"
系统: [实际执行 get_weather()，获取真实数据]
系统: [把天气数据返回给模型]
模型: "今天北京晴转多云，气温22-30℃，适合户外活动。"  ✅
```

### 17.2 工作流程

```
┌──────────────────────────────────────────────────────┐
│ 步骤1: 用户发送消息 + 可用工具定义                      │
│   messages: [{"role": "user", "content": "今天北京天气?"}]│
│   tools: [get_weather的定义]                          │
├──────────────────────────────────────────────────────┤
│ 步骤2: 模型返回"我要调用工具"的决策                     │
│   response.choices[0].finish_reason = "tool_calls"    │
│   response.choices[0].message.tool_calls = [          │
│     {name: "get_weather", arguments: '{"city":"北京"}'}│
│   ]                                                   │
├──────────────────────────────────────────────────────┤
│ 步骤3: 你的代码实际执行工具                             │
│   result = get_weather(city="北京")                    │
│   → {"temperature": 25, "condition": "晴"}            │
├──────────────────────────────────────────────────────┤
│ 步骤4: 把工具结果返回给模型                             │
│   messages.append({"role": "tool", "content": result}) │
├──────────────────────────────────────────────────────┤
│ 步骤5: 模型基于工具结果生成最终回答                      │
│   "今天北京晴，气温25℃..."                            │
└──────────────────────────────────────────────────────┘
```

---

## 第18章：定义工具函数

### 18.1 工具定义的 JSON Schema

工具需要用 JSON Schema 描述给模型（让它知道有哪些工具可用、每个工具的参数是什么）：

```python
# 一个天气查询工具的定义
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",  # 函数名
        "description": "查询指定城市的实时天气信息。返回温度、湿度、天气状况等。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：'北京'、'上海'、'杭州'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，celsius（摄氏度）或 fahrenheit（华氏度）"
                }
            },
            "required": ["city"],  # 必填参数
        }
    }
}
```

### 18.2 常用工具类型及定义

```python
# 工具1: 数据库查询
db_query_tool = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "执行SQL查询并返回结果。仅支持SELECT查询，不支持INSERT/UPDATE/DELETE等写操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "要执行的SQL查询语句，仅限SELECT语句"
                },
                "database": {
                    "type": "string",
                    "enum": ["orders_db", "logs_db", "metrics_db"],
                    "description": "要查询的数据库名称"
                }
            },
            "required": ["sql", "database"],
        }
    }
}

# 工具2: 日志搜索
search_logs_tool = {
    "type": "function",
    "function": {
        "name": "search_logs",
        "description": "在应用日志中搜索包含指定关键词的日志条目，支持时间范围过滤。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，支持多个关键词用空格分隔"
                },
                "service": {
                    "type": "string",
                    "description": "服务名称，如 'order-service', 'payment-service'"
                },
                "start_time": {
                    "type": "string",
                    "description": "开始时间，ISO 8601格式，如 '2024-06-15T14:00:00'"
                },
                "end_time": {
                    "type": "string",
                    "description": "结束时间，ISO 8601格式"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回的最大日志条数，默认50",
                    "default": 50,
                }
            },
            "required": ["keyword", "service"],
        }
    }
}

# 工具3: 查询监控指标
query_metrics_tool = {
    "type": "function",
    "function": {
        "name": "query_metrics",
        "description": "查询服务或主机的监控指标数据（CPU、内存、QPS、延迟等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["cpu_usage", "memory_usage", "qps", "latency_p99", "error_rate", "disk_usage"],
                    "description": "要查询的指标名称"
                },
                "target": {
                    "type": "string",
                    "description": "目标服务名或主机名"
                },
                "time_range_minutes": {
                    "type": "integer",
                    "description": "查询最近多少分钟的数据",
                    "default": 30,
                }
            },
            "required": ["metric", "target"],
        }
    }
}

# 工具4: 知识库搜索
search_knowledge_tool = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "在运维知识库中搜索相关文档、历史案例和排查手册（Runbook）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，描述你要找的知识内容"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的K个结果，默认3",
                    "default": 3,
                }
            },
            "required": ["query"],
        }
    }
}

# 工具5: 发送通知
send_notification_tool = {
    "type": "function",
    "function": {
        "name": "send_notification",
        "description": "向指定渠道发送通知消息（企业微信/钉钉/邮件）。注意：使用前需确认用户已授权。",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["wechat_work", "dingtalk", "email"],
                    "description": "通知渠道"
                },
                "recipient": {
                    "type": "string",
                    "description": "接收人/群组标识"
                },
                "message": {
                    "type": "string",
                    "description": "通知内容"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "normal", "low"],
                    "description": "消息优先级",
                    "default": "normal",
                }
            },
            "required": ["channel", "recipient", "message"],
        }
    }
}
```

### 18.3 用 Pydantic 生成工具定义

在实际项目中，手动写 JSON Schema 容易出错。推荐用 Pydantic 自动生成：

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json

class GetWeatherParams(BaseModel):
    """天气查询参数"""
    city: str = Field(description="城市名称，如'北京'、'上海'")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )

class QueryDatabaseParams(BaseModel):
    """数据库查询参数"""
    sql: str = Field(description="SQL查询语句，仅限SELECT")
    database: Literal["orders_db", "logs_db", "metrics_db"] = Field(
        description="目标数据库"
    )

def pydantic_to_tool(
    name: str,
    description: str,
    params_model: type[BaseModel],
) -> dict:
    """将 Pydantic 模型转换为 OpenAI tool 定义"""
    schema = params_model.model_json_schema()
    # 移除 Pydantic 特有的字段（如 title）
    schema.pop("title", None)
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        }
    }

# 使用
weather_tool = pydantic_to_tool(
    name="get_weather",
    description="查询指定城市的实时天气",
    params_model=GetWeatherParams,
)

print(json.dumps(weather_tool, ensure_ascii=False, indent=2))
```

---

## 第19章：完整的 Function Calling 流程

### 19.1 端到端实现

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-api-key")

# ============================================================
# 步骤1: 定义工具的 Schema（告诉模型有哪些工具可用）
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"],
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
def get_weather(city: str) -> str:
    """模拟天气查询（实际项目中替换为真实API）"""
    # 模拟数据
    weather_data = {
        "北京": "晴，22°C，湿度45%",
        "上海": "多云，25°C，湿度70%",
        "杭州": "小雨，20°C，湿度85%",
        "深圳": "雷阵雨，28°C，湿度90%",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 工具名到函数的映射
available_functions = {
    "get_weather": get_weather,
    "get_time": get_time,
}

# ============================================================
# 步骤3: 实现完整的对话循环
# ============================================================
def run_agent(user_message: str) -> str:
    """执行一次完整的 Agent 对话（可能包含多轮工具调用）"""
    
    messages = [
        {"role": "system", "content": "你是一个助手，可以查询天气和时间。请用中文回答。"},
        {"role": "user", "content": user_message},
    ]
    
    max_iterations = 5  # 防止无限循环
    
    for iteration in range(max_iterations):
        print(f"\n--- 第 {iteration + 1} 轮 ---")
        
        # 调用模型
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 让模型自动决定是否调用工具
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
# 测试1: 需要调用工具的查询
result = run_agent("今天北京天气怎么样？上海呢？")
print(f"\n最终回答:\n{result}")

# 测试2: 不需要工具的查询
result = run_agent("你好，请介绍一下你自己")
print(f"\n最终回答:\n{result}")
```

### 19.2 tool_choice 参数详解

```python
# tool_choice 控制工具调用的行为：

# 1. "auto"（默认）- 模型自动决定是否调用工具
tool_choice="auto"

# 2. "none" - 强制不调用工具（即使你定义了工具）
tool_choice="none"

# 3. "required" - 强制必须调用工具
tool_choice="required"

# 4. 指定工具 - 强制调用特定工具
tool_choice={"type": "function", "function": {"name": "get_weather"}}
```

---

## 第20章：多工具并行调用

### 20.1 并行调用

当用户的一次请求需要调用多个独立工具时，模型可以一次性返回多个 tool_calls：

```python
# 用户: "北京和上海的天气分别怎么样？"
# 模型可以并行调用两个 get_weather(city="北京") 和 get_weather(city="上海")

def run_parallel_tools(user_message: str):
    """支持并行工具调用的 Agent"""
    
    messages = [
        {"role": "system", "content": "你是天气助手，可以同时查询多个城市的天气。"},
        {"role": "user", "content": user_message},
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        # parallel_tool_calls 允许/禁止并行调用（OpenAI 默认开启）
        # parallel_tool_calls=True,  
    )
    
    choice = response.choices[0]
    
    if choice.finish_reason == "tool_calls":
        tool_calls = choice.message.tool_calls
        print(f"模型决定调用 {len(tool_calls)} 个工具:")
        
        # 并行执行（如果工具间没有依赖关系）
        import concurrent.futures
        
        def execute_tool(tool_call):
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func = available_functions.get(func_name)
            result = func(**func_args) if func else f"未知工具: {func_name}"
            return tool_call.id, result
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_tool, tc) for tc in tool_calls]
            results = {f.result()[0]: f.result()[1] for f in futures}
        
        # 构造返回消息
        messages.append(choice.message)
        for tc in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": results[tc.id],
            })
        
        # 再次调用模型生成最终回答
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content
    
    return choice.message.content
```

### 20.2 依赖工具调用（串行）

有些场景下，工具调用之间存在依赖关系：

```python
# 场景: 先查用户ID，再用ID查订单
def run_sequential_tools(user_name: str):
    """串行工具调用：先查用户，再查订单"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_id",
                "description": "根据用户名查询用户ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "用户名"}
                    },
                    "required": ["username"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_orders",
                "description": "根据用户ID查询订单列表",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "用户ID"}
                    },
                    "required": ["user_id"],
                }
            }
        }
    ]
    
    # 模拟函数实现
    users_db = {"张三": "U001", "李四": "U002"}
    orders_db = {
        "U001": ["订单A: ¥99", "订单B: ¥150"],
        "U002": ["订单C: ¥200"],
    }
    
    def get_user_id(username: str) -> str:
        return users_db.get(username, "未找到")
    
    def get_user_orders(user_id: str) -> str:
        return ", ".join(orders_db.get(user_id, ["无订单"]))
    
    available_funcs = {
        "get_user_id": get_user_id,
        "get_user_orders": get_user_orders,
    }
    
    messages = [
        {"role": "system", "content": "你可以查询用户和订单信息。"},
        {"role": "user", "content": f"{user_name}最近的订单有哪些？"},
    ]
    
    # 多轮工具调用循环
    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )
        
        choice = response.choices[0]
        
        if choice.finish_reason != "tool_calls":
            return choice.message.content
        
        messages.append(choice.message)
        
        for tc in choice.message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            func = available_funcs[name]
            result = func(**args)
            print(f"调用 {name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    
    return "达到最大调用次数"
```

---

## 第21章：工具调用错误处理

### 21.1 常见错误类型

```python
import traceback

def safe_tool_executor(tool_call, available_functions: dict) -> str:
    """
    安全地执行工具调用，处理各种错误情况
    """
    func_name = tool_call.function.name
    
    try:
        # 错误1: 工具不存在
        if func_name not in available_functions:
            return json.dumps({
                "error": "tool_not_found",
                "message": f"工具 '{func_name}' 不存在。可用工具: {list(available_functions.keys())}"
            }, ensure_ascii=False)
        
        # 错误2: 参数解析失败
        try:
            func_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": "invalid_arguments",
                "message": f"工具参数解析失败: {str(e)}",
                "raw_arguments": tool_call.function.arguments,
            }, ensure_ascii=False)
        
        # 错误3: 函数执行异常
        func = available_functions[func_name]
        try:
            result = func(**func_args)
            return str(result)
        except TypeError as e:
            return json.dumps({
                "error": "invalid_parameters",
                "message": f"参数类型错误: {str(e)}",
                "expected_params": list(func.__code__.co_varnames[:func.__code__.co_argcount]),
                "received_params": func_args,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "error": "execution_error",
                "message": f"工具执行失败: {str(e)}",
                "traceback": traceback.format_exc()[:500],
            }, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({
            "error": "unknown_error",
            "message": f"未知错误: {str(e)}",
        }, ensure_ascii=False)

# 错误处理最佳实践：
# 1. 工具返回错误时，用 JSON 格式（方便模型理解）
# 2. 错误信息包含足够上下文（但不暴露敏感信息）
# 3. 让模型根据错误信息调整策略（如换个参数重试）
```

### 21.2 重试机制

```python
def execute_with_retry(tool_call, available_functions: dict, max_retries: int = 2) -> str:
    """带重试的工具执行"""
    
    for attempt in range(max_retries + 1):
        result = safe_tool_executor(tool_call, available_functions)
        
        # 检查是否为可重试的错误
        try:
            error_info = json.loads(result)
            if error_info.get("error") == "execution_error":
                if "timeout" in error_info.get("message", "").lower():
                    if attempt < max_retries:
                        import time
                        wait = (attempt + 1) * 2  # 指数退避
                        print(f"超时重试，等待 {wait}s...")
                        time.sleep(wait)
                        continue
        except json.JSONDecodeError:
            pass
        
        return result
    
    return result  # 返回最后一次尝试的结果
```

---

## 第22章：工具调用的最佳实践

### 22.1 工具设计原则

```python
# ❌ 不好的工具设计：功能过多，描述模糊
bad_tool = {
    "name": "do_everything",
    "description": "执行各种操作",
    "parameters": {
        "properties": {
            "action": {"type": "string", "description": "要执行的操作"},
            "data": {"type": "object", "description": "操作数据"},
        }
    }
}

# ✅ 好的工具设计：单一职责，描述清晰
good_tool = {
    "name": "query_slow_sql_logs",
    "description": "查询数据库的慢SQL日志，返回最近N条执行时间超过阈值的SQL语句及其耗时、执行计划等详细信息。",
    "parameters": {
        "type": "object",
        "properties": {
            "hours": {
                "type": "integer",
                "description": "查询最近几小时的慢SQL，默认1小时",
                "default": 1,
                "minimum": 1,
                "maximum": 24,
            },
            "min_duration_ms": {
                "type": "integer",
                "description": "最小执行时间（毫秒），只返回超过此时间的SQL，默认1000ms",
                "default": 1000,
            },
            "limit": {
                "type": "integer",
                "description": "最多返回条数，默认20，最大100",
                "default": 20,
                "maximum": 100,
            },
        },
        "required": [],
    }
}
```

### 22.2 设计清单

1. **单一职责**：一个工具只做一件事
2. **描述清晰**：description 要详细，帮助模型理解何时使用
3. **参数约束**：使用 enum、minimum、maximum 限制参数范围
4. **错误友好**：工具失败时返回结构化错误信息
5. **幂等性**：查询类工具应该是幂等的（多次调用结果一致）
6. **安全第一**：写操作（删除、修改）需要额外确认机制
7. **返回格式**：工具返回 JSON 或结构化文本，方便模型理解

### 22.3 实际项目中的工具注册机制

```python
from typing import Callable, Dict, Any
from functools import wraps

class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, dict] = {}      # 工具的 Schema
        self._functions: Dict[str, Callable] = {}  # 工具的实现
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
    ):
        """注册工具（装饰器方式）"""
        def decorator(func: Callable):
            self._functions[name] = func
            self._tools[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            }
            
            @wraps(func)
            def wrapper(**kwargs):
                try:
                    return str(func(**kwargs))
                except Exception as e:
                    return json.dumps({
                        "error": str(e),
                        "tool": name,
                    }, ensure_ascii=False)
            
            return wrapper
        return decorator
    
    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 Schema 列表（发送给模型）"""
        return list(self._tools.values())
    
    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行指定工具"""
        if tool_name not in self._functions:
            return f'{{"error": "工具 {tool_name} 不存在"}}'
        return self._functions[tool_name](**arguments)

# 使用示例
registry = ToolRegistry()

@registry.register(
    name="query_metrics",
    description="查询监控指标",
    parameters={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["cpu", "memory", "qps"],
                "description": "指标名"
            },
            "service": {"type": "string", "description": "服务名"},
        },
        "required": ["metric", "service"],
    }
)
def query_metrics(metric: str, service: str) -> str:
    # 实际查询逻辑
    data = {"order-service": {"cpu": "45%", "memory": "60%", "qps": "1200"}}
    return str(data.get(service, {}).get(metric, "未找到"))

@registry.register(
    name="search_logs",
    description="搜索应用日志",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "搜索关键词"},
            "service": {"type": "string", "description": "服务名"},
        },
        "required": ["keyword", "service"],
    }
)
def search_logs(keyword: str, service: str) -> str:
    # 实际搜索逻辑
    return f"找到3条包含'{keyword}'的日志（{service}）"

# 获取所有工具定义
print(json.dumps(registry.get_tool_schemas(), ensure_ascii=False, indent=2))
```

---

# 第四部分：RAG 检索增强生成（进阶篇）

## 第23章：RAG 完整原理

### 23.1 为什么需要 RAG

LLM 有两个根本性限制：
1. **知识截止日期**：训练数据有截止时间，不知道训练后的新信息
2. **幻觉问题**：模型会"编造"听起来合理但实际不存在的答案
3. **私有知识缺失**：你公司的内部文档、业务规则、历史案例，模型完全不知道

RAG 解决这些问题的方式是：**在回答问题之前，先从外部知识库中检索相关信息，把这些信息作为上下文提供给模型。**

### 23.2 RAG 的完整 Pipeline

```
                        ┌──────────────────────┐
 离线阶段（索引）        │   文档/知识库          │
                        │       ↓               │
                        │   文档分块（Chunking）  │
                        │       ↓               │
                        │   向量嵌入（Embedding） │
                        │       ↓               │
                        │   存入向量数据库        │
                        └──────────────────────┘

                        ┌──────────────────────┐
 在线阶段（查询）        │   用户提问             │
                        │       ↓               │
                        │   问题向量化（Embedding）│
                        │       ↓               │
                        │   向量相似度检索        │
                        │       ↓               │
                        │   取回 Top-K 相关文档   │
                        │       ↓               │
                        │   拼接 Prompt:          │
                        │   系统提示 + 检索文档 + 用户问题 │
                        │       ↓               │
                        │   LLM 生成回答         │
                        └──────────────────────┘
```

### 23.3 RAG 的核心组件

| 组件 | 作用 | 常用选择 |
|------|------|----------|
| 文档加载器 | 读取各种格式的文件 | PyPDF, Unstructured, langchain loaders |
| 文本分割器 | 将长文档切成小块 | RecursiveCharacterTextSplitter |
| Embedding 模型 | 将文本转为向量 | text-embedding-3, bge-large-zh |
| 向量数据库 | 存储和检索向量 | Chroma, FAISS, Milvus, Qdrant |
| 检索器 | 根据查询检索相关文档 | 向量检索 + BM25 混合检索 |
| 生成器 | 基于检索结果生成回答 | GPT-4o, Claude, DeepSeek |

---

## 第24章：文档分块策略

### 24.1 为什么分块很重要

- Embedding 模型有最大输入长度限制（通常是 512-8192 tokens）
- 小块：检索更精确，但可能丢失上下文
- 大块：上下文更完整，但检索精度下降
- **关键是要找到平衡点**

### 24.2 常见的分块方法

```python
# 安装：pip install langchain langchain-text-splitters

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)
import tiktoken

# ============================================================
# 方法1: 按字符递归分割（最常用）
# ============================================================
text = """
# 数据库慢查询排查指南

## 1. 识别慢查询

慢查询是指执行时间超过阈值的SQL语句。在MySQL中，可以通过以下方式开启慢查询日志：

```sql
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;
```

## 2. 分析慢查询

使用EXPLAIN命令分析SQL执行计划...

## 3. 常见优化方法

### 3.1 索引优化
确保查询条件中的列有合适的索引...

### 3.2 SQL改写
避免SELECT *，只查询需要的列...
"""

# 递归字符分割器：优先按段落→句子→单词分割
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最大字符数
    chunk_overlap=50,      # 块与块之间的重叠字符数（保持上下文连续）
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],  # 分割优先级
    length_function=len,
)

chunks = char_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"=== Chunk {i+1} ({len(chunk)} 字符) ===")
    print(chunk[:200] + "...")
    print()

# ============================================================
# 方法2: Markdown 标题分割（适合结构化文档）
# ============================================================
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题在内容中
)

md_chunks = md_splitter.split_text(text)
for chunk in md_chunks:
    print(f"元数据: {chunk.metadata}")
    print(f"内容: {chunk.page_content[:100]}...")
    print()

# ============================================================
# 方法3: 按 Token 数量分割（精确控制 token 消耗）
# ============================================================
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4 的 tokenizer
    chunk_size=200,       # 每块 200 tokens
    chunk_overlap=20,     # 重叠 20 tokens
)

token_chunks = token_splitter.split_text(text)
print(f"按Token分割: 共 {len(token_chunks)} 块")
for i, chunk in enumerate(token_chunks):
    # 计算实际 token 数
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(chunk))
    print(f"Chunk {i+1}: {token_count} tokens, {len(chunk)} 字符")
```

### 24.3 分块策略选择指南

| 场景 | 推荐方法 | chunk_size | overlap |
|------|---------|-----------|---------|
| 通用文档 | 递归字符分割 | 500-1000 字符 | 10-20% |
| 代码 | 按函数/类分割 | 1000-2000 字符 | 0 |
| 法律/合同 | Markdown 标题分割 | 按条款 | 上下文保留 |
| FAQ/短问答 | 按条目分割 | 完整条目 | 0 |
| 中文长文档 | 递归 + 句号分割 | 300-500 字符 | 50-100 字符 |

### 24.4 针对中文的优化分块

```python
import re

class ChineseTextSplitter:
    """针对中文文本优化的分割器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> list[str]:
        """按中文语义边界分割"""
        # 先按段落分割
        paragraphs = text.split('\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前段加入后不超限，则合并
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n"
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果段落本身超限，按句子分割
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_paragraph(para)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_paragraph(self, para: str) -> list[str]:
        """按中文标点分割长段落"""
        sentences = re.split(r'([。！？；\n])', para)
        # 重新合并标点到句子末尾
        merged = []
        for i in range(0, len(sentences) - 1, 2):
            merged.append(sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))
        if len(sentences) % 2 == 1:
            merged.append(sentences[-1])
        
        chunks = []
        current = ""
        for sent in merged:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        
        if current:
            chunks.append(current)
        
        return chunks
```

---

## 第25章：Embedding 向量嵌入

### 25.1 什么是 Embedding

**Embedding（向量嵌入）** 是将文本转换成一个固定长度的数字向量（比如 1536 维的浮点数数组）。这个向量的神奇之处在于：**语义相近的文本，它们在向量空间中的距离也相近。**

```
"今天天气真好"  →  [0.023, -0.451, 0.789, ..., 0.123]  (1536维向量)
"天气不错"      →  [0.019, -0.447, 0.792, ..., 0.118]  (向量很接近！)
"数据库连接超时" →  [-0.834, 0.291, -0.156, ..., 0.567]  (向量很远！)
```

### 25.2 使用 OpenAI Embedding

```python
from openai import OpenAI
import numpy as np

client = OpenAI(api_key="your-key")

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """获取文本的向量嵌入"""
    # 清理文本（去除多余换行）
    text = text.replace("\n", " ")
    
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding

# 测试
text1 = "数据库连接超时导致服务不可用"
text2 = "MySQL连接池耗尽，应用报错"
text3 = "今天天气很好适合出去散步"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# 计算余弦相似度
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度（-1到1，1表示完全相同）"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
```

### 25.3 使用开源中文 Embedding 模型

```python
# 安装：pip install sentence-transformers

from sentence_transformers import SentenceTransformer

# 加载中文 Embedding 模型（首次会自动下载）
# 模型选项:
# - BAAI/bge-large-zh-v1.5 (推荐，效果好)
# - moka-ai/m3e-base (轻量)
# - shibing624/text2vec-base-chinese (均衡)

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 获取 embedding
texts = [
    "数据库慢查询如何优化",
    "MySQL SQL调优方法",
    "前端页面加载速度优化",
]

embeddings = model.encode(texts, normalize_embeddings=True)

print(f"向量维度: {embeddings.shape[1]}")
print(f"查询优化 vs SQL调优: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"查询优化 vs 前端优化: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")

# BGE 模型的特殊处理：query 需要加前缀
query = "如何查找慢SQL"
query_embedding = model.encode(
    "为这个句子生成表示以用于检索相关文章：" + query,
    normalize_embeddings=True
)

# 文档不需要加前缀
doc_embedding = model.encode(
    "使用EXPLAIN分析SQL执行计划，关注type、rows、Extra字段",
    normalize_embeddings=True
)

print(f"查询-文档相似度: {cosine_similarity(query_embedding, doc_embedding):.4f}")
```

### 25.4 批量处理优化

```python
def batch_embed(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """批量 Embedding，避免内存溢出"""
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=True,  # 显示进度条
        )
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

# 使用
docs = [f"文档{i}: 这是关于Agent开发的第{i}条知识" for i in range(1000)]
embeddings = batch_embed(docs, model)
print(f"嵌入完成: {embeddings.shape}")
```

---

## 第26章：向量数据库选型与使用

### 26.1 Chroma（最易上手）

Chroma 是最简单的向量数据库，适合原型开发和中小规模应用：

```python
# 安装：pip install chromadb

import chromadb
from chromadb.utils import embedding_functions

# 创建客户端
client = chromadb.PersistentClient(path="./chroma_db")  # 持久化存储
# client = chromadb.Client()  # 内存模式（测试用）

# 使用 OpenAI Embedding（或自定义 embedding function）
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small",
)

# 创建或获取集合
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=openai_ef,
    metadata={"description": "运维知识库"},
)

# 添加文档
documents = [
    "当数据库CPU超过90%时，首先检查慢查询日志，查看是否有执行时间异常长的SQL",
    "Redis内存使用超过maxmemory时，会根据maxmemory-policy策略进行淘汰",
    "Nginx 502错误通常表示上游服务（如应用服务器）无响应或超时",
]
ids = ["doc_001", "doc_002", "doc_003"]
metadatas = [
    {"category": "数据库", "source": "runbook_v1"},
    {"category": "缓存", "source": "runbook_v1"},
    {"category": "网络", "source": "runbook_v1"},
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

# 查询
results = collection.query(
    query_texts=["数据库性能突然下降怎么办"],
    n_results=2,
    # where={"category": "数据库"},  # 元数据过滤
)

print("查询结果:")
for i, doc in enumerate(results["documents"][0]):
    print(f"  {i+1}. [距离: {results['distances'][0][i]:.4f}] {doc[:100]}...")

# 更新
collection.update(
    ids=["doc_001"],
    documents=["更新后的文档内容..."],
)

# 删除
collection.delete(ids=["doc_003"])

# 统计
print(f"集合中总文档数: {collection.count()}")
```

### 26.2 FAISS（高性能本地检索）

FAISS 是 Meta 开源的高性能向量检索库，适合大规模数据：

```python
# 安装：pip install faiss-cpu  (或 faiss-gpu 如果有GPU)

import faiss
import numpy as np

class FAISSVectorStore:
    """基于 FAISS 的向量存储"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP: 内积索引（适合归一化后的向量，相当于余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
    
    def add(self, embeddings: np.ndarray, documents: list[str], metadatas: list[dict] = None):
        """添加文档"""
        # FAISS 要求 float32
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(documents)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """搜索最相似的 k 个文档"""
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 表示没有结果
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(dist),
                })
        return results
    
    def save(self, path: str):
        """持久化"""
        faiss.write_index(self.index, f"{path}.faiss")
        import pickle
        with open(f"{path}.docs.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadatas": self.metadatas}, f)
    
    def load(self, path: str):
        """加载"""
        self.index = faiss.read_index(f"{path}.faiss")
        import pickle
        with open(f"{path}.docs.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]

# 使用示例
store = FAISSVectorStore(dimension=1024)

# 准备数据（使用之前加载的 BGE 模型）
texts = ["文档1", "文档2", "文档3"]
embeddings = model.encode(texts, normalize_embeddings=True)

store.add(embeddings, texts, [{"id": i} for i in range(len(texts))])

# 搜索
query_emb = model.encode("查询文本", normalize_embeddings=True)
results = store.search(query_emb, k=2)
for r in results:
    print(f"文档: {r['document']}, 相似度: {r['score']:.4f}")
```

### 26.3 向量数据库对比

| 数据库 | 类型 | 适用规模 | 特点 |
|--------|------|---------|------|
| Chroma | 嵌入式 | < 10万文档 | 最简单，Python原生，自带Embedding |
| FAISS | 库 | > 100万文档 | 极快，需自己管理元数据 |
| Milvus | 分布式数据库 | > 1亿文档 | 企业级，支持混合检索 |
| Qdrant | 独立服务 | > 100万文档 | Rust编写，高性能，过滤强大 |
| Weaviate | 独立服务 | > 100万文档 | GraphQL接口，内置多种向量化 |
| pgvector | PostgreSQL插件 | < 100万文档 | SQL兼容，便于和业务数据联动 |

---

## 第27章：完整的 RAG Pipeline 实现

### 27.1 端到端 RAG 系统

```python
"""
完整的 RAG (Retrieval-Augmented Generation) Pipeline 实现
包含: 文档加载 → 分块 → Embedding → 存储 → 检索 → 生成
"""
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict

# ============================================================
# 配置
# ============================================================
class RAGConfig:
    """RAG 系统配置"""
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    CHUNK_SIZE = 500       # 分块大小（字符）
    CHUNK_OVERLAP = 50     # 重叠大小
    TOP_K = 3              # 检索返回的文档数
    COLLECTION_NAME = "knowledge_base"

# ============================================================
# RAG 系统
# ============================================================
class RAGSystem:
    """完整的 RAG 问答系统"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # 初始化 LLM 客户端
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 初始化 Chroma 向量数据库
        self.chroma_client = chromadb.PersistentClient(path="./rag_db")
        
        # 使用 OpenAI embedding
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=self.config.EMBEDDING_MODEL,
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
    
    # ---------- 文档处理 ----------
    
    def _split_text(self, text: str) -> List[str]:
        """将长文本分割为小块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.CHUNK_SIZE
            
            # 尝试在句子边界分割
            if end < len(text):
                # 寻找最近的句号、换行等自然断点
                for sep in ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ']:
                    pos = text.rfind(sep, start, end)
                    if pos > start + self.config.CHUNK_SIZE // 2:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.config.CHUNK_OVERLAP
        
        return chunks
    
    def add_document(self, text: str, metadata: Dict = None) -> int:
        """添加文档到知识库"""
        chunks = self._split_text(text)
        
        if not chunks:
            return 0
        
        # 生成 ID（基于当前文档数量）
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]
        
        metadatas = [metadata or {}] * len(chunks)
        # 为每个块添加索引
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
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # 转换距离为相似度
            })
        
        return documents
    
    # ---------- 生成 ----------
    
    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """构建包含检索上下文的 Prompt"""
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
        """执行 RAG 问答"""
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
        """清空知识库"""
        self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        print("✅ 知识库已清空")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    rag = RAGSystem()
    
    # 添加知识文档
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
    
    # 提问
    result = rag.ask("数据库CPU突然飙到95%怎么办")
    
    print("=" * 50)
    print("回答:")
    print(result["answer"])
    print("\n参考来源:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] {s['content'][:100]}...")
    print(f"\nToken用量: {result['usage']}")
```

---

## 第28章：检索质量优化

### 28.1 重排序（Re-ranking）

初始的向量检索可能不够精确，重排序模型可以进一步提高精度：

```python
# 安装：pip install sentence-transformers

from sentence_transformers import CrossEncoder

class Reranker:
    """重排序器：对初步检索结果进行精细排序"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # Cross-Encoder 直接计算 query-document 对的相关性分数
        # 比 embedding 相似度更准确，但更慢
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """对文档列表重排序"""
        # 构造 (query, document) 对
        pairs = [[query, doc] for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 排序
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [{"document": doc, "score": float(score)} for doc, score in ranked[:top_k]]

# 使用
reranker = Reranker()

# 假想的初步检索结果
candidate_docs = [
    "Redis内存使用超过maxmemory时会触发淘汰策略...",
    "数据库索引可以加快查询速度...",
    "Nginx配置反向代理可以负载均衡...",
]

query = "缓存满了怎么办"

results = reranker.rerank(query, candidate_docs, top_k=2)
for r in results:
    print(f"[{r['score']:.4f}] {r['document'][:80]}...")
```

### 28.2 查询改写（Query Rewriting）

用户的原始查询可能不适合直接检索，需要改写：

```python
def rewrite_query(original_query: str, llm_client: OpenAI) -> str:
    """利用 LLM 改写查询，提高检索效果"""
    
    prompt = f"""你是一个查询优化助手。将用户的原始问题改写为更适合文档检索的查询语句。

规则：
1. 提取核心关键词和概念
2. 补充同义词和相关术语
3. 如果是问题，转换为陈述句
4. 去除口语化和无关词汇

原始问题: "我的数据库咋突然变慢了，特别慢那种"
改写后: "数据库性能下降 响应变慢 慢查询 性能问题排查"

原始问题: "服务一直报502咋回事"
改写后: "HTTP 502错误 网关错误 上游服务不可用排查"

原始问题: "{original_query}"
改写后:"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
```

---

## 第29章：HyDE 假设文档嵌入

### 29.1 HyDE 原理

**HyDE（Hypothetical Document Embeddings）** 的核心思想很巧妙：

传统方法：`用户问题 → 向量化 → 在知识库中搜索`
HyDE 方法：`用户问题 → LLM生成假设答案 → 向量化 → 在知识库中搜索`

**为什么有效？** 问题通常很短很抽象，而文档很长很具体。用 LLM 先生成一个"假设的答案"，这个答案的向量会跟真实文档的向量更接近。

### 29.2 实现

```python
class HyDERetriever:
    """基于 HyDE 的检索增强器"""
    
    def __init__(self, llm_client: OpenAI, embed_model):
        self.llm = llm_client
        self.embed_model = embed_model
    
    def generate_hypothetical_doc(self, query: str) -> str:
        """生成假设的答案文档"""
        prompt = f"""请根据以下问题，生成一段假设的答案。即使你不确定答案是否正确，也请基于常识生成。

问题: {query}

假设答案（用一段话描述，包含可能的细节和技术术语）:"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 需要一定的创造性
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    
    def retrieve(self, query: str, vector_store, top_k: int = 5) -> List[Dict]:
        """HyDE 检索"""
        # 步骤1: 生成假设文档
        hypo_doc = self.generate_hypothetical_doc(query)
        print(f"假设文档: {hypo_doc[:200]}...")
        
        # 步骤2: 用假设文档的向量去检索
        # （假设 vector_store 有 search_by_embedding 方法）
        hypo_embedding = self.embed_model.encode(hypo_doc, normalize_embeddings=True)
        
        results = vector_store.search(hypo_embedding, k=top_k)
        
        # 也可以用原始query和假设文档的混合向量
        # query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        # combined = 0.5 * query_embedding + 0.5 * hypo_embedding
        
        return results
```

---

## 第30章：多路召回与融合排序

### 30.1 混合检索架构

实际生产环境中，单一检索方式往往不够。**多路召回 + 融合排序** 是业界最佳实践：

```
用户查询
    │
    ├──→ 向量检索（语义相似）
    │         ↓
    │     候选集A (top-20)
    │
    ├──→ BM25检索（关键词匹配）
    │         ↓
    │     候选集B (top-20)
    │
    └──→ (可选) 知识图谱检索
              ↓
          候选集C (top-10)
    
    三路候选集合并去重
              ↓
         重排序（Reranker）
              ↓
         最终结果 (top-5)
```

### 30.2 RRF 融合算法

```python
def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[tuple]:
    """
    RRF (Reciprocal Rank Fusion) 融合多路召回结果
    
    Args:
        ranked_lists: 多路召回的排序结果列表
        k: 平滑参数（通常 60）
    
    Returns:
        融合后的排序结果 [(doc_id, score), ...]
    """
    scores = {}
    
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            # RRF 公式: score = 1 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    
    # 按分数降序排列
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

# 使用示例
vector_results = ["doc_3", "doc_1", "doc_5", "doc_2"]  # 向量检索结果
bm25_results = ["doc_2", "doc_4", "doc_1", "doc_3"]    # BM25检索结果

fused = reciprocal_rank_fusion([vector_results, bm25_results])
print("融合后排序:")
for doc_id, score in fused:
    print(f"  {doc_id}: {score:.4f}")
```


# 第五部分：Agent 核心架构（精通篇）

## 第31章：Agent 的完整定义与架构

### 31.1 Agent 的正式定义

Agent（智能体）是一个能够**感知环境 → 推理决策 → 执行动作 → 接收反馈 → 自我改进**的自治系统。

与普通 Chatbot 的本质区别：
- Chatbot：你问 → 我答（单轮交互）
- Agent：你给目标 → 我规划 → 我执行 → 我检查 → 我调整 → 我汇报（多轮自治）

### 31.2 Agent 的六层架构

```
┌─────────────────────────────────────────────┐
│ 输入层（Input Layer）                        │
│ 用户消息 / Webhook / 文件 / API 请求          │
├─────────────────────────────────────────────┤
│ 感知层（Perception Layer）                   │
│ 意图识别 / 实体抽取 / 任务分类                │
├─────────────────────────────────────────────┤
│ 推理层（Reasoning Layer）                    │
│ 任务规划 / 工具选择 / 策略生成                │
├─────────────────────────────────────────────┤
│ 记忆层（Memory Layer）                       │
│ 短期记忆 / 长期记忆 / 工作记忆                │
├─────────────────────────────────────────────┤
│ 执行层（Execution Layer）                    │
│ 工具调用 / API 调用 / 数据库操作              │
├─────────────────────────────────────────────┤
│ 输出层（Output Layer）                       │
│ 文本回答 / JSON / 报告 / 下一步建议           │
└─────────────────────────────────────────────┘
```

### 31.3 Agent 循环

```python
class AgentLoop:
    """标准的 Agent 循环"""
    
    def __init__(self, llm, tools, memory, max_iterations: int = 10):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
    
    def run(self, task: str) -> str:
        """运行 Agent 循环直到完成任务或达到最大迭代次数"""
        self.memory.add("user", task)
        
        for i in range(self.max_iterations):
            print(f"\n=== 第 {i+1} 轮 ===")
            
            # 1. 思考：调用 LLM 决定下一步
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.memory.get_messages(),
                tools=self.tools.get_schemas(),
            )
            
            choice = response.choices[0]
            
            # 2. 如果模型直接回答 → 任务完成
            if choice.finish_reason == "stop":
                self.memory.add("assistant", choice.message.content)
                return choice.message.content
            
            # 3. 如果模型要调用工具 → 执行工具
            elif choice.finish_reason == "tool_calls":
                self.memory.add(choice.message)
                
                for tc in choice.message.tool_calls:
                    result = self.tools.execute(
                        tc.function.name,
                        json.loads(tc.function.arguments)
                    )
                    self.memory.add_tool_result(tc.id, result)
                
                # 继续循环
        
        return "达到最大迭代次数，任务未完成"
```

---

## 第32章：ReAct Agent 完整实现

### 32.1 ReAct 模式原理

**ReAct（Reasoning + Acting）** 是 Agent 领域最经典的范式。核心思想是让模型交替进行"推理"和"行动"：

```
Thought: 我需要查询北京天气来回答用户
Action: get_weather(city="北京")
Observation: 晴，22°C
Thought: 我已获得天气信息，可以回答了
Answer: 今天北京晴，22°C...
```

### 32.2 完整的 ReAct Agent

```python
"""
完整的 ReAct Agent 实现
"""

REACT_SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来完成任务。

## 工作方式
你需要按照以下格式工作：

Thought: 分析当前情况，思考下一步应该做什么
Action: 工具名称
Action Input: 工具参数（JSON格式）

当你有足够的工具结果后，你可以给出最终回答：
Thought: 我已经收集了足够的信息
Final Answer: 你的最终回答

## 规则
1. 每次只调用一个工具
2. 在调用工具前先思考是否必要
3. 如果工具返回错误，尝试分析原因并调整
4. 当你确定可以回答用户问题时，给出 Final Answer

## 可用工具
{tools_description}
"""

class ReActAgent:
    """ReAct Agent 实现"""
    
    def __init__(self, llm_client: OpenAI, tools: ToolRegistry):
        self.llm = llm_client
        self.tools = tools
        
    def _build_system_prompt(self) -> str:
        """构建包含工具描述的 System Prompt"""
        tools_desc = json.dumps(self.tools.get_tool_schemas(), ensure_ascii=False, indent=2)
        return REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)
    
    def _parse_action(self, text: str) -> tuple:
        """从模型输出中解析 Action"""
        import re
        
        # 匹配 Action: xxx
        action_match = re.search(r'Action:\s*(\w+)', text)
        # 匹配 Action Input: {...}
        input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
        # 匹配 Final Answer: xxx
        final_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        
        if final_match:
            return ("final_answer", final_match.group(1).strip())
        elif action_match and input_match:
            try:
                args = json.loads(input_match.group(1))
                return ("action", action_match.group(1), args)
            except json.JSONDecodeError:
                return ("error", "Action Input JSON 解析失败")
        else:
            return ("unknown", text)
    
    def run(self, user_task: str, max_iterations: int = 10) -> str:
        """运行 ReAct Agent"""
        
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_task},
        ]
        
        for i in range(max_iterations):
            print(f"\n--- ReAct 第 {i+1} 轮 ---")
            
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            
            assistant_output = response.choices[0].message.content
            print(f"模型输出:\n{assistant_output[:300]}...")
            
            # 解析输出
            parsed = self._parse_action(assistant_output)
            
            if parsed[0] == "final_answer":
                print(f"✅ 任务完成")
                return parsed[1]
            
            elif parsed[0] == "action":
                _, func_name, func_args = parsed
                print(f"🔧 调用工具: {func_name}({func_args})")
                
                result = self.tools.execute(func_name, func_args)
                print(f"📊 工具结果: {result[:200]}")
                
                # 把交互历史加入上下文
                messages.append({"role": "assistant", "content": assistant_output})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {result}"
                })
            
            elif parsed[0] == "error":
                messages.append({"role": "assistant", "content": assistant_output})
                messages.append({
                    "role": "user",
                    "content": f"格式错误: {parsed[1]}。请使用正确的 Action/Action Input 格式。"
                })
            
            else:
                # 格式不符，提示模型
                messages.append({"role": "assistant", "content": assistant_output})
                messages.append({
                    "role": "user",
                    "content": "请按格式输出 Thought/Action/Action Input 或 Final Answer。"
                })
        
        return "达到最大迭代次数"
```

---

## 第33章：Plan-and-Execute Agent

### 33.1 原理

Plan-and-Execute Agent 将任务分为两个阶段：

1. **Plan 阶段**：模型先制定完整的执行计划
2. **Execute 阶段**：按计划逐步执行，并在每步后根据结果调整

```python
class PlanAndExecuteAgent:
    """先规划后执行的 Agent"""
    
    def __init__(self, llm_client: OpenAI, tools: ToolRegistry):
        self.llm = llm_client
        self.tools = tools
    
    def make_plan(self, task: str) -> list[dict]:
        """制定执行计划"""
        tools_desc = "\n".join([
            f"- {name}: {info['function']['description']}"
            for name, info in self.tools._tools.items()
        ])
        
        prompt = f"""你是一个任务规划专家。请为以下任务制定详细的执行计划。

## 可用工具
{tools_desc}

## 任务
{task}

## 输出格式
请输出 JSON 格式的执行计划：
{{
  "计划概述": "一句话概述",
  "步骤": [
    {{
      "步骤编号": 1,
      "目标": "这一步要达成什么",
      "工具": "使用的工具名（如果需要）",
      "工具参数": {{}},
      "预期结果": "预期获得什么信息"
    }}
  ]
}}
"""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        
        plan = json.loads(response.choices[0].message.content)
        return plan.get("步骤", [])
    
    def execute_plan(self, plan: list[dict]) -> list[dict]:
        """执行计划并记录每步结果"""
        results = []
        
        for step in plan:
            step_num = step["步骤编号"]
            print(f"\n--- 执行步骤 {step_num}: {step['目标']} ---")
            
            tool_name = step.get("工具")
            if tool_name and tool_name in self.tools._functions:
                result = self.tools.execute(tool_name, step.get("工具参数", {}))
            else:
                result = "跳过（无需工具调用）"
            
            step_result = {
                "步骤": step_num,
                "目标": step["目标"],
                "结果": result,
            }
            results.append(step_result)
            print(f"结果: {result[:150]}")
        
        return results
    
    def summarize(self, task: str, plan: list[dict], results: list[dict]) -> str:
        """基于执行结果生成总结"""
        summary_prompt = f"""
## 任务
{task}

## 执行计划
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 执行结果
{json.dumps(results, ensure_ascii=False, indent=2)}

请基于以上信息，生成最终的分析总结报告。包括：任务完成情况、关键发现、建议。
"""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    
    def run(self, task: str) -> dict:
        """完整的 Plan-and-Execute 流程"""
        print("📋 制定计划...")
        plan = self.make_plan(task)
        
        print("🔧 执行计划...")
        results = self.execute_plan(plan)
        
        print("📊 生成总结...")
        summary = self.summarize(task, plan, results)
        
        return {
            "plan": plan,
            "results": results,
            "summary": summary,
        }
```

---

## 第34章：Agent 状态管理

### 34.1 为什么需要状态管理

Agent 通常是一个多步骤过程。你需要跟踪：
- 当前在哪个阶段
- 已经获得了哪些信息
- 还有哪些步骤待执行
- 是否有异常需要处理

### 34.2 状态机实现

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

class AgentPhase(str, Enum):
    IDLE = "idle"              # 等待输入
    PLANNING = "planning"       # 正在规划
    EXECUTING = "executing"     # 正在执行
    WAITING_TOOL = "waiting"    # 等待工具返回
    SUMMARIZING = "summarizing" # 正在总结
    DONE = "done"               # 完成
    ERROR = "error"             # 出错

@dataclass
class AgentState:
    """Agent 的完整状态"""
    # 基本状态
    session_id: str
    phase: AgentPhase = AgentPhase.IDLE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 任务信息
    task: str = ""
    task_type: str = ""
    
    # 执行状态
    current_step: int = 0
    total_steps: int = 0
    steps_completed: List[Dict] = field(default_factory=list)
    
    # 工具调用历史
    tool_calls: List[Dict] = field(default_factory=list)
    
    # 收集的证据/数据
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    errors: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "task": self.task,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "steps_completed": self.steps_completed,
            "tool_calls": self.tool_calls,
            "evidence": self.evidence,
            "errors": self.errors,
        }
    
    def add_tool_call(self, tool_name: str, args: dict, result: Any):
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "result": str(result)[:500],
            "timestamp": datetime.now().isoformat(),
        })
        self._touch()
    
    def add_evidence(self, key: str, value: Any):
        self.evidence[key] = value
        self._touch()
    
    def add_error(self, error: str, context: dict = None):
        self.errors.append({
            "error": error,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        })
        self._touch()
    
    def _touch(self):
        self.updated_at = datetime.now().isoformat()
```

---

## 第35章：Agent 记忆系统设计

### 35.1 三层记忆架构

```
┌──────────────────────────────────────────────────┐
│ 工作记忆（Working Memory）                        │
│ 当前任务中已获得的信息、中间推理结果                │
│ 生命周期: 当前任务                                 │
│ 存储: Python 字典 / Redis                         │
├──────────────────────────────────────────────────┤
│ 短期记忆（Short-term Memory）                     │
│ 当前会话的对话历史                                 │
│ 生命周期: 当前会话                                 │
│ 存储: 消息列表 / 数据库                            │
├──────────────────────────────────────────────────┤
│ 长期记忆（Long-term Memory）                      │
│ 历史案例、用户偏好、知识库                         │
│ 生命周期: 永久                                     │
│ 存储: 向量数据库 + 关系数据库                      │
└──────────────────────────────────────────────────┘
```

### 35.2 记忆系统实现

```python
from typing import List, Dict, Optional
import sqlite3
import json
from datetime import datetime

class MemorySystem:
    """Agent 的三层记忆系统"""
    
    def __init__(self, session_id: str, db_path: str = "agent_memory.db"):
        self.session_id = session_id
        
        # 工作记忆
        self.working_memory: Dict[str, Any] = {}
        
        # 短期记忆 - 对话历史
        self.conversation: List[Dict] = []
        
        # 长期记忆 - SQLite 存储
        self.db = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS tool_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                timestamp TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                updated_at TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_history(session_id);
        """)
        self.db.commit()
    
    # --- 工作记忆 ---
    def set_working(self, key: str, value: Any):
        self.working_memory[key] = value
    
    def get_working(self, key: str) -> Optional[Any]:
        return self.working_memory.get(key)
    
    def clear_working(self):
        self.working_memory.clear()
    
    # --- 短期记忆 ---
    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})
        
        # 同时持久化
        self.db.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, datetime.now().isoformat())
        )
        self.db.commit()
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """获取最近 n 条消息"""
        return self.conversation[-n:]
    
    def get_all_messages(self) -> List[Dict]:
        return self.conversation
    
    # --- 长期记忆 ---
    def get_historical_conversations(self, limit: int = 5) -> List[Dict]:
        """获取历史会话"""
        cursor = self.db.execute(
            "SELECT DISTINCT session_id FROM conversations WHERE session_id != ? ORDER BY timestamp DESC LIMIT ?",
            (self.session_id, limit)
        )
        return cursor.fetchall()
    
    def record_tool_call(self, tool_name: str, arguments: dict, result: str):
        """记录工具调用历史"""
        self.db.execute(
            "INSERT INTO tool_history (session_id, tool_name, arguments, result, timestamp) VALUES (?, ?, ?, ?, ?)",
            (self.session_id, tool_name, json.dumps(arguments), result, datetime.now().isoformat())
        )
        self.db.commit()
    
    def get_tool_history(self, limit: int = 20) -> List[Dict]:
        """获取工具调用历史"""
        cursor = self.db.execute(
            "SELECT tool_name, arguments, result, timestamp FROM tool_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, limit)
        )
        return [
            {"tool": row[0], "args": row[1], "result": row[2], "time": row[3]}
            for row in cursor.fetchall()
        ]
    
    def save_user_preferences(self, user_id: str, preferences: dict):
        """保存用户偏好"""
        self.db.execute(
            "INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at) VALUES (?, ?, ?)",
            (user_id, json.dumps(preferences), datetime.now().isoformat())
        )
        self.db.commit()
    
    def get_user_preferences(self, user_id: str) -> Optional[dict]:
        cursor = self.db.execute(
            "SELECT preferences FROM user_preferences WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None
    
    def close(self):
        self.db.close()
```

---

## 第36章：Agent 循环与终止条件

### 36.1 终止条件设计

```python
from abc import ABC, abstractmethod
from enum import Enum

class StopReason(str, Enum):
    TASK_COMPLETED = "task_completed"       # 任务完成
    MAX_ITERATIONS = "max_iterations"       # 达到最大轮次
    TOOL_ERROR = "tool_error"              # 工具连续失败
    LLM_REFUSED = "llm_refused"            # 模型拒绝继续
    HUMAN_INTERVENTION = "human_needed"     # 需要人工介入
    TIMEOUT = "timeout"                     # 超时
    LOOP_DETECTED = "loop_detected"        # 检测到死循环

class StopChecker:
    """Agent 终止条件检查"""
    
    def __init__(self):
        self.max_iterations = 10
        self.max_consecutive_errors = 3
        self.max_time_seconds = 300
        self.consecutive_errors = 0
        self.start_time = None
        self.previous_responses = []  # 用于检测循环
    
    def check(self, iteration: int, response: str, tool_results: list) -> tuple[bool, Optional[StopReason]]:
        """检查是否应该终止"""
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        # 1. 达到最大迭代次数
        if iteration >= self.max_iterations:
            return True, StopReason.MAX_ITERATIONS
        
        # 2. 超时
        if time.time() - self.start_time > self.max_time_seconds:
            return True, StopReason.TIMEOUT
        
        # 3. 工具连续失败
        if tool_results:
            all_failed = all("error" in str(r).lower() for r in tool_results)
            if all_failed:
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_consecutive_errors:
                    return True, StopReason.TOOL_ERROR
            else:
                self.consecutive_errors = 0
        
        # 4. 检测死循环（相同输出重复出现）
        normalized = response.strip()[:200]  # 取前200字符比较
        self.previous_responses.append(normalized)
        if len(self.previous_responses) > 3:
            self.previous_responses.pop(0)
            if len(set(self.previous_responses)) == 1:
                return True, StopReason.LOOP_DETECTED
        
        return False, None
```

---

# 第六部分：Agent 框架实战（实战篇）

## 第37章：LangChain Agent 实战

### 37.1 LangChain 简介

LangChain 是目前最流行的 LLM 应用开发框架。它提供了一整套工具来构建 Agent：
- **Chains**：将多个 LLM 调用和工具调用串联
- **Agents**：让 LLM 自主决定使用哪些工具
- **Tools**：预定义的工具接口
- **Memory**：对话记忆管理

### 37.2 LangChain Agent 示例

```python
# 安装：pip install langchain langchain-openai

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# 定义工具（使用 @tool 装饰器）
@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询。参数 sql 是要执行的 SELECT 语句。"""
    # 模拟数据库查询
    if "orders" in sql.lower():
        return "查询结果: 共1567条订单，平均金额¥234.50，今日新增23条"
    elif "users" in sql.lower():
        return "查询结果: 共8923名用户，今日活跃342人"
    return "查询结果: 无匹配数据"

@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool  
def calculate(expression: str) -> str:
    """执行数学计算。参数 expression 是数学表达式，如 '2+3*4'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# 创建 Agent
def create_langchain_agent():
    """创建一个 LangChain Agent"""
    
    # 1. LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key="your-key",
    )
    
    # 2. 工具列表
    tools = [query_database, get_current_time, calculate]
    
    # 3. Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个数据分析助手。你可以：
- 查询数据库
- 获取当前时间
- 执行数学计算

请用中文回答。如果信息不足，明确说明。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 4. 记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    
    # 5. 创建 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # 6. 创建 Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 打印详细日志
        max_iterations=5,
        handle_parsing_errors=True,
    )
    
    return agent_executor

# 使用
# agent = create_langchain_agent()
# result = agent.invoke({"input": "帮我查一下今天的订单总数，然后计算如果增长10%是多少"})
# print(result["output"])
```

---

## 第38章：AutoGen 多 Agent 实战

### 38.1 AutoGen 简介

AutoGen 是微软开源的多 Agent 对话框架。它允许多个 Agent 互相交流、协作完成任务。

```python
# 安装：pip install pyautogen

import autogen

# 配置 LLM
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": "your-api-key",
    }
]

# 创建用户代理（代表人类用户）
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="NEVER",  # 全自动模式；TERMINATE 表示关键步骤需要人工确认
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # 如果不想用Docker执行代码
    },
    system_message="你是用户代表，负责提出需求和确认结果。",
)

# 创建规划 Agent
planner = autogen.AssistantAgent(
    name="规划者",
    llm_config={"config_list": config_list},
    system_message="""你是一个任务规划专家。
    1. 分析用户需求
    2. 将任务分解为子任务
    3. 分配给合适的专家执行
    4. 汇总结果""",
)

# 创建数据分析 Agent
data_analyst = autogen.AssistantAgent(
    name="数据分析师",
    llm_config={"config_list": config_list},
    system_message="""你是一个数据分析专家。
    你可以：
    - 编写 Python 代码分析数据
    - 生成统计报告
    - 发现数据中的异常""",
)

# 创建运维排查 Agent
ops_agent = autogen.AssistantAgent(
    name="运维专家",
    llm_config={"config_list": config_list},
    system_message="""你是一个运维排查专家。
    你可以：
    - 分析告警信息
    - 给出排查步骤
    - 建议修复方案""",
)

# 启动多 Agent 协作
def run_autogen_task(task: str):
    """启动 AutoGen 多 Agent 协作"""
    
    # 方式1: 顺序对话
    groupchat = autogen.GroupChat(
        agents=[user_proxy, planner, data_analyst, ops_agent],
        messages=[],
        max_round=15,
        speaker_selection_method="auto",  # 自动选择下一个说话者
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list},
    )
    
    # 启动对话
    user_proxy.initiate_chat(
        manager,
        message=task,
    )

# 使用示例
# run_autogen_task("分析当前系统CPU告警原因，并给出解决方案")
```

---

## 第39章：CrewAI 多 Agent 协作

### 39.1 CrewAI 简介

CrewAI 是一个更简洁的多 Agent 框架，强调"角色扮演"模式：

```python
# 安装：pip install crewai

from crewai import Agent, Task, Crew, Process

# 创建 Agent
alert_receiver = Agent(
    role="告警接收员",
    goal="接收和理解告警信息，提取关键字段",
    backstory="你是一个经验丰富的运维监控员，擅长快速识别告警的严重程度和影响范围。",
    verbose=True,
    allow_delegation=True,
)

analyst = Agent(
    role="告警分析师",
    goal="深入分析告警根因，给出排查建议",
    backstory="你是一个资深系统分析师，有10年的故障排查经验。你擅长从日志、监控数据中找到问题根因。",
    verbose=True,
    allow_delegation=False,
)

reporter = Agent(
    role="报告生成员",
    goal="将分析结果整理成结构化的排查报告",
    backstory="你是一个技术写作专家，擅长将技术分析转化为清晰易读的报告。",
    verbose=True,
    allow_delegation=False,
)

# 创建任务
receive_task = Task(
    description="""
    接收以下告警信息并提取关键字段：
    【P1告警】订单服务数据库CPU使用率达到95%，当前QPS 2000，响应时间5s
    
    提取：告警级别、影响服务、关键指标、影响范围
    """,
    expected_output="结构化的告警信息JSON",
    agent=alert_receiver,
)

analyze_task = Task(
    description="""
    基于告警接收员提取的信息，深入分析可能的根因。
    考虑：慢SQL、连接池耗尽、锁等待、突发流量等
    给出排查步骤（按优先级排列）
    """,
    expected_output="详细的根因分析和排查建议",
    agent=analyst,
)

report_task = Task(
    description="""
    将分析结果整理成Markdown格式的排查报告。
    包含：告警摘要、根因分析、排查步骤、建议措施
    """,
    expected_output="Markdown格式的完整排查报告",
    agent=reporter,
)

# 创建 Crew
crew = Crew(
    agents=[alert_receiver, analyst, reporter],
    tasks=[receive_task, analyze_task, report_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

# 运行
# result = crew.kickoff()
# print(result)
```

---

## 第40章：从零手写 Agent 框架

### 40.1 为什么需要自己写框架

现成框架（LangChain、AutoGen、CrewAI）功能强大但可能过于复杂。对于初学者，自己动手实现一个简单框架是最好的学习方式。你能深入理解每个环节，也更容易按需定制。

### 40.2 迷你 Agent 框架

```python
"""
从零实现一个轻量级的 Agent 框架
"""
import json
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ============================================================
# 日志系统
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MiniAgent")

# ============================================================
# 消息模型
# ============================================================
@dataclass
class Message:
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)

# ============================================================
# 工具定义
# ============================================================
class Tool:
    def __init__(self, name: str, description: str, func: Callable, parameters: dict):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
    
    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def execute(self, **kwargs) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            logger.error(f"工具 {self.name} 执行失败: {e}")
            return json.dumps({"error": str(e)})

# ============================================================
# 记忆系统
# ============================================================
class Memory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.knowledge: Dict[str, Any] = {}
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            # 保留 system + 最近的
            system_msgs = [m for m in self.messages if m["role"] == "system"]
            other_msgs = [m for m in self.messages if m["role"] != "system"]
            self.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]
    
    def get_messages(self) -> List[Dict]:
        return self.messages
    
    def remember(self, key: str, value: Any):
        self.knowledge[key] = value
    
    def recall(self, key: str) -> Optional[Any]:
        return self.knowledge.get(key)

# ============================================================
# 核心 Agent
# ============================================================
class MiniAgent:
    """轻量级 Agent 框架核心"""
    
    def __init__(
        self,
        name: str,
        llm_client,  # OpenAI 客户端
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        tools: List[Tool] = None,
        max_iterations: int = 10,
    ):
        self.name = name
        self.llm = llm_client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        
        self.memory = Memory()
        if system_prompt:
            self.memory.add("system", system_prompt)
    
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def run(self, user_input: str) -> Dict:
        """运行 Agent"""
        logger.info(f"[{self.name}] 收到任务: {user_input[:100]}")
        
        self.memory.add("user", user_input)
        tool_schemas = [t.to_schema() for t in self.tools.values()]
        
        for i in range(self.max_iterations):
            logger.info(f"[{self.name}] 第 {i+1}/{self.max_iterations} 轮")
            
            try:
                # 调用 LLM
                if tool_schemas:
                    response = self.llm.chat.completions.create(
                        model=self.model,
                        messages=self.memory.get_messages(),
                        tools=tool_schemas,
                        tool_choice="auto",
                    )
                else:
                    response = self.llm.chat.completions.create(
                        model=self.model,
                        messages=self.memory.get_messages(),
                    )
                
                choice = response.choices[0]
                
                # 模型直接回答
                if choice.finish_reason == "stop":
                    answer = choice.message.content
                    self.memory.add("assistant", answer)
                    logger.info(f"[{self.name}] 任务完成")
                    return {
                        "success": True,
                        "answer": answer,
                        "iterations": i + 1,
                        "tool_calls_made": len(self.memory.knowledge),
                    }
                
                # 模型调用工具
                elif choice.finish_reason == "tool_calls":
                    self.memory.add("assistant", str(choice.message.tool_calls))
                    
                    for tc in choice.message.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
                        
                        logger.info(f"[{self.name}] 调用工具: {tool_name}({tool_args})")
                        
                        tool = self.tools.get(tool_name)
                        if tool:
                            result = tool.execute(**tool_args)
                        else:
                            result = json.dumps({"error": f"工具 {tool_name} 不存在"})
                        
                        self.memory.add("tool", result)
                        self.memory.remember(f"tool_{tool_name}_{i}", result)
                
                else:
                    logger.warning(f"[{self.name}] 未知 finish_reason: {choice.finish_reason}")
                    
            except Exception as e:
                logger.error(f"[{self.name}] 第 {i+1} 轮出错: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "iterations": i + 1,
                }
        
        logger.warning(f"[{self.name}] 达到最大迭代次数")
        return {
            "success": False,
            "error": "达到最大迭代次数",
            "iterations": self.max_iterations,
        }

# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    from openai import OpenAI
    
    client = OpenAI(api_key="your-key")
    
    # 创建工具
    def search_knowledge(query: str) -> str:
        kb = {
            "cpu高": "CPU过高的常见原因：1. 死循环或密集计算 2. GC频繁 3. 并发过高",
            "数据库慢": "数据库慢查询处理方法：1. EXPLAIN分析 2. 检查索引 3. 优化SQL",
        }
        for key, value in kb.items():
            if key in query.lower():
                return value
        return "未找到相关知识"
    
    tools = [
        Tool(
            name="search_knowledge",
            description="搜索运维知识库",
            func=search_knowledge,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            }
        )
    ]
    
    agent = MiniAgent(
        name="运维助手",
        llm_client=client,
        system_prompt="你是运维排错助手。先查知识库，再给出建议。用中文回答。",
        tools=tools,
    )
    
    result = agent.run("数据库突然变慢了怎么办")
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

# 第七部分：多 Agent 系统（架构篇）

## 第41章：多 Agent 架构模式

### 41.1 四种经典模式

```
1. 顺序模式（Sequential）
   Agent A → Agent B → Agent C → 输出
   适合: 流水线式任务

2. 层级模式（Hierarchical）
          Orchestrator
         /     |     \
     Agent1  Agent2  Agent3
   适合: 复杂任务分解

3. 辩论模式（Debate）
   Agent A ⇄ Agent B (辩论) → 综合
   适合: 需要多视角分析

4. 市场模式（Market）
   Agent 们竞标任务 → 最优者执行
   适合: 动态任务分配
```

### 41.2 层级模式实现

```python
class OrchestratorAgent:
    """编排 Agent：负责分解任务和调度子 Agent"""
    
    def __init__(self, llm_client, sub_agents: Dict[str, MiniAgent]):
        self.llm = llm_client
        self.sub_agents = sub_agents
    
    def decompose_task(self, task: str) -> List[Dict]:
        """将任务分解为子任务"""
        agents_desc = "\n".join([
            f"- {name}: {agent.system_prompt[:100]}"
            for name, agent in self.sub_agents.items()
        ])
        
        prompt = f"""将以下任务分解为子任务并分配给合适的Agent。

可用Agent:
{agents_desc}

任务: {task}

输出JSON:
[
  {{"agent": "Agent名称", "subtask": "子任务描述", "priority": 1}}
]
"""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    
    def run(self, task: str) -> Dict:
        """执行编排"""
        # 1. 分解任务
        subtasks = self.decompose_task(task)
        logger.info(f"任务分解为 {len(subtasks)} 个子任务")
        
        # 2. 分配给各 Agent 执行
        results = []
        for subtask in subtasks:
            agent_name = subtask["agent"]
            agent = self.sub_agents.get(agent_name)
            
            if agent:
                logger.info(f"→ {agent_name} 执行: {subtask['subtask'][:80]}")
                result = agent.run(subtask["subtask"])
                results.append({
                    "agent": agent_name,
                    "subtask": subtask["subtask"],
                    "result": result,
                })
            else:
                results.append({
                    "agent": agent_name,
                    "error": f"Agent '{agent_name}' 不存在",
                })
        
        # 3. 汇总结果
        summary_prompt = f"""
任务: {task}

各Agent执行结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

请汇总以上结果，给出最终的综合分析报告。
"""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=1000,
        )
        
        return {
            "subtasks": subtasks,
            "results": results,
            "summary": response.choices[0].message.content,
        }
```

---

## 第42章：Agent 间通信协议

### 42.1 消息格式标准

```python
@dataclass
class AgentMessage:
    """Agent 间通信的标准消息格式"""
    sender: str           # 发送者 ID
    receiver: str         # 接收者 ID（"broadcast" 表示广播）
    msg_type: str         # 消息类型: "request", "response", "notification", "error"
    content: Any          # 消息内容
    correlation_id: str   # 关联ID，用于请求-响应匹配
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

class MessageBus:
    """Agent 间消息总线"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[AgentMessage] = []
    
    def subscribe(self, agent_id: str, callback: Callable):
        """Agent 订阅消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    def publish(self, message: AgentMessage):
        """发布消息"""
        self.message_history.append(message)
        
        if message.receiver == "broadcast":
            # 广播给所有 Agent（除了发送者）
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender:
                    for cb in callbacks:
                        cb(message)
        elif message.receiver in self.subscribers:
            # 单播
            for cb in self.subscribers[message.receiver]:
                cb(message)
    
    def send_and_wait(self, message: AgentMessage, timeout: float = 30) -> Optional[AgentMessage]:
        """发送请求并等待响应"""
        import threading
        
        response_event = threading.Event()
        response_msg = [None]
        
        def response_callback(msg: AgentMessage):
            if msg.correlation_id == message.correlation_id:
                response_msg[0] = msg
                response_event.set()
        
        # 临时订阅响应
        original_sender = message.sender
        self.subscribe(original_sender, response_callback)
        
        self.publish(message)
        
        if response_event.wait(timeout):
            return response_msg[0]
        return None
```

---

## 第43章：Orchestrator 编排模式

### 43.1 工作流引擎

```python
class WorkflowEngine:
    """Agent 工作流编排引擎"""
    
    def __init__(self):
        self.workflows: Dict[str, Dict] = {}
    
    def define_workflow(self, name: str, steps: List[Dict]):
        """
        定义工作流
        steps: [
            {"id": "step1", "agent": "analyzer", "task": "...", "depends_on": []},
            {"id": "step2", "agent": "fixer", "task": "...", "depends_on": ["step1"]},
        ]
        """
        self.workflows[name] = {"steps": steps}
    
    def run_workflow(self, name: str, agents: Dict[str, MiniAgent], input_data: Dict) -> Dict:
        """运行工作流"""
        if name not in self.workflows:
            return {"error": f"工作流 '{name}' 不存在"}
        
        workflow = self.workflows[name]
        results = {}
        
        for step in workflow["steps"]:
            # 检查依赖是否都完成
            deps_met = all(
                dep in results and results[dep].get("success")
                for dep in step.get("depends_on", [])
            )
            
            if not deps_met:
                results[step["id"]] = {"error": "依赖步骤未完成"}
                continue
            
            # 构建任务（可以引用之前步骤的结果）
            task = step["task"]
            for dep_id in step.get("depends_on", []):
                dep_result = results.get(dep_id, {})
                task = task.replace(f"{{{dep_id}}}", str(dep_result.get("answer", "")))
            
            # 执行
            agent = agents.get(step["agent"])
            if agent:
                results[step["id"]] = agent.run(task)
            else:
                results[step["id"]] = {"error": f"Agent '{step['agent']}' 不存在"}
        
        return results

# 使用示例
engine = WorkflowEngine()
engine.define_workflow("alert_handling", [
    {
        "id": "analyze",
        "agent": "analyzer",
        "task": "分析这个告警: {input_data}",
        "depends_on": [],
    },
    {
        "id": "suggest",
        "agent": "advisor",
        "task": "基于分析结果 {analyze} 给出修复建议",
        "depends_on": ["analyze"],
    },
    {
        "id": "report",
        "agent": "reporter",
        "task": "基于 {analyze} 和 {suggest} 生成排查报告",
        "depends_on": ["analyze", "suggest"],
    },
])
```

---

# 第八部分：评估、监控与运维（工程篇）

## 第44章：Agent 评估体系

### 44.1 评估维度

| 维度 | 说明 | 评估方法 |
|------|------|----------|
| 任务完成率 | 是否完成了用户指定的任务 | 人工评估 / 规则判断 |
| 工具调用准确率 | 是否选了正确的工具和参数 | 自动比对 |
| 回答准确性 | 回答是否事实正确 | 人工评估 / RAGAS |
| 效率 | 完成任务的 token 消耗和轮次 | 自动统计 |
| 安全性 | 是否产生有害或不安全的输出 | 安全评估模型 |

### 44.2 评估框架实现

```python
class AgentEvaluator:
    """Agent 评估框架"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.test_cases = []
    
    def add_test_case(self, test_case: Dict):
        """
        test_case = {
            "id": "tc001",
            "input": "用户的输入",
            "expected_tools": ["query_database"],  # 期望调用的工具
            "expected_keywords": ["数据库", "索引"],  # 回答应包含的关键词
            "severity": "critical",  # 重要程度
        }
        """
        self.test_cases.append(test_case)
    
    def evaluate(self, agent, test_case: Dict) -> Dict:
        """评估单个测试用例"""
        result = agent.run(test_case["input"])
        
        scores = {}
        
        # 1. 任务完成评估
        if result.get("success"):
            scores["completion"] = 1.0
        else:
            scores["completion"] = 0.0
        
        # 2. 关键词检查
        if "expected_keywords" in test_case:
            answer = result.get("answer", "")
            matched = sum(1 for kw in test_case["expected_keywords"] if kw in answer)
            scores["keyword_match"] = matched / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 1.0
        
        # 3. 工具调用检查
        if "expected_tools" in test_case:
            # 从结果中提取实际调用的工具
            # （取决于你的 agent 如何记录工具调用）
            scores["tool_accuracy"] = 0.5  # 简化实现
        
        # 4. LLM 作为评判者
        judge_prompt = f"""
评估以下 Agent 的回答质量。

用户问题: {test_case['input']}
Agent 回答: {result.get('answer', '')}

请从以下维度打分(1-5):
1. 准确性: 回答是否事实正确
2. 完整性: 是否涵盖了所有必要信息
3. 清晰度: 是否易于理解
4. 有用性: 是否对用户有实际帮助

输出JSON: {{"准确性": 4, "完整性": 3, "清晰度": 5, "有用性": 4}}
"""
        try:
            judge_response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            llm_scores = json.loads(judge_response.choices[0].message.content)
            scores.update({f"llm_{k}": v / 5.0 for k, v in llm_scores.items()})
        except:
            pass
        
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "test_case_id": test_case["id"],
            "scores": scores,
            "result": result,
        }
    
    def run_all(self, agent) -> List[Dict]:
        """运行所有测试用例"""
        all_results = []
        for tc in self.test_cases:
            eval_result = self.evaluate(agent, tc)
            all_results.append(eval_result)
            print(f"  [{tc['id']}] {tc['input'][:50]}... → 得分: {eval_result['scores']['overall']:.2f}")
        
        avg_score = sum(r["scores"]["overall"] for r in all_results) / len(all_results) if all_results else 0
        print(f"\n平均得分: {avg_score:.2f}")
        return all_results
```

---

## 第45章：LangSmith / LangFuse 追踪

### 45.1 为什么需要追踪

在生产环境中，你不能靠 print 调试。追踪系统让你：
- 看到每次 LLM 调用的输入输出
- 追踪工具调用的链路
- 分析延迟和成本
- 发现和复现问题

### 45.2 自建简单追踪

```python
import time
import uuid
from dataclasses import dataclass, field

@dataclass
class TraceSpan:
    """追踪 span"""
    span_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    input_data: Any = None
    output_data: Any = None
    metadata: Dict = field(default_factory=dict)
    error: str = None
    children: List["TraceSpan"] = field(default_factory=list)
    
    def finish(self, output: Any = None, error: str = None):
        self.end_time = time.time()
        self.output_data = output
        self.error = error
    
    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "input": str(self.input_data)[:200] if self.input_data else None,
            "output": str(self.output_data)[:200] if self.output_data else None,
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }

class Tracer:
    """简单的追踪器"""
    
    def __init__(self):
        self.traces: List[TraceSpan] = []
        self._stack: List[TraceSpan] = []
    
    def start_span(self, name: str, input_data: Any = None) -> TraceSpan:
        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            input_data=input_data,
        )
        
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.traces.append(span)
        
        self._stack.append(span)
        return span
    
    def end_span(self, output: Any = None, error: str = None):
        if self._stack:
            span = self._stack.pop()
            span.finish(output, error)
    
    def get_traces(self) -> List[Dict]:
        return [t.to_dict() for t in self.traces]
    
    def print_tree(self):
        """打印追踪树"""
        def _print(span, indent=0):
            status = "❌" if span.error else "✅"
            print(f"{'  ' * indent}{status} {span.name} ({span.duration_ms:.0f}ms)")
            for child in span.children:
                _print(child, indent + 1)
        
        for trace in self.traces:
            _print(trace)

# 使用示例
# tracer = Tracer()
# span = tracer.start_span("agent_run", "分析告警")
# tool_span = tracer.start_span("tool_call", "query_database")
# tracer.end_span("查询结果")
# tracer.end_span("最终回答")
# tracer.print_tree()
```

---

## 第46章：成本分析与优化

### 46.1 成本追踪

```python
class CostTracker:
    """Agent 成本追踪器"""
    
    # 各模型价格（每百万token）
    PRICES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deepseek-chat": {"input": 0.14, "output": 0.28},  # $换算
    }
    
    def __init__(self):
        self.records = []  # 每次调用的用量记录
        self.total_input = 0
        self.total_output = 0
    
    def record(self, model: str, input_tokens: int, output_tokens: int):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.records.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": self._calculate_cost(model, input_tokens, output_tokens),
        })
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self.PRICES.get(model, {"input": 0, "output": 0})
        return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    
    def get_summary(self) -> Dict:
        total_cost = sum(r["cost"] for r in self.records)
        return {
            "total_calls": len(self.records),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_cost_usd": round(total_cost, 4),
            "per_call_avg_cost": round(total_cost / len(self.records), 6) if self.records else 0,
        }
```

### 46.2 成本优化策略

1. **缓存相同查询**：相同问题不重复调用 LLM
2. **选择合适模型**：简单任务用 gpt-4o-mini 而非 gpt-4o
3. **精简 Prompt**：去掉冗余的 system prompt 内容
4. **限制输出长度**：设置合理的 max_tokens
5. **批量处理**：合并多个小请求

```python
class CacheManager:
    """简单的 LLM 响应缓存"""
    
    def __init__(self, max_size: int = 1000):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def _make_key(self, messages: list, model: str) -> str:
        import hashlib
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: list, model: str) -> Optional[str]:
        key = self._make_key(messages, model)
        return self.cache.get(key)
    
    def set(self, messages: list, model: str, response: str):
        key = self._make_key(messages, model)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = response
```

---

## 第47章：安全与护栏

### 47.1 安全防护层

```python
class SafetyGuard:
    """Agent 安全护栏"""
    
    # 禁止的模式（简化版）
    FORBIDDEN_PATTERNS = [
        "DROP TABLE",
        "DELETE FROM",
        "rm -rf",
        "sudo ",
        "shutdown",
        "reboot",
    ]
    
    # 敏感信息模式
    SENSITIVE_PATTERNS = [
        r'sk-[a-zA-Z0-9]{20,}',        # OpenAI API Key
        r'AKIA[0-9A-Z]{16}',           # AWS Access Key
        r'password\s*[=:]\s*\S+',      # 密码
    ]
    
    @classmethod
    def check_input(cls, user_input: str) -> tuple[bool, str]:
        """检查用户输入"""
        # SQL 注入风险
        dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "EXEC", "UNION"]
        for kw in dangerous_keywords:
            if kw.upper() in user_input.upper():
                return False, f"输入包含潜在危险关键词: {kw}"
        
        return True, "OK"
    
    @classmethod
    def check_output(cls, output: str) -> tuple[bool, str]:
        """检查 Agent 输出"""
        import re
        
        # 检查敏感信息泄露
        for pattern in cls.SENSITIVE_PATTERNS:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return False, f"输出包含敏感信息: {match.group(0)[:20]}..."
        
        # 检查禁止的操作
        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.upper() in output.upper():
                return False, f"输出包含禁止操作: {pattern}"
        
        return True, "OK"
    
    @classmethod
    def check_tool_call(cls, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """检查工具调用是否安全"""
        # 危险工具需要额外确认
        DANGEROUS_TOOLS = ["execute_command", "delete_data", "modify_config"]
        
        if tool_name in DANGEROUS_TOOLS:
            return False, f"工具 '{tool_name}' 需要人工确认后才能执行"
        
        # 检查 SQL 工具的参数
        if "sql" in tool_name.lower():
            sql = arguments.get("sql", arguments.get("query", ""))
            dangerous_sql = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
            for kw in dangerous_sql:
                if kw.upper() in sql.upper():
                    return False, f"SQL工具不允许执行 {kw} 操作"
        
        return True, "OK"
```

---

# 第九部分：生产部署（落地篇）

## 第48章：FastAPI 部署 Agent 服务

### 48.1 完整的 Agent API 服务

```python
"""
Agent API 服务 - 使用 FastAPI 部署
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime

# ============================================================
# 数据模型
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话ID，不传则自动创建")
    stream: bool = Field(default=False, description="是否流式输出")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls_made: int
    tokens_used: int
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    last_active: str

# ============================================================
# 应用初始化
# ============================================================
app = FastAPI(
    title="Agent API Service",
    description="智能体问答服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 会话管理
# ============================================================
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "agent_state": {},
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })

session_manager = SessionManager()

# ============================================================
# API 路由
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Agent 对话接口"""
    # 创建或获取会话
    session_id = request.session_id or session_manager.create_session()
    
    if not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 执行 Agent
    # 这里替换为你的实际 Agent 逻辑
    result = {
        "answer": f"收到你的消息: {request.message}。这是Agent的回答。",
        "tool_calls_made": 2,
        "tokens_used": 500,
    }
    
    # 记录对话
    session_manager.add_message(session_id, "user", request.message)
    session_manager.add_message(session_id, "assistant", result["answer"])
    
    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        tool_calls_made=result["tool_calls_made"],
        tokens_used=result["tokens_used"],
        timestamp=datetime.now().isoformat(),
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口"""
    session_id = request.session_id or session_manager.create_session()
    
    async def generate():
        # 模拟流式输出
        answer = f"这是对'{request.message}'的流式回答。流式输出让用户体验更好。"
        
        # 先发送元数据
        yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        
        # 逐字发送内容
        for i, char in enumerate(answer):
            yield f"data: {json.dumps({'type': 'content', 'text': char}, ensure_ascii=False)}\n\n"
            import asyncio
            await asyncio.sleep(0.05)  # 模拟打字效果
        
        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done', 'tokens': len(answer)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取会话信息"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=len(session["messages"]),
        last_active=session["messages"][-1]["timestamp"] if session["messages"] else session["created_at"],
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "timestamp": datetime.now().isoformat(),
    }

# 运行命令:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 第49章：WebSocket 实时对话

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio

class ConnectionManager:
    """WebSocket 连接管理"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

ws_manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 实时对话"""
    await ws_manager.connect(session_id, websocket)
    
    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            
            if not user_message:
                continue
            
            # 发送"正在思考"状态
            await ws_manager.send_message(session_id, {
                "type": "status",
                "status": "thinking",
                "message": "正在分析...",
            })
            
            # 模拟 Agent 处理
            # 实际项目中这里调用你的 Agent
            import asyncio
            await asyncio.sleep(1)  # 模拟处理时间
            
            # 发送回答
            await ws_manager.send_message(session_id, {
                "type": "answer",
                "content": f"收到你的消息: {user_message}",
                "timestamp": datetime.now().isoformat(),
            })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
        print(f"会话 {session_id} 断开连接")
```

---

## 第50章：Docker 容器化部署

### 50.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 50.2 docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/agent
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: agent
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 50.3 requirements.txt

```text
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
openai==1.51.0
pydantic==2.9.0
chromadb==0.5.5
sentence-transformers==3.1.0
langchain==0.3.0
langchain-openai==0.2.0
jinja2==3.1.4
redis==5.1.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.35
python-dotenv==1.0.1
```

---

## 第51章：可观测性搭建

### 51.1 结构化日志

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """结构化日志"""
    
    def __init__(self, name: str = "agent"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(f"logs/agent_{datetime.now():%Y%m%d}.log")
        fh.setLevel(logging.INFO)
        
        # JSON 格式
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if hasattr(record, "extra_data"):
                    log_entry.update(record.extra_data)
                return json.dumps(log_entry, ensure_ascii=False)
        
        fh.setFormatter(JsonFormatter())
        self.logger.addHandler(fh)
    
    def log_llm_call(self, model: str, input_tokens: int, output_tokens: int, duration_ms: float):
        extra = logging.LogRecord(
            self.logger.name, logging.INFO, "", 0,
            f"LLM call: {model}", (), None
        )
        extra.extra_data = {
            "event": "llm_call",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": round(duration_ms, 2),
        }
        self.logger.handle(extra)
    
    def log_tool_call(self, tool_name: str, args: dict, result: str, duration_ms: float):
        extra = logging.LogRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Tool call: {tool_name}", (), None
        )
        extra.extra_data = {
            "event": "tool_call",
            "tool": tool_name,
            "args": args,
            "result_preview": str(result)[:200],
            "duration_ms": round(duration_ms, 2),
        }
        self.logger.handle(extra)
    
    def log_error(self, error_type: str, error_msg: str, context: dict = None):
        extra = logging.LogRecord(
            self.logger.name, logging.ERROR, "", 0,
            f"Error: {error_msg[:100]}", (), None
        )
        extra.extra_data = {
            "event": "error",
            "error_type": error_type,
            "error_message": error_msg,
            "context": context or {},
        }
        self.logger.handle(extra)

# 使用
# slog = StructuredLogger()
# slog.log_llm_call("gpt-4o-mini", 500, 200, 1500.5)
# slog.log_tool_call("query_database", {"sql": "SELECT..."}, "返回3行", 50.2)
```

---

# 第十部分：综合实战项目（项目篇）

## 第52章：告警排查 Agent 完整项目

### 52.1 项目结构

```
alert_agent/
├── main.py                 # FastAPI 入口
├── agent.py                # Agent 核心逻辑
├── tools.py                # 工具定义和实现
├── prompts.py              # Prompt 模板
├── memory.py               # 记忆系统
├── config.py               # 配置管理
├── models.py               # Pydantic 数据模型
├── requirements.txt
└── docker-compose.yml
```

### 52.2 核心代码

```python
# alert_agent/config.py
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AlertAgentConfig:
    """告警排查 Agent 配置"""
    # LLM 配置
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    llm_base_url: str = "https://api.deepseek.com/v1"
    
    # Agent 配置
    max_iterations: int = 10
    max_tool_errors: int = 3
    
    # 知识库配置
    knowledge_base_path: str = "./knowledge"
    chroma_db_path: str = "./chroma_db"
    
    # 告警分析配置
    alert_severity_levels: List[str] = field(default_factory=lambda: ["P0", "P1", "P2", "P3"])
    max_analysis_time_seconds: int = 300

# alert_agent/tools.py
from typing import Dict, Any
import json

class AlertTools:
    """告警排查专用工具集"""
    
    def __init__(self):
        pass
    
    def query_monitoring(self, metric: str, service: str, time_range: int = 30) -> str:
        """查询监控指标"""
        # 模拟监控数据
        mock_data = {
            "order-service": {
                "cpu_usage": "85%",
                "memory_usage": "72%",
                "qps": "2500",
                "error_rate": "2.3%",
                "latency_p99": "3500ms",
            },
            "payment-service": {
                "cpu_usage": "45%",
                "memory_usage": "60%",
                "qps": "800",
                "error_rate": "0.1%",
                "latency_p99": "200ms",
            },
        }
        
        service_data = mock_data.get(service, {})
        value = service_data.get(metric, "数据不可用")
        return f"{service} 的 {metric}: {value} (最近{time_range}分钟)"
    
    def query_slow_sql(self, hours: int = 1, limit: int = 10) -> str:
        """查询慢SQL"""
        mock_slow_sql = [
            {"sql": "SELECT * FROM orders WHERE status='pending' ORDER BY created_at",
             "duration_ms": 5200, "rows_examined": 500000},
            {"sql": "SELECT o.*, u.* FROM orders o JOIN users u ON o.user_id=u.id WHERE o.amount>1000",
             "duration_ms": 3800, "rows_examined": 200000},
        ]
        return json.dumps(mock_slow_sql, ensure_ascii=False, indent=2)
    
    def query_error_logs(self, service: str, keyword: str = "ERROR", limit: int = 20) -> str:
        """查询错误日志"""
        mock_logs = [
            f"[{service}] ConnectionTimeout: 连接数据库超时 (30s)",
            f"[{service}] OutOfMemoryError: Java heap space",
            f"[{service}] TooManyConnections: 连接池耗尽 (200/200)",
        ]
        return "\n".join(mock_logs[:limit])
    
    def check_service_status(self, service: str) -> str:
        """检查服务状态"""
        statuses = {
            "order-service": {"status": "degraded", "uptime": "99.2%", "last_restart": "3天前"},
            "payment-service": {"status": "healthy", "uptime": "99.9%", "last_restart": "7天前"},
        }
        return json.dumps(statuses.get(service, {"status": "unknown"}), ensure_ascii=False)

# alert_agent/prompts.py
ALERT_AGENT_SYSTEM_PROMPT = """你是一个专业的运维告警排查助手。

## 你的工作流程
1. 理解告警内容，提取关键信息
2. 查询相关监控指标，了解当前系统状态
3. 根据症状，查询慢SQL、错误日志等
4. 综合分析，给出根因和排查建议

## 规则
- 先查数据再下结论，不臆测
- 对于每个发现，标注证据来源
- 不确定时标注"待进一步确认"
- 区分"已确认"和"推测"
- 输出使用 Markdown 格式

## 输出结构
每次分析包含：
1. 告警摘要
2. 关键指标快照
3. 根因分析（标注置信度）
4. 排查步骤（按优先级）
5. 是否需要升级
"""

# alert_agent/agent.py
from openai import OpenAI
import json
from typing import Dict, List

class AlertAnalysisAgent:
    """告警分析 Agent"""
    
    def __init__(self, config: AlertAgentConfig):
        self.config = config
        self.llm = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        self.tools = AlertTools()
        self.tool_schemas = self._build_tool_schemas()
    
    def _build_tool_schemas(self) -> List[Dict]:
        """构建工具 Schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_monitoring",
                    "description": "查询指定服务的监控指标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string", "enum": ["cpu_usage", "memory_usage", "qps", "error_rate", "latency_p99"], "description": "指标名"},
                            "service": {"type": "string", "description": "服务名"},
                            "time_range": {"type": "integer", "description": "查询时间范围（分钟）", "default": 30},
                        },
                        "required": ["metric", "service"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_slow_sql",
                    "description": "查询最近的慢SQL日志",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hours": {"type": "integer", "description": "最近几小时", "default": 1},
                            "limit": {"type": "integer", "description": "返回条数", "default": 10},
                        },
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_error_logs",
                    "description": "查询应用错误日志",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "服务名"},
                            "keyword": {"type": "string", "description": "搜索关键词", "default": "ERROR"},
                            "limit": {"type": "integer", "description": "返回条数", "default": 20},
                        },
                        "required": ["service"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_service_status",
                    "description": "检查服务运行状态",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "服务名"},
                        },
                        "required": ["service"],
                    }
                }
            }
        ]
    
    def analyze(self, alert_message: str, session_id: str = None) -> Dict:
        """分析告警"""
        
        messages = [
            {"role": "system", "content": ALERT_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下告警:\n\n{alert_message}"},
        ]
        
        tool_calls_made = []
        
        for i in range(self.config.max_iterations):
            response = self.llm.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
            )
            
            choice = response.choices[0]
            
            if choice.finish_reason == "stop":
                return {
                    "success": True,
                    "analysis": choice.message.content,
                    "tool_calls_made": tool_calls_made,
                    "iterations": i + 1,
                }
            
            elif choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                
                for tc in choice.message.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)
                    
                    # 执行工具
                    func = getattr(self.tools, func_name, None)
                    if func:
                        result = func(**func_args)
                    else:
                        result = f"工具 {func_name} 不存在"
                    
                    tool_calls_made.append({
                        "tool": func_name,
                        "args": func_args,
                        "result": str(result)[:200],
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
        
        return {
            "success": False,
            "error": "达到最大分析轮次",
            "tool_calls_made": tool_calls_made,
        }

# alert_agent/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="告警排查 Agent")

agent = AlertAnalysisAgent(AlertAgentConfig())

class AlertRequest(BaseModel):
    alert: str
    session_id: str = None

class AlertResponse(BaseModel):
    success: bool
    analysis: str = None
    tool_calls: list = None
    error: str = None

@app.post("/analyze", response_model=AlertResponse)
async def analyze_alert(request: AlertRequest):
    """分析告警"""
    result = agent.analyze(request.alert, request.session_id)
    
    if result["success"]:
        return AlertResponse(
            success=True,
            analysis=result["analysis"],
            tool_calls=result["tool_calls_made"],
        )
    else:
        return AlertResponse(
            success=False,
            error=result.get("error", "分析失败"),
        )

@app.get("/health")
async def health():
    return {"status": "ok"}

# 运行: uvicorn main:app --reload
```

---

## 第53章：数据质量检测 Agent

```python
"""
数据质量检测 Agent
"""
import json
from typing import List, Dict, Any
from openai import OpenAI

class DataQualityAgent:
    """数据质量自动检测 Agent"""
    
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
    
    def detect_anomalies(self, data: List[Dict], rules: List[Dict] = None) -> Dict:
        """检测数据异常"""
        
        # 1. 先做规则检测
        rule_violations = self._rule_based_check(data, rules or [])
        
        # 2. 再做 LLM 智能分析
        llm_analysis = self._llm_analysis(data, rule_violations)
        
        return {
            "total_records": len(data),
            "rule_violations": rule_violations,
            "llm_analysis": llm_analysis,
        }
    
    def _rule_based_check(self, data: List[Dict], rules: List[Dict]) -> List[Dict]:
        """基于规则的检测"""
        violations = []
        
        for i, record in enumerate(data):
            for rule in rules:
                field = rule.get("field")
                check = rule.get("check")
                
                if field not in record:
                    continue
                
                value = record[field]
                
                # 空值检查
                if check == "not_null" and (value is None or value == ""):
                    violations.append({
                        "row": i + 1,
                        "field": field,
                        "rule": "not_null",
                        "value": value,
                        "message": f"第{i+1}行 {field} 为空",
                    })
                
                # 范围检查
                elif check == "range":
                    min_val = rule.get("min")
                    max_val = rule.get("max")
                    if value is not None:
                        try:
                            num_val = float(value)
                            if (min_val is not None and num_val < min_val) or \
                               (max_val is not None and num_val > max_val):
                                violations.append({
                                    "row": i + 1,
                                    "field": field,
                                    "rule": f"range[{min_val}, {max_val}]",
                                    "value": value,
                                })
                        except (ValueError, TypeError):
                            pass
                
                # 正则检查
                elif check == "pattern":
                    import re
                    pattern = rule.get("pattern")
                    if pattern and not re.match(pattern, str(value)):
                        violations.append({
                            "row": i + 1,
                            "field": field,
                            "rule": f"pattern:{pattern}",
                            "value": value,
                        })
        
        return violations
    
    def _llm_analysis(self, data: List[Dict], rule_violations: List[Dict]) -> str:
        """LLM 智能分析"""
        
        # 取数据样本（太多会超 token）
        sample = data[:20]
        
        prompt = f"""
你是数据质量分析专家。请分析以下数据样本的质量问题。

## 规则检测结果
{json.dumps(rule_violations, ensure_ascii=False, indent=2)}

## 数据样本（前{len(sample)}行）
{json.dumps(sample, ensure_ascii=False, indent=2)}

请分析：
1. 数据整体质量评估
2. 除规则检测外，还有哪些潜在的异常模式
3. 可能的产生原因
4. 修复建议

输出Markdown格式。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content

# 使用示例
# agent = DataQualityAgent(client)
# data = [
#     {"id": "001", "amount": 100, "date": "2024-01-15"},
#     {"id": "002", "amount": None, "date": "2024-01-16"},  # 空值
#     {"id": "003", "amount": -50, "date": "2024-13-01"},   # 负值 + 无效日期
# ]
# rules = [
#     {"field": "amount", "check": "not_null"},
#     {"field": "amount", "check": "range", "min": 0},
# ]
# result = agent.detect_anomalies(data, rules)
```

---

## 第54章：智能知识库问答系统

```python
"""
完整的智能知识库问答系统
结合 RAG + Agent 的架构
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict
import json

class KnowledgeBaseQA:
    """知识库问答系统"""
    
    def __init__(self, api_key: str = None):
        self.llm = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # 向量数据库
        self.chroma = chromadb.PersistentClient(path="./kb_chroma")
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.collection = self.chroma.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.ef,
        )
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """批量添加文档
        documents: [{"content": "文档内容", "title": "标题", "category": "分类"}, ...]
        """
        texts = [doc["content"] for doc in documents]
        ids = [f"kb_{i}" for i in range(self.collection.count(), self.collection.count() + len(texts))]
        metadatas = [{"title": doc.get("title", ""), "category": doc.get("category", "")} for doc in documents]
        
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
        return len(texts)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索知识库"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            docs.append({
                "content": doc,
                "title": results["metadatas"][0][i].get("title", ""),
                "category": results["metadatas"][0][i].get("category", ""),
                "score": 1 - results["distances"][0][i],
            })
        return docs
    
    def ask(self, question: str, top_k: int = 5) -> Dict:
        """问答"""
        # 1. 检索相关知识
        relevant_docs = self.search(question, top_k)
        
        if not relevant_docs:
            return {
                "answer": "抱歉，知识库中没有找到相关信息。",
                "sources": [],
            }
        
        # 2. 构建上下文
        context = "\n\n---\n\n".join([
            f"[文档{i+1}: {doc['title']}]\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 3. 生成回答
        prompt = f"""你是一个基于知识库的问答助手。

## 知识库内容
{context}

## 用户问题
{question}

## 回答要求
1. 只基于知识库内容回答
2. 引用具体的文档编号
3. 如果信息不足，明确说明
4. 用Markdown格式组织回答"""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"title": doc["title"], "score": round(doc["score"], 4), "category": doc["category"]}
                for doc in relevant_docs
            ],
            "total_docs": self.collection.count(),
        }
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count(),
            "categories": list(set(
                m.get("category", "") for m in self.collection.get()["metadatas"]
            )),
        }

# 使用示例
# kb = KnowledgeBaseQA()
# kb.add_documents([
#     {"content": "MySQL慢查询优化：1. 使用EXPLAIN分析 2. 添加索引 3. 优化SQL", "title": "MySQL优化", "category": "数据库"},
#     {"content": "Redis缓存策略：LRU、LFU、TTL过期策略的选择和配置", "title": "Redis缓存", "category": "缓存"},
# ])
# result = kb.ask("数据库查询慢怎么办")
# print(result["answer"])
```

---

# 附录

## 附录A：常用命令速查

```bash
# 创建 Python 虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 安装核心依赖
pip install openai langchain chromadb sentence-transformers fastapi uvicorn pydantic jinja2

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Docker 部署
docker build -t agent-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=xxx agent-api
docker-compose up -d
```

## 附录B：学习资源推荐

### 必读论文
- **Attention Is All You Need** (2017)：Transformer 架构奠基论文
- **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)：ReAct Agent 范式
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2020)：RAG 原始论文

### 推荐框架文档
- LangChain 官方文档: https://python.langchain.com
- AutoGen 官方文档: https://microsoft.github.io/autogen
- CrewAI 官方文档: https://docs.crewai.com
- Chroma 官方文档: https://docs.trychroma.com

### 实践建议
1. **先跑通最小闭环**：LLM API 调用 → Prompt 调试 → 工具调用 → RAG → Agent
2. **用真实场景驱动**：找一个工作中的实际问题（如告警排查），用 Agent 去解决
3. **不要追求完美**：Agent 的幻觉和不确定性是常态，通过工程手段（规则、日志、评估）来应对
4. **成本意识**：从小模型（gpt-4o-mini / deepseek-chat）开始，确认效果后再考虑大模型
5. **持续迭代**：Agent 不是一个"写完就完了"的项目，需要根据实际效果不断调整 prompt、工具和流程

---

> **文档版本**: v2.0  
> **适用范围**: Agent 全栈开发从入门到实战  
> **更新日期**: 2025年
