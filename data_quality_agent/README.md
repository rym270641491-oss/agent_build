# 数据质量排查 Agent

每天早上收到校验平台的“源表与目标表数据量不一致”报告后，本 Agent 会参考
`data/user_experience/` 下的历史排查经验文档，按 ReAct 循环逐步推理，
输出一份结构化排查报告（严重等级、根因列表、验证手段、修复建议、已检查/未检查项）。

> 重要：本项目的观测面是「阿里云 Flink UI + 校验平台数据量」，不假设能直连数据库。
> 根因列表中的 `flink_ui_checks` 是主验证手段（Flink UI / 校验平台检查项），
> `validation_sql` 是可选辅助项，需提交给有库权限的数据工程师人工执行。

> 本项目的全部文件已整合在 `data_quality_agent/` 目录下，使用前先
> `cd data_quality_agent`。

## 技术选型

| 组件 | 选型 |
| --- | --- |
| 编排框架 | LangChain + LangGraph（自定义 ReAct 循环） |
| 大模型 | DeepSeek-V4-Flash（OpenAI 兼容接口） |
| 向量库 | Chroma（持久化到 `data/vector_store`） |
| 状态管理 | InMemorySaver（每次运行独立 thread_id） |
| 推理模式 | ReAct，最多 8 步、单请求 60 秒、总时长 180 秒（可配置）、每步最多 3 个假设 |

## 目录结构

```text
.
├── main.py                    # 命令行入口
├── config/settings.py         # 全局配置（模型、步数、超时、Embedding 等）
├── models/schemas.py          # Pydantic 数据模型（输入 / 假设 / 报告）
├── tools/retriever.py         # 经验文档读取 + Chroma 向量索引 + 融合检索
├── tools/flink_checklist.py   # 按假设类型生成 Flink UI / 校验平台检查清单（主验证手段）
├── tools/sql_generator.py     # 按假设类型生成辅助验证 SQL（可选，需人工执行）
├── agent/state.py             # LangGraph 状态定义
├── agent/prompts.py           # 系统提示词与用户提示词
├── agent/graph.py             # ReAct 图：检索 -> 推理 -> 工具 -> 收尾
├── api.py                     # Web 聊天服务（FastAPI + 内置页面）
├── data/user_experience/      # 经验文档（Markdown，含实际问题总结与示例，可自行替换）
├── data/vector_store/         # Chroma 持久化目录（自动生成）
├── tests/smoke_test.py        # 离线冒烟测试
└── requirements-agent.txt     # 依赖清单
```

## 快速开始

```bash
# 1. 安装依赖（项目已有 .venv 时先激活）
cd data_quality_agent
source ../.venv/bin/activate
pip install -r requirements-agent.txt

# 2. 配置密钥：把 .env.example 复制为 .env 并填入 DEEPSEEK_API_KEY
cp .env.example .env

# 3. 运行（示例输入）
python main.py --demo

# 或传入真实输入 JSON
python main.py input.json -o report.json
```

说明：仓库根目录已有一个 `.env`（学习示例脚本共用），配置会自动回退读取；
你也可以在本文件夹内放自己的 `.env`，文件夹内的优先生效。

输入 JSON 示例：

```json
{
  "source_table": "ods_orders",
  "target_table": "dwd_orders",
  "source_count": 1285430,
  "target_count": 1209876,
  "time_window_start": "2026-08-01 00:00:00",
  "time_window_end": "2026-08-04 00:00:00",
  "source_schema": "ods",
  "target_schema": "dwd",
  "extra_context": "每日凌晨 1 点收到校验平台告警"
}
```

输出为结构化 JSON 报告，包含 `severity`（严重等级）、`root_causes`
（每条含 `flink_ui_checks`、可选 `validation_sql` 与 `fix_suggestion`）、`checked_items`、
`unchecked_items` 与 `steps_used`。

## 工作原理

1. **启动建索引**：`ExperienceRetriever` 读取 `data/user_experience/` 下所有
   Markdown，按标题切块后写入 Chroma（默认每次启动重建，文档量小成本低）。
2. **表名优先检索**：收到输入后，先用“源表名 + 目标表名 + 时间窗口”构造查询，
   走“表名关键词 + 向量相似度”融合检索，把最相关的历史案例喂给模型。
3. **ReAct 循环**：LangGraph 状态图按 `retrieve_cases -> agent -> run_tools -> agent...`
   循环。`agent` 每步最多输出 3 个假设并指定工具（检索经验文档 / 生成 Flink UI 检查清单 /
   生成辅助验证 SQL / 列出文档）；`run_tools` 执行工具并把观察结果写回状态，供下一步推理。
   知识库中记录了“差异形态判断方向”的方法论：突然缺口查作业状态与 Sink 指标，
   稳定偏差审阅作业 SQL，随时间累积查水位线/背压/Checkpoint。
4. **终止条件**：找到根因立即输出结论；达到 8 步或总时长超过
   `MAX_TOTAL_TIME_SECONDS`（默认 180 秒，单次模型请求超时 60 秒）时，输出
   “信息不足”报告并列出已检查/未检查项。大模型调用异常（网络、超时、JSON 解析失败）
   也会自动降级为兜底报告，不会中断流程。

## 常用命令

```bash
# 终端对话模式（启动后可持续聊天）
python main.py --chat

# Web 聊天模式（浏览器打开 http://127.0.0.1:8000/）
python main.py --serve --port 8000

# 离线冒烟测试（不调用大模型，验证检索与图流程）
python main.py --no-llm
python tests/smoke_test.py

# 列出当前经验文档
python main.py --list-docs

# 从 stdin 读取输入
cat input.json | python main.py -
```

## 对话模式说明

- `--chat` 与 `--serve` 启动后不会退出：先粘贴一条排查输入 JSON（或输入 `/demo`），
  之后可以直接追问“为什么怀疑这个根因”“换个时间窗口再分析”等。
- 两种消息的处理方式不同：
  - JSON 输入（以 `{` 开头）→ 走完整 ReAct 排查流程，返回结构化报告；
  - 自然语言描述问题（如“目标表比源表大且两边都在减少”）→ 先尝试自动提取
    表名/数据量并转入排查；提取不到时基于知识库直接分析问题模式并给出排查建议，
    不会要求必须提供 JSON。
  - 已有排查后的普通追问 → 直接结合对话历史与最近报告用自然语言回答。
- 对话通过 LangGraph 的 messages 通道 + InMemorySaver 跨轮保留：
  每轮消息进入 `ingest` 节点（解析 JSON / 保留上一轮输入），同一会话固定 thread_id，
  提示词中会带上最近几轮问答。
- 命令：`/demo` 内置示例、`/new` 开启新会话、`/json` 查看上轮完整报告、
  `/help` 帮助、`/quit` 退出。

## 如何积累自己的排查经验

直接在 `data/user_experience/` 下新增 Markdown 文档即可，Agent 每次启动都会
重新建索引。建议每个案例包含：背景、现象、排查步骤（Flink UI + 校验平台）、根因、
UI 检查项与判定标准、修复建议。
文档中多写表名和关键词（如 `ods_orders`、`分区缺失`），检索命中率会更高。

## 限制与扩展点

- **主验证手段是 Flink UI + 校验平台**：每个假设类型都会生成对应的 Flink UI 检查清单；
  验证 SQL 只是辅助项，默认不自动执行。若之后有了库权限并想自动执行，可在 `.env`
  中开启 `SQL_EXECUTION_ENABLED=true` 并配置 `DATA_DB_URL`，再在
  `agent/graph.py` 的 `_execute_tool` 中补充执行逻辑（需另装 sqlalchemy）。
- **假设类型覆盖 Flink 作业侧场景**：Checkpoint 失败、背压/资源不足、
  Exactly-Once 重复消费、维表 JOIN 行数变化、水位线停滞等。
- **Embedding**：默认 `EMBEDDING_MODE=auto`，配置了独立的 `EMBEDDING_API_KEY`
  就走 OpenAI 兼容嵌入接口；否则自动降级为本地哈希嵌入（离线可用、效果一般）。
  注意 DeepSeek 不提供嵌入接口，请勿把 `DEEPSEEK_API_KEY` 当嵌入密钥用；
  追求更高检索质量时建议配置独立的语义 Embedding 服务。
- **SQL 模板字段**：模板默认分区字段 `dt`、主键 `id`，与真实表结构不一致时，
  模型会在提示词中看到字段说明，数据工程师执行前请按实际情况调整。
- **严重等级阈值**：`models/schemas.py` 的 `compute_severity` 中 10%/1% 阈值可调。
