# -*- coding: utf-8 -*-
"""验证 SQL 生成工具。

按“假设类型”生成用于验证根因的 SQL 片段（辅助手段）。

注意：当前场景下主验证手段是「阿里云 Flink UI + 校验平台数据量」，
不假设能直连数据库；这里的 SQL 是可选辅助，由有库权限的数据工程师
在数据平台上人工执行，用于最终确认。

模板中用到了占位符，生成时会替换为真实的表名与时间窗口：
- {src} / {tgt}：源表 / 目标表（含 schema）
- {start} / {end}：时间窗口起止
- {part_col}：分区字段（默认 dt，请按实际表结构调整）
- {pk}：主键字段（默认 id）
- {key_col}：业务键字段（默认 id）
"""
from __future__ import annotations

from typing import Dict, List, Optional

from models.schemas import DataMismatchInput

# 假设类型 -> SQL 模板
HYPOTHESIS_SQL_TEMPLATES: Dict[str, str] = {
    # 通用计数对账：先确认差异本身
    "count_mismatch": """
-- 通用计数对账：对比源表与目标表在时间窗口内的记录数
SELECT COUNT(*) AS cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';

SELECT COUNT(*) AS cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 分区缺失
    "partition_missing": """
-- 场景：某天分区未生成/未写入
-- 按分区统计源表与目标表，检查是否存在缺分区：
SELECT {part_col} AS partition_day, COUNT(*) AS cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}'
GROUP BY {part_col}
ORDER BY {part_col};

SELECT {part_col} AS partition_day, COUNT(*) AS cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}'
GROUP BY {part_col}
ORDER BY {part_col};
""",
    # 增量任务失败/未执行
    "incremental_job_failed": """
-- 场景：增量任务失败或未调度
-- 1) 到调度平台（Airflow/DolphinScheduler 等）核对任务最近一次运行状态；
-- 2) 用下列 SQL 检查目标表最近是否有新数据写入：
SELECT MAX({part_col}) AS latest_partition, COUNT(*) AS cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 去重口径差异
    "dedup_difference": """
-- 场景：两边去重口径不一致（如源表按主键去重，目标表未去重）
SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT {pk}) AS distinct_cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';

SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT {pk}) AS distinct_cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 时间窗口/时区边界差异
    "time_window_difference": """
-- 场景：时间窗口边界/时区处理不一致（如 start 边界是否含当日、UTC 与本地时间）
SELECT MIN({part_col}) AS min_dt, MAX({part_col}) AS max_dt, COUNT(*) AS cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';

SELECT MIN({part_col}) AS min_dt, MAX({part_col}) AS max_dt, COUNT(*) AS cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 字段映射/类型转换差异
    "field_mapping_difference": """
-- 场景：字段映射/类型转换导致数据丢失（如大数转 decimal 溢出、枚举值映射为空）
-- 检查两边字段映射与类型定义，并抽查源表有、目标表没有的行：
SELECT COUNT(*) AS missing_cnt
FROM {src} s
LEFT JOIN {tgt} t ON s.{pk} = t.{pk}
WHERE t.{pk} IS NULL;
""",
    # 空值/非法值过滤差异
    "null_filter_difference": """
-- 场景：空值/非法值过滤口径不一致
SELECT COUNT(*) AS null_cnt
FROM {src}
WHERE {key_col} IS NULL;

SELECT COUNT(*) AS null_cnt
FROM {tgt}
WHERE {key_col} IS NULL;
""",
    # 同步延迟
    "sync_lag": """
-- 场景：同步延迟导致目标表数据尚未刷出
-- 对比两边最新分区时间差：
SELECT MAX({part_col}) AS latest_src FROM {src};
SELECT MAX({part_col}) AS latest_tgt FROM {tgt};
""",
    # 源表重复数据
    "duplicate_in_source": """
-- 场景：源表本身存在重复数据
SELECT {pk}, COUNT(*) AS dup_cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}'
GROUP BY {pk}
HAVING COUNT(*) > 1
LIMIT 20;
""",
    # 抽样/限量差异
    "sampling_difference": """
-- 场景：目标表建设时抽样或限量（如只同步部分数据）
SELECT COUNT(*) AS cnt, COUNT(DISTINCT {pk}) AS distinct_cnt
FROM {src}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';

SELECT COUNT(*) AS cnt, COUNT(DISTINCT {pk}) AS distinct_cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # Checkpoint 失败：主要靠 Flink UI 的 Checkpoint 指标确认，SQL 仅辅助核对数据缺口
    "checkpoint_failure": """
-- 场景：Checkpoint 反复失败，作业频繁重启或无法恢复
-- 主验证手段：Flink UI → Checkpoint，查看最近 Checkpoint 状态（COMPLETED/FAILED）、
--             失败原因、barrier 对齐时长；以及作业重启次数与 TaskManager 日志。
-- 以下 SQL 仅辅助确认目标表最新数据落点：
SELECT MAX({part_col}) AS latest_partition, COUNT(*) AS cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 背压/资源不足：主要靠 Flink UI 反压与资源指标确认
    "backpressure": """
-- 场景：背压（Backpressure）导致吞吐下降、数据积压
-- 主验证手段：Flink UI → 作业拓扑，查看各算子背压状态（OK/LOW/HIGH）；
--             指标看 CPU/内存使用率与 Sink 端 records-sent 走势。
-- 以下 SQL 仅辅助确认目标表是否落后于源表：
SELECT MAX({part_col}) AS latest_src FROM {src};
SELECT MAX({part_col}) AS latest_tgt FROM {tgt};
""",
    # Exactly-Once / 重复消费：主要靠作业配置与 Checkpoint/重启记录确认
    "exactly_once_duplication": """
-- 场景：作业为 at-least-once 且重启后重复消费，Sink 未做幂等，目标表数据偏多
-- 主验证手段：Flink UI → 作业参数（checkpointing mode）、重启次数与恢复方式；
--             若目标表总数明显大于去重数，则高度怀疑重复写入。
-- 辅助 SQL（需有库权限）：
SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT {pk}) AS distinct_cnt
FROM {tgt}
WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 维表 JOIN 行数放大/缩小
    "dimension_join_amplification": """
-- 场景：维表 JOIN 导致行数放大（一对多）或缩小（关联不上被过滤）
-- 主验证手段：Flink UI → 作业 SQL/拓扑，查看 JOIN 类型（INNER/LEFT/Lookup）、
--             维表缓存策略；对比 JOIN 前后算子 records-in/records-out 行数。
-- 辅助 SQL（需有库权限）：分别统计源表行数与目标表行数，评估放大/缩小比例
SELECT COUNT(*) AS src_cnt FROM {src} WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
SELECT COUNT(*) AS tgt_cnt FROM {tgt} WHERE {part_col} >= '{start}' AND {part_col} < '{end}';
""",
    # 水位线停滞/迟到数据
    "watermark_stall": """
-- 场景：水位线长时间停滞，窗口结果迟迟不输出或迟到数据被丢弃
-- 主验证手段：Flink UI → 指标，查看 Source 端 Watermark 与
--             currentFetchEventTimeLag；确认上游是否停止生产数据。
-- 辅助 SQL（需有库权限）：对比两边最新数据时间
SELECT MAX({part_col}) AS latest_src FROM {src};
SELECT MAX({part_col}) AS latest_tgt FROM {tgt};
""",
    # 自定义类型：没有内置模板
    "custom": """
-- 该假设类型为自定义，未匹配到内置模板。
-- 请根据以下假设描述人工编写验证 SQL：
-- {description}
""",
}

# 假设类型的中文说明（会写入系统提示词，供模型选择）
HYPOTHESIS_TYPES: Dict[str, str] = {
    "count_mismatch": "通用计数对账（先确认差异本身）",
    "partition_missing": "分区缺失",
    "incremental_job_failed": "增量任务失败/未执行",
    "dedup_difference": "去重口径差异",
    "time_window_difference": "时间窗口/时区边界差异",
    "field_mapping_difference": "字段映射/类型转换差异",
    "null_filter_difference": "空值/非法值过滤差异",
    "sync_lag": "同步延迟",
    "duplicate_in_source": "源表重复数据",
    "sampling_difference": "抽样/限量差异",
    # ---- Flink 作业侧常见根因（Flink UI 可直接观测） ----
    "checkpoint_failure": "Checkpoint 失败/超时（Flink）",
    "backpressure": "背压/资源不足导致吞吐下降（Flink）",
    "exactly_once_duplication": "Exactly-Once 配置/重复消费（Flink）",
    "dimension_join_amplification": "维表 JOIN 行数放大/缩小",
    "watermark_stall": "水位线停滞/迟到数据未处理（Flink）",
    "custom": "自定义",
}


def _quote(identifier: str) -> str:
    """给标识符加反引号，并去掉可能存在的空格/反引号，防止注入。"""
    return "`{}`".format(identifier.strip().replace("`", ""))


def _full_name(schema: Optional[str], table: str) -> str:
    """拼出 schema.table 形式（可带反引号）。"""
    if schema:
        return "{}.{}".format(_quote(schema), _quote(table))
    return _quote(table)


def list_hypothesis_types() -> List[str]:
    """返回“类型: 说明”列表，供系统提示词使用。"""
    return ["{}: {}".format(key, label) for key, label in HYPOTHESIS_TYPES.items()]


def generate_validation_sql(
    input_data: DataMismatchInput,
    hypothesis_type: str,
    description: str = "",
) -> str:
    """根据假设类型生成验证 SQL。

    参数：
    - input_data: 排查输入（提供表名、schema、时间窗口）
    - hypothesis_type: 假设类型，见 HYPOTHESIS_TYPES
    - description: 假设描述，自定义类型时嵌入 SQL 注释
    """
    src = _full_name(input_data.source_schema, input_data.source_table)
    tgt = _full_name(input_data.target_schema, input_data.target_table)
    template = HYPOTHESIS_SQL_TEMPLATES.get(
        hypothesis_type, HYPOTHESIS_SQL_TEMPLATES["custom"]
    )
    sql = template.format(
        src=src,
        tgt=tgt,
        start=input_data.time_window_start,
        end=input_data.time_window_end,
        # 占位符默认值：分区字段 dt、主键 id，请按实际表结构调整
        part_col="dt",
        pk="id",
        key_col="id",
        description=description or hypothesis_type,
    ).strip()
    header = (
        "-- 假设说明：{}\n"
        "-- 注意：主验证手段为 Flink UI 检查（见 generate_flink_checklist），\n"
        "-- 以下 SQL 为可选辅助，需人工在数据平台执行；分区字段/主键请按实际表结构调整"
    ).format(
        description or HYPOTHESIS_TYPES.get(hypothesis_type, hypothesis_type)
    )
    return "{}\n{}".format(header, sql)


def generate_count_check_sql(input_data: DataMismatchInput) -> str:
    """生成最基础的计数对账 SQL（报告与兜底逻辑使用）。"""
    return generate_validation_sql(input_data, "count_mismatch")
