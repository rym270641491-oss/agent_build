# -*- coding: utf-8 -*-
"""Flink UI / 校验平台检查清单生成器。

当前场景的约束：只能看到「阿里云 Flink UI」和「校验平台的数据量」，
不能直连数据库执行 SQL。因此每个假设类型对应的“验证手段”应是一组
Flink UI 可观测项 + 校验平台数据量对比，而不是 SQL。

每条检查项包含四个要素：
- where     : 去哪里看（Flink UI 的哪个页面/入口）
- check     : 看什么（具体指标或状态）
- expected  : 若假设成立，应当观察到什么现象
- conclusion: 根据观察结果能得到什么判断
"""
from __future__ import annotations

from typing import Dict, List

from models.schemas import DataMismatchInput

# 每个检查项的结构：位置 / 检查内容 / 预期现象 / 判定结论
FLINK_CHECKLISTS: Dict[str, List[Dict[str, str]]] = {
    # 通用计数对账：先确认差异本身与整体走势
    "count_mismatch": [
        {
            "where": "校验平台 → 数据量对比",
            "check": "源表与目标表在告警窗口内的数据量及差异比例",
            "expected": "确认差异量级与方向（源多/目标多）",
            "conclusion": "锁定差异规模和方向，作为后续排查基线",
        },
        {
            "where": "Flink UI → 作业详情 → 指标",
            "check": "Sink 端 records-sent / numRecordsOut 的走势",
            "expected": "某天/某时段输出骤降或为 0",
            "conclusion": "定位差异发生的时间点，缩小到具体分区或时段",
        },
    ],
    # 分区缺失
    "partition_missing": [
        {
            "where": "校验平台 → 数据量对比（按天）",
            "check": "逐天对比源表与目标表数据量",
            "expected": "缺口集中在某一个具体日期，其他日期一致",
            "conclusion": "确认缺失分区日期，缩小排查范围",
        },
        {
            "where": "Flink UI → 作业详情 → 指标",
            "check": "缺口当天 Sink 端 records-sent / numRecordsOut",
            "expected": "缺口当天 Sink 无输出或明显偏低",
            "conclusion": "对应日期没有数据写入目标表，支持“分区缺失”",
        },
        {
            "where": "Flink UI → 作业状态",
            "check": "缺口当天作业是否 RUNNING / RESTARTING / FAILED",
            "expected": "作业整体失败时缺口为整个窗口；RUNNING 但无输出则问题在作业内部或上游",
            "conclusion": "区分“作业失败”与“作业内逻辑/上游数据问题”",
        },
        {
            "where": "Flink UI → 日志（TaskManager）",
            "check": "缺口当天的异常栈与报错日志",
            "expected": "任务失败、反序列化错误、上游连接超时等",
            "conclusion": "定位具体失败原因",
        },
    ],
    # 增量任务失败/未执行
    "incremental_job_failed": [
        {
            "where": "Flink UI → 作业列表/作业状态",
            "check": "作业最近几天的运行状态与重启次数",
            "expected": "作业多次 RESTARTING / FAILED，或长时间未重启恢复",
            "conclusion": "确认作业层故障，数据停更时间与告警窗口吻合",
        },
        {
            "where": "Flink UI → 日志",
            "check": "失败时刻的异常日志",
            "expected": "连接超时、资源不足、SQL 校验失败等",
            "conclusion": "确定失败根因类别",
        },
        {
            "where": "Flink UI → Checkpoint",
            "check": "最近 Checkpoint 是否成功、恢复点是否有效",
            "expected": "Checkpoint 失败导致无法恢复到最新状态",
            "conclusion": "评估补跑时数据会不会丢失或重复",
        },
    ],
    # 去重口径差异
    "dedup_difference": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "作业中是否使用 DISTINCT / GROUP BY / ROW_NUMBER 去重",
            "expected": "源端查询去重、目标端直接 INSERT，两边口径不一致",
            "conclusion": "确认去重逻辑差异",
        },
        {
            "where": "Flink UI → 作业拓扑",
            "check": "JOIN / Aggregate 算子前后 records-in 与 records-out 行数",
            "expected": "目标链路少了去重算子，输出行数多于源表去重后行数",
            "conclusion": "量化行数差异来源",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "差异是否长期、稳定存在",
            "expected": "每天差异量级基本一致，随天数线性累积",
            "conclusion": "口径类问题通常是稳定偏差，而非某天突然缺口",
        },
    ],
    # 时间窗口/时区边界差异
    "time_window_difference": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "WATERMARK、窗口函数（TUMBLE/HOP/SESSION）与时间字段选择",
            "expected": "窗口边界或时间字段口径与校验平台不一致",
            "conclusion": "确认窗口语义差异",
        },
        {
            "where": "Flink UI → 作业参数",
            "check": "table.local-time-zone（时区）配置",
            "expected": "作业用 UTC、校验平台用本地时间（相差 8 小时）",
            "conclusion": "确认时区错位，边界数据归属不同分区",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "窗口边界前后 1 小时的数据量",
            "expected": "差异恰好等于边界错位的那部分数据",
            "conclusion": "验证边界/时区假设是否成立",
        },
    ],
    # 字段映射/类型转换差异
    "field_mapping_difference": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "字段映射、CAST / UDF 转换逻辑",
            "expected": "存在大数转精度丢失、枚举映射为空等转换",
            "conclusion": "确认转换逻辑是否可能导致数据被丢弃",
        },
        {
            "where": "Flink UI → 日志",
            "check": "Sink 写入报错（类型不匹配、字段缺失）",
            "expected": "部分记录被 Sink 拒绝并计入错误指标",
            "conclusion": "确认有记录在写入阶段被丢弃",
        },
        {
            "where": "Flink UI → 指标",
            "check": "Sink 端 numRecordsOutErrors 或类似错误计数",
            "expected": "错误计数与数据量缺口量级吻合",
            "conclusion": "量化转换/写入失败造成的缺口",
        },
    ],
    # 空值/非法值过滤差异
    "null_filter_difference": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "WHERE / CASE WHEN 中的空值、非法值过滤条件",
            "expected": "目标链路过滤条件比源表更严格或更宽松",
            "conclusion": "确认过滤口径差异",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "差异是否稳定、与空值数据量相关",
            "expected": "差异集中在存在空值的分区/时段",
            "conclusion": "佐证过滤差异假设",
        },
    ],
    # 同步延迟
    "sync_lag": [
        {
            "where": "Flink UI → 指标",
            "check": "Source 端 Watermark 与 currentFetchEventTimeLag",
            "expected": "水位线明显落后于当前时间，滞后量持续增大",
            "conclusion": "确认数据在消费/处理链路积压",
        },
        {
            "where": "Flink UI → 指标（Kafka 消费组）",
            "check": "各分区消费 lag",
            "expected": "lag 持续增长或某分区长期不前进",
            "conclusion": "定位是上游生产慢还是消费侧处理慢",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "多天差异走势",
            "expected": "差异随时间累积、目标表始终落后源表一段",
            "conclusion": "同步延迟类问题通常是持续累积而非单点缺口",
        },
    ],
    # 源表重复数据
    "duplicate_in_source": [
        {
            "where": "Flink UI → 作业拓扑（Source）",
            "check": "Source 端 records-consumed 与去重算子前后行数",
            "expected": "源端本身存在重复键，未做去重直接写入目标表",
            "conclusion": "确认重复数据来源在源侧",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "目标表是否明显大于源表",
            "expected": "目标表总数大于源表，且差异稳定",
            "conclusion": "提示去重或重复写入问题，与 dedup_difference 配合判断",
        },
    ],
    # 抽样/限量差异
    "sampling_difference": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "是否存在 LIMIT / 抽样 / 维表限量等逻辑",
            "expected": "目标链路只同步了部分数据",
            "conclusion": "确认抽样/限量逻辑",
        },
        {
            "where": "校验平台 → 数据量对比",
            "check": "差异比例是否整体稳定（如恒为 90%）",
            "expected": "目标表恒为源表的某个固定比例",
            "conclusion": "抽样/限量类问题呈固定比例差异",
        },
    ],
    # Flink：Checkpoint 失败
    "checkpoint_failure": [
        {
            "where": "Flink UI → Checkpoint",
            "check": "最近 Checkpoint 状态（COMPLETED/FAILED）、失败原因",
            "expected": "Checkpoint 反复失败或长时间未完成",
            "conclusion": "确认 Checkpoint 故障",
        },
        {
            "where": "Flink UI → Checkpoint 详情",
            "check": "barrier 对齐时长、Checkpoint 大小、对齐失败子任务",
            "expected": "对齐超时、数据倾斜导致个别子任务拖慢整体",
            "conclusion": "定位 Checkpoint 失败的具体环节",
        },
        {
            "where": "Flink UI → 作业状态",
            "check": "作业重启次数与恢复方式",
            "expected": "重启后从旧 Checkpoint 恢复，导致数据重复或缺失",
            "conclusion": "评估 Checkpoint 问题对数据一致性的影响",
        },
    ],
    # Flink：背压/资源不足
    "backpressure": [
        {
            "where": "Flink UI → 作业拓扑",
            "check": "各算子背压状态（OK / LOW / HIGH）",
            "expected": "某个算子持续 HIGH，上游数据积压",
            "conclusion": "确认背压瓶颈算子",
        },
        {
            "where": "Flink UI → 资源指标",
            "check": "TaskManager CPU / 内存 / GC 情况",
            "expected": "CPU 打满或频繁 Full GC，吞吐下降",
            "conclusion": "确认是资源不足还是算子效率问题",
        },
        {
            "where": "Flink UI → 指标",
            "check": "Sink 端 records-sent 走势",
            "expected": "输出速率明显低于源端消费速率",
            "conclusion": "量化延迟累积速度",
        },
    ],
    # Flink：Exactly-Once / 重复消费
    "exactly_once_duplication": [
        {
            "where": "Flink UI → 作业参数",
            "check": "checkpointing mode（EXACTLY_ONCE / AT_LEAST_ONCE）",
            "expected": "at-least-once 且作业发生过重启",
            "conclusion": "存在重复消费的可能性",
        },
        {
            "where": "Flink UI → 作业状态/日志",
            "check": "重启次数与恢复时间点",
            "expected": "重启后从旧位点重新消费，重复写入目标表",
            "conclusion": "确认重复窗口",
        },
        {
            "where": "Flink UI → 作业 SQL",
            "check": "Sink 是否幂等（upsert / 按主键去重写入）",
            "expected": "Sink 为普通 INSERT，无幂等保护",
            "conclusion": "确认重复数据会真实落库",
        },
    ],
    # Flink：维表 JOIN 行数放大/缩小
    "dimension_join_amplification": [
        {
            "where": "Flink UI → 作业 SQL",
            "check": "JOIN 类型（INNER / LEFT / Lookup）与维表关联键",
            "expected": "一对多关联导致放大，或关联不上被过滤导致缩小",
            "conclusion": "确认 JOIN 语义是否导致行数变化",
        },
        {
            "where": "Flink UI → 作业拓扑",
            "check": "JOIN 算子前后 records-in / records-out 行数",
            "expected": "输出行数与输入行数差异显著",
            "conclusion": "量化放大/缩小比例，与数据量差异比对",
        },
        {
            "where": "Flink UI → 作业参数",
            "check": "维表缓存策略（LRU / ALL）与缓存失效时间",
            "expected": "缓存过期后维表查询失败，关联不上导致行数减少",
            "conclusion": "确认维表侧故障是否造成缺口",
        },
    ],
    # Flink：水位线停滞/迟到数据
    "watermark_stall": [
        {
            "where": "Flink UI → 指标",
            "check": "Source 端 Watermark 与 currentFetchEventTimeLag",
            "expected": "Watermark 长时间停滞不前",
            "conclusion": "确认水位线停滞，窗口结果延迟或不再触发",
        },
        {
            "where": "Flink UI → 作业 SQL",
            "check": "WATERMARK 定义与 allowedLateness / 迟到数据处理",
            "expected": "迟到数据被丢弃或窗口迟迟不输出",
            "conclusion": "确认迟到数据口径",
        },
        {
            "where": "Flink UI → 指标（Source 端）",
            "check": "上游数据是否仍在生产（records-consumed）",
            "expected": "上游停止生产导致水位线停滞，目标表数据不再增长",
            "conclusion": "区分是上游断流还是作业处理问题",
        },
    ],
    # 自定义：通用兜底检查清单
    "custom": [
        {
            "where": "Flink UI → 作业状态",
            "check": "作业运行状态、重启次数、运行时长",
            "expected": "发现异常状态或频繁重启",
            "conclusion": "先排除作业层故障",
        },
        {
            "where": "Flink UI → Checkpoint",
            "check": "最近 Checkpoint 状态与耗时",
            "expected": "Checkpoint 失败或异常缓慢",
            "conclusion": "排除一致性/恢复类问题",
        },
        {
            "where": "Flink UI → 指标",
            "check": "Source/Sink 端 records 走势与 Watermark",
            "expected": "输入输出速率不匹配或水位线停滞",
            "conclusion": "判断是上游、作业内部还是下游问题",
        },
        {
            "where": "Flink UI → 日志",
            "check": "异常日志与报错栈",
            "expected": "发现明确异常信息",
            "conclusion": "直接定位根因或缩小范围",
        },
    ],
}


def list_checklist_types() -> List[str]:
    """返回已配置检查清单的假设类型列表。"""
    return sorted(FLINK_CHECKLISTS.keys())


def generate_flink_checklist(
    input_data: DataMismatchInput,
    hypothesis_type: str,
    description: str = "",
) -> str:
    """生成某个假设类型的 Flink UI / 校验平台检查清单。"""
    items = FLINK_CHECKLISTS.get(hypothesis_type, FLINK_CHECKLISTS["custom"])
    lines = [
        "【Flink UI / 校验平台检查清单】假设类型: {}（{}）".format(
            hypothesis_type,
            description or "自定义",
        )
    ]
    for i, item in enumerate(items, start=1):
        lines.append(
            "{}. 查看位置: {}\n"
            "   检查内容: {}\n"
            "   预期现象: {}\n"
            "   判定结论: {}".format(
                i,
                item["where"],
                item["check"],
                item["expected"],
                item["conclusion"],
            )
        )
    return "\n".join(lines)
