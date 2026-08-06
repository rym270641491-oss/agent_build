# -*- coding: utf-8 -*-
"""数据模型定义（Pydantic）。

包含：
- 输入 JSON 的数据结构（源表、目标表、数据量、时间窗口）
- 推理过程中产生的假设对象
- 最终输出的结构化报告（严重等级、根因列表、已检查/未检查项）
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DataMismatchInput(BaseModel):
    """用户输入的排查请求（来自校验平台报告的 JSON）。"""

    source_table: str = Field(..., description="源表名，例如 ods_orders")
    target_table: str = Field(..., description="目标表名，例如 dwd_orders")
    source_count: Optional[int] = Field(None, description="源表数据量")
    target_count: Optional[int] = Field(None, description="目标表数据量")
    time_window_start: str = Field(..., description="时间窗口开始，例如 2026-08-01 00:00:00")
    time_window_end: str = Field(..., description="时间窗口结束，例如 2026-08-04 00:00:00")
    source_schema: Optional[str] = Field(None, description="源表所在 schema/库名")
    target_schema: Optional[str] = Field(None, description="目标表所在 schema/库名")
    extra_context: Optional[str] = Field(None, description="附加上下文，例如任务名、报错信息")


class Hypothesis(BaseModel):
    """一条候选假设：由模型在 ReAct 循环的某一步提出。"""

    id: int = Field(..., description="假设编号（全局递增）")
    description: str = Field(..., description="假设内容描述")
    hypothesis_type: str = Field(..., description="假设类型，见 sql_generator 中的类型映射")
    tool_name: str = Field(
        ...,
        description="要调用的工具名：search_experience / generate_validation_sql / list_experience_docs",
    )
    tool_query: str = Field("", description="传给工具的参数，例如检索关键词或假设描述")
    step: int = Field(0, description="提出该假设时所在的步数")
    status: str = Field(
        "proposed",
        description="假设状态：proposed / evidence_collected / confirmed / refuted / unknown",
    )


class RootCause(BaseModel):
    """报告中的一条根因结论。"""

    description: str = Field("", description="根因描述")
    evidence: str = Field("", description="支撑证据（检索到的经验、数据对比等）")
    # Flink UI / 校验平台检查项：当前主验证手段（不假设能直连数据库）
    flink_ui_checks: List[str] = Field(
        default_factory=list, description="Flink UI / 校验平台检查项（验证手段）"
    )
    # 验证 SQL：可选辅助手段，需提交给有库权限的平台人工执行
    validation_sql: str = Field("", description="验证 SQL（可选辅助，需人工在数据平台执行）")
    fix_suggestion: str = Field("", description="修复建议")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="置信度 0~1")


class CheckedItem(BaseModel):
    """已检查项。"""

    item: str = Field(..., description="检查项名称")
    result: str = Field(..., description="检查结果")
    evidence: str = Field("", description="证据说明")


class UncheckedItem(BaseModel):
    """未检查项（信息不足时列出）。"""

    item: str = Field(..., description="未检查项名称")
    reason: str = Field("", description="未能检查的原因")


class DataQualityReport(BaseModel):
    """最终结构化报告。"""

    # root_cause_found：找到根因；insufficient_info：信息不足
    status: Literal["root_cause_found", "insufficient_info"] = Field(
        ..., description="结论状态"
    )
    # 严重等级：high / medium / low / unknown
    severity: Literal["high", "medium", "low", "unknown"] = Field(
        "unknown", description="严重等级"
    )
    summary: str = Field("", description="结论摘要")
    # 根因列表（每条包含验证 SQL 与修复建议）
    root_causes: List[RootCause] = Field(default_factory=list, description="根因列表")
    checked_items: List[CheckedItem] = Field(default_factory=list, description="已检查项")
    unchecked_items: List[UncheckedItem] = Field(default_factory=list, description="未检查项")
    steps_used: int = Field(0, description="实际使用的步数")
    total_steps: int = Field(8, description="允许的最大步数")


def compute_severity(
    source_count: Optional[int], target_count: Optional[int]
) -> str:
    """根据数据量差异比例估算严重等级（可自行调整阈值）。

    规则：
    - 差异比例 >= 10%：high
    - 差异比例 >= 1% ：medium
    - 有差异但小于 1%：low
    - 数据量缺失或完全一致：unknown
    """
    if source_count is None or target_count is None or source_count == target_count:
        return "unknown"
    diff = abs(source_count - target_count)
    base = max(source_count, target_count, 1)
    ratio = diff / base
    if ratio >= 0.10:
        return "high"
    if ratio >= 0.01:
        return "medium"
    return "low"
