# 案例：Checkpoint 反复失败 + 非幂等写入导致目标表数据偏多

## 背景
`ods_trade` 与 `dwd_trade` 数据量不一致，目标表总数比去重数多出约 1.5 万行，
差异集中在某几次作业重启之后。

## 现象
- 校验平台显示目标表总数明显大于源表去重后的行数。
- 差异增量出现在作业重启的时间点附近。

## 排查步骤（Flink UI）
1. Flink UI → Checkpoint：最近 Checkpoint 反复 FAILED，或长时间未 COMPLETED。
2. Flink UI → 作业状态：作业多次重启，恢复方式是从旧 Checkpoint / 无状态恢复。
3. Flink UI → 作业参数：checkpointing mode 为 AT_LEAST_ONCE，重启后重复消费。
4. Flink UI → 作业 SQL：Sink 是普通 INSERT，没有 upsert / 主键去重，不幂等。

## 根因
作业为 at-least-once 语义且 Checkpoint 反复失败，重启后从旧位点重新消费；
又因为 Sink 非幂等，同一批数据被写入多次，目标表数据量偏大。

## UI 检查项与判定标准
- Checkpoint 状态与失败原因：反复失败是问题起点。
- checkpointing mode：AT_LEAST_ONCE + 重启 → 存在重复消费窗口。
- Sink 是否幂等：普通 INSERT → 重复数据真实落库。

## 可选辅助 SQL（需人工在数据平台执行）
```sql
-- 目标表总数 vs 去重数，差距即重复量
SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT trade_id) AS distinct_cnt
FROM `dwd`.`dwd_trade`
WHERE dt >= '2026-08-01' AND dt < '2026-08-04';
```

## 修复建议
- 排查 Checkpoint 失败原因（状态后端容量、barrier 对齐超时、资源不足）。
- 作业参数改为 EXACTLY_ONCE，Sink 改为 upsert 幂等写入。
- 清理存量重复数据后，再核对两边数据量。
