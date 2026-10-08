# 原研究仿真与统计程序

本目录恢复原始研究文件：阻力模型、POD 数据库与训练、水动力/推进耦合、在线收益估计、四动作枚举、同规则贪心、LinUCB/Thompson、四种工况、统一实验和原测试。恢复文件的出处和原始哈希见根目录 PROVENANCE.json。除 baseline2_rom.py 的旧 CFD 数据说明已更正为合成数据外，原研究文件按字节保留；数值实现未改动。`smoke.py`、`verify_statistics.py`、`reproduce.py` 是新增便携入口。

在仓库根目录用 Python 3.12 执行：

```bash
python -m pip install -r requirements-engine.txt
python projects/multifidelity-scheduler/engine/smoke.py
python projects/multifidelity-scheduler/engine/verify_statistics.py
```

第一条程序重新生成 48 工况的合成数据库并训练 POD，执行一条 900 步耦合轨迹上的八策略、四预算共 32 次运行。使用原冻结成本、收益先验与上下文策略超参数；这是新执行检查，不是 2,048 次实验的新统计复现。Fixed-L2 为非预算可行对照，其余策略检查调用额度可行性。测得耗时写入报告，不作为精度通过门槛。

第二条程序从原 2,048 条逐次记录和 64 条直接控制计时重算 ROM 簇均值、Student t 区间、配对 p 值、12 项主比较及四项消融的独立 Holm 校正。它没有重新生成历史计时。

运行完整统一实验（含先在独立校准种子上调参，再执行全部策略）：

```bash
python projects/multifidelity-scheduler/engine/reproduce.py
```

完整运行比检查慢，产生新目录 `reports/multifidelity-<时间>/`，保留协议、逐次数据和统计。本次交付没有重跑全部 2,048 次实验。需要指定目录时使用 `--out`，目录必须不存在或为空。原冻结成本与先验来自保留的 2026-09-24 校准计划；重新测成本不是默认实验内容。

原 11 项测试从本目录执行：

```bash
python -m unittest test_revision_20261002 recovered_source.tests.test_baseline7_online_allocation recovered_source.tests.test_baseline9_strong_baselines
```

数据由数值模型生成，当前没有重新进行 CFD、实船试航或硬实时资源控制。单步枚举覆盖有限动作集合，不能声称多步全局最优或已实现端到端加速。
