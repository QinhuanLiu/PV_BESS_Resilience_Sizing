"""Economic and PV utilization metrics derived from streaming dispatch totals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .costing import pv_bess_annual_cost_from_totals
from .resilience_metrics import DEFAULT_TOLERANCE


PRIMARY_COLUMNS = ["annual_cost_sgd_per_year", "SSR", "SCR", "SCR_defined", "SCR_status"]


def economic_metrics(
    totals: dict[str, float], sizing: dict[str, float], pv_capacity: float, config: dict[str, Any],
) -> dict[str, Any]:
    if not all(np.isfinite(value) for value in totals.values()):
        raise ValueError("Dispatch statistics contain nonfinite values")
    if totals["load_kwh_total"] <= 0:
        raise ValueError("SSR requires positive building demand")
    costs = pv_bess_annual_cost_from_totals(totals, sizing, {"pv_capacity_kw": pv_capacity}, config)
    pv_to_load = totals["pv_to_load_kwh_total"] + totals["battery_discharge_from_pv_kwh_total"]
    local_pv_use = totals["pv_to_load_kwh_total"] + totals["pv_to_battery_kwh_total"]
    generated = totals["pv_available_kwh_total"]
    count, capacity = totals["dispatch_timesteps"], sizing["battery_energy_kwh"]
    return {
        **totals,
        **costs,
        "annual_cost_sgd_per_year": costs["pv_bess_system_annual_cost_sgd"],
        "SSR": pv_to_load / totals["load_kwh_total"],
        "SCR": local_pv_use / generated if generated > 0 else None,
        "SCR_defined": bool(generated > 0),
        "SCR_status": "defined" if generated > 0 else "not_applicable_no_pv",
        "pv_self_served_load_kwh_total": pv_to_load,
        "pv_local_use_kwh_total": local_pv_use,
        "battery_soc_mean_fraction": totals["battery_soc_kwh_sum"] / (count * capacity) if capacity > 0 else 0.0,
    }


def validate_economic_results(results: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    columns = [column for column in results.select_dtypes(include="number").columns if column != "SCR"]
    if not np.isfinite(results[columns].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite values outside the nullable SCR column")
    if results.drop(columns=["SCR"]).isna().any().any():
        raise ValueError("Missing values outside the nullable SCR column")
    has_pv = results.pv_available_kwh_total > 0
    if not results.SCR_defined.eq(has_pv).all() or not results.SCR.isna().eq(~has_pv).all():
        raise ValueError("SCR must be defined exactly when PV generation is positive")
    if not results.loc[~has_pv, "SCR_status"].eq("not_applicable_no_pv").all():
        raise ValueError("Undefined SCR is not labelled")
    if not results.loc[has_pv, "SCR_status"].eq("defined").all():
        raise ValueError("Defined SCR is incorrectly labelled")
    for values in (results.SSR, results.loc[has_pv, "SCR"]):
        values = values.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < -1e-10).any() or (values > 1 + 1e-10).any():
            raise ValueError("SSR or SCR outside [0, 1]")
    if not results.loc[~has_pv, "SSR"].eq(0).all():
        raise ValueError("Zero PV must have SSR=0")

    comparisons = {
        "annual_cost_components": (
            results.annual_cost_sgd_per_year,
            results.annualized_pv_cost_sgd + results.pv_om_cost_sgd
            + results.annualized_battery_cost_sgd + results.battery_om_cost_sgd
            + results.pv_bess_net_grid_cost_sgd + results.unserved_energy_cost_sgd_per_year,
        ),
        "annual_cost_alias": (results.annual_cost_sgd_per_year, results.pv_bess_system_annual_cost_sgd),
        "net_grid_cost": (
            results.pv_bess_net_grid_cost_sgd,
            (results.grid_cost_sgd_total - results.pv_export_revenue_sgd_total) / results.analysis_years,
        ),
        "outage_cost": (
            results.unserved_energy_cost_sgd_per_year,
            results.unserved_energy_kwh_total / results.analysis_years * config["active_scenario"]["voll_sgd_per_kwh"],
        ),
        "pv_balance": (
            results.pv_available_kwh_total,
            results.pv_to_load_kwh_total + results.pv_to_battery_kwh_total
            + results.pv_curtailed_kwh_total + results.pv_export_kwh_total,
        ),
        "load_balance": (
            results.load_kwh_total,
            results.pv_to_load_kwh_total + results.grid_to_load_kwh_total
            + results.battery_discharge_kwh_total + results.unserved_energy_kwh_total,
        ),
        "grid_balance": (results.grid_import_kwh_total, results.grid_to_load_kwh_total + results.grid_to_battery_kwh_total),
        "battery_charge_origins": (results.battery_charge_kwh_total, results.pv_to_battery_kwh_total + results.grid_to_battery_kwh_total),
        "battery_discharge_origins": (
            results.battery_discharge_kwh_total,
            results.battery_discharge_from_pv_kwh_total + results.battery_discharge_from_grid_kwh_total
            + results.battery_discharge_from_initial_kwh_total,
        ),
        "SSR_formula": (results.SSR, results.pv_self_served_load_kwh_total / results.load_kwh_total),
        "SCR_formula": (results.loc[has_pv, "SCR"], results.loc[has_pv, "pv_local_use_kwh_total"] / results.loc[has_pv, "pv_available_kwh_total"]),
        "raw_EUE_difference": (
            results.EUE_threshold_difference_kwh_total,
            results.unserved_energy_kwh_total - results.EUE_kwh_total,
        ),
    }
    eta = float(config["battery"]["roundtrip_efficiency"]) ** 0.5
    comparisons["battery_energy_balance"] = (
        results.battery_soc_final_kwh,
        results.battery_soc_initial_kwh + eta * results.battery_charge_kwh_total - results.battery_discharge_kwh_total / eta,
    )
    comparisons["battery_final_origins"] = (
        results.battery_soc_final_kwh,
        results.battery_soc_pv_final_kwh + results.battery_soc_grid_final_kwh + results.battery_soc_initial_origin_final_kwh,
    )
    failed = [name for name, (a, b) in comparisons.items()
              if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9, atol=1e-7)]
    if failed:
        raise ValueError(f"Cost/energy reconciliation failed: {failed}")
    difference = results.EUE_threshold_difference_kwh_total
    if (difference < -1e-9).any() or (difference > results.dispatch_timesteps * DEFAULT_TOLERANCE + 1e-9).any():
        raise ValueError("Raw versus thresholded EUE difference exceeds the documented tolerance")
    return {
        "economic_metrics_passed": True,
        "cost_and_energy_checks": list(comparisons),
        "SCR_not_applicable_rows": int((~has_pv).sum()),
        "five_metrics_rows": len(results),
        "max_EUE_threshold_difference_kwh_total": float(difference.max()),
    }


ECONOMIC_METRICS_README = """# 年化成本、EUE、LOLP、SSR、SCR 逐场景结果

`scenario_results.csv` 的每一行对应一个 PV 容量、电池容量和完整十五年停电场景。
`pv_capacity_kw + battery_energy_kwh + scenario_id` 是唯一键。五项指标均逐场景保留，
容量汇总表只是附加输出。

| 主列 | 口径 |
|---|---|
| annual_cost_sgd_per_year | SGD/年；PV/电池年化投资＋运维＋净购电费用＋停电损失 |
| EUE_kwh_per_year | kWh/年；十五年未供电量除以 15 |
| EUE_kwh_total | kWh；十五年累计未供电量 |
| LOLP | 原项目的能量型比例：未供电量／停电期间需求 |
| SSR | （光伏直供负荷＋光伏来源电池放电）／建筑总电当量需求 |
| SCR | （光伏直供＋光伏充入电池）／可用光伏发电量 |

SSR、SCR、LOLP 均保存为 0–1 比例，0.1 表示 10%。SSR 分母包括全部需求，未供电部分
也保留在分母内。SSR 不计电网充电和初始电池电量；电池放电来源沿用模型的 PV 优先
归属规则。SCR 包含充入电池后产生的损耗及期末储能，不计弃光或上网电量。
无光伏时 SSR=0，SCR 数值留空、SCR_defined=false、SCR_status=not_applicable_no_pv。
本次 99×100 网格应有 900 行这样的 SCR，不是漏算。

成本公式沿用原模型。PV 和电池投资通过各自寿命及折现率进行资本回收年化；运行电费、
上网收入及停电损失取十五年平均值，不额外引入逐年折现现金流。电池经济寿命由充放电
吞吐量及平均 SOC 推算衰减得到，并受原参数更换年限上限约束；调度中的可用容量不衰减。
`battery_replacement_years` 对零容量电池只是公式的占位值，其投资及运维成本均为零。

原成本函数使用原始未供电量 `unserved_energy_kwh_total`；已有 EUE 在每小时剔除不大于
1e-6 kWh 的微小未供电量。`EUE_threshold_difference_kwh_total` 保存两者差值。
停电损失使用原始未供电量×VoLL／分析年数，已有 EUE 不改写。

保留所有成本分项、累计购电量和费用、光伏直供/充电/弃光/上网电量、电池不同来源的
放电量、SOC 总和及期初期末储能，供后续分析或换算口径。小时输出仍仅有可靠性所需
数据，新增指标通过流式累计得到，完整小时明细不落盘。

`capacity_summary.csv` 提供每组容量的场景均值、样本标准差、中位数、P95、最小值及
最大值；SCR 不适用时汇总值也为空。所有均值都是逐场景指标的算术平均。
`yearly_results.csv` 保留逐年 EUE/LOLP，成本、SSR、SCR 的主交付粒度为整个十五年场景。

`manifest.json` 记录代码、输入、定义及源结果的指纹；`checkpoints/` 支持断点续跑。
仅当 run_status.json 的 state 为 complete 且 validation.json 的 passed 为 true 时，
才表示所有场景完成且核对通过。
"""


ENRICHMENT_README = ECONOMIC_METRICS_README + """

本目录由已完成的可靠性扫描补充得到：输入、有效配置与停电序列直接来自
`source_reference.json` 指向的旧运行，使用保存的稀疏可用率重建，不重新抽样。原参数
快照仍可能写 analysis_years=1，实际运行年数以 effective_config.json 为准。原运行文件
不会被改写。
"""
