"""9단계 발표 3숫자 — V9 9절 "1지도 3숫자". 각 숫자 옆에 가정을 함께 적는다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_per_charger_multiple(access, share=0.10):
    """숫자 1 — 충전기 1기당 EV 수의 상위 10% 평균 ÷ 하위 10% 평균, 대상 수. 충전기 0기·EV 0대인 동네는 뺀다."""
    if "ev_per_charger" not in access:
        return np.nan, 0
    v = pd.to_numeric(access["ev_per_charger"], errors="coerce")
    v = v[np.isfinite(v) & (v > 0)].sort_values()
    if len(v) < 2:
        return np.nan, len(v)
    k = max(1, int(np.ceil(len(v) * share)))
    return float(v.tail(k).mean() / v.head(k).mean()), len(v)


def _energy_at(results, n_new, multiplier, codes):
    g = results[(results["mode"] == "forecast") & (results["new_chargers"] == n_new)
                & np.isclose(results["multiplier"], multiplier)]
    if codes is not None:
        g = g[g["bjd_code"].astype(str).isin(codes)]
    if g.empty:
        return None
    return {"before": g["baseline_exceedance_kwh"].mean(), "after": g["exceedance_kwh"].mean(),
            "cost": g["energy_cost_difference_won"].mean(), "n": g["bjd_code"].nunique()}


def headline_table(access, results, params, utilization_scale, codes=None):
    """codes 를 주면 그 법정동만 집계한다(PNG 는 소표본 법정동을 뺀 목록을 넘긴다)."""
    n_new = int(params["scenario_new_chargers"])
    base_m = float(params["base_multiplier"])
    rows = []
    if access is not None:
        multiple, n = ev_per_charger_multiple(access)
        rows.append({"번호": "숫자 1", "내용": "충전기 1기당 전기차 대수의 지역 간 배수(상위10%÷하위10% 평균)",
                     "값": multiple, "범위": "",
                     "가정": f"충전소를 법정동 중심점 최근접에 배정(폴리곤 공간조인 아님) · 대상 {n}곳(충전기·EV 0 제외)"})
        km = pd.to_numeric(access.get("nearest_charger_km"), errors="coerce").dropna()
        if len(km):
            rows.append({"번호": "보조", "내용": "최근접 공용충전기 거리 중앙값 / 90퍼센타일(km)", "값": float(km.median()),
                         "범위": f"p90 {km.quantile(0.9):.2f}", "가정": "법정동 중심점 기준 근사(격자점 아님)"})
    if results is not None:
        assume = f"가정: 증설 {n_new}대, 상한 배율 {base_m}, 이용률 배율 {utilization_scale}"
        at = _energy_at(results, n_new, base_m, codes)
        multipliers = sorted(float(m) for m in pd.to_numeric(results["multiplier"], errors="coerce").dropna().unique())
        ranged = {m: _energy_at(results, n_new, m, codes) for m in multipliers}
        ranged = {m: v for m, v in ranged.items() if v}
        if at:
            def span(key):
                vals = [v[key] for v in ranged.values() if np.isfinite(v[key])]
                return f"상한 {min(ranged)}~{max(ranged)}: {min(vals):.1f}~{max(vals):.1f}" if vals else ""
            n_regions = f" · {at['n']}곳 · 지역·일 평균"
            rows.append({"번호": "숫자 2", "내용": "ESS 미적용 평균 상한 초과 kWh", "값": at["before"],
                         "범위": span("before"), "가정": assume + n_regions})
            rows.append({"번호": "숫자 3", "내용": "ESS 적용 후 잔여 초과 kWh", "값": at["after"],
                         "범위": span("after"), "가정": assume + n_regions})
            rows.append({"번호": "숫자 3", "내용": "SMP 기준 비용 차이(원, ESS 적용 시)", "값": at["cost"],
                         "범위": span("cost"), "가정": assume + n_regions + " · SMP 없으면 빈 값"})
    if not rows:
        raise ValueError("8-C 접근성과 8-B ESS 결과가 모두 없음")
    return pd.DataFrame(rows)