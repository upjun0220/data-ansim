"""8-B: 정책적 충전 부하 상한에 대한 1일 ESS 배치 시뮬레이션.

변압기 용량·정전 확률을 추정하지 않는다. 열화·설비비·전기요금은 미포함.
초과 시에만 방전하는 제한된 운전 정책으로 충방전 동시 수행과 전력 판매를 배제한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import log

DEFAULT_ESS = {"power_kw": 14.0, "energy_kwh": 40.0, "eta_c": 0.95, "eta_d": 0.95,
               "soc_min": 0.1, "soc_max": 0.9, "soc_initial": 0.5}
DEFAULT_CANDIDATES = [[0, 0], [7, 20], [14, 40], [28, 80], [56, 160], [112, 320], [224, 640]]
TOL = 1e-6


def _inputs(load, target, ess_params):
    load = np.asarray(load, dtype=float)
    p = {**DEFAULT_ESS, **(ess_params or {})}
    if load.shape != (24,) or not np.isfinite(load).all() or (load < 0).any():
        raise ValueError("부하는 유한한 비음수 24시간 평균 kW여야 함")
    if not np.isfinite(target) or target < 0:
        raise ValueError("정책적 목표 상한은 유한한 비음수")
    if not all(np.isfinite(p[k]) for k in DEFAULT_ESS):
        raise ValueError("ESS 파라미터에 비유한 값")
    if p["power_kw"] < 0 or p["energy_kwh"] < 0 or not (0 < p["eta_c"] <= 1 and 0 < p["eta_d"] <= 1):
        raise ValueError("ESS 출력·용량·효율 범위 오류")
    if not 0 <= p["soc_min"] <= p["soc_initial"] <= p["soc_max"] <= 1 or p["soc_min"] == p["soc_max"]:
        raise ValueError("SOC 최소 < 최대, 초기 SOC는 그 범위 안이어야 함")
    return load, p


def _schedule(load, target, p, charge, discharge, method, reason=""):
    charge, discharge = np.asarray(charge), np.asarray(discharge)
    e0 = p["energy_kwh"] * p["soc_initial"]
    energy = np.r_[e0, e0 + np.cumsum(p["eta_c"] * charge - discharge / p["eta_d"])]
    grid = load + charge - discharge
    violation = max(0.0, float(np.max(grid - target)), float(np.max(-grid)),
                    float(np.max(p["soc_min"] * p["energy_kwh"] - energy)),
                    float(np.max(energy - p["soc_max"] * p["energy_kwh"])),
                    float(np.max(charge - p["power_kw"])), float(np.max(discharge - p["power_kw"])),
                    float(np.max(-charge)), float(np.max(-discharge)), abs(float(energy[-1] - e0)))
    simultaneous = bool(((charge > TOL) & (discharge > TOL)).any())
    return {"charge_kw": charge, "discharge_kw": discharge, "energy_kwh": energy, "grid_kw": grid,
            "feasible": violation <= TOL and not simultaneous, "max_violation": violation,
            "method": method, "reason": reason, "ess_params": p, "target_kw": float(target)}


def greedy_peak_shaving(load, target, ess_params):
    """초과분 방전·미래 방전에 필요한 양만 충전. 모든 제약과 종단 잔량을 사후 검증한다."""
    load, p = _inputs(load, target, ess_params)
    c, d = np.zeros(24), np.zeros(24)
    required = np.maximum(load - target, 0)
    e0 = p["energy_kwh"] * p["soc_initial"]
    energy = e0
    for t in range(24):
        if required[t] > 0:
            d[t] = min(required[t], p["power_kw"], max(0, energy - p["soc_min"] * p["energy_kwh"]) * p["eta_d"])
        else:
            desired = min(p["energy_kwh"] * p["soc_max"], e0 + required[t:].sum() / p["eta_d"])
            c[t] = min(p["power_kw"], max(0, target - load[t]), max(0, desired - energy) / p["eta_c"])
        energy += p["eta_c"] * c[t] - d[t] / p["eta_d"]
    return _schedule(load, target, p, c, d, "greedy")


def solve_peak_shaving(load, smp, target, ess_params, solver="auto"):
    """초과분은 고정 방전, 비첨두 재충전을 LP로 최적화한다. 최적화 실패는 명시한다.

    SMP 없음: 피크 최소화. scipy 없음: 그리디(비용 최적해가 아님).
    """
    load, p = _inputs(load, target, ess_params)
    if solver not in {"auto", "greedy"}:
        raise ValueError("solver는 auto 또는 greedy")
    price = None if smp is None else np.asarray(smp, dtype=float)
    if price is not None and (price.shape != (24,) or not np.isfinite(price).all()):
        raise ValueError("SMP는 유한한 24개 가격이어야 함")
    fallback_reason = "설정에 따른 그리디"
    if solver == "auto":
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            fallback_reason = f"scipy 미설치: {exc}"
        else:
            required = np.maximum(load - target, 0)
            if required.max() > p["power_kw"] + TOL:
                result = greedy_peak_shaving(load, target, p)
                result["reason"] = "필요 방전 출력이 후보 정격을 초과"
                return result
            # 변수: 충전 c[24], 순부하 피크 z. 방전은 목표 초과량으로 고정한다.
            lower = np.tril(np.ones((24, 24))) * p["eta_c"]
            cumulative_d = np.cumsum(required / p["eta_d"])
            e0 = p["energy_kwh"] * p["soc_initial"]
            a = np.vstack([np.c_[lower, np.zeros(24)], np.c_[-lower, np.zeros(24)],
                           np.c_[np.eye(24), -np.ones(24)]])
            b = np.r_[p["energy_kwh"] * p["soc_max"] - e0 + cumulative_d,
                      e0 - p["energy_kwh"] * p["soc_min"] - cumulative_d, required - load]
            equality = np.r_[np.full(24, p["eta_c"]), 0.0][None, :]
            objective = np.r_[np.zeros(24), 1.0] if price is None else np.r_[price, 0.0]
            bounds = [(0, float(min(p["power_kw"], max(0, target - v)))) for v in load] + [(0, target)]
            res = linprog(objective, A_ub=a, b_ub=b, A_eq=equality,
                          b_eq=[required.sum() / p["eta_d"]], bounds=bounds, method="highs")
            if res.success:
                result = _schedule(load, target, p, np.maximum(res.x[:24], 0), required,
                                   "lp_smp" if price is not None else "lp_peak")
                if result["feasible"]:
                    return result
            fallback_reason = f"LP 실패/검증 불합격: {res.message}"
    log.warning("ESS 폴백: %s", fallback_reason)
    result = greedy_peak_shaving(load, target, p)
    result["reason"] = fallback_reason
    return result


def size_ess(calibration_loads, target, ess_params=None, candidates=None, solver="auto"):
    """평가 시작 전 완전한 일별 곡선에서 후보를 검증. 전역 최적 용량을 주장하지 않는다.

    후보는 (에너지, 출력) 오름차순으로 선택한다. 각 날짜의 초기 SOC는 고정이며 일별 독립 실험이다.
    """
    curves = np.asarray(calibration_loads, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != 24 or len(curves) == 0:
        raise ValueError("사이징에는 완전한 과거 일별 24시간 곡선이 필요")
    for curve in curves:
        _inputs(curve, target, ess_params)
    pairs = DEFAULT_CANDIDATES if candidates is None else candidates
    if not pairs:
        raise ValueError("ESS 후보 목록 없음")
    p = {**DEFAULT_ESS, **(ess_params or {})}
    # 이 값은 당일 재충전 없는 경우의 에너지 참고값이며 최적 용량·필요조건으로 쓰지 않는다.
    reference = np.maximum(curves - target, 0).sum(axis=1).max() / (p["eta_d"] * (p["soc_max"] - p["soc_min"]))
    for power, energy in sorted(pairs, key=lambda x: (x[1], x[0])):
        candidate = {**p, "power_kw": float(power), "energy_kwh": float(energy)}
        checks = [solve_peak_shaving(curve, None, target, candidate, solver) for curve in curves]
        if all(check["feasible"] for check in checks):
            return {"feasible": True, "params": candidate, "reference_kwh": float(reference),
                    "calibration_days": len(curves), "reason": "열거 후보 중 에너지·출력 순 첫 실행가능 후보"}
    return {"feasible": False, "params": None, "reference_kwh": float(reference),
            "calibration_days": len(curves), "reason": "후보 범위에서 제약을 만족한 용량 없음"}


def evaluate_realized(schedule, actual, smp, target):
    """계획을 실측에 재현하되 SOC·역송 금지로 제한한 운전과 조정량을 별도 보고한다.

    예측 오차로 목표 초과·종단 SOC 불일치가 생기면 숨기지 않는다. 다음날 자동 연결은 하지 않는다.
    """
    actual, p = _inputs(actual, target, schedule["ess_params"])
    energy = p["energy_kwh"] * p["soc_initial"]
    c, d = np.zeros(24), np.zeros(24)
    for t in range(24):
        c[t] = min(schedule["charge_kw"][t], max(0, p["soc_max"] * p["energy_kwh"] - energy) / p["eta_c"])
        d[t] = min(schedule["discharge_kw"][t], actual[t], max(0, energy - p["soc_min"] * p["energy_kwh"]) * p["eta_d"])
        energy += c[t] * p["eta_c"] - d[t] / p["eta_d"]
    net = actual + c - d
    excess = np.maximum(net - target, 0)
    before = np.maximum(actual - target, 0)
    price = None if smp is None else np.asarray(smp, dtype=float)
    if price is not None and (price.shape != (24,) or not np.isfinite(price).all()):
        raise ValueError("평가 SMP 값 오류")
    adjustment = np.abs(c - schedule["charge_kw"]) + np.abs(d - schedule["discharge_kw"])
    # 과잉 방전: 실측 기준으로는 필요 없었던 방전(실측이 상한 아래인데 방전한 양). P90 운전의 "크게 잡아서 줄였다" 반론 대응.
    over_discharge = np.maximum(d - np.maximum(actual - target, 0), 0)
    return {"exceedance_kw": float(excess.max()), "exceedance_hours": int((excess > TOL).sum()),
            "over_discharge_kwh": float(over_discharge.sum()),
            "exceedance_kwh": float(excess.sum()), "baseline_exceedance_kwh": float(before.sum()),
            "peak_reduction_kw": float(actual.max() - net.max()),
            "energy_cost_difference_won": float(np.dot(price, d - c)) if price is not None else np.nan,
            "terminal_error_kwh": float(energy - p["energy_kwh"] * p["soc_initial"]),
            "adjusted_kwh": float(adjustment.sum()), "grid_kw": net, "charge_kw": c, "discharge_kw": d}


def compute_added_load(u_t, N_new, P_rated_charger=7.0):
    """u는 0~1의 가정 이용률. CAN 기반이어도 상대 모양이며 모집단 이용률이 아니다."""
    u = np.asarray(u_t, dtype=float)
    if u.shape != (24,) or not np.isfinite(u).all() or ((u < 0) | (u > 1)).any():
        raise ValueError("이용률 가정은 0~1의 24개 값")
    if not np.isfinite(N_new) or N_new < 0 or N_new != int(N_new) or not np.isfinite(P_rated_charger) or P_rated_charger <= 0:
        raise ValueError("증설 대수·정격출력 오류")
    return u * N_new * P_rated_charger


def utilization_profile(params, utilization_shape=None):
    """8-A·8-B 공용 u(t) = 시간대 모양 × 가정 배율. CAN 모양이 있으면 그것(상대 모양, 실제 이용률 아님)."""
    scale = float(params["utilization_scale"])
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("가정 이용률 배율은 0~1")
    shape = params["utilization_shape"] if utilization_shape is None else utilization_shape
    source = "가정 시간대 모양" if utilization_shape is None else "사전 CAN 상대 모양(실제 이용률 아님)"
    return np.asarray(shape, dtype=float) * scale, source


def policy_target(calibration_peak_kw, multiplier):
    """8-A·8-B 공용 목표 상한 = 교정기간 관측 최대 × 배율. 공학 기준·변압기 용량이 아닌 정책 가정."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("정책 상한 배수 오류")
    return float(calibration_peak_kw) * float(multiplier)


def load_smp(path):
    """정규화 CSV(timestamp,smp)만 수용. EPSIS 원본 열·단위·시간대는 반입 전 확인한다.

    KST의 시간 시작 시각, 원/kWh. 15분 자료는 네 관측값이 모두 있을 때만 시간 평균한다.
    """
    if not path:
        raise FileNotFoundError("SMP 경로 없음")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if not {"timestamp", "smp"} <= set(frame):
        raise ValueError("SMP 정규화 열 timestamp,smp 필요")
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    if ts.dt.tz is not None:
        raise ValueError("SMP는 KST로 변환한 뒤 시간대 없는 시각으로 입력")
    values = pd.to_numeric(frame["smp"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or ts.isna().any() or ts.duplicated().any() or len(ts) == 0:
        raise ValueError("SMP 중복·결측·비유한 값")
    s = pd.Series(values, index=pd.DatetimeIndex(ts)).sort_index()
    if (s.index.second != 0).any() or (s.index.microsecond != 0).any() or (s.index.nanosecond != 0).any():
        raise ValueError("SMP 시각은 정각 또는 15분 단위")
    if (s.index.minute == 0).all():
        return s
    if not np.isin(s.index.minute, [0, 15, 30, 45]).all():
        raise ValueError("SMP 시각은 정각 또는 15분 단위")
    counts = s.resample("h").count()
    if (counts != 4).any():
        raise ValueError("SMP 15분 자료는 시간당 4개 관측 필요")
    return s.resample("h").mean()


def public_priority(results, public_evidence=None, multiplier=1.2):
    """N_new=0 기준 잠정 검토표. 확인되지 않은 공공성은 점수 0으로 치환하지 않는다."""
    base = results[(results["new_chargers"] == 0) & np.isclose(results["multiplier"], multiplier)
                   & (results["mode"] == "forecast")].copy()
    if base.empty:
        raise ValueError("잠정 우선순위용 N_new=0·기준배수 결과 없음")
    rows = []
    evidence = {} if public_evidence is None else public_evidence
    for code, g in base.groupby("bjd_code"):
        e = evidence.get(str(code), {})
        confirmed = e.get("status") == "confirmed" and bool(e.get("source")) and bool(e.get("checked_on"))
        valid = bool(g["plan_feasible"].all() and g["forecast_reliable"].all()
                     and (g["terminal_error_kwh"].abs() <= TOL).all())
        excess = float(g["baseline_exceedance_kwh"].mean())
        group = "자료 보완 후 검토" if not valid or not confirmed else (
            "현장 우선 검토" if excess > TOL else "현 시나리오에서 초과 없음")
        rows.append({"bjd_code": str(code), "review_group": group,
                     "public_evidence": "확인" if confirmed else "미확보·확인 필요",
                     "baseline_exceedance_kwh": excess,
                     "residual_exceedance_kwh": g["exceedance_kwh"].mean(),
                     "peak_reduction_kw": g["peak_reduction_kw"].mean(), "calculation_valid": valid})
    out = pd.DataFrame(rows)
    # 공공 투자 순위가 아닌 부하 점검 순서. 동일 값은 법정동코드로 고정하며 공공성 우열을 의미하지 않는다.
    out = out.sort_values(["baseline_exceedance_kwh", "residual_exceedance_kwh", "bjd_code"],
                          ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "load_review_order", np.arange(1, len(out) + 1))
    return out


INPUTS = ("baseline", "p50", "p90")


def run_scenarios(hourly, regions, params, validation, smp=None, utilization_shape=None, ai_pred=None):
    """평가 시작일 이전에 상한·용량을 고정하고 동일 용량으로 예측 운전(forecast)과 oracle을 비교한다.

    params["ess"](v11, 없으면 V10 그대로 기준 모델 운전):
      forecast_input: "p90"(기본) | "p50" | "baseline" — forecast 행의 스케줄을 짜는 예측 입력.
      compare_inputs: True 면 증설 compare_new_chargers 대(기본 3)에서 나머지 입력도 같은 용량으로 돌려
        mode="compare" 행으로 남긴다(AI 효과 계산용). ai_pred(bjd_code,date,hour,q50,q90)가 없으면 AI 입력은
        쓸 수 없어 기준 모델로 운전하고 forecast_input 에 "baseline(AI 없음)"을 적는다.
    """
    from loadforecast import daily_matrix, forecast_day

    start = pd.Timestamp(params["evaluation_start"]).normalize()
    days = pd.date_range(start, periods=int(params["evaluation_days"]), freq="D")
    if len(days) == 0 or int(params["calibration_days"]) < 1:
        raise ValueError("평가일·교정일 수는 양수")
    ess = {"forecast_input": "baseline", "compare_inputs": False, "compare_new_chargers": 3, **params.get("ess", {})}
    chosen = ess["forecast_input"]
    if chosen not in INPUTS:
        raise ValueError(f"ess.forecast_input 은 {INPUTS} 중 하나 — 받은 값: {chosen!r}")
    wide = {}
    if ai_pred is not None and len(ai_pred):
        pred = ai_pred.assign(bjd_code=ai_pred["bjd_code"].astype(str), date=pd.to_datetime(ai_pred["date"]).dt.normalize())
        wide = {q: pred.pivot_table(index=["bjd_code", "date"], columns="hour", values=q).reindex(columns=range(24))
                for q in ("q50", "q90")}
    label = chosen
    if chosen != "baseline" and not wide:
        log.warning("ESS: AI 예측 없음 — forecast_input %s 대신 기준 모델로 운전", chosen)
        chosen, label = "baseline", "baseline(AI 없음)"
    u, shape_source = utilization_profile(params, utilization_shape)
    rows, schedules = [], []
    for region in regions:
        matrix = daily_matrix(hourly, region)
        calibration = matrix.reindex(pd.date_range(start - pd.Timedelta(days=int(params["calibration_days"])),
                                                   start - pd.Timedelta(days=1), freq="D"))
        if calibration.isna().any().any():
            raise ValueError(f"{region}: 교정기간 완전한 일별 곡선 부족")
        v = validation.loc[validation["bjd_code"].astype(str) == str(region)]
        if len(v) != 1 or "error" in v and pd.notna(v.iloc[0].get("error")):
            reliable = False
        else:
            v = v.iloc[0]
            reliable = bool(v["n_days"] >= params["min_validation_days"]
                            and v["mae_naive_seasonal"] <= v["mae_persistence"]
                            and abs(v["bias_actual_minus_forecast"]) <= params["max_relative_bias"] * max(calibration.to_numpy().mean(), TOL))

        def curve(inp, day):
            """(24시간 예측 또는 None, 실패 사유)."""
            if inp == "baseline":
                try:
                    return forecast_day(matrix, day, params["weeks"], params["holidays"]).to_numpy(), ""
                except ValueError as exc:
                    log.warning("%s 기준 모델 예측 실패: %s", region, exc)
                    return None, str(exc)
            key = (str(region), day)
            q = "q50" if inp == "p50" else "q90"
            if key not in wide[q].index or wide[q].loc[key].isna().any():
                return None, f"{day.date()}: AI {inp} 예측 없음"
            return wide[q].loc[key].to_numpy(float), ""

        for multiplier in params["multipliers"]:
            target = policy_target(calibration.to_numpy().max(), multiplier)
            for n_new in params["new_chargers"]:
                added = compute_added_load(u, n_new, params["charger_kw"])
                size = size_ess(calibration.to_numpy() + added, target, params["ess_params"],
                                params["candidates"], params["solver"])
                runs = [("forecast", chosen), ("oracle", "oracle")]
                if ess["compare_inputs"] and wide and n_new == int(ess["compare_new_chargers"]):
                    runs += [("compare", x) for x in INPUTS if x != chosen]
                for day in days:
                    actual = matrix.reindex([day]).iloc[0].to_numpy(float) + added
                    _inputs(actual, target, params["ess_params"])
                    price = None if smp is None else smp.reindex(pd.date_range(day, periods=24, freq="h")).to_numpy(float)
                    if price is not None and not np.isfinite(price).all():
                        log.warning("%s SMP 24시간 부족 — 해당 날짜 피크 목적함수", day.date())
                        price = None
                    chosen_ok = True
                    for mode, inp in runs:
                        if mode == "oracle":
                            load, error = actual, ""
                        else:
                            load, error = curve(inp, day)
                            load = None if load is None else load + added
                            if mode == "forecast":
                                chosen_ok = load is not None
                        reason = size["reason"] if not size["feasible"] else error
                        if load is None or not size["feasible"]:
                            # 실패 행도 미적용 기준값을 남기되 계획 실행가능/투자 검증으로 처리하지 않는다.
                            candidate = size["params"] if size["feasible"] else {
                                **params["ess_params"], "power_kw": 0, "energy_kwh": 0}
                            _, candidate = _inputs(actual, target, candidate)
                            plan = _schedule(actual, target, candidate, np.zeros(24), np.zeros(24), "unavailable")
                            plan["feasible"] = False
                            plan["method"] = "unavailable"
                            plan["reason"] = reason
                        else:
                            plan = solve_peak_shaving(load, price, target, size["params"], params["solver"])
                        realized = evaluate_realized(plan, actual, price, target)
                        key = {"bjd_code": str(region), "date": str(day.date()), "new_chargers": n_new,
                               "multiplier": multiplier, "mode": mode}
                        row = {**key, "forecast_input": label if mode == "forecast" else inp,
                               "target_kw": target, "plan_feasible": plan["feasible"],
                               "forecast_reliable": reliable and chosen_ok,
                               "power_kw": plan["ess_params"]["power_kw"], "energy_kwh": plan["ess_params"]["energy_kwh"],
                               "method": plan["method"], "reason": plan["reason"], "shape_source": shape_source,
                               "calibration_end": str((start - pd.Timedelta(days=1)).date()),
                               "smp_available": price is not None}
                        row.update({k: v for k, v in realized.items() if not isinstance(v, np.ndarray)})
                        rows.append(row)
                        if mode == "compare":
                            continue   # 비교 운전은 스케줄 표를 늘리지 않는다
                        for hour in range(24):
                            schedules.append({**key, "hour": hour, "plan_charge_kw": plan["charge_kw"][hour],
                                              "plan_discharge_kw": plan["discharge_kw"][hour],
                                              "plan_energy_kwh": plan["energy_kwh"][hour],
                                              "actual_grid_kw": realized["grid_kw"][hour],
                                              "actual_charge_kw": realized["charge_kw"][hour],
                                              "actual_discharge_kw": realized["discharge_kw"][hour]})
    if not rows:
        raise ValueError("ESS 시나리오 대상 또는 조합 없음")
    return pd.DataFrame(rows), pd.DataFrame(schedules)


def ai_effect(results):
    """법정동 × 상한 배율 × 증설 대수별 AI 효과 = 1 − 초과kWh(AI P90) / 초과kWh(기준 모델).

    같은 사전 선정 용량으로 기준 모델·P50·P90·oracle 을 운전해 실측에 재현한 초과 kWh 합으로 계산한다.
    모든 입력의 계획이 있는 날(공통일)만 합산하고(휴일 등 기준 모델 실패일 제외), 기준 모델 초과가 0인 동네는
    분모가 없어 ai_effect 를 비우고 ai_effect_eligible=False 로 둔다. 음수·0 근처여도 그대로 낸다.
    """
    runs = results[results["mode"].isin(["forecast", "compare", "oracle"])].copy()
    runs["input"] = np.where(runs["mode"] == "oracle", "oracle", runs["forecast_input"].astype(str).str.split("(").str[0])
    keys = ["bjd_code", "multiplier", "new_chargers"]
    have = runs.groupby(keys)["input"].transform(lambda s: len(set(INPUTS) - set(s)) == 0)
    runs = runs[have.astype(bool)]
    if runs.empty:
        return pd.DataFrame()
    ok = runs[runs["method"] != "unavailable"].groupby(keys + ["date"])["input"].nunique()
    common = ok[ok == len(INPUTS) + 1].reset_index()[keys + ["date"]]
    runs = runs.merge(common, on=keys + ["date"])
    if runs.empty:
        return pd.DataFrame()
    agg = runs.pivot_table(index=keys, columns="input", values=["exceedance_kwh", "over_discharge_kwh"], aggfunc="sum")
    agg.columns = [f"{v}_{i}" for v, i in agg.columns]
    out = agg.reset_index().merge(common.groupby(keys).size().rename("n_common_days").reset_index(), on=keys)
    base = out["exceedance_kwh_baseline"]
    out["ai_effect_eligible"] = base > TOL
    out["ai_effect"] = np.where(out["ai_effect_eligible"], 1 - out["exceedance_kwh_p90"] / base.where(base > TOL), np.nan)
    return out


def ai_effect_summary(effect):
    """상한 배율·증설 대수별 AI 효과 평균·범위와 대상 수(기준 모델 초과 0인 동네 제외)."""
    if effect.empty:
        return pd.DataFrame()
    g = effect.groupby(["multiplier", "new_chargers"])
    return pd.DataFrame({"n_regions": g.size(), "n_eligible": g["ai_effect_eligible"].sum(),
                         "ai_effect_mean": g["ai_effect"].mean(), "ai_effect_min": g["ai_effect"].min(),
                         "ai_effect_max": g["ai_effect"].max(),
                         "over_discharge_kwh_p90": g["over_discharge_kwh_p90"].sum(),
                         "over_discharge_kwh_baseline": g["over_discharge_kwh_baseline"].sum()}).reset_index()
