"""공학 제약·데이터 누수·실패 표시의 회귀 검증. 수치는 모두 합성 자료다."""
import builtins
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS
from bjdmapping import load_bjd_master
from kepcoloader import load_kepco_hourly
from loadforecast import daily_matrix, forecast_load, backtest_forecast
from essoptimizer import (solve_peak_shaving, greedy_peak_shaving, size_ess,
                           evaluate_realized, compute_added_load, load_smp, public_priority, run_scenarios)
import mock_data


def history(growth=False):
    dates = pd.date_range("2025-09-01", "2025-12-31")
    return pd.DataFrame([{"bjd_code": "1100010100", "date": d, "hour": h,
                          "kw": 2 + (5 if h == 18 else 0) + (i / 10 if growth else d.dayofweek)}
                         for i, d in enumerate(dates) for h in range(24)])


def test_forecast_exact_weeks_and_no_future():
    data = history()
    pred = forecast_load(data, "2025-12-25", "1100010100")
    assert pred[18] == 10
    data.loc[data.date >= "2025-12-25", "kw"] = 99999
    pd.testing.assert_series_equal(pred, forecast_load(data, "2025-12-25", "1100010100"))


def test_growth_bias_is_reported():
    result = backtest_forecast(history(True), "1100010100", end_date="2025-12-24")
    assert result["n_obs"] == 56 * 24
    assert result["growth_bias"] == pytest.approx(1.75)
    assert result["mae_naive_seasonal"] > result["mae_persistence"]


def test_missing_history_and_holiday_do_not_become_oracle():
    data = history()
    with pytest.raises(ValueError, match="이력 부족"):
        forecast_load(data.iloc[:-1], "2026-01-07", "1100010100")
    with pytest.raises(ValueError, match="이력 부족"):
        forecast_load(data, "2025-12-25", "1100010100", holidays={"2025-12-25": "성탄절"})


@pytest.mark.parametrize("bad", ["duplicate", "negative", "hour"])
def test_matrix_rejects_bad_cells(bad):
    data = history()
    if bad == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif bad == "negative":
        data.loc[0, "kw"] = -1
    else:
        data.loc[0, "hour"] = 24
    with pytest.raises(ValueError):
        daily_matrix(data, "1100010100")


@pytest.mark.parametrize("price", [None, np.full(24, 100), np.linspace(-100, 200, 24)])
def test_lp_enforces_all_physical_constraints(price):
    load = np.full(24, 2.0)
    load[17:20] = 9
    p = {"power_kw": 4, "energy_kwh": 20}
    result = solve_peak_shaving(load, price, 5, p)
    assert result["feasible"], result["reason"]
    assert result["grid_kw"].min() >= -1e-6
    assert result["grid_kw"].max() <= 5 + 1e-6
    assert result["energy_kwh"].min() >= 2 - 1e-6
    assert result["energy_kwh"].max() <= 18 + 1e-6
    assert result["energy_kwh"][-1] == pytest.approx(10)
    assert result["charge_kw"].max() <= 4 + 1e-6
    assert result["discharge_kw"].max() <= 4 + 1e-6
    assert not ((result["charge_kw"] > 1e-6) & (result["discharge_kw"] > 1e-6)).any()


def test_initial_soc_makes_midnight_peak_infeasible():
    load = np.zeros(24)
    load[0] = 15
    p = {"power_kw": 10, "energy_kwh": 10 / .9 / .8, "eta_d": .9}
    assert not solve_peak_shaving(load, None, 5, p)["feasible"]


def test_candidate_capacity_uses_soc_and_daily_recharge():
    load = np.full(24, 2.0)
    load[18:20] = 10
    result = size_ess([load], 5, candidates=[[5, 10], [5, 20]])
    assert result["feasible"]
    assert result["params"]["energy_kwh"] == 20
    assert result["reference_kwh"] == pytest.approx(10 / .95 / .8)
    assert not size_ess([np.full(24, 10)], 5, candidates=[[100, 1000]])["feasible"]


def test_no_scipy_fallback_is_explicit(monkeypatch, caplog):
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise ImportError("시험용 미설치")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    load = np.full(24, 2.0)
    load[18] = 9
    result = solve_peak_shaving(load, None, 5, {"power_kw": 4, "energy_kwh": 20})
    assert result["feasible"] and result["method"] == "greedy"
    assert "scipy 미설치" in result["reason"] and "ESS 폴백" in caplog.text


def test_realized_clips_export_and_reports_terminal_error():
    load = np.full(24, 2.0)
    load[18] = 9
    plan = greedy_peak_shaving(load, 5, {"power_kw": 4, "energy_kwh": 20})
    actual = load.copy()
    actual[18] = 0
    result = evaluate_realized(plan, actual, np.full(24, 100), 5)
    assert result["grid_kw"].min() >= 0
    assert result["adjusted_kwh"] > 0 and result["terminal_error_kwh"] > 0


def test_smp_quarter_hour_and_missing_slot(tmp_path):
    path = tmp_path / "smp.csv"
    data = pd.DataFrame({"timestamp": pd.date_range("2025-12-25", periods=96, freq="15min"), "smp": np.arange(96)})
    data.to_csv(path, index=False)
    assert load_smp(path).iloc[0] == 1.5 and len(load_smp(path)) == 24
    data.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="4개"):
        load_smp(path)
    data.loc[0, "timestamp"] = pd.NaT
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="결측"):
        load_smp(path)


@pytest.mark.parametrize("n,u", [(-1, np.ones(24)), (1.5, np.ones(24)), (2, np.full(24, 1.1))])
def test_added_load_rejects_invalid_assumptions(n, u):
    with pytest.raises(ValueError):
        compute_added_load(u, n)


def test_scenarios_forecast_oracle_and_missing_price():
    data = history()
    ep = copy.deepcopy(DEFAULT_PARAMS["energy"])
    ep.update(evaluation_days=2, multipliers=[1.2], new_chargers=[0, 2])
    val = pd.DataFrame([backtest_forecast(data, "1100010100", end_date="2025-12-24")])
    results, schedules = run_scenarios(data, ["1100010100"], ep, val)
    assert len(results) == 8 and len(schedules) == 8 * 24
    assert results["energy_cost_difference_won"].isna().all()
    assert results["method"].eq("lp_peak").all()
    assert results["plan_feasible"].all()
    assert (schedules["actual_grid_kw"] >= -1e-6).all()
    for _, g in results.groupby(["date", "new_chargers"]):
        assert g.energy_kwh.nunique() == 1 and g.power_kw.nunique() == 1
    priority = public_priority(results)
    assert priority.public_evidence.eq("미확보·확인 필요").all()
    assert priority.review_group.eq("자료 보완 후 검토").all()
    changed = data.copy()
    changed.loc[changed.date >= ep["evaluation_start"], "kw"] *= 2
    revised, _ = run_scenarios(changed, ["1100010100"], ep, val)
    pd.testing.assert_series_equal(results.energy_kwh, revised.energy_kwh)
    pd.testing.assert_series_equal(results.target_kw, revised.target_kw)


def test_forecast_failure_stays_unavailable():
    data = history()
    data = data[~((data.date == "2025-12-18") & (data.hour == 0))]
    ep = copy.deepcopy(DEFAULT_PARAMS["energy"])
    ep.update(evaluation_start="2025-12-25", evaluation_days=1, calibration_days=6, multipliers=[1.2], new_chargers=[0])
    val = pd.DataFrame([backtest_forecast(data, "1100010100", end_date="2025-12-24")])
    result, _ = run_scenarios(data, ["1100010100"], ep, val)
    forecast = result[result["mode"] == "forecast"].iloc[0]
    assert not forecast.plan_feasible and forecast.method == "unavailable"
    assert result[result["mode"] == "oracle"].iloc[0].method == "lp_peak"


def _ai_pred(data, days, scale50=1.0, scale90=1.3):
    """시험용 AI 예측: 실측 × 배율(정답을 아는 예측이 아니라 입력 경로 시험용)."""
    d = data[data.date.isin(days)]
    return d.assign(q50=d.kw * scale50, q90=d.kw * scale90)[["bjd_code", "date", "hour", "q50", "q90"]]


def test_ess_uses_ai_input_and_compares_inputs_with_same_capacity():
    data = history()
    ep = copy.deepcopy(DEFAULT_PARAMS["energy"])
    ep.update(evaluation_start="2025-12-25", evaluation_days=2, multipliers=[1.2], new_chargers=[0, 3],
              ess={"forecast_input": "p90", "compare_inputs": True, "compare_new_chargers": 3})
    val = pd.DataFrame([backtest_forecast(data, "1100010100", end_date="2025-12-24")])
    days = pd.date_range("2025-12-25", periods=2)
    results, _ = run_scenarios(data, ["1100010100"], ep, val, ai_pred=_ai_pred(data, days))
    fc = results[results["mode"] == "forecast"]
    assert fc["forecast_input"].eq("p90").all()
    cmp = results[results["mode"] == "compare"]
    assert set(cmp["forecast_input"]) == {"baseline", "p50"} and cmp["new_chargers"].eq(3).all()
    for _, g in results[results["new_chargers"] == 3].groupby("date"):
        assert g["energy_kwh"].nunique() == 1                     # 입력만 바꾸고 용량은 같다
    from essoptimizer import ai_effect, ai_effect_summary
    effect = ai_effect(results)
    assert len(effect) == 1 and {"ai_effect", "over_discharge_kwh_p90", "n_common_days"} <= set(effect)
    assert effect.at[0, "n_common_days"] == 2
    assert not ai_effect_summary(effect).empty
    no_ai, _ = run_scenarios(data, ["1100010100"], ep, val)       # AI 없으면 기준 모델로 운전·표시
    assert no_ai.loc[no_ai["mode"] == "forecast", "forecast_input"].eq("baseline(AI 없음)").all()


def test_ai_effect_skips_zero_baseline_denominator_and_keeps_negative():
    from essoptimizer import ai_effect
    rows = []
    for code, base_exc, p90_exc in (("A", 0.0, 0.0), ("B", 10.0, 12.0)):
        for mode, inp, exc in (("forecast", "p90", p90_exc), ("compare", "baseline", base_exc),
                               ("compare", "p50", base_exc), ("oracle", "oracle", 0.0)):
            rows.append({"bjd_code": code, "multiplier": 1.2, "new_chargers": 3, "date": "2025-12-25", "mode": mode,
                         "forecast_input": inp, "method": "lp_peak", "exceedance_kwh": exc, "over_discharge_kwh": 0.0})
    out = ai_effect(pd.DataFrame(rows)).set_index("bjd_code")
    assert not out.at["A", "ai_effect_eligible"] and np.isnan(out.at["A", "ai_effect"])
    assert out.at["B", "ai_effect"] == pytest.approx(-0.2)          # AI가 더 나빠도 그대로 보고


def test_hourly_loader_preserves_days_and_rejects_monthly(tmp_path):
    rng = np.random.default_rng(42)
    reg = mock_data.build_regions(rng)
    mock_data.write_master(reg, tmp_path / "master.txt")
    mock_data.write_kepco(reg.iloc[:2], rng, tmp_path / "001.csv", tmp_path / "002.csv",
                          daily_period=("2025-12-01", "2025-12-31"))
    master = load_bjd_master(tmp_path / "master.txt", DEFAULT_COLUMNS["bjd"])
    result = load_kepco_hourly(tmp_path / "001.csv", DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"], master)
    assert result.date.nunique() == 31 and len(result) == 2 * 31 * 24
    raw = pd.read_csv(tmp_path / "001.csv", encoding="cp949")
    raw["조회기간"] = 202512
    raw.to_csv(tmp_path / "monthly.csv", index=False, encoding="cp949")
    with pytest.raises(ValueError, match="월 집계"):
        load_kepco_hourly(tmp_path / "monthly.csv", DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"], master)


def test_public_evidence_and_order_are_reproducible():
    rows = [{"bjd_code": code, "new_chargers": 0, "multiplier": 1.2, "mode": "forecast",
             "plan_feasible": True, "forecast_reliable": True, "terminal_error_kwh": 0,
             "baseline_exceedance_kwh": excess, "exceedance_kwh": 0, "peak_reduction_kw": 1}
            for code, excess in [("03", 0), ("02", 5), ("01", 5), ("04", 9)]]
    data = pd.DataFrame(rows)
    evidence = {c: {"status": "confirmed", "source": "합성 시험 증빙", "checked_on": "2026-09-17"}
                for c in ["01", "03", "04"]}
    data.loc[data.bjd_code == "04", "terminal_error_kwh"] = 1
    result = public_priority(data, evidence)
    assert result.bjd_code.tolist() == ["04", "01", "02", "03"]
    labels = result.set_index("bjd_code").review_group.to_dict()
    assert labels == {"01": "현장 우선 검토", "02": "자료 보완 후 검토",
                      "03": "현 시나리오에서 초과 없음", "04": "자료 보완 후 검토"}
