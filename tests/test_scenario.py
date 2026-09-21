"""v11 8-F 충전 부하 시나리오 검증. 수치는 전부 합성이며 예측이 아니다."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loadscenario as ls  # noqa: E402
import mock_data  # noqa: E402
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS  # noqa: E402


def test_years_ahead_counts_months():
    assert ls.years_ahead("2026-06", 2030) == 4.5
    assert ls.years_ahead("2026-06", 2028) == 2.5


def test_k_goal_bisection_hits_target_multiple():
    ev_now, g = np.array([100.0, 300.0, 50.0]), np.array([0.2, 0.1, 0.35])
    m_goal = 4_200_000 / 1_095_218
    k = ls.solve_k_goal(ev_now, g, 4.5, m_goal)
    assert ls.project_ev(ev_now, g, k, 4.5).sum() / ev_now.sum() == pytest.approx(m_goal, rel=1e-6)
    assert ls.solve_k_goal(ev_now, np.zeros(3), 4.5, m_goal) is None       # 증가율 0 → 해 없음
    assert ls.solve_k_goal(ev_now, g, 4.5, np.nan) is None


def test_beta_falls_back_to_one_when_unstable_and_keeps_stable_estimate():
    rng = np.random.default_rng(0)
    ev = pd.Series(rng.uniform(50, 500, 40))
    stable = ls.estimate_beta(ev ** 0.9 * rng.lognormal(0, 0.05, 40), ev, n_boot=199)
    assert not stable["fallback"] and stable["beta"] == pytest.approx(0.9, abs=0.1) and stable["sensitivity"] == []
    noise = ls.estimate_beta(pd.Series(rng.lognormal(0, 1, 40)), ev, n_boot=199)
    assert noise["fallback"] and noise["beta"] == 1.0 and noise["sensitivity"] == [0.8, 1.2]
    steep = ls.estimate_beta(ev ** 3.0, ev, bounds=(0.3, 2.0), n_boot=199)
    assert steep["fallback"] and "범위" in steep["reason"]


def test_history_mapping_splits_by_weight_filters_fuel_and_reports_failure(tmp_path):
    reg = mock_data.build_regions(np.random.default_rng(42))
    mock_data.write_ev_history(reg, np.random.default_rng(1), tmp_path / "h.csv", tmp_path / "m.csv")
    hist = ls.load_ev_history(tmp_path / "h.csv", DEFAULT_COLUMNS["ev_history"], "전기", "11")
    assert hist["ev"].max() < 5000                                            # 휘발유 행 제외
    mapping = ls.load_hdong_bjd(tmp_path / "m.csv", DEFAULT_COLUMNS["hdong_bjd"])
    ev, fail = ls.map_to_bjd(hist, mapping, base_month="2026-06")
    at = ev[ev["ym"] == "2026-06"].set_index("bjd_code")["ev_count"]
    codes = reg["bjd_code"].tolist()
    assert at[codes[0]] / at[codes[1]] == pytest.approx(0.6 / 0.4)           # 한 행정동을 가중치로 나눔
    assert 0 < fail < 0.01                                                   # 대응표에 없는 행정동 1곳


def _inputs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"11110{101 + i:03d}00" for i in range(n)]
    months = pd.period_range("2023-01", "2026-08", freq="M").strftime("%Y-%m")
    rows = []
    for i, c in enumerate(codes):
        now, g = 100.0 + 50 * i, 0.15 + 0.03 * i
        for j, ym in enumerate(months):
            rows.append((c, ym, now * (1 + g) ** ((j - list(months).index("2026-06")) / 12)))
    ev = pd.DataFrame(rows, columns=["bjd_code", "ym", "ev_count"])
    curve = 1 + 2 * np.exp(-((np.arange(24) - 19) / 2) ** 2)
    curves = {c: curve * (1 + 0.1 * i) * rng.uniform(0.95, 1.05) for i, c in enumerate(codes)}
    calib = pd.Series({c: curves[c].max() * 0.95 for c in codes})
    access = pd.DataFrame({"bjd_code": codes, "chargers_assigned": [5, 10, 0, 3, 8, 2]})
    sp = copy.deepcopy(DEFAULT_PARAMS["scenario"])
    sp.update(n_boot=99, seoul_base=None)
    ep = copy.deepcopy(DEFAULT_PARAMS["energy"])
    return ev, curves, calib, access, sp, ep


def test_scenarios_run_high_skipped_without_goal_and_seoul_check_warns():
    ev, curves, calib, access, sp, ep = _inputs()
    scen, region, beta, notes = ls.run_scenarios_8f(ev, curves, calib, access, sp, ep, 1.2)
    assert set(scen["scenario"]) == {"저", "중", "고"} and not notes
    main = scen[(scen["beta_case"] == "main") & (scen["multiplier"] == 1.2)].set_index(["year", "scenario"])
    assert main.at[(2030, "고"), "ev_total_Y"] / main.at[(2030, "고"), "ev_total_now"] == pytest.approx(
        4_200_000 / 1_095_218, rel=1e-5)
    assert main.at[(2030, "중"), "n_over_limit"] >= main.at[(2028, "저"), "n_over_limit"]
    assert "over_only_high" in region and region["bjd_code"].str.startswith("11").all()
    no_goal = dict(sp, goal_national=None)
    scen2, _, _, notes2 = ls.run_scenarios_8f(ev, curves, calib, access, no_goal, ep, 1.2)
    assert "고" not in set(scen2["scenario"]) and any("고 시나리오 생략" in n for n in notes2)
    _, _, _, notes3 = ls.run_scenarios_8f(ev, curves, calib, access, dict(sp, seoul_base=118_967), ep, 1.2)
    assert any("정의 차이" in n for n in notes3)


def test_scenario_does_not_use_tree_model():
    import inspect
    assert "loadforecast" not in inspect.getsource(ls) and "predict" not in inspect.getsource(ls.run_scenarios_8f)
