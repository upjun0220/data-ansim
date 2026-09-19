"""V9 접근성·결합점수 최소 회귀 검증. 수치는 전부 합성이다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equity_access import add_access_indicators, compute_2sfca
from headline import ev_per_charger_multiple, headline_table
from priority_score import ahp_consistency_ratio, axes_correlation, build_priority, combine_scores


def test_2sfca_is_reversed_into_equity_need():
    stations = pd.DataFrame({"lat": [37.5], "lon": [127.0], "chargers": [2]})
    demand = pd.DataFrame({"bjd_code": ["A", "B"], "lat": [37.5, 37.52], "lon": [127.0, 127.0],
                           "ev_count": [100, 100]})
    out = compute_2sfca(stations, demand)
    row = out.set_index("bjd_code")
    assert row.at["A", "access_2sfca"] > row.at["B", "access_2sfca"]
    assert row.at["A", "equity_need_norm"] < row.at["B", "equity_need_norm"]


def test_missing_axis_is_renormalized_not_zero_filled():
    energy = pd.DataFrame({"bjd_code": ["A", "B"], "safety_raw": [10, 0], "economy_raw": [0, 10],
                           "forecast_confidence": [1, 1], "plan_confidence": [1, 1]})
    access = pd.DataFrame({"bjd_code": ["A"], "access_2sfca": [0.2], "access_data_present": [True]})
    out = combine_scores(energy, access).set_index("bjd_code")
    assert out.at["B", "missing_axis_count"] == 1
    assert out.at["B", "PriorityScore"] == 0.5
    assert out.at["B", "confidence"] < out.at["A", "confidence"]


def test_priority_sensitivity_and_ahp_consistency():
    codes = list("ABCDE")
    rows = []
    for multiplier in (1.1, 1.2, 1.3):
        for i, code in enumerate(codes):
            rows.append({"bjd_code": code, "new_chargers": 0, "multiplier": multiplier, "mode": "forecast",
                         "baseline_exceedance_kwh": 10 - i, "energy_cost_difference_won": 5 + i,
                         "plan_feasible": True, "forecast_reliable": True})
    access = pd.DataFrame({"bjd_code": codes, "access_2sfca": np.arange(5, dtype=float),
                           "access_data_present": True})
    params = {"weights": [1 / 3] * 3, "base_multiplier": 1.2, "multipliers": [1.1, 1.2, 1.3],
              "scenario_new_chargers": 0, "weight_delta": 0.15, "top_share": 0.2, "robust_share": 0.9}
    result = build_priority(pd.DataFrame(rows), access, params)
    assert len(result) == 5 and result["PriorityScore"].between(0, 1).all()
    assert {"robust_top", "boundary", "confidence"} <= set(result)
    assert ahp_consistency_ratio(np.ones((3, 3))) == 0


def _ranking_fixture():
    # A~E: 8-B 평가 대상(안전·경제성 있음). X,Y: 접근성만 있는 미평가 동네. E는 접근성 자료 없음(data_missing).
    codes = list("ABCDE")
    rows = [{"bjd_code": c, "new_chargers": 0, "multiplier": m, "mode": "forecast",
             "baseline_exceedance_kwh": 10 - i, "energy_cost_difference_won": 5 + i * 2,
             "plan_feasible": True, "forecast_reliable": True}
            for m in (1.1, 1.2, 1.3) for i, c in enumerate(codes)]
    access = pd.DataFrame({"bjd_code": list("ABCDXY"), "access_2sfca": [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],
                           "access_data_present": True})
    params = {"weights": [1 / 3] * 3, "base_multiplier": 1.2, "multipliers": [1.1, 1.2, 1.3],
              "scenario_new_chargers": 0, "weight_delta": 0.15, "top_share": 0.2, "robust_share": 0.9}
    return pd.DataFrame(rows), access, params


def test_single_axis_area_is_not_ranked_or_robust_top():
    results, access, params = _ranking_fixture()
    out = build_priority(results, access, params).set_index("bjd_code")
    # X,Y는 형평성 축 하나뿐 — 재정규화하면 점수 1.0이 나오지만 순위 대상이 아니다.
    assert out.loc[["X", "Y"], "axes_present"].eq(1).all()
    assert not out.loc[["X", "Y"], "rank_eligible"].any()
    assert not out.loc[["X", "Y"], "robust_top"].any() and not out.loc[["X", "Y"], "boundary"].any()
    assert out.loc[["X", "Y"], "missing_reason"].eq("not_assessed_low_concentration").all()
    assert out.at["E", "missing_reason"] == "data_missing" and not out.at["E", "rank_eligible"]
    assert out.loc[out["robust_top"], "axes_present"].eq(3).all() and out["robust_top"].any()
    assert out["rank"].notna().sum() == out["rank_eligible"].sum() == 4


def test_min_axes_relaxes_ranking_pool():
    results, access, params = _ranking_fixture()
    out = build_priority(results, access, {**params, "min_axes": 1}).set_index("bjd_code")
    assert out.loc[["X", "Y", "E"], "rank_eligible"].all()


def test_axes_correlation_flags_redundant_axes():
    df = pd.DataFrame({"safety_raw": np.arange(10.0), "economy_raw": np.arange(10.0) * 3 + 1})
    corr = axes_correlation(df)
    assert corr["axes_redundant"] and corr["pearson"] > 0.99
    mixed = pd.DataFrame({"safety_raw": [1, 2, 3, 4, 5.0], "economy_raw": [3, 1, 5, 2, 4.0]})
    assert not axes_correlation(mixed)["axes_redundant"]

def test_access_indicators_use_nearest_centroid_and_headline_multiple():
    stations = pd.DataFrame({"lat": [37.5, 37.5], "lon": [127.0, 127.0], "chargers": [4, 2]})
    demand = pd.DataFrame({"bjd_code": ["A", "B", "C"], "lat": [37.5, 37.6, 37.8], "lon": [127.0] * 3,
                           "ev_count": [100, 100, 100]})
    out = add_access_indicators(compute_2sfca(stations, demand), stations, demand).set_index("bjd_code")
    assert out.at["A", "chargers_assigned"] == 6 and out.at["A", "ev_per_charger"] == 100 / 6
    assert out.loc[["B", "C"], "ev_per_charger"].isna().all()          # 충전기 0기 — 0으로 나누지 않는다
    assert out["nearest_charger_km"].is_monotonic_increasing
    assert not {"lat", "lon"} & set(out.reset_index().columns)          # 좌표 열은 표에 남기지 않는다
    multiple, n = ev_per_charger_multiple(pd.DataFrame({"ev_per_charger": np.arange(1.0, 21.0)}))
    assert n == 20 and multiple == pytest.approx((19.5) / (1.5))          # 상위 10%=19,20 · 하위 10%=1,2


def test_headline_table_carries_assumptions_and_range():
    rows = [{"bjd_code": c, "new_chargers": 3, "multiplier": m, "mode": "forecast",
             "baseline_exceedance_kwh": 10.0 * m, "exceedance_kwh": 2.0 * m, "energy_cost_difference_won": 100.0 * m}
            for c in ("A", "B") for m in (1.1, 1.2, 1.3)]
    params = {"scenario_new_chargers": 3, "base_multiplier": 1.2}
    out = headline_table(None, pd.DataFrame(rows), params, 0.5)
    first = out.iloc[0]
    assert first["값"] == pytest.approx(12.0) and "증설 3대" in first["가정"] and "이용률 배율 0.5" in first["가정"]
    assert "상한 1.1~1.3" in first["범위"]
    assert headline_table(None, pd.DataFrame(rows), params, 0.5, codes={"A"}).iloc[0]["가정"].endswith("1곳 · 지역·일 평균")

def test_filter_sido_code_keeps_only_listed_prefixes():
    from common import filter_sido_code
    chunk = pd.DataFrame({"sido_cd": ["11", "41", "11.0", "26"], "v": [1, 2, 3, 4]})
    assert filter_sido_code(chunk, "sido_cd", ["11"])["v"].tolist() == [1, 3]
    assert filter_sido_code(chunk, "sido_cd", ["11", "41"])["v"].tolist() == [1, 2, 3]
    assert len(filter_sido_code(chunk, "sido_cd", None)) == 4   # 설정 없으면 전체

def test_import_bundle_check_rules(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from check_import_bundle import check, collect
    for name in ("code.zip", "a.csv", "b.csv", "font.ttf"):
        (tmp_path / name).write_bytes(b"x")
    rows, problems = check(collect([tmp_path]))
    assert len(rows) == 4 and problems == []                 # 코드 1 + 데이터 2 + 폰트 1
    (tmp_path / "c.xlsx").write_bytes(b"x")
    assert any("허용되지 않는 형식" in p for p in check(collect([tmp_path]))[1])
    for i in range(6):
        (tmp_path / f"d{i}.csv").write_bytes(b"x")
    assert any("최대 10개" in p for p in check(collect([tmp_path]))[1])