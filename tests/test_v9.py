"""V9 접근성·결합점수 최소 회귀 검증. 수치는 전부 합성이다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equity_access import compute_2sfca
from priority_score import ahp_consistency_ratio, build_priority, combine_scores


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
