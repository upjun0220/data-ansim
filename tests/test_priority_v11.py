"""v11 8-E 두 축 결합 점수·9-H 발표 숫자 검증. 수치는 전부 합성이다."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import priorityscore as ps  # noqa: E402
from config import DEFAULT_PARAMS  # noqa: E402
from headline import headline_table_v11  # noqa: E402


def _risk(surge, peak=None, status=None, codes=None):
    codes = codes or [f"R{i}" for i in range(len(surge))]
    rows = []
    for m in (1.1, 1.2, 1.3):
        for i, c in enumerate(codes):
            rows.append({"bjd_code": c, "multiplier": m, "status": (status or ["assessed"] * len(codes))[i],
                         "surge_risk": surge[i], "peak_ratio": (peak or [0.9] * len(codes))[i],
                         "tp_ai": 1, "alerts_ai": 2, "actual_ai": 2, "tp_base": 0, "alerts_base": 1, "actual_base": 2,
                         "surge_risk_with_new_3": 0.9})
    return pd.DataFrame(rows)


def _access(codes, values):
    return pd.DataFrame({"bjd_code": codes, "access_2sfca": values, "access_data_present": True,
                         "ev_per_charger": np.arange(1.0, len(codes) + 1)})


def _params(**kw):
    return {**copy.deepcopy(DEFAULT_PARAMS["priority"]), **kw}


def test_two_axes_with_min_axes_two_and_missing_reasons():
    codes = ["A", "B", "C", "D", "E"]
    risk = _risk([0.3, 0.1, 0.2, 0.0, 0.5], status=["assessed"] * 4 + ["not_assessed_low_load"], codes=codes)
    access = _access(["A", "B", "C", "E", "X"], [0.1, 0.4, 0.2, 0.3, 0.0])       # D 는 접근성 없음, X 는 위험 없음
    out = ps.build_priority_v11(risk, access, _params(risk_metric="surge_risk")).set_index("bjd_code")
    assert set(out.index[out["rank_eligible"]]) == {"A", "B", "C"}
    assert out.at["D", "missing_reason"] == "data_missing" and out.at["X", "missing_reason"] == "data_missing"
    assert out.at["E", "missing_reason"] == "not_assessed_low_load" and not out.at["E", "rank_eligible"]
    # 정규화는 분석 대상 집합 전체에서: A 위험 최대(1.0), 접근성은 X(0.0)가 최저라 A 형평성 = 1 − 0.1/0.4 = 0.75
    assert out.at["A", "PriorityScore"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.75)
    assert out.loc[out["robust_top"], "rank_eligible"].all()
    assert out.attrs["risk_metric_used"] == "surge_risk"


def test_risk_axis_is_unaffected_by_new_charger_layer():
    codes = list("ABCD")
    access = _access(codes, [0.1, 0.2, 0.3, 0.4])
    r1 = _risk([0.3, 0.1, 0.2, 0.0], codes=codes)
    r2 = r1.assign(surge_risk_with_new_3=0.0)                      # 증설 가정만 바꿈
    a = ps.build_priority_v11(r1, access, _params(risk_metric="surge_risk")).set_index("bjd_code")
    b = ps.build_priority_v11(r2, access, _params(risk_metric="surge_risk")).set_index("bjd_code")
    pd.testing.assert_series_equal(a["PriorityScore"], b["PriorityScore"])


def test_auto_switches_to_peak_ratio_when_surge_mostly_zero():
    codes = list("ABCD")
    risk = _risk([0.0, 0.0, 0.0, 0.2], peak=[0.8, 1.1, 0.9, 1.0], codes=codes)
    out = ps.build_priority_v11(risk, _access(codes, [0.1, 0.2, 0.3, 0.4]), _params())
    assert out.attrs["risk_metric_used"] == "peak_ratio" and out.attrs["share_zero_surge_risk"] == 0.75
    raw = out.set_index("bjd_code")["risk_raw"].reindex(list("ABCD"))
    assert raw.tolist() == [0.8, 1.1, 0.9, 1.0]
    fixed = ps.build_priority_v11(risk, _access(codes, [0.1, 0.2, 0.3, 0.4]), _params(risk_metric="surge_risk"))
    assert fixed.attrs["risk_metric_used"] == "surge_risk"


@pytest.mark.parametrize("w", [{"risk": 0.6, "equity": 0.6}, {"risk": -0.1, "equity": 1.1}, [0.5]])
def test_weights_must_sum_to_one_and_be_nonnegative(w):
    with pytest.raises(ValueError):
        ps.validate_weights_v11(w)
    assert ps.validate_weights_v11({"risk": 0.35, "equity": 0.65}).tolist() == [0.35, 0.65]


def test_headline_v11_rows_and_assumptions():
    codes = list("ABCD")
    risk = _risk([0.3, 0.05, 0.2, 0.0], codes=codes)
    effect = pd.DataFrame({"bjd_code": codes * 3, "multiplier": np.repeat([1.1, 1.2, 1.3], 4), "new_chargers": 3,
                           "ai_effect": [0.2, np.nan, 0.4, -0.1] * 3, "ai_effect_eligible": [True, False, True, True] * 3})
    region = pd.DataFrame({"bjd_code": codes * 2, "year": 2030, "beta_case": "main", "beta": 1.0,
                           "scenario": ["중"] * 4 + ["고"] * 4, "over_limit_1.2": [True, False, False, False, True, True, False, False]})
    metrics = pd.DataFrame([{"bjd_code": "", "mae_ai_p50": 1.0, "mae_baseline": 2.0, "p90_coverage": 0.9,
                             "model_used": "sklearn", "reference_only": False}])
    ai = {"model_used": "sklearn", "weather_mode": "forecast", "metrics": metrics}
    params = {**DEFAULT_PARAMS["priority"], "utilization_scale": 0.5, "risk_threshold": 0.1, "min_p90_coverage": 0.8}
    out = headline_table_v11(_access(codes, [0.1, 0.2, 0.3, 0.4]), risk, effect, region, ai, params)
    rows = out.set_index("번호")
    assert rows.loc["숫자 3(a)", "값"] == 2                        # 급증위험 ≥ 0.1: A, C
    assert rows.loc["숫자 3(b)", "값"] == 1 and rows.loc["숫자 3(b)", "비교"] == 2
    effect_row = out[out["내용"].str.contains("AI 효과")].iloc[0]
    assert effect_row["값"] == pytest.approx(np.mean([0.2, 0.4, -0.1])) and "대상 3/4곳" in effect_row["가정"]
    assert all("sklearn" in a for a in out.loc[out["번호"].str.startswith("숫자 2"), "가정"])
    bad = headline_table_v11(None, risk, None, None, {**ai, "metrics": metrics.assign(p90_coverage=0.5)}, params)
    two = bad["번호"] == "숫자 2"                                  # AI 상태 표시는 숫자 2 행에만
    assert bad.loc[two, "가정"].str.contains("AI 개선 없음").all()
    none = headline_table_v11(None, risk, None, None, None, params)
    assert none.loc[none["번호"] == "숫자 2", "가정"].str.contains("미실행").all()
