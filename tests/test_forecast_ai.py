"""v11 8-A AI 분위수 예측·급증 위험 검증. 수치는 전부 합성이며 성능 근거가 아니다."""
import builtins
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loadforecast as lf  # noqa: E402
from config import DEFAULT_PARAMS  # noqa: E402


def synthetic(codes=("A", "B", "C"), start="2025-06-01", days=150, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    rows = []
    for i, code in enumerate(codes):
        for j, d in enumerate(dates):
            level = (2 + i) * (1 + 0.002 * j) * (0.85 if d.dayofweek >= 5 else 1.0)
            for h in range(24):
                rows.append((code, d, h, level * (1 + 2 * np.exp(-((h - 19) / 2) ** 2)) * rng.lognormal(0, 0.05)))
    return pd.DataFrame(rows, columns=["bjd_code", "date", "hour", "kw"])


def test_features_do_not_see_the_forecast_day_or_later():
    data = synthetic(days=60)
    day = pd.Timestamp("2025-07-20")
    before = lf.build_features(lf._matrices(data), [day])
    future = data.copy()
    future.loc[future["date"] >= day, "kw"] *= 50          # 예측일 이후 값을 바꿔도
    after = lf.build_features(lf._matrices(future), [day])
    cols = ["lag1_kw", "lag7_kw", "baseline_kw"]
    pd.testing.assert_frame_equal(before[cols], after[cols])  # 특성은 그대로여야 한다
    lf.assert_no_leakage(before)


def test_leakage_guard_rejects_future_inputs():
    feats = lf.build_features(lf._matrices(synthetic(days=40)), [pd.Timestamp("2025-07-05")])
    with pytest.raises(ValueError, match="정보 누설"):
        lf.assert_no_leakage(feats.assign(src_max_date=feats["date"]))         # 당일 관측을 섞음
    late = feats.assign(temp_c=1.0, temp_issued_at=pd.Timestamp("2025-07-04 20:00"))
    with pytest.raises(ValueError, match="예보 발표"):
        lf.assert_no_leakage(late, "forecast", "18:00")
    lf.assert_no_leakage(late.assign(temp_issued_at=pd.Timestamp("2025-07-04 17:00")), "forecast", "18:00")
    lf.assert_no_leakage(late, "observed", "18:00")                            # 실측 참고 모델은 기온 검사 제외


def test_feature_baseline_matches_v10_baseline_including_holidays():
    data = synthetic(days=90)
    matrices = lf._matrices(data)
    holidays = {"2025-06-06": "holiday", "2025-07-01": "holiday", "2025-07-15": "holiday",
                "2025-08-01": "holiday", "2025-08-15": "holiday"}
    normal, holiday = pd.Timestamp("2025-08-20"), pd.Timestamp("2025-08-15")
    feats = lf.build_features(matrices, [normal, holiday], holidays).set_index(["bjd_code", "date", "hour"])
    for d in (normal, holiday):
        expected = lf.forecast_baseline(matrices["A"], d, 4, holidays).to_numpy()
        np.testing.assert_allclose(feats.loc[("A", d), "baseline_kw"].to_numpy(), expected)
    assert (feats.loc[("A", holiday), "holiday_type"] == 1).all()
    few = {"2025-08-15": "holiday"}                                            # 같은 유형 휴일 4개 미만 → 실패(NaN)
    assert lf.build_features(matrices, [holiday], few)["baseline_kw"].isna().all()


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_quantile_crossing_is_prevented():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    out = lf.predict_quantiles({0.5: (_Const(5.0), {}), 0.9: (_Const(3.0), {})}, frame, ["x"])
    assert (out["q90"] >= out["q50"]).all() and out["q90"].tolist() == [5.0, 5.0]


def test_model_fallback_order(monkeypatch):
    original = builtins.__import__

    def block(names):
        def fake(name, *args, **kwargs):
            if any(name == n or name.startswith(n + ".") for n in names):
                raise ImportError(f"시험용 미설치: {name}")
            return original(name, *args, **kwargs)
        return fake
    monkeypatch.setattr(builtins, "__import__", block(["lightgbm"]))
    assert lf.select_model("auto")[0] == "sklearn"
    monkeypatch.setattr(builtins, "__import__", block(["lightgbm", "sklearn"]))
    name, factory = lf.select_model("auto")
    assert name == "baseline" and factory is None
    with pytest.raises(ValueError):
        lf.select_model("xgboost")


def _risk_pred():
    # A 동네 4일: q90 피크 [12, 12, 8, 8], 기준 [12, 8, 12, 8], 실측 [12, 12, 12, 8], 상한 10(피크 10 × 1.0)
    rows = []
    for i, (ai, base, act) in enumerate([(12, 12, 12), (12, 8, 12), (8, 12, 12), (8, 8, 8)]):
        for h in range(24):
            peak = h == 19
            rows.append({"bjd_code": "A", "date": pd.Timestamp("2025-12-01") + pd.Timedelta(days=i), "hour": h,
                         "q50": 1.0, "q90": ai if peak else 1.0, "baseline_kw": base if peak else 1.0,
                         "actual_kw": act if peak else 1.0})
    return pd.DataFrame(rows)


def test_alert_precision_recall_by_hand():
    risk = lf.surge_risk(_risk_pred(), pd.Series({"A": 10.0}), [1.0]).iloc[0]
    assert risk["surge_risk"] == 0.5                           # 경보 2/4일
    assert risk["precision_ai"] == 1.0 and risk["recall_ai"] == pytest.approx(2 / 3)
    assert risk["precision_base"] == 1.0 and risk["recall_base"] == pytest.approx(2 / 3)
    assert risk["peak_ratio"] == pytest.approx(np.mean([12, 12, 8, 8]) / 10)
    summary = lf.risk_summary(pd.DataFrame([risk]))
    assert summary.at[0, "precision_ai"] == 1.0


def test_risk_axis_ignores_new_charger_assumption():
    pred, peak = _risk_pred(), pd.Series({"A": 10.0})
    small = lf.surge_risk(pred, peak, [1.0], {3: np.full(24, 0.5)}).iloc[0]
    big = lf.surge_risk(pred, peak, [1.0], {3: np.full(24, 5.0), 5: np.full(24, 9.0)}).iloc[0]
    assert small["surge_risk"] == big["surge_risk"] == 0.5     # 점수용 위험 축은 ΔL과 무관
    assert small["surge_risk_with_new_3"] < big["surge_risk_with_new_3"] == 1.0


def test_low_calibration_peak_is_not_assessed():
    risk = lf.surge_risk(_risk_pred(), pd.Series({"A": 0.5}), [1.2], min_calib_peak_kw=1.0).iloc[0]
    assert risk["status"] == "not_assessed_low_load" and pd.isna(risk.get("surge_risk"))


def test_holiday_baseline_failure_is_excluded_from_comparison():
    pred = _risk_pred()
    pred.loc[pred["date"] == "2025-12-02", "baseline_kw"] = np.nan
    risk = lf.surge_risk(pred, pd.Series({"A": 10.0}), [1.0]).iloc[0]
    assert risk["n_excluded_days"] == 1 and risk["n_common_days"] == 3
    assert risk["surge_risk"] == 0.5                          # AI 단독 지표는 모든 평가일


def test_ai_forecast_end_to_end_sklearn():
    pytest.importorskip("sklearn.ensemble")
    params = copy.deepcopy(DEFAULT_PARAMS["energy"])
    params.update(evaluation_start="2025-10-01", evaluation_days=7, holdout_weeks=2)
    fp = copy.deepcopy(DEFAULT_PARAMS["forecast"])
    fp.update(model="sklearn", train_days=60, max_iter=30, cv_folds=2,
              grid={"learning_rate": [0.1], "max_leaf_nodes": [15], "min_samples_leaf": [20]})
    ai = lf.run_ai_forecast(synthetic(days=130), dict(params, forecast=fp))
    assert ai["model_used"] == "sklearn" and (ai["eval"]["q90"] >= ai["eval"]["q50"]).all()
    overall = ai["metrics"].iloc[0]
    assert overall["bjd_code"] == "" and overall["n_days"] == 14 and 0 <= overall["p90_coverage"] <= 1
    assert set(ai["metrics"]["bjd_code"]) == {"", "A", "B", "C"}
    assert ai["train_days"][1] < ai["valid_days"][0] < "2025-10-01"                 # 학습 < 검증 < 평가


def test_lightgbm_path_when_installed():
    pytest.importorskip("lightgbm")
    assert lf.select_model("auto")[0] == "lightgbm"
