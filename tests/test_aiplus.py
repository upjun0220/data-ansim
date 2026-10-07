"""8-A+ AI 강화 — 합성 자료로 동작·누설·보정만 확인한다(합성 자료라 AI 성능 근거가 아니다)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aiplus  # noqa: E402
from config import DEFAULT_PARAMS  # noqa: E402


def _hourly(n_codes=12, days=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for i in range(n_codes):
        code = f"11{110 + 30 * (i % 2)}{10100 + 100 * i:05d}"[:10]
        level = rng.uniform(5, 50)
        night = i % 3 == 0
        shape = np.array([1.5 if (h >= 21 or h <= 5) == night else 0.7 for h in range(24)])
        ar = 0.0
        for d in dates:
            ar = 0.7 * ar + rng.normal(0, 0.15)
            week = 0.8 if d.dayofweek >= 5 else 1.0
            kw = np.maximum(level * shape * week * (1 + ar) * (1 + d.dayofyear / 1000) + rng.normal(0, 1, 24), 0)
            rows += [(code, d, h, kw[h], 5.0) for h in range(24)]
    return pd.DataFrame(rows, columns=["bjd_code", "date", "hour", "kw", "cust"])


def test_ai_plus_runs_and_calibrates_p90():
    hourly = _hourly()
    params = dict(copy.deepcopy(DEFAULT_PARAMS["energy"]), evaluation_start="2025-09-29",
                  forecast=dict(DEFAULT_PARAMS["forecast"], train_days=150),
                  forecast_plus=dict(DEFAULT_PARAMS["forecast_plus"], max_iter=60, importance_rows=3000),
                  min_calib_peak_kw=1.0)
    calib = hourly[hourly["date"] < "2025-09-29"].groupby("bjd_code")["kw"].max()
    res = aiplus.run_ai_plus(hourly, params, None, calib, None, visible=set(hourly["bjd_code"].unique()[:8]))
    h = res["hourly"].set_index(["기간", "범위"])
    assert h.loc[("평가", "전체"), "n_obs"] > 1000 and "skill_ai_plus" in h
    assert 0.8 <= h.loc[("평가", "전체"), "p90cov_ai_plus_conformal"] <= 0.97          # 컨포멀 보정 후 목표 0.9 근처
    assert h.loc[("평가", "전체"), "mae_ai_plus"] <= 1.2 * h.loc[("평가", "전체"), "mae_persistence"]   # 어제값에서 출발
    assert set(res["peak"]["기간"]) == {"검증", "평가"} and res["alerts"]["전체"]["방법"].nunique() == 3
    assert set(res["agg"]["단위"]) == {"구", "서울 합계"}
    assert len(res["importance"]) == len(aiplus.FEATURES) - 1 - len(aiplus.LIVING)   # 기온·생활인구 입력 없음 → 그 특성만 빠짐
    assert abs(res["importance"]["비중"].sum() - 1) < 1e-6
    assert len(res["type_table"]) == 4 and res["types"].between(1, 4).all()
    pre = aiplus.preregistered(res)
    assert pre.iloc[0]["구분"] == "대표(숫자 B)" and pre.iloc[0]["판정"] in ("본문 주장 유지", "보조 근거로만", "AI 주장 철회 — 제목 변경")


def test_panel_features_use_only_previous_days():
    hourly = _hourly(n_codes=2, days=60)
    m = {c: g.pivot(index="date", columns="hour", values="kw") for c, g in hourly.groupby("bjd_code")}
    target = pd.Timestamp("2025-02-20")
    panel = aiplus.build_panel(m, [target])
    code = next(iter(m))
    row = panel[(panel["bjd_code"] == code) & (panel["hour"] == 7)].iloc[0]
    day = m[code]
    assert np.isclose(row["lag2_kw"], day.loc[target - pd.Timedelta(days=2), 7], rtol=1e-5)
    assert np.isclose(row["yday_peak_kw"], day.loc[target - pd.Timedelta(days=1)].max(), rtol=1e-5)
    assert np.isclose(row["roll7_kw"], day.loc[target - pd.Timedelta(days=7):target - pd.Timedelta(days=1), 7].mean(), rtol=1e-5)
    assert (panel["src_max_date"] < panel["date"]).all()


def test_conformal_offset_hits_target_coverage():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 2000)
    off = aiplus.conformal_offset(e, 0.9)
    assert abs((e <= off).mean() - 0.9) < 0.01


def test_living_population_features_use_week_old_values_and_enter_model():
    hourly = _hourly()
    codes = hourly["bjd_code"].unique()
    dates = pd.date_range("2024-12-01", "2025-10-31")
    rng = np.random.default_rng(3)
    living = pd.DataFrame([(d, c, 1000 + 100 * (d.dayofweek >= 5) + rng.normal(0, 10), 2000.0, 1500.0)
                           for d in dates for c in codes], columns=["date", "bjd_code", "lp_night", "lp_day", "lp_eve"])
    f = aiplus.living_features(living)
    c, d = codes[0], pd.Timestamp("2025-06-15")
    lv = living.set_index(["bjd_code", "date"])["lp_night"]
    want = lv[(c, d - pd.Timedelta(days=7))] / lv.loc[c].loc[d - pd.Timedelta(days=35):d - pd.Timedelta(days=8)].mean()
    got = f.set_index(["bjd_code", "date"]).loc[(c, d), "lp_night_r"]
    assert np.isclose(got, want)                                          # d−7 값 ÷ d−35~d−8 평균(누설 없음)
    params = dict(copy.deepcopy(DEFAULT_PARAMS["energy"]), evaluation_start="2025-09-29",
                  forecast=dict(DEFAULT_PARAMS["forecast"], train_days=150),
                  forecast_plus=dict(DEFAULT_PARAMS["forecast_plus"], max_iter=40, importance_rows=2000),
                  min_calib_peak_kw=1.0)
    res = aiplus.run_ai_plus(hourly, params, living=living)
    assert {"생활인구 야간(1주 전·4주 대비)", "생활인구 야간÷주간(1주 전)"} <= set(res["importance"]["특성"])
