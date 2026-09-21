"""현장 실행 실패를 막는 T1~T5 회귀 검증."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import identification
import kepcoloader
import priorityscore
from config import DEFAULT_PARAMS
from outputs import setup_korean_font
from pipeline import (_energy_load_window, _exclude_small_activation, _exclude_small_cells, _kepco_export,
                      _period_customer_count, _warn_energy_calendar)


def test_mde_null_effect_mean_is_within_two_empirical_se():
    rng = np.random.default_rng(7)
    codes = [f"R{i:02d}" for i in range(14)]
    months = range(24300, 24316)
    panel = pd.DataFrame([
        {"bjd_code": code, "mi": month, "y_main": rng.normal(0, 0.08),
         "y_placebo": rng.normal(0, 0.08)}
        for code in codes for month in months
    ])
    activation = pd.DataFrame({
        "bjd_code": codes,
        "status": ["treated"] * 6 + ["never_treated"] * 8,
        "T_r_mi": [24306, 24307, 24308, 24306, 24307, 24308] + [np.nan] * 8,
    })
    params = dict(DEFAULT_PARAMS["identification"], k_min=-3, k_max=3, min_units_per_k=2)
    result = identification.estimate_mde(panel, activation, params, repetitions=20, seed=42)
    assert (result["가짜효과평균"].abs() <= 2 * result["경험_SE"]).all()


def test_energy_load_window_is_minimum_needed_period():
    params = dict(DEFAULT_PARAMS["energy"])
    start, end = _energy_load_window(params)                       # v11 기본: 평가 12-04부터 28일
    assert start == pd.Timestamp("2025-09-11")
    assert end == pd.Timestamp("2025-12-31")
    assert _energy_load_window(params, train_days=182)[0] == pd.Timestamp("2025-03-13")   # AI 학습기간 포함


def test_configured_font_file_is_registered():
    path = font_manager.findfont("DejaVu Sans")
    assert setup_korean_font(path) == font_manager.FontProperties(fname=path).get_name()


def test_energy_start_is_required_and_calendar_warns(caplog):
    params = dict(DEFAULT_PARAMS["energy"], evaluation_start=None)
    with pytest.raises(ValueError, match="evaluation_start"):
        _energy_load_window(params)
    with caplog.at_level("WARNING", logger="ripplemap"):
        _warn_energy_calendar(DEFAULT_PARAMS["energy"])
    assert "holidays가 비어 있음" in caplog.text and "12/24~1/1" in caplog.text


def test_kepco_small_cells_are_only_hidden_in_png():
    frame = pd.DataFrame({"bjd_code": ["A", "B"], "daily_kwh": [10.0, 20.0]})
    csv, png = _kepco_export(frame, pd.Series({"A": 2, "B": 3}), 3, ["daily_kwh"])
    assert csv["daily_kwh"].tolist() == [10.0, 20.0]
    assert csv["소표본억제"].tolist() == [True, False]
    assert np.isnan(png.loc[0, "daily_kwh"]) and png.loc[1, "daily_kwh"] == 20.0


# ---------------------------------------------------------------- R1: 소표본 억제 기준(max/min)

def test_period_customer_count_max_avoids_single_low_hour_suppression():
    """새벽 한 시간만 고객이 적어도(A) 낮에 고객이 많으면 최댓값 기준으로는 가리지 않는다."""
    mh = pd.DataFrame({
        "bjd_code": ["A", "A", "B", "B"],
        "mi": [1, 1, 1, 1],
        "hour": [3, 14, 3, 14],
        "cust_min": [1, 20, 1, 2],
    })
    max_count = _period_customer_count(mh, "cust_min", "max")
    min_count = _period_customer_count(mh, "cust_min", "min")
    assert max_count.to_dict() == {"A": 20, "B": 2}
    assert min_count.to_dict() == {"A": 1, "B": 1}
    _, png = _kepco_export(pd.DataFrame({"bjd_code": ["A", "B"], "daily_kwh": [10.0, 20.0]}),
                           max_count, 3, ["daily_kwh"])
    assert png.loc[0, "daily_kwh"] == 10.0    # A: 새벽 1명뿐이어도 낮 20명 있으면 억제 안 함
    assert np.isnan(png.loc[1, "daily_kwh"])  # B: 최댓값도 2명 미만 → 여전히 억제(보수적 유지)


def test_period_customer_count_rejects_unknown_basis():
    mh = pd.DataFrame({"bjd_code": ["A"], "mi": [1], "cust_min": [5]})
    with pytest.raises(ValueError, match="suppress_basis"):
        _period_customer_count(mh, "cust_min", "average")


# ---------------------------------------------------------------- R2: 전국 합계표·예시 그림 억제

def test_exclude_small_cells_removes_contribution_from_png_aggregate_only():
    daily = pd.DataFrame({
        "bjd_code": ["A", "A", "B", "B"],
        "mi": [1, 2, 1, 2],
        "kwh_per_day": [10.0, 10.0, 5.0, 5.0],
    })
    # A: 1월 대표 고객호수 2(소표본), 2월은 10(정상). B는 항상 정상.
    rep = pd.Series({("A", 1): 2, ("A", 2): 10, ("B", 1): 10, ("B", 2): 10})
    rep.index = pd.MultiIndex.from_tuples(rep.index, names=["bjd_code", "mi"])
    visible = _exclude_small_cells(daily, rep, 3, "test")
    assert set(visible[["bjd_code", "mi"]].itertuples(index=False, name=None)) == {("A", 2), ("B", 1), ("B", 2)}
    csv_total = daily.groupby("mi")["kwh_per_day"].sum()
    png_total = visible.groupby("mi")["kwh_per_day"].sum().reindex(csv_total.index).fillna(0)
    assert csv_total[1] == 15.0   # CSV(전체)는 A의 소표본 1월 값도 포함
    assert png_total[1] == 5.0    # PNG 집계는 A의 1월 기여분을 뺀다


def test_exclude_small_activation_drops_small_treated_regions_from_examples():
    act = pd.DataFrame({"bjd_code": ["A", "B", "C"], "status": ["treated", "treated", "never_treated"]})
    rep = pd.Series({"A": 1, "B": 10, "C": 10})
    eligible, n_excluded = _exclude_small_activation(act, rep, 3)
    assert n_excluded == 1
    assert set(eligible["bjd_code"]) == {"B", "C"}


# ---------------------------------------------------------------- R3: 계절 커버리지는 8-A 필터와 무관

def test_seasonal_coverage_scan_ignores_8a_date_window(tmp_path):
    dates = pd.date_range("2025-01-01", periods=370, freq="D")
    path = tmp_path / "hourly.csv"
    pd.DataFrame({"조회기간": [f"{d.strftime('%Y%m%d')}00" for d in dates]}).to_csv(
        path, index=False, encoding="utf-8")
    # 8-A 평가창처럼 12월만 보는 좁은 date_start/date_end가 섞여 있어도 무시하고 전체를 스캔해야 한다.
    narrow_8a_params = {"chunksize": 100, "date_start": "2025-12-01", "date_end": "2025-12-31"}
    scanned = kepcoloader.scan_observed_dates(path, {"period": "조회기간"}, narrow_8a_params)
    assert len(scanned) == 370
    coverage = priorityscore.seasonal_coverage(scanned)
    assert coverage["season_status"] == "검증 가능"
    assert coverage["missing_seasons"] == ""


def test_no_korean_font_falls_back_to_ascii_labels(tmp_path, monkeypatch):
    import outputs
    monkeypatch.setattr(outputs, "_ASCII", False)  # 테스트 뒤 원래 값으로 복원
    monkeypatch.setattr(outputs, "_find_korean_font", lambda font_path=None: None)
    writer = outputs.OutputWriter(tmp_path, font_fallback="ascii")
    assert writer.font is None and "대체 라벨" in writer.font_label
    for s in ("실행 요약 — 단계별 상태", "완료", "알 수 없는 문자열", "일평균 충전량 (kWh/일)"):
        assert not outputs._HANGUL.search(outputs.png_text(s)), s
    df = pd.DataFrame({"단계": ["8-E"], "상태": ["완료"], "bjd_code": ["1171010900"]})
    _csv, png = writer.table(df, "fb", "실행 요약")
    assert png.is_file()
    assert outputs.OutputWriter(tmp_path, font_fallback="none").font_label == "없음"
    assert outputs.png_text("완료") == "완료"  # fallback=none 이면 한글 그대로

def test_output_min_cell_count_overrides_can_alias():
    from config import _resolve_min_cell_count, deep_merge
    only_can = _resolve_min_cell_count(deep_merge(DEFAULT_PARAMS, {"can": {"min_cell_count": 5}}))
    assert only_can["output"]["min_cell_count"] == 5
    both = _resolve_min_cell_count(deep_merge(DEFAULT_PARAMS, {"can": {"min_cell_count": 5}, "output": {"min_cell_count": 7}}))
    assert both["output"]["min_cell_count"] == both["can"]["min_cell_count"] == 7

def test_window_shortfall_clip_and_label():
    w = ["2025-03", "2025-09"]
    dec = 2025 * 12 + 11
    assert kepcoloader.window_shortfall(dec, w, 3) == 0            # 12월까지 있으면 사후 3개월 충분
    aug = 2025 * 12 + 7
    assert kepcoloader.window_shortfall(aug, w, 3) == 3            # 8월까지면 7~9월 후보가 못 채움
    assert kepcoloader.clip_window(w, aug) == ["2025-03", "2025-08"]
    assert kepcoloader.clip_window(w, dec) == w
    assert kepcoloader.activation_label("002") == "공용(사업자 채널) 충전 활성화"
    assert kepcoloader.activation_label("001") == "충전 활성화(전체)"
    assert kepcoloader.activation_label("002", "내 이름") == "내 이름"