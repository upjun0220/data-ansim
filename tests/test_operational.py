"""현장 실행 실패를 막는 T1~T5 회귀 검증."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import identification
from config import DEFAULT_PARAMS
from outputs import setup_korean_font
from pipeline import _energy_load_window, _kepco_export, _warn_energy_calendar


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
    start, end = _energy_load_window(params)
    assert start == pd.Timestamp("2025-10-02")
    assert end == pd.Timestamp("2025-12-31")


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
