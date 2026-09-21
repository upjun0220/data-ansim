"""v11 기본 경로 — 상권 단계 비활성·SHC 없음·서울 한정·기온/공휴일 입력. 수치는 전부 합성이다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_data  # noqa: E402
import weatherloader  # noqa: E402
from common import keep_region  # noqa: E402
from config import DEFAULT_PARAMS, deep_merge, resolve_region  # noqa: E402
from killcriteria import run_checks  # noqa: E402
from pipeline import channel_share, channel_share_table, run  # noqa: E402


@pytest.fixture(scope="module")
def v11_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v11")
    data = base / "data"
    mock_data.generate(data, seed=42)
    mock_data.generate_energy(data, seed=42)
    names = {"bjd_master": "bjd_master.txt", "emd_centroids": "bjd_centroids.csv", "kepco_001": "kepco_001.csv",
             "kepco_002": "kepco_002.csv", "kepco_hourly": "kepco_001_hourly.csv", "smp": "smp_mock.csv",
             "weather": "weather_mock.csv", "holidays": "holidays.yaml", "access_stations": "access_stations.csv",
             "ev_registration": "ev_registration.csv", "can_m": "can_m_individual.csv", "kep007": "kep007.csv",
             "ev_history": "ev_history.csv", "hdong_bjd": "hdong_bjd.csv"}
    cfg = {"paths": {**{k: str(data / v) for k, v in names.items()}, "out_dir": str(base / "out")},
           "params": {"energy": {"enabled": True}, "priority": {"enabled": True},
                      "kill": {"sample_rows": 200000, "compare_rows": 200000},
                      "scenario": {"seoul_base": None, "n_boot": 199}}}
    path = base / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return {"base": base, "data": data, "config": path}


@pytest.fixture(scope="module")
def v11_result(v11_env):
    return run(v11_env["config"])


def test_mock_has_no_shc_by_default(v11_env, tmp_path):
    assert not list(v11_env["data"].glob("shc*.csv"))
    mock_data.generate(tmp_path, seed=1, with_shc=True)
    assert (tmp_path / "shc001.csv").exists() and (tmp_path / "shc002.csv").exists()


def test_full_run_without_shc(v11_result):
    stages = v11_result["stages"]
    assert v11_result["status"] == "완료", stages.to_string()
    off = stages[stages["상태"] == "비활성"]
    assert len(off) == 1 and off.iloc[0]["비고"] == "상권 단계 비활성(v11)"
    assert not stages["단계"].isin(["2", "4", "4.5", "5", "6", "6.5", "7"]).any()
    assert stages["분석지역"].iloc[0] == "서울특별시(11)" and stages["지역단위수"].iloc[0] == 30
    out = Path(v11_result["out_dir"])
    for name in ("s1_channel_share", "s8_load_type", "s8_load_type_counts", "s0_external_inputs", "s8e_priority",
                 "s9_headline", "s8a_forecast_metrics", "s8a_risk", "s8a_risk_summary", "s8f_scenario",
                 "s8f_scenario_region", "s8f_beta"):
        assert (out / "csv" / f"{name}.csv").exists() and (out / "png" / f"{name}.png").exists(), name
    assert not (out / "csv" / "s8_quadrants.csv").exists() and not (out / "csv" / "s4_ydd_panel.csv").exists()
    assert v11_result["state"]["can"]["treated_excl_base"] is None
    assert v11_result["state"]["can"]["charging_shape"] is not None          # u(t)은 그대로
    assert (out / "png" / "s8f_scenario_2030_mid.png").exists()
    ai = v11_result["state"]["ai"]
    assert ai["model_used"] in {"lightgbm", "sklearn"} and ai["weather_mode"] == "forecast"
    risk = v11_result["state"]["risk"]
    assert (risk["status"] == "assessed").mean() > 0.9                     # 급증위험은 모든 법정동에 계산
    import loadscenario
    assert loadscenario.SCENARIO_TITLE == "시나리오(예측 아님), 충전기 추가 설치 없음 가정"   # 8-F 표·그림 제목에 붙임


def test_external_inputs_feed_calendar_and_weather_mode(v11_result):
    table = pd.read_csv(Path(v11_result["out_dir"]) / "csv" / "s0_external_inputs.csv", encoding="utf-8-sig")
    row = table.set_index("항목")
    assert row.at["기온 모드", "값"] == "forecast"
    assert "마감 뒤 발표" in row.at["예보 사용 가능 시각 수", "비고"]
    assert v11_result["state"]["weather_mode"] == "forecast"


def test_kill_criteria_without_shc(v11_env):
    table = run_checks(v11_env["config"], write=False).set_index("번호")
    assert not (table["판정"] == "오류").any(), table.to_string()
    assert (table.loc[[2, 3, 4, 6, 10], "판정"] == "해당 없음").all()
    assert "대조 가능" in table.at[16, "근거"]


def test_channel_share_is_ratio_not_sum():
    keys = {"bjd_code": ["A", "A"], "mi": [1, 1], "hour": [18, 19]}
    mh001 = pd.DataFrame({**keys, "kwh": [10.0, 0.0]})
    mh002 = pd.DataFrame({**keys, "kwh": [4.0, 3.0]})
    cells = channel_share(mh001, mh002)
    assert cells["ratio"].tolist() == [0.4]                      # 001 이 0 인 셀은 비율에서 뺀다
    assert "kwh" not in cells and channel_share_table(cells).at[0, "비율_중앙값"] == 0.4


def test_region_overrides_kepco_sido_alias():
    p = resolve_region(deep_merge(DEFAULT_PARAMS, {"kepco": {"sido": ["부산"]}}))
    assert p["kepco"]["sido"] == ["서울특별시"] and p["shc"]["sido"] == ["11"]
    alias = resolve_region(deep_merge(DEFAULT_PARAMS, {"kepco": {"sido": ["부산"]},
                                                      "region": {"sido_name": None, "sido_prefix": None}}))
    assert alias["kepco"]["sido"] == ["부산"]
    df = pd.DataFrame({"bjd_code": ["1111010100", "4111010100"]})
    assert keep_region(df, "11")["bjd_code"].tolist() == ["1111010100"] and len(keep_region(df, None)) == 2


def _weather(rows):
    return pd.DataFrame(rows, columns=["timestamp", "station_or_grid", "temp_obs_c", "temp_fcst_c", "fcst_issued_at"]) \
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]), fcst_issued_at=lambda d: pd.to_datetime(d["fcst_issued_at"]))


def test_forecast_uses_latest_issue_before_cutoff_only():
    w = _weather([("2025-12-25 18:00", "108", 1.0, 2.0, "2025-12-24 05:00"),
                  ("2025-12-25 18:00", "108", 1.0, 3.0, "2025-12-24 17:00"),     # 마감 전 최신 → 이것
                  ("2025-12-25 18:00", "108", 1.0, 9.0, "2025-12-24 20:00")])    # 마감 후 → 쓰면 안 됨
    f = weatherloader.forecast_temps(w, "18:00")
    assert f["temp_c"].tolist() == [3.0]
    weatherloader.assert_forecast_before_cutoff(f, "18:00")
    with pytest.raises(ValueError, match="정보 누설"):
        weatherloader.assert_forecast_before_cutoff(w.rename(columns={"temp_fcst_c": "temp_c"}), "18:00")


def test_weather_single_station_is_shared_and_multi_needs_coords():
    cent = pd.DataFrame({"bjd_code": ["A", "B"], "lat": [37.5, 37.6], "lon": [127.0, 127.0]})
    one = _weather([("2025-12-25 00:00", "108", 1.0, np.nan, None)])
    assert weatherloader.assign_stations(one, cent)["station_or_grid"].tolist() == ["108", "108"]
    two = _weather([("2025-12-25 00:00", "S1", 1.0, np.nan, None), ("2025-12-25 00:00", "S2", 2.0, np.nan, None)])
    with pytest.raises(ValueError, match="station_lat"):
        weatherloader.assign_stations(two, cent)
    two = two.assign(station_lat=[37.5, 37.61], station_lon=[127.0, 127.0])
    assert weatherloader.assign_stations(two, cent)["station_or_grid"].tolist() == ["S1", "S2"]


def test_holiday_types_come_only_from_listed_dates(tmp_path):
    types = weatherloader.classify_holidays(mock_data.MOCK_HOLIDAYS)
    assert types["2025-10-06"] == "long_holiday" and types["2025-10-03"] == "long_holiday"   # 10/3~10/9 연속
    assert types["2025-10-02"] == types["2025-10-10"] == "pre_post_holiday"
    assert types["2025-12-25"] == "holiday"                                                 # 목요일 단일
    extra = set(types) - set(mock_data.MOCK_HOLIDAYS)                                     # 새 휴일을 만들지 않음
    assert {"2025-10-02", "2025-10-10"} <= extra and all(types[d] == "pre_post_holiday" for d in extra)
    path = tmp_path / "holidays.yaml"
    path.write_text(weatherloader.dump_holidays({"2025-12-25": {"public_holiday": True, "name": "성탄절"}}),
                    encoding="utf-8")
    assert weatherloader.load_holidays(path) == {"2025-12-25": "holiday"}
    assert weatherloader.load_holidays(tmp_path / "없음.yaml") == {}
    path.write_text('{"2025-12-25": {"type": "festival"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="휴일유형"):
        weatherloader.load_holidays(path)


def test_fetch_holidays_build_keeps_api_dates_and_marks_source():
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_holidays import build
    raw = {"2025-12-25": {"name": "기독탄신일", "public_holiday": True},
           "2025-10-06": {"name": "추석", "public_holiday": True}}
    out = build(raw)
    assert out["2025-12-25"]["type"] == "holiday" and out["2025-12-25"]["source"].startswith("한국천문연구원")
    assert all(d in raw or e["type"] == "pre_post_holiday" for d, e in out.items())
