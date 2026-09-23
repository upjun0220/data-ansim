"""KEPCO 시간대별 가로(wide) 형식 회귀 — 현장 KEPCO_001 실제 헤더(조회기간 YYYYMMDD + 01:00~24:00). 합성 자료."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS
from bjdmapping import load_bjd_master
import kepcoloader
import mock_data

META = ["조회기간", "시도", "시군구", "읍면동", "고객호수"]


def _write_pair(tmp_path, label):
    """mock long(YYYYMMDDHH, 시각 0~23)과 같은 값을 하루 한 행·구간 끝 시각(1~24) 열로 펼친 wide 를 쓴다."""
    rng = np.random.default_rng(7)
    reg = mock_data.build_regions(rng)
    mock_data.write_master(reg, tmp_path / "master.txt")
    mock_data.write_kepco(reg.iloc[:2], rng, tmp_path / "long.csv", tmp_path / "002.csv",
                          daily_period=("2025-12-01", "2025-12-07"))
    raw = pd.read_csv(tmp_path / "long.csv", encoding="cp949", dtype=str)
    raw["일자"] = raw["조회기간"].str[:8]
    raw["시각"] = (raw["조회기간"].str[8:10].astype(int) + 1).map(label)
    keys = ["일자", "시도", "시군구", "읍면동"]
    wide = raw.pivot(index=keys, columns="시각", values="시간대별 사용량(kWh)").reset_index()
    cust = raw.assign(c=raw["고객호수"].astype(int)).groupby(keys)["c"].max().rename("고객호수").reset_index()
    wide = cust.merge(wide, on=keys).rename(columns={"일자": "조회기간"})
    wide[META + [label(h) for h in range(1, 25)]].to_csv(tmp_path / "wide.csv", index=False, encoding="utf-8-sig")
    return load_bjd_master(tmp_path / "master.txt", DEFAULT_COLUMNS["bjd"])


def _cols(**over):
    return {**copy.deepcopy(DEFAULT_COLUMNS["kepco"]), **over}


def _params(**over):
    return {**copy.deepcopy(DEFAULT_PARAMS["kepco"]), **over}


@pytest.mark.parametrize("label", [lambda h: f"{h:02d}:00", lambda h: f"시간대별 사용량{h:02d}"],
                         ids=["현장 01:00", "접두형"])
def test_wide_equals_long_with_default_config(tmp_path, label):
    master = _write_pair(tmp_path, label)
    header, _, _ = kepcoloader.read_header(tmp_path / "wide.csv")
    assert kepcoloader._layout(header, _cols(), "auto") == "wide"

    long_m = kepcoloader.load_kepco_monthly_hour(tmp_path / "long.csv", _cols(), _params())
    wide_m = kepcoloader.load_kepco_monthly_hour(tmp_path / "wide.csv", _cols(), _params())
    assert sorted(wide_m["hour"].unique()) == list(range(24))
    pd.testing.assert_series_equal(wide_m.groupby("hour")["kwh"].sum(), long_m.groupby("hour")["kwh"].sum())

    long_h = kepcoloader.load_kepco_hourly(tmp_path / "long.csv", _cols(), _params(), master)
    wide_h = kepcoloader.load_kepco_hourly(tmp_path / "wide.csv", _cols(), _params(), master)
    both = long_h.merge(wide_h, on=["bjd_code", "date", "hour"], suffixes=("_l", "_w"), validate="one_to_one")
    assert len(both) == len(long_h) == 2 * 7 * 24
    np.testing.assert_allclose(both["kw_w"], both["kw_l"])


def test_wide_hours_collapsing_to_same_hour_is_rejected(tmp_path):
    _write_pair(tmp_path, lambda h: f"{h:02d}:00")
    # 빈 접두 + 기본 정규식은 '01:00' 의 끝 '00' 을 시각으로 읽어 24개 열이 모두 0시가 된다 — 조용히 합치지 않는다.
    with pytest.raises(ValueError, match="같은 시각"):
        kepcoloader.load_kepco_monthly_hour(tmp_path / "wide.csv", _cols(kwh=""), _params(layout="wide"))


def test_wide_chunk_counts_melted_rows(tmp_path):
    _write_pair(tmp_path, lambda h: f"{h:02d}:00")
    agg = kepcoloader.load_kepco_monthly_hour(tmp_path / "wide.csv", _cols(), _params(chunksize=48), max_chunks=1)
    assert agg["n_days"].sum() == 48          # 청크 48행 = wide 2행 × 24시간(펼치기 전 48행이 아님)
