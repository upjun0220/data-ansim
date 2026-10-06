"""실데이터 규모 대비 — CAN 적재 메모리 절감, 벡터화한 점유 시간, 거주지 추정 성능, 분석 지역 청크 필터."""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import canloader
import kepcoloader
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS


def _occupancy_reference(sessions):
    """예전 행 단위 구현 — 벡터화 결과와 대조용."""
    occ = np.zeros(24)
    for row in sessions.itertuples():
        start, end = pd.Timestamp(row.start), pd.Timestamp(row.end)
        if end <= start:
            continue
        for hour in pd.date_range(start.floor("h"), end.floor("h"), freq="h"):
            occ[hour.hour] += max(0, (min(end, hour + pd.Timedelta(hours=1)) - max(start, hour)).total_seconds()) / 3600
    return occ


def test_vectorized_occupancy_matches_row_loop():
    rng = np.random.default_rng(3)
    start = pd.Timestamp("2023-03-01") + pd.to_timedelta(rng.integers(0, 30 * 24 * 60, 400), unit="min")
    dur = pd.to_timedelta(rng.choice([0, 5, 37, 60, 61, 125, 600, 1439], 400), unit="min")
    s = pd.DataFrame({"start": start, "end": start + dur})
    s.loc[:4, "end"] = s.loc[:4, "start"].dt.floor("h") + pd.Timedelta(hours=2)   # 정시에 끝나는 세션
    np.testing.assert_allclose(canloader._occupancy(s), _occupancy_reference(s), atol=1e-9)
    assert canloader._occupancy(s.iloc[0:0]).sum() == 0


def test_infer_home_is_fast_for_thousands_of_vehicles():
    rng = np.random.default_rng(0)
    n_veh, n_nights, per = 3000, 20, 8                                  # 48만 밤 기록
    vid = np.repeat(np.arange(n_veh), n_nights * per)
    night = np.tile(np.repeat(np.arange(n_nights), per), n_veh)
    home_lat = 37.45 + rng.uniform(0, 0.25, n_veh)
    df = pd.DataFrame({"vehicle_id": vid,
                       "time": pd.Timestamp("2023-03-01 22:00") + pd.to_timedelta(night, unit="D")
                       + pd.to_timedelta(np.tile(np.arange(per) * 15, n_veh * n_nights), unit="min"),
                       "lat": home_lat[vid], "lon": 127.0, "charging": False, "soc": 50.0})
    t0 = time.perf_counter()
    home = canloader.infer_home(df, DEFAULT_PARAMS["can"])
    assert len(home) == n_veh and (home["nights"] == n_nights).all()
    np.testing.assert_allclose(home.sort_values("vehicle_id")["home_lat"], home_lat, atol=1e-6)
    assert time.perf_counter() - t0 < 30                                 # 예전 차량별 루프는 이 규모에서 수 분


def test_can_loader_drops_far_rows_caps_rows_and_codes_ids(tmp_path):
    rows = []
    for v, (lat, lon) in enumerate([(37.55, 126.98), (35.10, 129.03), (37.60, 127.05)]):   # 서울 · 부산 · 서울
        for k in range(10):
            rows.append({"발생시간": f"2023-03-01 10:{k:02d}:00", "차종_식별번호": f"VEH-{v}",
                         "충전중여부": "1", "배터리상태_SOC": 50 + k, "위도": lat, "경도": lon})
    rows.append({"발생시간": "2023-03-01 11:00:00", "차종_식별번호": "VEH-9", "충전중여부": "0",
                 "배터리상태_SOC": 40, "위도": "", "경도": ""})                                       # 좌표 없음 → 남김
    pd.DataFrame(rows).to_csv(tmp_path / "can.csv", index=False, encoding="utf-8-sig")
    params = copy.deepcopy(DEFAULT_PARAMS["can"])
    df = canloader.load_can_m(tmp_path / "can.csv", DEFAULT_COLUMNS, params)
    assert len(df) == 21 and df["vehicle_id"].dtype == "int32" and df["vehicle_id"].nunique() == 3
    assert df["lat"].dtype == "float32" and "범위 밖 좌표 10행 제외" in df.attrs["read_note"]
    capped = canloader.load_can_m(tmp_path / "can.csv", DEFAULT_COLUMNS, dict(params, max_rows=12))
    assert len(capped) == 12 and "중단" in capped.attrs["read_note"]


def test_observed_dates_scan_is_reused_until_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "k.csv"
    pd.DataFrame({"조회기간": ["2025030101", "2025030201"]}).to_csv(path, index=False)
    calls = []
    real = kepcoloader.read_columns
    monkeypatch.setattr(kepcoloader, "read_columns", lambda *a, **k: calls.append(1) or real(*a, **k))
    cols, params = {"period": "조회기간"}, {"chunksize": 10}
    first = kepcoloader.scan_observed_dates(path, cols, params)
    again = kepcoloader.scan_observed_dates(path, cols, params)
    assert first["date"].tolist() == again["date"].tolist() == ["20250301", "20250302"] and len(calls) == 1
    pd.DataFrame({"조회기간": ["2025030101", "2025030201", "2025030301"]}).to_csv(path, index=False)   # 파일이 바뀌면 다시 훑음
    assert kepcoloader.scan_observed_dates(path, cols, params)["date"].tolist()[-1] == "20250303" and len(calls) == 2


def test_hourly_loader_drops_missing_hours_instead_of_stopping(tmp_path):
    import mock_data
    from bjdmapping import load_bjd_master
    rng = np.random.default_rng(42)
    reg = mock_data.build_regions(rng)
    mock_data.write_master(reg, tmp_path / "master.txt")
    mock_data.write_kepco(reg.iloc[:2], rng, tmp_path / "001.csv", tmp_path / "002.csv", daily_period=("2025-12-01", "2025-12-10"))
    master = load_bjd_master(tmp_path / "master.txt", DEFAULT_COLUMNS["bjd"])
    raw = pd.read_csv(tmp_path / "001.csv", encoding="cp949", dtype=str)
    raw.loc[[5, 30], "시간대별 사용량(kWh)"] = ["*", ""]                    # 마스킹·빈 칸 2시간
    raw.to_csv(tmp_path / "gap.csv", index=False, encoding="cp949")
    out = kepcoloader.load_kepco_hourly(tmp_path / "gap.csv", DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"], master)
    assert len(out) == 2 * 10 * 24 - 2 and out.attrs["missing_hours"] == 2 and out["kw"].notna().all()   # 0으로 채우지 않음
    raw.loc[: len(raw) // 2, "시간대별 사용량(kWh)"] = "*"                      # 절반이 결측이면 멈춤
    raw.to_csv(tmp_path / "bad.csv", index=False, encoding="cp949")
    import pytest
    with pytest.raises(ValueError, match="결측·마스킹"):
        kepcoloader.load_kepco_hourly(tmp_path / "bad.csv", DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"], master)
    raw.loc[0, "시간대별 사용량(kWh)"] = "-1"
    raw.loc[1:, "시간대별 사용량(kWh)"] = "1.0"
    raw.to_csv(tmp_path / "neg.csv", index=False, encoding="cp949")
    with pytest.raises(ValueError, match="음수"):
        kepcoloader.load_kepco_hourly(tmp_path / "neg.csv", DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"], master)


def test_kepco_chunks_without_analysis_region_are_skipped(tmp_path):
    hours = [f"{h:02d}:00" for h in range(1, 25)]
    rows = [["20250301", "부산광역시", "해운대구", "우동", 5] + [1.0] * 24 for _ in range(30)] + \
           [["20250301", "서울특별시", "종로구", "청운동", 5] + [2.0] * 24 for _ in range(3)]
    pd.DataFrame(rows, columns=["조회기간", "시도", "시군구", "읍면동", "고객호수"] + hours).to_csv(
        tmp_path / "k.csv", index=False, encoding="utf-8-sig")
    params = dict(copy.deepcopy(DEFAULT_PARAMS["kepco"]), sido=["서울특별시"], chunksize=24 * 10)   # 청크 = wide 10행
    agg = kepcoloader.load_kepco_monthly_hour(tmp_path / "k.csv", DEFAULT_COLUMNS["kepco"], params, max_chunks=1)
    assert set(agg["sido"]) == {"서울"} and agg["kwh"].sum() == 2.0 * 24 * 3   # 앞 3청크(부산)를 건너뛰고 서울을 읽음


def test_copy_path_from_server_root_is_found_from_subfolder(tmp_path, monkeypatch):
    from common import locate, read_header, resolve_csv_path
    (tmp_path / "import_data").mkdir()
    (tmp_path / "team" / "nb").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "import_data" / "x.csv", index=False)
    monkeypatch.chdir(tmp_path / "team" / "nb")                               # 노트북은 두 단계 아래에서 돈다
    assert resolve_csv_path("import_data/x.csv") == tmp_path / "import_data" / "x.csv"
    assert read_header("import_data/x.csv")[0] == ["a"]
    assert locate("import_data") == tmp_path / "import_data" and not locate("nope.csv").exists()


def test_absent_days_are_reported_and_optionally_zero_filled(tmp_path):
    """실데이터 1차(10/6): 많은 동네가 검증 56일 중 하루만 완전 — 날짜 행 자체가 없는 날. 진단하고, 옵션으로 0을 채운다."""
    import mock_data
    from bjdmapping import load_bjd_master
    rng = np.random.default_rng(1)
    reg = mock_data.build_regions(rng)
    mock_data.write_master(reg, tmp_path / "master.txt")
    mock_data.write_kepco(reg.iloc[:2], rng, tmp_path / "001.csv", tmp_path / "002.csv", daily_period=("2025-12-01", "2025-12-10"))
    master = load_bjd_master(tmp_path / "master.txt", DEFAULT_COLUMNS["bjd"])
    raw = pd.read_csv(tmp_path / "001.csv", encoding="cp949", dtype=str)
    day = raw["조회기간"].str[:8]
    second = raw["읍면동"] == raw["읍면동"].unique()[1]
    raw[~(day.isin(["20251203", "20251207"]) & second)].to_csv(tmp_path / "gap.csv", index=False, encoding="cp949")  # 둘째 동네 이틀 빠짐
    params = copy.deepcopy(DEFAULT_PARAMS["kepco"])
    out = kepcoloader.load_kepco_hourly(tmp_path / "gap.csv", DEFAULT_COLUMNS["kepco"], params, master)
    assert out.attrs["absent_day_share"] == pytest.approx(2 / 20) and out.attrs["filled_days"] == 0
    filled = kepcoloader.load_kepco_hourly(tmp_path / "gap.csv", DEFAULT_COLUMNS["kepco"], dict(params, fill_missing_days=True), master)
    assert filled.attrs["filled_days"] == 2 and len(filled) == 2 * 10 * 24
    assert filled.groupby("bjd_code")["date"].nunique().eq(10).all() and (filled["kw"] >= 0).all()


def test_updated_import_names_take_priority(tmp_path):
    """같은 이름은 다시 반입할 수 없어 2026까지 늘린 SMP·기온은 11.csv·12.csv — 있으면 5.csv·9.csv 대신 쓴다."""
    import fieldday1
    for n in ("5.csv", "9.csv", "12.csv"):
        (tmp_path / n).write_text("x\n", encoding="utf-8")
    assert fieldday1._import_name(tmp_path, "weather") == "12.csv"
    assert fieldday1._import_name(tmp_path, "smp") == "5.csv"                 # 11.csv 가 없으면 옛 이름
    assert fieldday1._import_name(tmp_path, "font") is None


def test_small_units_merge_within_hdong_and_detail_marks_members():
    import headline
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from detail_from_export import check_page

    cust = pd.Series({"a": 10, "b": 1, "c": 1, "d": 2, "e": 2, "f": 1, "g": 5, "h": 2}, dtype=float)
    hdong = pd.DataFrame({"hdong_code": ["H1"] * 3 + ["H2"] * 2 + ["H3"] + ["H4"] * 2 + ["H9"],
                          "bjd_code": list("abcdefgh") + ["b"], "weight": [1.0] * 8 + [0.1]})
    groups = kepcoloader.small_unit_groups(cust, hdong, 3)
    # 소표본끼리 모자라면 일반 동(a·g)에, 모이면 그들끼리(d), 혼자 남은 소표본(f)은 그대로
    assert groups == {"b": "a", "c": "a", "e": "d", "h": "g"}
    mh = pd.DataFrame({"bjd_code": list("abc"), "mi": 1, "hour": 0, "kwh": [5.0, 1.0, 2.0], "n_days": 30,
                       "cust_sum": [10.0, 1.0, 1.0], "cust_min": [10.0, 1.0, 1.0], "region_name": "x"})
    m = kepcoloader.merge_units(mh, groups, ["mi", "hour"], kepcoloader.MH_AGGS)
    assert m[["bjd_code", "kwh", "cust_min"]].values.tolist() == [["a", 8.0, 12.0]]

    codes = ["1111010100", "1111010200", "1111010300", "1111010400"]
    pri = pd.DataFrame({"bjd_code": codes, "rank": [1, np.nan, np.nan, 2], "PriorityScore": [0.9, np.nan, np.nan, 0.5],
                        "risk_raw": [1.2, np.nan, np.nan, 0.8],
                        "risk_status": ["assessed", "missing", "missing", "assessed"], "robust_top": [True, False, False, False]})
    g = {codes[1]: codes[0], codes[2]: codes[3]}
    detail = headline.detail_table(pri, groups=g)
    assert detail["위험상태"].tolist() == ["A", "G", "G", "A"] and detail["묶음"].tolist() == [codes[0], codes[0], codes[3], codes[3]]
    cmin = pd.Series({codes[0]: 5.0, codes[3]: 1.0})                      # 대표 0400 은 묶은 뒤에도 소표본
    (_, png), = headline.detail_pages(detail, cmin, 3, rows=10)
    assert png["위험상태"].tolist()[:4] == ["A", "G", "G", "S"]
    body, problems = check_page(png.astype(object).where(png.notna(), "—").astype(str))
    assert problems == [] and body["묶음"].tolist() == [codes[0], codes[0], codes[3], codes[3]]
