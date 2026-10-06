"""CAN 고유 데이터 활용 — 거주지 추정·원정 충전·접근성 검증·시간 이동 여지. 정답을 아는 합성 자료."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import canloader
from config import DEFAULT_PARAMS

P = dict(DEFAULT_PARAMS["can"], centroid_max_km=5.0)
HOME_A, FAR = (37.5000, 127.0000), (37.5000, 127.0300)          # 약 2.6km 떨어짐
CENT = pd.DataFrame({"bjd_code": ["1100000001", "1100000002"], "lat": [37.5, 37.5], "lon": [127.0, 127.03]})


def _nights(vid, places, hour=23):
    return [{"vehicle_id": vid, "time": pd.Timestamp("2023-03-01") + pd.Timedelta(days=i, hours=hour),
             "lat": lat, "lon": lon, "charging": False, "soc": 50.0} for i, (lat, lon) in enumerate(places)]


def test_home_is_the_usual_night_parking_cell_and_unstable_vehicles_are_dropped():
    a = _nights("A", [HOME_A] * 5 + [FAR])                                   # 6밤 중 5밤 같은 곳
    b = _nights("B", [(37.5 + 0.01 * i, 127.0) for i in range(4)])            # 밤마다 다른 곳
    day = [{"vehicle_id": "A", "time": pd.Timestamp("2023-03-02 13:00"), "lat": 37.6, "lon": 127.1,
            "charging": True, "soc": 50.0}]                                  # 낮 기록은 무시
    home = canloader.infer_home(pd.DataFrame(a + b + day), P)
    assert home["vehicle_id"].tolist() == ["A"]
    assert abs(home.at[0, "home_lat"] - HOME_A[0]) < 1e-6 and home.at[0, "nights"] == 5


def test_late_night_drive_after_charging_does_not_move_home():
    recs = []
    for night in range(4):   # 밤마다 집에서 10개 기록(충전) 뒤 새벽 2시에 먼 곳 주행 기록 1개
        base = pd.Timestamp("2023-03-01") + pd.Timedelta(days=night, hours=21)
        recs += [{"vehicle_id": "E", "time": base + pd.Timedelta(minutes=10 * k), "lat": HOME_A[0], "lon": HOME_A[1],
                  "charging": True, "soc": 50.0} for k in range(10)]
        recs.append({"vehicle_id": "E", "time": base + pd.Timedelta(hours=5), "lat": FAR[0] + 0.01 * night, "lon": FAR[1],
                     "charging": False, "soc": 80.0})
    home = canloader.infer_home(pd.DataFrame(recs), P)
    assert home["vehicle_id"].tolist() == ["E"] and abs(home.at[0, "home_lon"] - HOME_A[1]) < 1e-6


def _session(vid, lat, lon, hour=20, soc=30.0):
    start = pd.Timestamp("2023-03-05") + pd.Timedelta(hours=hour)
    return {"vehicle_id": vid, "start": start, "end": start + pd.Timedelta(minutes=40), "lat": lat, "lon": lon,
            "soc_start": soc}


def test_away_share_no_home_vehicles_and_small_region_suppression():
    home = pd.DataFrame({"vehicle_id": ["A", "C", "D"], "home_lat": [37.5, 37.5, 37.5], "home_lon": [127.0, 127.0, 127.03],
                         "nights": 5})
    sessions = pd.DataFrame([_session("A", *HOME_A)] * 3 + [_session("A", *FAR)] * 2 +     # A: 원정 2/5
                            [_session("C", *FAR)] * 4 +                                    # C: 거주지 충전 없음
                            [_session("D", *FAR)])                                         # D: 다른 동네 거주
    reg, export, summary = canloader.away_charging(sessions, home, CENT, P, min_n=2)
    r = reg.set_index("bjd_code")
    assert r.at["1100000001", "거주추정_차량수"] == 2
    assert abs(r.at["1100000001", "원정충전_세션비율"] - 6 / 9) < 1e-9
    assert r.at["1100000001", "거주지충전없음_차량비율"] == 0.5                              # C 만
    assert np.isnan(export.set_index("bjd_code").at["1100000002", "거주지충전없음_차량비율"])   # 1대 → 가림
    assert "3 / 3대" in summary.loc[0, "값"]
    assert "vehicle_id" not in reg and "vehicle_id" not in export                            # 차량 단위 없음


def test_access_check_negative_when_low_access_means_more_away_charging():
    away = pd.DataFrame({"bjd_code": [f"11000000{i:02d}" for i in range(10)], "거주추정_차량수": 5,
                         "원정충전_세션비율": np.linspace(0.9, 0.1, 10), "거주지충전없음_차량비율": 0.0})
    access = pd.DataFrame({"bjd_code": away["bjd_code"], "access_2sfca": np.linspace(0.0, 1.0, 10)})
    check = canloader.away_access_check(away, access, min_n=3)
    assert check.attrs["rho"] < -0.99 and check.attrs["low"] > check.attrs["high"] and check.attrs["n"] == 10
    assert "계산 불가" in check["값"].iloc[3]                        # 보조 지표가 모두 0 이면 nan 대신 이유
    few = canloader.away_access_check(away.assign(거주추정_차량수=1), access, min_n=3)
    assert "검증 불가" in few.loc[0, "값"]


def test_soc_flexibility_counts_peak_sessions_and_handles_fraction_scale():
    s = pd.DataFrame([_session("A", *HOME_A, hour=19, soc=v) for v in (10, 60, 70, 90, 30)] +
                     [_session("A", *HOME_A, hour=10, soc=95)])                  # 피크 밖은 제외
    bands, summary = canloader.soc_flexibility(s, [18, 19, 20, 21], 50.0, min_n=3)
    assert summary.attrs["share"] == 0.6 and "5" in summary.loc[1, "값"]
    frac = s.assign(soc_start=s["soc_start"] / 100)
    assert canloader.soc_flexibility(frac, [19], 50.0, min_n=3)[1].attrs["share"] == 0.6
    assert canloader.peak_hours_from(None) == [18, 19, 20, 21]


def test_charging_flag_accepts_numeric_and_text_codes():
    """실데이터 1차(10/6): 충전 레코드 0건 → CAN 단계 AttributeError. '1.0' 같은 숫자 표기도 충전으로 인정한다."""
    import canloader
    tv = {"1", "Y", "TRUE", "충전중", "CONNECTED"}
    s = pd.Series(["1.0", "0.0", "1", "0", "Y", "N", "충전중", "", None, "2"])
    assert canloader.charging_flag(s, tv).tolist() == [True, False, True, False, True, False, True, False, False, True]


def test_can_reads_random_vehicles_not_file_head_when_over_max_rows(tmp_path):
    import copy
    from config import DEFAULT_COLUMNS, DEFAULT_PARAMS
    rows = [{"발생시간": f"2023-01-01 00:{m:02d}:00", "차종_식별번호": f"V{v:02d}", "충전중여부": 0,
             "배터리상태_SOC": 50, "위도": 37.5, "경도": 127.0} for v in range(20) for m in range(10)]   # 차량 순 정렬
    path = tmp_path / "can.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    params = copy.deepcopy(DEFAULT_PARAMS["can"]) | {"max_rows": 50, "chunksize": 37}
    cols = {"can_m": {k: v[0] for k, v in canloader.CAN_COLUMN_ALIASES.items()}}
    df = canloader.load_can_m(path, cols, params)
    per = df.groupby("vehicle_id").size()
    assert len(per) == 5 and (per == 10).all()                       # 차량 5대를 통째로(앞부분 50행이면 V00~V04)
    assert "무작위 표본" in df.attrs["read_note"] and "5/20대" in df.attrs["read_note"]
    head = canloader.load_can_m(path, cols, params | {"sample_by_vehicle": False})
    assert head.attrs["read_note"].endswith("(앞부분 표본)") and len(head) == 50
