"""CAN × KEPCO_001 교차검증(canloader.kepco_cross_check) — 정답을 아는 합성 자료."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import canloader

PROFILE = np.array([1] * 7 + [2] * 10 + [6, 9, 6] + [2] * 4, float)   # 저녁 18시 피크


def _sessions(month, counts):
    """법정동별 counts 개 세션. 시작 시각은 PROFILE 비율대로, 길이 30분."""
    rows, rng = [], np.random.default_rng(0)
    hours = rng.choice(24, size=sum(counts.values()), p=PROFILE / PROFILE.sum())
    codes = [c for c, n in counts.items() for _ in range(n)]
    for code, h in zip(codes, hours):
        start = pd.Timestamp(f"{month}-10") + pd.Timedelta(hours=int(h))
        rows.append({"bjd_code": code, "start": start, "end": start + pd.Timedelta(minutes=30)})
    return pd.DataFrame(rows)


def _kepco(month, kwh_by_code):
    y, m = map(int, month.split("-"))
    return pd.DataFrame([{"bjd_code": c, "mi": y * 12 + m - 1, "hour": h, "kwh": total * PROFILE[h] / PROFILE.sum()}
                         for c, total in kwh_by_code.items() for h in range(24)])


def test_same_shape_and_rank_give_high_agreement():
    counts = {"1100000001": 400, "1100000002": 250, "1100000003": 120, "1100000004": 60}
    sessions = _sessions("2023-05", counts)
    kepco = _kepco("2023-05", {c: n * 10.0 for c, n in counts.items()})
    shape, summary = canloader.kepco_cross_check(sessions, kepco, min_n=3)
    got = dict(zip(summary["항목"], summary["값"]))
    assert float(got["시간대 모양 상관(피어슨)"]) > 0.9
    assert got["피크 시각 CAN / KEPCO"] == "18시 / 18시"
    assert float(got["법정동 순위 상관(스피어만)"]) == 1.0
    assert abs(shape["CAN_충전점유_비중"].sum() - 1) < 1e-9 and abs(shape["KEPCO_사용량_비중"].sum() - 1) < 1e-9
    assert got["비교 기간"].startswith("2023-05~2023-05")


def test_small_regions_excluded_and_no_overlap_reported():
    counts = {"1100000001": 50, "1100000002": 2}                     # 2건짜리 동은 공간 비교에서 빠진다
    sessions = _sessions("2023-05", counts)
    shape, summary = canloader.kepco_cross_check(sessions, _kepco("2023-05", {"1100000001": 5.0, "1100000002": 9.0}), 3)
    got = dict(zip(summary["항목"], summary["값"]))
    assert got["공간 비교 법정동 수(CAN 세션 3건 이상)"] == "1"
    assert got["법정동 순위 상관(스피어만)"].startswith("—")

    shape, summary = canloader.kepco_cross_check(sessions, _kepco("2025-05", {"1100000001": 5.0}), 3)
    assert shape is None and summary["값"].iloc[0].startswith("없음")
