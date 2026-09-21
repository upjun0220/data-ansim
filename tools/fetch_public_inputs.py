"""반입용 외부 공개데이터 받기 — 5.csv(SMP)·9.csv(기온). 인터넷이 되는 곳에서 한 번 돌린다(파이프라인은 호출하지 않음).

    python tools/fetch_public_inputs.py 2024-01-01 2025-12-31 [--out dist/import]

5.csv  전력거래소 EPSIS 시간별 SMP(육지). EPSIS 화면이 쓰는 공개 조회 주소를 월 단위로 부른다.
       EPSIS 의 "1시"는 00:00~01:00 구간이므로 시간 시작 시각(KST)으로 바꿔 timestamp,smp(원/kWh)로 저장한다.
9.csv  Open-Meteo(무료·키 없음, CC BY 4.0) — 서울 ASOS 108 지점 좌표(37.5714, 126.9658) 한 점.
       temp_fcst_c  = 기상청(KMA) 예보모델(kma_seamless)이 대상 시각 48시간 전에 낸 예보(previous_day2).
       fcst_issued_at = timestamp − 48시간(보수적 근사). 모든 값이 전날 18:00 마감 이전 발표분이 된다.
       temp_obs_c   = ERA5 재분석 기온(archive API). 기상청 ASOS 관측값이 아니다 — 실측 기온은 "observed" 참고 모델에만 쓴다.
       설계서의 1순위 출처(기상청 ASOS·동네예보 과거자료)는 로그인·API 키가 필요하다. 팀이 그 자료를 확보하면 같은 열로 바꿔 넣는다.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SMP_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SEOUL = {"latitude": 37.5714, "longitude": 126.9658}   # 서울 ASOS 108 지점 좌표
STATION = "seoul108"


def _get(url, data=None, timeout=60):
    body = urllib.parse.urlencode(data).encode() if data else None
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_smp(text):
    """EPSIS 응답(자바스크립트): 하루치 c1~c24(1~24시 구간) 값 다음에 gridData.push({"Date":"YYYY/MM/DD" ...})."""
    rows = []
    for block in re.split(r"gridData\.push\(", text)[:-1]:
        vals = dict(re.findall(r"c(\d+)\s*=\s*textFormmat\(\"([^\"]*)\"", block))
        rows.append(vals)
    dates = re.findall(r"gridData\.push\(\{\"Date\":\"(\d{4}/\d{2}/\d{2})\"", text)
    out = []
    for vals, date in zip(rows, dates):
        day = pd.Timestamp(date.replace("/", "-"))
        for h in range(1, 25):
            v = vals.get(str(h))
            if v not in (None, ""):
                out.append((day + pd.Timedelta(hours=h - 1), float(v)))
    return out


def fetch_smp(start, end):
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        b = max(pd.Timestamp(start), month.start_time).strftime("%Y%m%d")
        e = min(pd.Timestamp(end), month.end_time.normalize()).strftime("%Y%m%d")
        rows += parse_smp(_get(SMP_URL, {"beginDate": b, "endDate": e, "selYear": "", "selMonth": "",
                                         "selKind": "land", "locale": ""}))
    df = pd.DataFrame(rows, columns=["timestamp", "smp"]).drop_duplicates("timestamp").sort_values("timestamp")
    return df


def fetch_weather(start, end):
    q = {**SEOUL, "start_date": start, "end_date": end, "timezone": "Asia/Seoul"}
    fc = json.loads(_get(PREV_URL + "?" + urllib.parse.urlencode(
        {**q, "hourly": "temperature_2m_previous_day2", "models": "kma_seamless"}), timeout=180))
    ob = json.loads(_get(ARCHIVE_URL + "?" + urllib.parse.urlencode({**q, "hourly": "temperature_2m"}), timeout=180))
    f = pd.DataFrame({"timestamp": pd.to_datetime(fc["hourly"]["time"]),
                      "temp_fcst_c": fc["hourly"]["temperature_2m_previous_day2"]})
    o = pd.DataFrame({"timestamp": pd.to_datetime(ob["hourly"]["time"]), "temp_obs_c": ob["hourly"]["temperature_2m"]})
    df = o.merge(f, on="timestamp", how="outer").sort_values("timestamp")
    df.insert(1, "station_or_grid", STATION)
    df["fcst_issued_at"] = (df["timestamp"] - pd.Timedelta(hours=48)).where(df["temp_fcst_c"].notna())
    return df[["timestamp", "station_or_grid", "temp_obs_c", "temp_fcst_c", "fcst_issued_at"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--out", default="dist/import")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smp = fetch_smp(a.start, a.end)
    smp.to_csv(out / "5.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    w = fetch_weather(a.start, a.end)
    w.to_csv(out / "9.csv", index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.1f")
    print(f"5.csv {len(smp):,}행 {smp['timestamp'].min()}~{smp['timestamp'].max()}")
    print(f"9.csv {len(w):,}행 · 예보 결측 {int(w['temp_fcst_c'].isna().sum())} · 실측 결측 {int(w['temp_obs_c'].isna().sum())}")


if __name__ == "__main__":
    main()
