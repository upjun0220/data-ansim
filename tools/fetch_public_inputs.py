"""반입용 외부 공개데이터 받기 — 5.csv(SMP)·9.csv(기온). 인터넷이 되는 곳에서 한 번 돌린다(파이프라인은 호출하지 않음).

    python tools/fetch_public_inputs.py 2024-01-01 2025-12-31 [--out dist/import]

5.csv  전력거래소 EPSIS 시간별 SMP(육지). EPSIS 화면이 쓰는 공개 조회 주소를 월 단위로 부른다.
       EPSIS 의 "1시"는 00:00~01:00 구간이므로 시간 시작 시각(KST)으로 바꿔 timestamp,smp(원/kWh)로 저장한다.
14.csv 서울 공용 충전소(환경공단 충전소 정보 API, 이용자 제한·삭제 제외) — python tools/fetch_public_inputs.py --chargers
13.csv 서울 생활인구(OA-14991, 공공누리 1유형) 법정동·일 평균(0~6·9~17·18~23시) — python tools/fetch_public_inputs.py --living 2508-2605 --hdong 8.csv
9.csv  Open-Meteo(무료·키 없음, CC BY 4.0) — 서울 ASOS 108 지점 좌표(37.5714, 126.9658) 한 점.
       temp_fcst_c  = ECMWF IFS 0.25° 예보모델(ecmwf_ifs025)이 대상 시각 48시간 전에 낸 예보(previous_day2), 빈 시각만
                      JMA(jma_seamless)로 채운다. 기상청 모델(kma_seamless)은 Open-Meteo 에서 2026-04-12 이후가 없어
                      (2026-10-06 확인) 평가 기간(2026-05)을 못 채우므로, 기간 전체를 한 모델로 맞추려고 바꿨다.
       fcst_issued_at = timestamp − 48시간(보수적 근사). 모든 값이 전날 18:00 마감 이전 발표분이 된다.
       temp_obs_c   = ERA5 재분석 기온(archive API). 기상청 ASOS 관측값이 아니다 — 실측 기온은 "observed" 참고 모델에만 쓴다.
       설계서의 1순위 출처(기상청 ASOS·동네예보 과거자료)는 로그인·API 키가 필요하다. 팀이 그 자료를 확보하면 같은 열로 바꿔 넣는다.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
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


ELEV_URL = "https://api.open-meteo.com/v1/elevation"


def fetch_elevation(lat, lon, batch=100):
    """Open-Meteo 표고(Copernicus DEM GLO-90, 약 90m 격자, CC BY 4.0) — 점마다 표고(m). 한 번에 100점.
    v12.1: 지형 보정(8-C 경사 반영 2SFCA)은 공개 좌표만 쓰므로 센터 LX DEM 대신 센터 밖에서 이 값으로 계산한다."""
    out = []
    for i in range(0, len(lat), batch):
        q = {"latitude": ",".join(f"{v:.5f}" for v in lat[i:i + batch]),
             "longitude": ",".join(f"{v:.5f}" for v in lon[i:i + batch])}
        if i:
            time.sleep(11)   # 무료 API 는 좌표 하나를 호출 하나로 센다 — 분당 600개 이내로
        for wait in (0, 61, 61, 121, 121):   # 그래도 호출 제한(429)이면 기다렸다 다시
            time.sleep(wait)
            try:
                out += json.loads(_get(ELEV_URL + "?" + urllib.parse.urlencode(q)))["elevation"]
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    raise
        else:
            raise RuntimeError("Open-Meteo 표고 호출 제한 — 잠시 뒤 다시(받은 만큼은 캐시에 남음)")
    return out


def fetch_weather(start, end, models=("ecmwf_ifs025", "jma_seamless")):
    q = {**SEOUL, "start_date": start, "end_date": end, "timezone": "Asia/Seoul"}
    f = None
    for model in models:   # 앞 모델이 우선, 빈 시각만 다음 모델로
        fc = json.loads(_get(PREV_URL + "?" + urllib.parse.urlencode(
            {**q, "hourly": "temperature_2m_previous_day2", "models": model}), timeout=180))
        g = pd.Series(fc["hourly"]["temperature_2m_previous_day2"], index=pd.to_datetime(fc["hourly"]["time"]), dtype=float)
        f = g if f is None else f.combine_first(g)
    ob = json.loads(_get(ARCHIVE_URL + "?" + urllib.parse.urlencode({**q, "hourly": "temperature_2m"}), timeout=180))
    f = pd.DataFrame({"timestamp": f.index, "temp_fcst_c": f.to_numpy()})
    o = pd.DataFrame({"timestamp": pd.to_datetime(ob["hourly"]["time"]), "temp_obs_c": ob["hourly"]["temperature_2m"]})
    df = o.merge(f, on="timestamp", how="outer").sort_values("timestamp")
    df.insert(1, "station_or_grid", STATION)
    df["fcst_issued_at"] = (df["timestamp"] - pd.Timedelta(hours=48)).where(df["temp_fcst_c"].notna())
    return df[["timestamp", "station_or_grid", "temp_obs_c", "temp_fcst_c", "fcst_issued_at"]]


LIVING_URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
LIVING_BANDS = {"lp_night": range(0, 7), "lp_day": range(9, 18), "lp_eve": range(18, 24)}


def fetch_living_population(yymm, out_dir):
    """서울 열린데이터광장 OA-14991 행정동 단위 서울 생활인구(내국인) 월별 ZIP(약 45MB) — yymm 예: "2605".
    공공누리 1유형(출처 표시). 행정동 단위 생산은 2026-07 로 끝났다(이후 250m 격자)."""
    path = Path(out_dir) / f"LOCAL_PEOPLE_DONG_20{yymm}.zip"
    if not path.is_file():
        data = urllib.parse.urlencode({"infId": "OA-14991", "seqNo": "", "seq": yymm, "infSeq": "3"}).encode()
        req = urllib.request.Request(LIVING_URL, data=data,
                                     headers={"Referer": "https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            path.write_bytes(resp.read())
    return path


def living_population(zips, hdong_bjd):
    """월별 ZIP → 13.csv 표(date, bjd_code, lp_night·lp_day·lp_eve = 0~6시·9~17시·18~23시 평균 생활인구).
    행정동(8자리) → 법정동은 8.csv 면적 가중치(행정동마다 합 1)로 나눈다. 반환: (표, 대응 안 된 생활인구 몫)."""
    import zipfile

    import numpy as np

    w = pd.read_csv(hdong_bjd, dtype=str, encoding="utf-8-sig")
    w.columns = ["hdong", "bjd", "weight"]
    w["hdong"], w["weight"] = w["hdong"].str[:8], pd.to_numeric(w["weight"])
    w["weight"] = w["weight"] / w.groupby("hdong")["weight"].transform("sum")
    parts, lost, total = [], 0.0, 0.0
    for z in zips:
        with zipfile.ZipFile(z) as f:
            d = pd.read_csv(f.open(f.namelist()[0]), header=None, skiprows=1, usecols=[0, 1, 2, 3],
                            dtype={0: str, 1: int, 2: str, 3: float}, encoding="latin-1")   # 헤더만 월마다 UTF-8/CP949 — 건너뛰고 숫자 행만 읽는다
        d.columns = ["date", "hour", "hdong", "pop"]
        band = pd.Series(np.nan, index=d.index, dtype=object)
        for name, hours in LIVING_BANDS.items():
            band[d["hour"].isin(hours)] = name
        d = d.dropna(subset=["pop"]).assign(band=band).dropna(subset=["band"])
        day = d.groupby(["date", "hdong", "band"])["pop"].mean().unstack("band").reset_index()
        m = day.merge(w, on="hdong", how="left")
        total += float(day[list(LIVING_BANDS)].sum().sum())
        lost += float(m.loc[m["bjd"].isna(), list(LIVING_BANDS)].sum().sum())
        m = m.dropna(subset=["bjd"])
        for c in LIVING_BANDS:
            m[c] = m[c] * m["weight"]
        parts.append(m.groupby(["date", "bjd"], as_index=False)[list(LIVING_BANDS)].sum())
    out = pd.concat(parts, ignore_index=True).rename(columns={"bjd": "bjd_code"})
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return out.sort_values(["date", "bjd_code"]).round(1), (lost / total if total else 0.0)


CHARGER_URL = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"


def _data_go_kr_key():
    """공공데이터포털 인증키 — 환경변수 DATA_GO_KR_KEY(Windows 는 setx 로 저장한 사용자 환경변수도 읽는다). 출력·저장하지 않는다."""
    import os
    import sys

    k = os.environ.get("DATA_GO_KR_KEY")
    if not k and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            try:
                k = winreg.QueryValueEx(h, "DATA_GO_KR_KEY")[0]
            except FileNotFoundError:
                k = None
    if not k:
        raise SystemExit("DATA_GO_KR_KEY 없음 — 공공데이터포털 인증키를 환경변수로 저장할 것")
    return k


def fetch_chargers(zcode="11", rows=9999):
    """한국환경공단 전기자동차 충전소 정보 API(공공데이터포털 15076352, 실시간 갱신) — 시도(zcode) 충전기 전체(충전기 한 줄)."""
    key, items, page = _data_go_kr_key(), [], 1
    while True:
        q = urllib.parse.urlencode({"serviceKey": key, "pageNo": page, "numOfRows": rows, "zcode": zcode, "dataType": "JSON"})
        body = json.loads(_get(CHARGER_URL + "?" + q, timeout=120))
        got = body["items"]["item"] if body.get("items") else []
        items += got
        if not got or len(items) >= int(body.get("totalCount", 0)):
            return pd.DataFrame(items)
        page += 1


def charger_stations(chargers, public_only=True):
    """충전기 표 → 14.csv(6.csv 와 같은 열: 위도·경도·충전기수). 삭제된 충전기는 빼고, public_only 면 이용자 제한
    충전기(아파트 입주민 전용 등, limitYn=Y)도 뺀다 — 2SFCA 는 누구나 쓸 수 있는 공용 충전기 접근성이다."""
    c = chargers[chargers["delYn"].ne("Y")]
    if public_only:
        c = c[c["limitYn"].ne("Y")]
    c = c.assign(lat=pd.to_numeric(c["lat"], errors="coerce"), lng=pd.to_numeric(c["lng"], errors="coerce")).dropna(subset=["lat", "lng"])
    st = c.groupby("statId").agg(위도=("lat", "first"), 경도=("lng", "first"), 충전기수=("chgerId", "size")).reset_index(drop=True)
    return st.round({"위도": 6, "경도": 6})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start", nargs="?")
    ap.add_argument("end", nargs="?")
    ap.add_argument("--out", default="dist/import")
    ap.add_argument("--living", help="생활인구 13.csv 만들기: 월 범위 YYMM-YYMM(예 2508-2605). --hdong 8.csv 필요")
    ap.add_argument("--hdong", help="행정동→법정동 대응표(8.csv)")
    ap.add_argument("--chargers", action="store_true", help="서울 공용 충전소 14.csv(공공데이터포털 인증키 DATA_GO_KR_KEY)")
    a = ap.parse_args()
    if a.chargers:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        ch = fetch_chargers()
        ch.to_csv(Path(a.out) / "chargers_raw_seoul.csv", index=False, encoding="utf-8-sig")   # 밖에서 검토용(반입 안 함)
        st = charger_stations(ch)
        st.to_csv(Path(a.out) / "14.csv", index=False, encoding="utf-8-sig")
        live = ch["delYn"].ne("Y")
        print(f"충전기 {int(live.sum()):,}기(삭제 제외) · 이용자 제한 {int((live & ch['limitYn'].eq('Y')).sum()):,}기 · "
              f"14.csv 공용 {int(st['충전기수'].sum()):,}기 / {len(st):,}곳")
        return
    if a.living:
        lo, hi = a.living.split("-")
        months = [m.strftime("%y%m") for m in pd.period_range(f"20{lo[:2]}-{lo[2:]}", f"20{hi[:2]}-{hi[2:]}", freq="M")]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        zips = [fetch_living_population(m, a.out) for m in months]
        lp, lost = living_population(zips, a.hdong)
        lp.to_csv(Path(a.out) / "13.csv", index=False)
        print(f"13.csv {len(lp):,}행 {lp['date'].min()}~{lp['date'].max()} · 법정동 {lp['bjd_code'].nunique()} · 대응 안 된 생활인구 {lost:.1%}")
        return
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
