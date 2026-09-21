"""외부 입력 로더(v11) — 기온(실측·과거 예보)과 공휴일 달력.

기온: paths.weather 정규화 CSV(KST)
    timestamp,station_or_grid,temp_obs_c,temp_fcst_c,fcst_issued_at[,station_lat,station_lon]
  - 같은 시각·지점에 발표 시각이 다른 예보가 여러 행 있을 수 있다. 결측 시간은 보간하지 않는다.
  - 예측일 d 에는 발표 시각이 d−1일 weather.fcst_cutoff(기본 18:00) 이전인 예보 중 가장 늦은 발표분만 쓴다.
  - 서울은 관측 지점이 적어 사실상 서울 공통 기온이다. 지점이 하나면 모든 법정동에 같은 값을, 여럿이면
    station_lat/station_lon 으로 법정동 중심점에서 가장 가까운 지점을 배정한다(좌표 열은 입력 전용, 반출 표에 싣지 않는다).
공휴일: paths.holidays(config/holidays.yaml). YAML 은 JSON 의 상위집합이라 JSON 문법으로 저장해 표준 json 으로 읽는다
    (안심구역에 PyYAML 이 없을 수 있음). 형식 {"YYYY-MM-DD": {"type", "public_holiday", "name", "source"}}.
    휴일을 규칙으로 만들지 않는다 — 날짜는 파일(특일 정보 API·관보)에서만 온다. 유형 구분만 classify_holidays 가 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import log, nearest_region

WEATHER_COLUMNS = ["timestamp", "station_or_grid", "temp_obs_c", "temp_fcst_c", "fcst_issued_at"]
HOLIDAY_TYPES = ("holiday", "long_holiday", "pre_post_holiday")


def load_weather(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"station_or_grid": str})
    missing = set(WEATHER_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"기온 CSV 정규화 열 없음: {sorted(missing)} (필요: {','.join(WEATHER_COLUMNS)})")
    out = frame.copy()
    for col in ("timestamp", "fcst_issued_at"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
        if out[col].dt.tz is not None:
            raise ValueError("기온 시각은 KST로 변환한 뒤 시간대 없는 시각으로 입력")
    if out["timestamp"].isna().any():
        raise ValueError("기온 timestamp 해석 실패 행이 있음")
    for col in ("temp_obs_c", "temp_fcst_c"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if (out["temp_fcst_c"].notna() & out["fcst_issued_at"].isna()).any():
        raise ValueError("예보 기온에 발표 시각(fcst_issued_at)이 없는 행이 있음 — 시점 검사를 할 수 없음")
    log.info("기온 %d행 · 지점 %d · %s~%s · 예보 %d행", len(out), out["station_or_grid"].nunique(),
             out["timestamp"].min(), out["timestamp"].max(), int(out["temp_fcst_c"].notna().sum()))
    return out


def fcst_cutoff(timestamps, cutoff="18:00"):
    """각 예측 대상 시각의 예보 발표 마감 = 대상일 전날 cutoff."""
    hh, mm = (int(v) for v in str(cutoff).split(":"))
    return pd.to_datetime(timestamps).dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm)


def observed_temps(weather):
    """(지점, 시각)별 실측 기온. 중복 행의 실측값은 같아야 한다."""
    obs = weather.dropna(subset=["temp_obs_c"]).groupby(["station_or_grid", "timestamp"])["temp_obs_c"]
    if (obs.nunique() > 1).any():
        raise ValueError("같은 지점·시각에 실측 기온이 서로 다름")
    return obs.first().rename("temp_c").reset_index()


def forecast_temps(weather, cutoff="18:00"):
    """(지점, 시각)별로 마감 이전 발표분 중 가장 늦은 예보. 반환에 fcst_issued_at 을 남겨 누설 검사를 할 수 있게 한다."""
    f = weather.dropna(subset=["temp_fcst_c"])
    f = f[f["fcst_issued_at"] <= fcst_cutoff(f["timestamp"], cutoff)]
    f = f.sort_values("fcst_issued_at").groupby(["station_or_grid", "timestamp"]).tail(1)
    return f[["station_or_grid", "timestamp", "temp_fcst_c", "fcst_issued_at"]].rename(
        columns={"temp_fcst_c": "temp_c"}).reset_index(drop=True)


def assert_forecast_before_cutoff(frame, cutoff="18:00"):
    """예보 입력의 발표 시각이 마감 이후면 예외(정보 누설). 킬 19번이 이 예외를 실패로 표시한다."""
    late = frame["fcst_issued_at"] > fcst_cutoff(frame["timestamp"], cutoff)
    if late.any():
        raise ValueError(f"정보 누설: 예보 발표 시각이 전날 {cutoff} 이후인 입력 {int(late.sum())}행 "
                         f"(예: {frame.loc[late, 'timestamp'].iloc[0]} ← 발표 {frame.loc[late, 'fcst_issued_at'].iloc[0]})")
    return frame


def assign_stations(weather, centroids):
    """법정동 → 기온 지점. 지점이 하나면 전부 그 지점, 여럿이면 중심점 최근접(지점 좌표 필요)."""
    stations = weather.drop_duplicates("station_or_grid")
    codes = centroids["bjd_code"].astype(str).drop_duplicates()
    if len(stations) == 1:
        return pd.DataFrame({"bjd_code": codes, "station_or_grid": stations["station_or_grid"].iloc[0]})
    if not {"station_lat", "station_lon"} <= set(weather):
        raise ValueError("기온 지점이 여럿인데 station_lat·station_lon 이 없어 법정동 배정 불가")
    pts = stations.rename(columns={"station_or_grid": "bjd_code", "station_lat": "lat", "station_lon": "lon"})
    cent = centroids.drop_duplicates("bjd_code")
    nearest, _ = nearest_region(cent["lat"], cent["lon"], pts[["bjd_code", "lat", "lon"]], np.inf)
    return pd.DataFrame({"bjd_code": cent["bjd_code"].astype(str).to_numpy(), "station_or_grid": np.asarray(nearest)})


# ---------------------------------------------------------------- 공휴일

def classify_holidays(public_dates, long_min_days=3, count_weekends=True):
    """공휴일 날짜 목록 → {날짜: 유형}. 날짜를 만들지 않고, 주어진 공휴일의 유형만 나눈다.

    쉬는 날(공휴일 + count_weekends 면 토·일)이 long_min_days 이상 이어지고 그 안에 공휴일이 있으면 그 공휴일은
    long_holiday, 연휴 바로 앞·뒤의 평일은 pre_post_holiday, 나머지 공휴일은 holiday.
    """
    public = sorted({pd.Timestamp(d).normalize() for d in public_dates})
    if not public:
        return {}
    days = pd.date_range(public[0] - pd.Timedelta(days=7), public[-1] + pd.Timedelta(days=7))
    off = pd.Series(days.isin(public), index=days)
    if count_weekends:
        off |= days.dayofweek >= 5
    run_id = (off != off.shift()).cumsum()
    out = {}
    for _, run in off[off].groupby(run_id[off]):
        long_run = len(run) >= int(long_min_days)
        for d in run.index:
            if d in public:
                out[d] = "long_holiday" if long_run else "holiday"
        if long_run and any(d in public for d in run.index):
            for d in (run.index[0] - pd.Timedelta(days=1), run.index[-1] + pd.Timedelta(days=1)):
                out.setdefault(d, "pre_post_holiday")
    return {d.strftime("%Y-%m-%d"): t for d, t in sorted(out.items())}


def load_holidays(path, params=None):
    """holidays.yaml → {YYYY-MM-DD: 유형}. 파일이 없으면 빈 dict(공휴일 미보정).

    type 이 없으면 public_holiday=true 인 날짜로 classify_holidays 를 돌려 채운다.
    """
    if not path or not Path(path).is_file():
        log.warning("공휴일 달력 없음(%s) — 공휴일 미보정으로 진행", path)
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(f"{path}: JSON 문법이 아니고 PyYAML 도 없음 — tools/fetch_holidays.py 형식으로 저장할 것") from exc
        raw = yaml.safe_load(text)
    p = {"long_min_days": 3, "count_weekends": True, **(params or {})}
    entries = {str(pd.Timestamp(k).date()): (v or {}) for k, v in raw.items() if not str(k).startswith("_")}
    typed = {d: e["type"] for d, e in entries.items() if e.get("type")}
    bad = {d: t for d, t in typed.items() if t not in HOLIDAY_TYPES}
    if bad:
        raise ValueError(f"알 수 없는 휴일유형: {bad} (허용: {HOLIDAY_TYPES})")
    if len(typed) < len(entries):
        public = [d for d, e in entries.items() if e.get("public_holiday", True)]
        typed = {**classify_holidays(public, p["long_min_days"], p["count_weekends"]), **typed}
    log.info("공휴일 달력 %s: %d일 %s", Path(path).name, len(typed), pd.Series(typed).value_counts().to_dict())
    return typed


def dump_holidays(entries):
    """{날짜: {...}} → holidays.yaml 본문(JSON 문법 = 유효한 YAML)."""
    return json.dumps(dict(sorted(entries.items())), ensure_ascii=False, indent=1) + "\n"
