"""8-C 단계 — 법정동별 공용 충전 접근성(2SFCA)과 접근성 부족도."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bjd_mapping import canonicalize
from common import clean_code, read_columns, to_num


def load_access_inputs(station_path, ev_path, columns, centroids, crosswalk=None):
    """외부 공개자료를 최소 공통 스키마로 읽는다. EV 자료는 법정동 매핑 완료본이어야 한다."""
    st = read_columns(station_path, columns["access_station"], "공용충전소")
    ev = read_columns(ev_path, columns["ev_registration"], "전기차등록")
    stations = pd.DataFrame({
        "lat": to_num(st["lat"]), "lon": to_num(st["lon"]), "chargers": to_num(st["chargers"]),
    }).dropna()
    stations = stations[(stations["chargers"] > 0) & stations["lat"].between(-90, 90)
                        & stations["lon"].between(-180, 180)].reset_index(drop=True)
    demand = pd.DataFrame({"bjd_code": clean_code(ev["code"], 10), "ev_count": to_num(ev["ev_count"])}).dropna()
    demand["bjd_code"] = canonicalize(demand["bjd_code"], crosswalk)
    demand = demand[demand["ev_count"] >= 0].groupby("bjd_code", as_index=False)["ev_count"].sum()
    points = centroids[["bjd_code", "lat", "lon"]].copy()
    points["bjd_code"] = canonicalize(points["bjd_code"].astype(str), crosswalk)
    points = points.drop_duplicates("bjd_code").merge(demand, on="bjd_code", how="left")
    return stations, points


def _distance_km(lat1, lon1, lat2, lon2):
    """브로드캐스팅 가능한 haversine 거리."""
    a1, a2 = np.radians(lat1), np.radians(lat2)
    dlat = a2 - a1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _minmax(values, reverse=False):
    s = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = np.isfinite(s)
    if not ok.any():
        return out
    lo, hi = s[ok].min(), s[ok].max()
    out.loc[ok] = 0.5 if np.isclose(lo, hi) else (s[ok] - lo) / (hi - lo)
    return 1 - out if reverse else out


def compute_2sfca(stations, demand_points, radii_m=(300, 500, 800), default_radius_m=500):
    """충전소 공급/반경 내 EV 수요로 2SFCA를 계산한다.

    ponytail: 점-충전소 전체 거리행렬이다. 수도권 격자 규모에서 메모리가 문제가 될 때만 공간 인덱스로 교체한다.
    """
    if len(stations) == 0 or len(demand_points) == 0:
        raise ValueError("2SFCA 계산에 충전소와 법정동 수요점이 각각 1개 이상 필요")
    radii = sorted({int(r) for r in radii_m})
    if default_radius_m not in radii or any(r <= 0 for r in radii):
        raise ValueError("기본 반경은 양수 radii_m 중 하나여야 함")
    for name, df, cols in (("충전소", stations, ["lat", "lon", "chargers"]),
                           ("수요점", demand_points, ["bjd_code", "lat", "lon", "ev_count"])):
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"{name} 필수 열 없음: {sorted(missing)}")
    d = demand_points.reset_index(drop=True).copy()
    s = stations.reset_index(drop=True).copy()
    dist = _distance_km(d["lat"].to_numpy()[:, None], d["lon"].to_numpy()[:, None],
                        s["lat"].to_numpy()[None, :], s["lon"].to_numpy()[None, :])
    ev = pd.to_numeric(d["ev_count"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    chargers = pd.to_numeric(s["chargers"], errors="raise").to_numpy(float)
    out = d[["bjd_code"]].copy()
    for radius in radii:
        inside = dist <= radius / 1000.0
        station_demand = (inside * ev[:, None]).sum(axis=0)
        ratio = np.divide(chargers, station_demand, out=np.zeros_like(chargers), where=station_demand > 0)
        out[f"2sfca_{radius}m"] = (inside * ratio[None, :]).sum(axis=1)
    raw = out[f"2sfca_{default_radius_m}m"]
    out["access_2sfca"] = raw
    out["equity_need_norm"] = _minmax(raw, reverse=True)
    out["access_data_present"] = d["ev_count"].notna()
    return out


def add_access_indicators(access, stations, demand_points):
    """지표 1(충전기 1기당 EV 수)·지표 2(최근접 공용충전기 거리)를 8-C 표에 붙인다.

    충전소는 가장 가까운 법정동 중심점에 배정하고 거리는 중심점 기준이다. 폴리곤 공간조인도 격자점도 아닌
    근사이며, 행정경계 근처 충전소는 이웃 동네로 배정될 수 있다(경계 도형 확보 후 대체).
    충전기가 0기인 동네는 지표 1을 NaN 으로 두고(0으로 나눔 방지) chargers_assigned=0 으로 남긴다.
    """
    d = demand_points.reset_index(drop=True)
    s = stations.reset_index(drop=True)
    dist = _distance_km(d["lat"].to_numpy()[:, None], d["lon"].to_numpy()[:, None],
                        s["lat"].to_numpy()[None, :], s["lon"].to_numpy()[None, :])
    chargers = pd.to_numeric(s["chargers"], errors="raise").to_numpy(float)
    out = access.reset_index(drop=True).copy()
    out["chargers_assigned"] = np.bincount(dist.argmin(axis=0), weights=chargers, minlength=len(d))
    ev = pd.to_numeric(d["ev_count"], errors="coerce")
    out["ev_per_charger"] = (ev / out["chargers_assigned"]).where(out["chargers_assigned"] > 0)
    out["nearest_charger_km"] = dist.min(axis=1)
    return out
