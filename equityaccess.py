"""8-C 단계 — 법정동별 공용 충전 접근성(2SFCA)과 접근성 부족도."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bjdmapping import canonicalize
from common import clean_code, read_columns, to_num


def load_access_inputs(station_path, ev_path, columns, centroids, crosswalk=None, ev_counts=None):
    """외부 공개자료를 최소 공통 스키마로 읽는다. EV 자료는 법정동 매핑 완료본이어야 한다.

    ev_counts(bjd_code, ev_count)를 주면 ev_path 대신 쓴다 — v11 은 8-F 등록 이력의 기준월 값을 행정동→법정동
    대응표로 배분해 넘긴다(반입 파일 하나를 줄이려고).
    """
    st = read_columns(station_path, columns["access_station"], "공용충전소")
    ev = read_columns(ev_path, columns["ev_registration"], "전기차등록") if ev_counts is None else \
        ev_counts.rename(columns={"bjd_code": "code"})
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


def compute_2sfca(stations, demand_points, radii_m=(300, 500, 800), default_radius_m=500, dist_km=None):
    """충전소 공급/반경 내 EV 수요로 2SFCA를 계산한다. dist_km(수요점×충전소)를 주면 직선거리 대신 쓴다.

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
                        s["lat"].to_numpy()[None, :], s["lon"].to_numpy()[None, :]) if dist_km is None else dist_km
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


def slope_factor(grade):
    """경사(오르막 +)를 걸을 때 평지 대비 시간 배율. Tobler 보행 함수 v = 6·exp(−3.5·|g + 0.05|) 기준이며,
    충전기에 꽂아 두고 걸어서 오가는 왕복이라 갈 때·올 때 배율을 평균한다(평지 = 1)."""
    g = np.asarray(grade, float)
    flat = np.exp(-3.5 * 0.05)
    return (flat / np.exp(-3.5 * np.abs(g + 0.05)) + flat / np.exp(-3.5 * np.abs(-g + 0.05))) / 2


def slope_adjusted_2sfca(stations, demand_points, elev_points, elev_stations, radii_m=(300, 500, 800),
                         default_radius_m=500, max_grade=0.3, min_dist_m=50.0):
    """중심점·충전소 표고로 직선 경사를 구해 거리에 slope_factor 를 곱한 뒤 2SFCA 를 다시 계산한다.

    직선 경사는 실제 보행로의 오르내림이 아니라 근사다. 표고가 없는 쌍과 min_dist_m 미만 쌍은 경사 0(평지),
    경사는 ±max_grade 로 자른다(다리·절개지 같은 이상값 방지). 반환: (보정 2SFCA 표, 반경 안 쌍의 |경사| 배열).
    """
    d = demand_points.reset_index(drop=True)
    s = stations.reset_index(drop=True)
    dist = _distance_km(d["lat"].to_numpy()[:, None], d["lon"].to_numpy()[:, None],
                        s["lat"].to_numpy()[None, :], s["lon"].to_numpy()[None, :])
    rise = np.asarray(elev_stations, float)[None, :] - np.asarray(elev_points, float)[:, None]
    grade = np.divide(rise, dist * 1000.0, out=np.zeros_like(dist), where=dist * 1000.0 >= min_dist_m)
    grade = np.clip(np.nan_to_num(grade, nan=0.0), -max_grade, max_grade)
    out = compute_2sfca(s, d, radii_m, default_radius_m, dist_km=dist * slope_factor(grade))
    near = dist <= max(radii_m) / 1000.0
    return out, np.abs(grade[near])


def compare_slope(base, slope, grades, n_points, n_stations, share=0.20):
    """기본 2SFCA 대 경사 보정 2SFCA — 법정동 집계 요약만(표고 값·동네 목록은 싣지 않음)."""
    a, b = _present(base), _present(slope).reindex(_present(base).index)
    low_a, low_b = set(bottom_codes(base, share)), set(bottom_codes(slope, share))
    rho = a.rank().corr(b.rank()) if len(a) > 2 and a.nunique() > 1 and b.nunique() > 1 else np.nan
    rows = [("표고 확보 법정동", f"{n_points[0]} / {n_points[1]}", "중심점 표고. 없으면 그 동네 쌍은 평지로 봄"),
            ("표고 확보 충전소", f"{n_stations[0]} / {n_stations[1]}", ""),
            ("반경 안 쌍의 |경사| 중앙값", float(np.median(grades)) if len(grades) else np.nan, "직선 경사(높이차 ÷ 거리)"),
            ("반경 안 쌍 중 경사 5% 이상 비율", float((grades >= 0.05).mean()) if len(grades) else np.nan, ""),
            ("순위 상관(스피어만)", rho, "기본 대 경사 보정, 수요 있는 동네"),
            (f"하위 {share:.0%} 동네 수(기본 → 보정)", f"{len(low_a)} → {len(low_b)}", "동률 포함"),
            (f"하위 {share:.0%} 유지", len(low_a & low_b), "기본 하위 중 보정 후에도 하위"),
            (f"하위 {share:.0%} 새로 진입", len(low_b - low_a), "경사 때문에 새로 하위가 된 동네"),
            ("자카드(하위 집합)", len(low_a & low_b) / len(low_a | low_b) if low_a | low_b else np.nan, ""),
            ("접근성 0 동네(기본 → 보정)", f"{int((a <= 0).sum())} → {int((b <= 0).sum())}", ""),
            ("평균 접근성 변화율", float(b.mean() / a.mean() - 1) if a.mean() > 0 else np.nan, "음수 = 경사로 멀어짐")]
    out = pd.DataFrame(rows, columns=["항목", "값", "비고"])
    out.attrs.update(keep=len(low_a & low_b), n_low=len(low_a), rho=rho)   # s9_story '지형 강건성' 행
    return out


def _present(access):
    return access.loc[access["access_data_present"]].set_index("bjd_code")["access_2sfca"].astype(float)


def bottom_codes(access, share=0.20):
    """접근성 하위 share 동네(동률 포함 — 0 이 많아도 임의로 자르지 않는다). 수요가 있는 동네만."""
    a = _present(access)
    cut = a.nsmallest(max(1, int(np.ceil(len(a) * share)))).max()
    return list(a.index[a <= cut])


def access_gap(access, codes):
    """codes(개선 전 하위 동네로 고정) 평균 2SFCA, 전체 평균, 접근성 0 동네 수."""
    a = _present(access)
    return {"bottom_mean": float(a.reindex(codes).mean()), "mean": float(a.mean()), "zero": int((a <= 0).sum()),
            "n": len(a)}


def simulate_added_chargers(stations, demand_points, strategies, chargers_each, radii_m=(300, 500, 800),
                            default_radius_m=500, share=0.20):
    """전략별로 대상 법정동 중심점에 가상 충전소(chargers_each 기)를 더해 2SFCA 를 다시 계산한다.

    strategies: {전략 이름: 대상 bjd_code 목록}. 입지 최적화가 아니라 '어디부터 더하면 격차가 더 줄어드나'를
    같은 충전기 수로 비교하는 가정 실험이다(중심점 근사·반경 내 균등 이용 가정).
    하위 동네는 개선 전 기준으로 고정해 전략 간에 같은 동네를 비교한다.
    """
    base = compute_2sfca(stations, demand_points, radii_m, default_radius_m)
    low = bottom_codes(base, share)
    before = access_gap(base, low)
    pts = demand_points.drop_duplicates("bjd_code").set_index("bjd_code")
    rows = []
    for name, codes in strategies.items():
        codes = [c for c in dict.fromkeys(map(str, codes)) if c in pts.index]
        extra = pd.DataFrame({"lat": pts.loc[codes, "lat"].to_numpy(), "lon": pts.loc[codes, "lon"].to_numpy(),
                              "chargers": float(chargers_each)})
        after = access_gap(compute_2sfca(pd.concat([stations, extra], ignore_index=True), demand_points,
                                         radii_m, default_radius_m), low)
        rows.append({"전략": name, "대상 동네 수": len(codes), "추가 충전기(기)": len(codes) * chargers_each,
                     f"하위 {share:.0%} 동네 수(동률 포함)": len(low),
                     f"하위 {share:.0%} 평균 접근성(전)": before["bottom_mean"],
                     f"하위 {share:.0%} 평균 접근성(후)": after["bottom_mean"],
                     f"하위 {share:.0%} 개선율": after["bottom_mean"] / before["bottom_mean"] - 1
                     if before["bottom_mean"] > 0 else np.nan,
                     "접근성 0 동네(전)": before["zero"], "접근성 0 동네(후)": after["zero"],
                     "전체 평균 개선율": after["mean"] / before["mean"] - 1 if before["mean"] > 0 else np.nan,
                     "비교 동네 수": before["n"]})
    return pd.DataFrame(rows)


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
    out["ev_count"] = ev   # 8-A AI 동네 특성·8-F 시나리오 입력
    out["ev_per_charger"] = (ev / out["chargers_assigned"]).where(out["chargers_assigned"] > 0)
    out["nearest_charger_km"] = dist.min(axis=1)
    return out
