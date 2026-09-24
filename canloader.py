"""3단계 처치 정제 — 티벨 CAN M-Type(순간 상태 이벤트 로그).

`차종_식별번호`가 개별 차량인지 차종 모델 코드인지 현장 검증 전까지 불명 → verify_join_key 로 판정해 분기한다.

  verify_join_key → "individual" : 경로 A — 차량별 충전 세션 재구성 + 충전 시작 위치 DBSCAN 으로 거점성 판별
                  → "model"/"unknown" : 경로 B — 개별 차량 클러스터링을 하지 않고 법정동 충전 위치 밀집도만 집계
  세션 길이 분포(20~40분 체류 전제 검증)는 두 경로 공통으로 항상 산출한다.
  경로 B 의 세션은 '식별번호 + 위치격자' 로 근사한다 — 같은 모델 코드의 다른 차량이 같은 격자·같은 시간에
  충전하면 한 세션으로 합쳐질 수 있어 '정황상 보조 근거' 수준으로만 표기한다.

CHARGE_KWH 컬럼이 없다 → 충전량은 배터리상태_SOC 차분(%p)으로만 근사한다.
GPS 규칙: 이 모듈이 돌려주는 반출용 표에는 좌표가 없다(법정동 집계만). 세션 좌표는 내부 계산에만 쓴다.

격리: run_can_stage 전체를 pipeline 이 try/except 로 감싼다. 여기서 실패해도 다른 단계는 영향이 없다.
센터 첫 확인 때는 COPY_PATH 한 곳만 채운 뒤 직접 실행해 원본을 저장하지 않는 집계 진단을 할 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EARTH_RADIUS_KM, detect_columns, haversine_km, log, mi_to_ym, nearest_region, read_columns,
                    resolve_csv_path, to_num, ym_to_mi)

DURATION_BINS = [0, 10, 20, 30, 40, 60, 90, 120, 240, 480, 1440]

# 안심데이터센터 Jupyter 첫 확인용. 직원에게 받은 Copy Path를 따옴표 안에 그대로 붙여 넣는다.
# 예상 후보(현장 확인 필요): r"Import_data/TBE/TB_TBE_TERMINAL_LOGMOCEAN.csv"
COPY_PATH = r""

CAN_COLUMN_ALIASES = {
    "time": ("발생시간", "발생일시", "OCCUR_DTM", "EVENT_DTM", "LOG_DTM"),
    "vehicle_id": ("차종_식별번호", "차종식별번호", "VHCL_ID", "VEHICLE_ID", "CAR_ID"),
    "charging": ("충전중여부", "충전여부", "CHRG_YN", "CHARGING_YN", "CHARGERCONNECTION"),
    "soc": ("배터리상태_SOC", "배터리상태SOC", "SOC", "BATTERY_SOC", "BATTERYSOC"),
    "lat": ("위도", "LAT", "LATITUDE", "GPSLAT", "LTD"),
    "lon": ("경도", "LON", "LONGITUDE", "GPSLON", "LNGT"),
}


def load_can_m(path, columns, params, nrows=None):
    path = resolve_csv_path(path, "CAN_M")
    true_values = {str(v).upper() for v in params["charging_true_values"]}
    parts = []
    for chunk in read_columns(path, columns["can_m"], "CAN_M", chunksize=params["chunksize"], nrows=nrows):
        lat, lon = to_num(chunk["lat"]), to_num(chunk["lon"])
        bad = ~(lat.between(*params["lat_range"]) & lon.between(*params["lon_range"]))
        parts.append(pd.DataFrame({
            "vehicle_id": chunk["vehicle_id"].fillna("").astype(str).str.strip(),
            "time": pd.to_datetime(chunk["time"], errors="coerce"),
            "charging": chunk["charging"].fillna("").astype(str).str.strip().str.upper().isin(true_values),
            "soc": to_num(chunk["soc"]),
            "lat": lat.where(~bad), "lon": lon.where(~bad),
        }))
    df = pd.concat(parts, ignore_index=True)
    bad_time = df["time"].isna() | (df["vehicle_id"] == "")
    if bad_time.any():
        log.warning("CAN: 시간/식별번호 결측 %d행 제외", int(bad_time.sum()))
    df = df[~bad_time].sort_values(["vehicle_id", "time"], kind="mergesort").reset_index(drop=True)
    log.info("CAN M-Type: %s행 · 식별번호 %d · %s ~ %s · 충전중 %.1f%%", f"{len(df):,}", df["vehicle_id"].nunique(),
             df["time"].min(), df["time"].max(), 100 * df["charging"].mean())
    return df


def inspect_can_copy_path(path, nrows=200_000):
    """센터 CAN CSV 일부를 읽어 원본 행·식별번호·좌표 없이 집계 진단만 반환한다."""
    from config import DEFAULT_PARAMS

    path = resolve_csv_path(path, "TB_TBE_TERMINAL_LOGMOCEAN")
    detected = detect_columns(
        path, CAN_COLUMN_ALIASES, "TB_TBE_TERMINAL_LOGMOCEAN", required=set(CAN_COLUMN_ALIASES)
    )
    params = dict(DEFAULT_PARAMS["can"])
    params["chunksize"] = min(params["chunksize"], nrows)
    can = load_can_m(path, {"can_m": detected}, params, nrows=nrows)
    verdict, evidence = verify_join_key(can, params)
    mode = "vehicle" if verdict == "individual" else "geo_cell"
    sessions = build_sessions(can, params, mode)
    path_label = "A(개별 차량)" if verdict == "individual" else "B(집계 근사, 정황상 보조 근거)"
    duration_table, duration_summary = duration_distribution(sessions, path_label)
    summary = pd.DataFrame([
        {"항목": "사용 파일", "값": path.name},
        {"항목": "자동 판별 컬럼", "값": str(detected)},
        {"항목": "점검 범위", "값": f"앞 {len(can):,}행(최대 {nrows:,}행)"},
        {"항목": "원본 저장", "값": "안 함"},
        {"항목": "식별번호·좌표 반환", "값": "안 함"},
    ])
    return {
        "summary": summary,
        "evidence": evidence,
        "duration_summary": duration_summary,
        "duration_table": duration_table,
    }


def verify_join_key(can_df, params):
    """차종_식별번호가 개별 차량 단위인지 판정.

    근거 1: 식별번호당 평균 레코드 수(너무 적으면 판정 불가).
    근거 2: 같은 식별번호 안에서 짧은 간격(≤ pair_max_gap_min) 연속 레코드가 한 차량의 궤적으로 그럴듯한지 —
            순간이동(max_speed_kmh 초과)이나 SOC 급변(max_soc_jump 초과)이 드물면 개별 차량.
            모델 코드라면 서로 다른 곳의 차량 기록이 시간순으로 뒤섞여 이상치가 많이 생긴다.
    반환: ("individual" | "model" | "unknown", 근거표)
    """
    df = can_df
    n_rows, n_ids = len(df), df["vehicle_id"].nunique()
    rows_per_id = n_rows / n_ids if n_ids else float("nan")
    same = df["vehicle_id"].eq(df["vehicle_id"].shift())
    dt_min = (df["time"] - df["time"].shift()).dt.total_seconds() / 60.0
    pair = same & dt_min.between(0, params["pair_max_gap_min"])
    dist = pd.Series(haversine_km(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"]), index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = dist / (dt_min / 60.0)
    speed = speed.where(dt_min > 0, np.where(dist > 0.5, np.inf, 0.0))
    teleport = pair & (speed > params["max_speed_kmh"])
    soc_jump = pair & ((df["soc"] - df["soc"].shift()).abs() > params["max_soc_jump"])
    anomaly = teleport | soc_jump
    n_pairs = int(pair.sum())
    rate = anomaly.sum() / n_pairs if n_pairs else float("nan")
    per_id = pd.DataFrame({"id": df["vehicle_id"], "pair": pair, "anom": anomaly}).groupby("id")[["pair", "anom"]].sum()
    per_id = per_id[per_id["pair"] >= 5]
    id_bad_share = float((per_id["anom"] / per_id["pair"] > params["anomaly_model_min"]).mean()) if len(per_id) else float("nan")

    # ⚠ [9/15 mock] 식별번호 수 부족을 먼저 보면, 모델 코드 2개에 수천 행씩 몰린 경우(이상치 57%)도 'unknown' 이 된다.
    #   식별번호가 적고 번호당 레코드가 많은 것 자체가 모델 코드 정황이므로, 이상치가 뚜렷하면 먼저 model 로 판정한다.
    #   식별번호 수·레코드 수 조건은 'individual' 판정에만 요구한다.
    if n_pairs < 50:
        verdict, why = "unknown", f"연속 레코드 쌍 {n_pairs}개 — 표본 부족"
    elif rate >= params["anomaly_model_min"]:
        verdict, why = "model", f"궤적 이상치 {rate:.2%} ≥ {params['anomaly_model_min']:.0%} — 여러 차량이 섞인 코드"
    elif n_ids < params["min_ids"] or rows_per_id < params["min_rows_per_id"]:
        verdict, why = "unknown", f"식별번호 {n_ids}개 · 번호당 {rows_per_id:,.1f}행 — 개별 차량 판정에 표본 부족"
    elif rate <= params["anomaly_individual_max"]:
        verdict, why = "individual", f"궤적 이상치 {rate:.2%} ≤ {params['anomaly_individual_max']:.0%}"
    else:
        verdict, why = "unknown", f"궤적 이상치 {rate:.2%} — 개별/모델 경계 구간"
    evidence = pd.DataFrame([
        {"항목": "레코드 수", "값": f"{n_rows:,}"},
        {"항목": "distinct 식별번호", "값": f"{n_ids:,}"},
        {"항목": "식별번호당 평균 레코드", "값": f"{rows_per_id:,.1f}"},
        {"항목": f"연속 레코드 쌍(≤{params['pair_max_gap_min']:.0f}분)", "값": f"{n_pairs:,}"},
        {"항목": f"순간이동 비율(>{params['max_speed_kmh']:.0f}km/h)", "값": f"{teleport.sum() / n_pairs:.2%}" if n_pairs else "—"},
        {"항목": f"SOC 급변 비율(>{params['max_soc_jump']:.0f}%p)", "값": f"{soc_jump.sum() / n_pairs:.2%}" if n_pairs else "—"},
        {"항목": "궤적 이상치 비율(합)", "값": f"{rate:.2%}" if n_pairs else "—"},
        {"항목": f"이상치 >{params['anomaly_model_min']:.0%} 식별번호 비율", "값": f"{id_bad_share:.1%}" if np.isfinite(id_bad_share) else "—"},
        {"항목": "판정", "값": verdict},
        {"항목": "근거", "값": why},
    ])
    log.info("CAN verify_join_key → %s (%s)", verdict, why)
    return verdict, evidence


def build_sessions(can_df, params, mode):
    """충전 세션 재구성.

    mode="vehicle"  (경로 A): 식별번호별 충전중여부 0→1 전이(또는 간격 초과)에서 세션 시작
    mode="geo_cell" (경로 B): 충전중 레코드만 '식별번호 + 위치격자'로 묶고 간격 초과에서 끊는 근사
    """
    gap = params["session_max_gap_min"]
    df = can_df
    if mode == "vehicle":
        prev_chg = df.groupby("vehicle_id")["charging"].shift(fill_value=False)
        dt = df.groupby("vehicle_id")["time"].diff().dt.total_seconds() / 60.0
        chg = df[df["charging"]].copy()
        start = (~prev_chg[chg.index]) | (dt[chg.index] > gap) | dt[chg.index].isna()
        chg["key"] = chg["vehicle_id"]
        chg["sid"] = start.astype(int).groupby(chg["key"]).cumsum()
    else:
        chg = df[df["charging"] & df["lat"].notna()].copy()
        deg = params["geo_cell_m"] / 111_000.0
        chg["key"] = chg["vehicle_id"] + "|" + np.floor(chg["lat"] / deg).astype(int).astype(str) + "|" + \
            np.floor(chg["lon"] / (deg / 0.8)).astype(int).astype(str)
        chg = chg.sort_values(["key", "time"], kind="mergesort")
        dt = chg.groupby("key")["time"].diff().dt.total_seconds() / 60.0
        chg["sid"] = ((dt > gap) | dt.isna()).astype(int).groupby(chg["key"]).cumsum()
    if chg.empty:
        return pd.DataFrame(columns=["vehicle_id", "start", "duration_min", "soc_gain", "lat", "lon", "date"])
    s = chg.groupby(["key", "sid"]).agg(vehicle_id=("vehicle_id", "first"), start=("time", "min"), end=("time", "max"),
                                         soc_start=("soc", "first"), soc_end=("soc", "last"),
                                         lat=("lat", "median"), lon=("lon", "median")).reset_index(drop=True)
    s["duration_min"] = (s["end"] - s["start"]).dt.total_seconds() / 60.0
    s["soc_gain"] = s["soc_end"] - s["soc_start"]
    s["date"] = s["start"].dt.normalize()
    keep = s["duration_min"].between(params["session_min_min"], params["session_max_min"])
    log.info("CAN 세션(%s): %d개 재구성 · 길이 조건(%.0f~%.0f분) 밖 %d개 제외 — ⚠ 길이는 샘플 간격만큼 과소추정",
             mode, len(s), params["session_min_min"], params["session_max_min"], int((~keep).sum()))
    return s[keep].reset_index(drop=True)


def duration_distribution(sessions, label):
    cats = pd.cut(sessions["duration_min"], DURATION_BINS, right=False)
    counts = cats.value_counts(sort=False)
    table = pd.DataFrame({"bin": [f"{int(i.left)}~{int(i.right)}분" for i in counts.index], "n": counts.to_numpy()})
    table["비율"] = table["n"] / table["n"].sum() if table["n"].sum() else np.nan
    table.insert(0, "경로", label)
    in_dwell = sessions["duration_min"].between(20, 40, inclusive="left").mean() if len(sessions) else np.nan
    summary = pd.DataFrame([{"경로": label, "세션수": len(sessions),
                             "중앙값_분": sessions["duration_min"].median(),
                             "평균_분": sessions["duration_min"].mean(),
                             "20~40분_비율": in_dwell,
                             "평균_SOC증가_%p": sessions["soc_gain"].mean()}])
    return table, summary


def mark_base_sessions(sessions, params):
    """경로 A — 차량별 충전 위치 DBSCAN. 서로 다른 날 repeat_min_days 회 이상 반복되는 클러스터 = 거점성(자가·직장)."""
    from sklearn.cluster import DBSCAN

    s = sessions.copy()
    s["is_base"] = False
    eps = params["dbscan_eps_m"] / 1000.0 / EARTH_RADIUS_KM
    for vid, g in s.dropna(subset=["lat", "lon"]).groupby("vehicle_id"):
        if len(g) < params["dbscan_min_samples"]:
            continue
        labels = DBSCAN(eps=eps, min_samples=params["dbscan_min_samples"], metric="haversine").fit_predict(
            np.radians(g[["lat", "lon"]].to_numpy()))
        g = g.assign(cluster=labels)
        days = g[g["cluster"] >= 0].groupby("cluster")["date"].nunique()
        base_clusters = set(days[days >= params["repeat_min_days"]].index)
        s.loc[g.index[g["cluster"].isin(base_clusters)], "is_base"] = True
    log.info("CAN 경로A 거점성 세션 %.1f%% (%d/%d)", 100 * s["is_base"].mean(), int(s["is_base"].sum()), len(s))
    return s


def station_agreement(points, stations, radius_m):
    """충전 위치(내부 좌표) 중 KEP_007 설치장소 반경 안에 드는 비율. 집계 수치만 돌려준다."""
    from sklearn.neighbors import BallTree

    st = stations.dropna(subset=["lat", "lon"])
    pts = points.dropna(subset=["lat", "lon"])
    if st.empty or pts.empty:
        return np.nan, len(pts)
    tree = BallTree(np.radians(st[["lat", "lon"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.radians(pts[["lat", "lon"]].to_numpy()), k=1)
    return float((d[:, 0] * EARTH_RADIUS_KM * 1000 <= radius_m).mean()), len(pts)


# ---------------------------------------------------------------- 고유 데이터 활용: 원정 충전·시간 이동 여지

def infer_home(can_df, params):
    """차량별 '밤 주차 위치'로 거주지를 추정한다(개별 차량 판정일 때만 의미가 있다).

    밤(home_hours, 기본 21~06시) 레코드로 밤마다 '가장 오래 머문 격자'(레코드가 가장 많은 geo_cell_m 격자)를 구하고,
    그 격자들의 최빈값을 거주지로 본다. 마지막 위치를 쓰지 않는 것은 새벽 주행 기록이 마지막이 되기 때문이다.
    최빈 격자에 home_min_nights 밤 이상, 전체 밤의 home_min_share 이상 머문 차량만 인정한다.
    ⚠ 주차 중 기록을 남기지 않는 차량은 밤 기록이 없어 거주지를 못 잡는다 — 결과는 '거주지 추정 차량' 기준이다.
    반환: vehicle_id, home_lat, home_lon, nights — 차량 단위라 센터 밖으로 내보내지 않는다(내부 계산 전용).
    """
    start_h, end_h = params["home_hours"]
    d = can_df.dropna(subset=["lat", "lon"])
    h = d["time"].dt.hour
    d = d[(h >= start_h) | (h < end_h)].copy()
    if d.empty:
        return pd.DataFrame(columns=["vehicle_id", "home_lat", "home_lon", "nights"])
    d["night"] = (d["time"] - pd.Timedelta(hours=end_h)).dt.normalize()
    deg = params["geo_cell_m"] / 111_000.0
    d["cell"] = np.floor(d["lat"] / deg).astype(int).astype(str) + "|" + \
        np.floor(d["lon"] / (deg / 0.8)).astype(int).astype(str)
    stay = d.groupby(["vehicle_id", "night", "cell"]).size().rename("n").reset_index()
    nightly = stay.sort_values(["vehicle_id", "night", "n", "cell"], ascending=[True, True, False, True]) \
        .drop_duplicates(["vehicle_id", "night"])
    rows = []
    for vid, g in nightly.groupby("vehicle_id"):
        counts = g["cell"].value_counts()
        top, n_top = counts.index[0], int(counts.iloc[0])
        if n_top >= params["home_min_nights"] and n_top / len(g) >= params["home_min_share"]:
            at = d[(d["vehicle_id"] == vid) & (d["cell"] == top)]
            rows.append({"vehicle_id": vid, "home_lat": at["lat"].median(), "home_lon": at["lon"].median(), "nights": n_top})
    return pd.DataFrame(rows, columns=["vehicle_id", "home_lat", "home_lon", "nights"])


def away_charging(sessions, home, centroids, params, min_n):
    """거주지에서 away_km 넘게 떨어진 곳의 충전 = 원정 충전. 거주 법정동별로 집계한다.

    반환: (법정동 표[내부], 법정동 표[반출용: 거주 차량 min_n 미만 가림], 요약 표). 차량 단위 표는 돌려주지 않는다.
    '거주지 충전 없음' = 기간 중 거주지 반경 안 충전이 한 번도 없는 차량 — 자가충전 사각지대의 행동 신호(대리지표 아님).
    """
    s = sessions.merge(home, on="vehicle_id", how="inner")
    total_vehicles = sessions["vehicle_id"].nunique()
    if s.empty:
        summary = pd.DataFrame([{"항목": "거주지 추정 차량", "값": f"0 / {total_vehicles}대 — 원정 충전 분석 불가"}])
        return None, None, summary
    s["dist_km"] = haversine_km(s["lat"], s["lon"], s["home_lat"], s["home_lon"])
    s["away"] = s["dist_km"] > params["away_km"]
    per = s.groupby("vehicle_id").agg(n=("away", "size"), n_away=("away", "sum"),
                                      away_km=("dist_km", lambda v: v[v > params["away_km"]].mean()),
                                      home_lat=("home_lat", "first"), home_lon=("home_lon", "first")).reset_index()
    per["no_home"] = per["n_away"] == per["n"]
    per["bjd_code"], _ = nearest_region(per["home_lat"], per["home_lon"], centroids, params["centroid_max_km"])
    reg = per.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
        거주추정_차량수=("vehicle_id", "size"), 세션수=("n", "sum"), 원정세션수=("n_away", "sum"),
        거주지충전없음_차량비율=("no_home", "mean"), 원정_평균거리_km=("away_km", "mean")).reset_index()
    reg["원정충전_세션비율"] = reg["원정세션수"] / reg["세션수"]
    reg = reg[["bjd_code", "거주추정_차량수", "원정충전_세션비율", "거주지충전없음_차량비율", "원정_평균거리_km"]]
    summary = pd.DataFrame([
        {"항목": "거주지 추정 차량", "값": f"{len(per)} / {total_vehicles}대 (밤 {params['home_hours'][0]}~{params['home_hours'][1]}시 "
                                       f"주차 위치 최빈 격자, {params['home_min_nights']}밤 이상)"},
        {"항목": "원정 충전 세션 비율", "값": f"{s['away'].mean():.1%} (거주지에서 {params['away_km']:g}km 초과)"},
        {"항목": "거주지 충전이 없는 차량 비율", "값": f"{per['no_home'].mean():.1%}"},
        {"항목": "원정 충전 평균 거리", "값": f"{per['away_km'].mean():.2f} km" if per["away_km"].notna().any() else "—"},
        {"항목": "집계 법정동(거주 차량 {0}대 이상)".format(min_n), "값": f"{int((reg['거주추정_차량수'] >= min_n).sum())}곳"},
        {"항목": "해석", "값": "CAN 은 일부 차량 표본(2022.11~2023.11 무렵) — 경향 확인용. 차량 단위 결과는 반출하지 않음"},
    ])
    return reg, _suppress(reg, "거주추정_차량수", min_n), summary


def away_access_check(away_region, access, min_n, share=0.20):
    """원정 충전(고유 데이터 행동 신호)으로 형평성 축(2SFCA)을 검증한다. 거주 차량 min_n 이상 법정동만."""
    a = away_region[away_region["거주추정_차량수"] >= min_n].merge(
        access[["bjd_code", "access_2sfca"]].assign(bjd_code=access["bjd_code"].astype(str)), on="bjd_code", how="inner")
    if len(a) < 3:
        return pd.DataFrame([{"항목": "비교 법정동 수", "값": f"{len(a)} — 3곳 미만이라 검증 불가"}])
    low = a["access_2sfca"] <= a["access_2sfca"].quantile(share)
    rows, stats = [{"항목": "비교 법정동 수", "값": f"{len(a)}"}], {}
    # 주 지표 = 원정 충전 세션 비율(동네마다 값이 갈린다). 보조 = 거주지 충전 없는 차량 비율(0 에 몰리기 쉽다).
    for col, label in (("원정충전_세션비율", "원정 충전 세션 비율"), ("거주지충전없음_차량비율", "거주지 충전 없는 차량 비율")):
        v = a[col].astype(float)
        rho = v.rank().corr(a["access_2sfca"].rank()) if v.nunique() > 1 else np.nan
        lo, hi = v[low].mean(), v[~low].mean()
        stats[col] = (rho, lo, hi)
        rows += [{"항목": f"순위 상관({label} vs 2SFCA, 스피어만)",
                  "값": f"{rho:.3f} (음수면 접근성 낮을수록 높음)" if np.isfinite(rho) else "— (동네 값이 모두 같아 계산 불가)"},
                 {"항목": f"{label}: 접근성 하위 {share:.0%} 동네 / 나머지", "값": f"{lo:.1%} / {hi:.1%}"}]
    rows.append({"항목": "해석", "값": "CAN(2022.11~2023.11 무렵)과 2SFCA(충전소·등록 기준 시점)의 기간이 다르다 — 방향 확인용"})
    out = pd.DataFrame(rows)
    rho, lo, hi = stats["원정충전_세션비율"]
    out.attrs.update(rho=float(rho), low=float(lo), high=float(hi), n=len(a), no_home=stats["거주지충전없음_차량비율"])
    return out


def soc_flexibility(sessions, peak_hours, flex_soc, min_n):
    """피크 시간대에 시작한 충전 중 시작 SOC 가 flex_soc 이상인 비율 — 충전 시간 이동(DR) 여지의 상한 신호.

    출차 시각을 모르므로 '옮길 수 있다'가 아니라 '배터리가 급하지 않았다'는 뜻이다. 반환: (SOC 구간 표, 요약 표).
    """
    s = sessions.dropna(subset=["soc_start"])
    soc = s["soc_start"].astype(float)
    if len(soc) and soc.max() <= 1.0:   # 0~1 표기면 %로
        soc = soc * 100
    peak = s["start"].dt.hour.isin(peak_hours).to_numpy()
    soc_p = soc[peak]
    bands = pd.cut(soc_p, [0, 20, 40, 60, 80, 100.01], right=False,
                   labels=["0~20%", "20~40%", "40~60%", "60~80%", "80~100%"]).value_counts(sort=False)
    table = pd.DataFrame({"시작 SOC": bands.index.astype(str), "피크 세션 수": bands.to_numpy()})
    table["비율"] = table["피크 세션 수"] / max(1, len(soc_p))
    table.loc[table["피크 세션 수"] < min_n, ["피크 세션 수", "비율"]] = np.nan
    share = float((soc_p >= flex_soc).mean()) if len(soc_p) >= min_n else np.nan
    summary = pd.DataFrame([
        {"항목": "피크 시간대(KEPCO 사용량 상위)", "값": ", ".join(f"{h}시" for h in sorted(peak_hours))},
        {"항목": "피크 시간대 시작 세션 수", "값": f"{len(soc_p):,}"},
        {"항목": f"시작 SOC {flex_soc:g}% 이상 비율", "값": f"{share:.1%}" if np.isfinite(share) else "— (세션 부족)"},
        {"항목": "시작 SOC 중앙값", "값": f"{soc_p.median():.0f}%" if len(soc_p) >= min_n else "—"},
        {"항목": "해석", "값": "출차 시각을 몰라 실제로 옮길 수 있는지는 모름 — DR 여지의 상한 신호"},
    ])
    summary.attrs["share"] = share
    return table, summary


def peak_hours_from(kepco_mh, n=4):
    """KEPCO 서울 전체 시간대별 사용량 상위 n 시각. 없으면 저녁 18~21시."""
    if kepco_mh is None or kepco_mh.empty:
        return [18, 19, 20, 21]
    return sorted(int(h) for h in kepco_mh.groupby("hour")["kwh"].sum().nlargest(n).index)


def run_can_stage(path, columns, params, centroids, stations, activation, feature_before, kepco_mh=None):
    """3단계 전체. 반환 dict 의 표는 전부 법정동 집계(좌표 없음).

    activation=None(v11 상권 단계 비활성): 처치 정제(treated_excl_base)와 변화점 신뢰도 표시를 만들지 않는다.
    세션 시간대 모양 u(t)(charging_shape)과 반복 충전 위치 비율(공간 반복성 참고)은 그대로 낸다.
    """
    can = load_can_m(path, columns, params)
    verdict, evidence = verify_join_key(can, params)
    path_label = "A(개별 차량)" if verdict == "individual" else "B(법정동 밀집도, 정황상 보조 근거)"
    sessions = build_sessions(can, params, "vehicle" if verdict == "individual" else "geo_cell")
    if centroids is None:
        raise ValueError("CAN 좌표를 법정동으로 보내려면 paths.emd_centroids 가 필요")
    codes, _ = nearest_region(sessions["lat"], sessions["lon"], centroids, params["centroid_max_km"])
    sessions["bjd_code"] = codes
    dist_table, dist_summary = duration_distribution(sessions, path_label)
    min_n = params["min_cell_count"]
    before = pd.Timestamp(mi_to_ym(ym_to_mi(feature_before)) + "-01")
    feat_sessions = sessions[sessions["start"] < before]
    out = {"verdict": verdict, "path": path_label, "evidence": evidence,
           "duration_table": dist_table, "duration_summary": dist_summary}
    if kepco_mh is not None:
        out["kepco_shape"], out["kepco_check"] = kepco_cross_check(sessions, kepco_mh, min_n)
    peak = params["peak_hours"] or peak_hours_from(kepco_mh)
    out["flex_bands"], out["flex_summary"] = soc_flexibility(sessions, peak, params["flex_soc"], min_n)
    if verdict == "individual":   # 거주지 추정은 식별번호가 개별 차량일 때만 의미가 있다
        out["away_region"], out["away_region_export"], out["away_summary"] = \
            away_charging(sessions, infer_home(can, params), centroids, params, min_n)
    else:
        out["away_region"], out["away_region_export"] = None, None
        out["away_summary"] = pd.DataFrame([{"항목": "원정 충전 분석", "값": f"생략 — 식별번호 판정 {verdict}(개별 차량 아님)"}])

    if verdict == "individual":
        sessions = mark_base_sessions(sessions, params)
        feat_sessions = sessions[sessions["start"] < before]
        by_type = []
        for label, sub in (("거점성", sessions[sessions["is_base"]]), ("비거점(공용)", sessions[~sessions["is_base"]])):
            t, sm = duration_distribution(sub, label)
            by_type.append(sm)
        out["duration_by_type"] = pd.concat(by_type, ignore_index=True)
        reg = sessions.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
            세션수=("start", "size"), 차량수=("vehicle_id", "nunique"), 거점성비율=("is_base", "mean"),
            평균세션_분=("duration_min", "mean")).reset_index()
        out["region_table"] = reg
        feats = feat_sessions.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
            CAN_평균세션_분=("duration_min", "mean"), CAN_거점성비율=("is_base", "mean"), _n=("start", "size")).reset_index()
        out["treated_excl_base"] = None
        if activation is not None:
            reg["변화점_신뢰도"] = np.where(reg["거점성비율"] >= params["base_share_flag"], "낮음(거점성 다수)", "보통")
            low = set(reg.loc[reg["변화점_신뢰도"].str.startswith("낮음"), "bjd_code"])
            treated = activation[activation["status"] == "treated"][["bjd_code", "T_r"]].copy()
            treated["CAN_거점성비율"] = treated["bjd_code"].map(reg.set_index("bjd_code")["거점성비율"])
            treated["거점성제외_포함"] = ~treated["bjd_code"].isin(low)
            out["treated_excl_base"] = treated
        agree, n_pts = station_agreement(sessions[~sessions["is_base"]], stations, params["station_match_m"]) \
            if stations is not None else (np.nan, 0)
        out["region_table_export"] = _suppress(reg, "차량수", min_n)
    else:
        reg = sessions.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
            충전세션수_근사=("start", "size"), 충전일수=("date", "nunique"), 평균세션_분=("duration_min", "mean")).reset_index()
        total_days = max(1, int(sessions["date"].nunique()))
        reg["밀집도_일평균세션"] = reg["충전세션수_근사"] / total_days
        reg = reg.sort_values("충전세션수_근사", ascending=False)
        out["region_table"] = reg
        feats = feat_sessions.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
            CAN_평균세션_분=("duration_min", "mean"), _n=("start", "size")).reset_index()
        out["treated_excl_base"] = None
        agree, n_pts = station_agreement(sessions, stations, params["station_match_m"]) \
            if stations is not None else (np.nan, 0)
        out["region_table_export"] = _suppress(reg, "충전세션수_근사", min_n)
    feats = feats[feats["_n"] >= min_n].drop(columns="_n")
    out["features"] = feats
    out["charging_shape"] = charging_shape(feat_sessions[feat_sessions["end"] < before]) if verdict == "individual" else None
    out["spatial"] = pd.DataFrame([{"경로": path_label, "비교 충전위치 수": n_pts,
                                    f"KEP_007 {params['station_match_m']:.0f}m 이내 비율": agree,
                                    "비고": "KEP_007 은 2019 기준 — 이후 설치분은 불일치로 잡힘"}])
    log.info("3단계 CAN 완료: 경로 %s · 공간 일치율 %s", path_label, f"{agree:.1%}" if np.isfinite(agree) else "—")
    return out


def _occupancy(sessions):
    """세션이 시각(0~23)별로 충전 중이던 시간(h)의 합."""
    occupancy = np.zeros(24)
    for row in sessions.itertuples():
        start, end = pd.Timestamp(row.start), pd.Timestamp(row.end)
        if end <= start:
            continue
        for hour in pd.date_range(start.floor("h"), end.floor("h"), freq="h"):
            overlap = (min(end, hour + pd.Timedelta(hours=1)) - max(start, hour)).total_seconds()
            occupancy[hour.hour] += max(0, overlap) / 3600
    return occupancy


def charging_shape(sessions):
    """사전기간 세션의 시간대 점유 모양(최대=1). 충전기 수 분모가 없어 실제 이용률은 아니다."""
    if len(sessions) < 3:
        return None
    occupancy = _occupancy(sessions)
    return occupancy / occupancy.max() if occupancy.max() > 0 else None


def kepco_cross_check(sessions, kepco_mh, min_n):
    """CAN 충전 세션과 KEPCO_001을 겹치는 달에서 맞대 본다 — 서로 다른 안심구역 원천의 교차검증.

    kepco_mh: 법정동×월×시각 KEPCO 집계(소표본 셀을 뺀 반출용). 둘 다 서울 전체·법정동 집계만 쓴다.
    ① 시간대 모양: CAN 충전 점유 시간 비중 vs KEPCO 사용량 비중(각각 합 1) — 피어슨 상관·피크 시각
    ② 공간 순위: 세션 min_n 이상 법정동의 CAN 세션 수 vs KEPCO 사용량 — 스피어만 순위 상관
    CAN 은 일부 차량 표본이라 모집단 부하가 아니다. 상관이 낮아도 그대로 보고한다.
    반환: (시각별 비중 표 또는 None, 요약 표)
    """
    s = sessions.dropna(subset=["bjd_code"])
    s = s.assign(mi=s["start"].dt.year * 12 + s["start"].dt.month - 1)
    months = sorted(set(s["mi"]) & set(kepco_mh["mi"]))
    if not months:
        can_span = f"{mi_to_ym(s['mi'].min())}~{mi_to_ym(s['mi'].max())}" if len(s) else "세션 없음"
        kep_span = f"{mi_to_ym(kepco_mh['mi'].min())}~{mi_to_ym(kepco_mh['mi'].max())}" if len(kepco_mh) else "없음"
        return None, pd.DataFrame([{"항목": "겹치는 달", "값": f"없음 (CAN {can_span} · KEPCO {kep_span})"}])
    s, k = s[s["mi"].isin(months)], kepco_mh[kepco_mh["mi"].isin(months)]
    can_h = _occupancy(s)
    kep_h = k.groupby("hour")["kwh"].sum().reindex(range(24), fill_value=0).to_numpy(float)
    shape = pd.DataFrame({"시각": range(24), "CAN_충전점유_비중": can_h / can_h.sum() if can_h.sum() else np.nan,
                          "KEPCO_사용량_비중": kep_h / kep_h.sum() if kep_h.sum() else np.nan})
    r_shape = shape["CAN_충전점유_비중"].corr(shape["KEPCO_사용량_비중"])

    counts = s.groupby("bjd_code").size()
    counts = counts[counts >= min_n]
    kwh = k.groupby("bjd_code")["kwh"].sum()
    both = pd.concat([counts.rename("can"), kwh.rename("kepco")], axis=1, join="inner")
    rho = both["can"].rank().corr(both["kepco"].rank()) if len(both) >= 3 else np.nan
    summary = pd.DataFrame([
        {"항목": "비교 기간", "값": f"{mi_to_ym(months[0])}~{mi_to_ym(months[-1])} ({len(months)}개월)"},
        {"항목": "CAN 세션 수(기간 내)", "값": f"{len(s):,}"},
        {"항목": "시간대 모양 상관(피어슨)", "값": f"{r_shape:.3f}" if np.isfinite(r_shape) else "—"},
        {"항목": "피크 시각 CAN / KEPCO", "값": f"{int(np.argmax(can_h))}시 / {int(np.argmax(kep_h))}시"},
        {"항목": f"공간 비교 법정동 수(CAN 세션 {min_n}건 이상)", "값": f"{len(both)}"},
        {"항목": "법정동 순위 상관(스피어만)", "값": f"{rho:.3f}" if np.isfinite(rho) else "— (3곳 미만)"},
        {"항목": "해석", "값": "CAN 은 일부 차량 표본 — 모양·순위 일치 여부만 참고, 부하 크기 추정에 쓰지 않음"},
    ])
    return shape, summary


def _suppress(df, count_col, min_n):
    out = df.copy()
    small = out[count_col] < min_n
    for c in out.columns:
        if c != "bjd_code" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("float64")
            out.loc[small, c] = np.nan
    return out


if __name__ == "__main__":
    if not COPY_PATH:
        raise ValueError("canloader.py 상단 COPY_PATH에 센터의 CAN CSV Copy Path를 붙여 넣으세요")
    result = inspect_can_copy_path(COPY_PATH)
    for name, table in result.items():
        print(f"\n[{name}]\n{table.to_string(index=False)}")
