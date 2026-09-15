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
단독 실행하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import EARTH_RADIUS_KM, haversine_km, log, mi_to_ym, nearest_region, read_columns, to_num, ym_to_mi

DURATION_BINS = [0, 10, 20, 30, 40, 60, 90, 120, 240, 480, 1440]


def load_can_m(path, columns, params, nrows=None):
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


def run_can_stage(path, columns, params, centroids, stations, activation, feature_before):
    """3단계 전체. 반환 dict 의 표는 전부 법정동 집계(좌표 없음)."""
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
        reg["변화점_신뢰도"] = np.where(reg["거점성비율"] >= params["base_share_flag"], "낮음(거점성 다수)", "보통")
        out["region_table"] = reg
        feats = feat_sessions.dropna(subset=["bjd_code"]).groupby("bjd_code").agg(
            CAN_평균세션_분=("duration_min", "mean"), CAN_거점성비율=("is_base", "mean"), _n=("start", "size")).reset_index()
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
    out["spatial"] = pd.DataFrame([{"경로": path_label, "비교 충전위치 수": n_pts,
                                    f"KEP_007 {params['station_match_m']:.0f}m 이내 비율": agree,
                                    "비고": "KEP_007 은 2019 기준 — 이후 설치분은 불일치로 잡힘"}])
    log.info("3단계 CAN 완료: 경로 %s · 공간 일치율 %s", path_label, f"{agree:.1%}" if np.isfinite(agree) else "—")
    return out


def _suppress(df, count_col, min_n):
    out = df.copy()
    small = out[count_col] < min_n
    for c in out.columns:
        if c != "bjd_code" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("float64")
            out.loc[small, c] = np.nan
    return out
