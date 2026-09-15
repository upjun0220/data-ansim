"""3단계 보조 — 전기차충전소설치현황(KEP_007).

용도는 두 가지뿐이다.
  1) 공간 sanity check — CAN 충전 위치가 실제 설치장소 근처인지(공간 일치율)
  2) 정적 인프라 스톡 변수 — 법정동별 급속·완속 대수(CATE 피처)
⚠ 설치일자 컬럼이 없다(확인된 부재). 포털 메타데이터 2019-12-27 갱신. '변화점 = 설치시점' 대조에 쓰지 않는다.
⚠ 좌표(LTD/LNGT)는 내부 계산에만 쓰고 산출물에는 법정동 집계만 내보낸다.
단독 실행하지 않는다.
"""
from __future__ import annotations

import pandas as pd

from common import clean_code, log, nearest_region, norm_text, read_columns, to_num


def load_kep007(path, columns, params, centroids=None, centroid_max_km=3.0, lat_range=(33, 39), lon_range=(124, 132)):
    df = read_columns(path, columns["kep007"], "KEP_007",
                      required=["sido_cd", "sgg_cd", "fast", "slow", "lat", "lon"])
    sido = clean_code(df["sido_cd"], 2).replace(params["sido_code_remap"])
    sgg = clean_code(df["sgg_cd"])
    # SGNG_CD 가 5자리(시도 포함)인지 3자리인지 불명 → 둘 다 받는다. 시도코드 개편(42→51 등)은 앞 2자리에도 적용
    sgg5 = sgg.where(sgg.str.len() >= 5, sido + sgg.fillna("").str.zfill(3))
    sgg5 = sgg5.str[:2].replace(params["sido_code_remap"]) + sgg5.str[2:5]
    st = pd.DataFrame({
        "sido_code": sido, "sigungu_code": sgg5,
        "place": norm_text(df["place"]) if "place" in df else "",
        "fast": to_num(df["fast"]).fillna(0), "slow": to_num(df["slow"]).fillna(0),
        "lat": to_num(df["lat"]), "lon": to_num(df["lon"]),
    })
    bad = ~(st["lat"].between(*lat_range) & st["lon"].between(*lon_range))
    if bad.any():
        log.warning("KEP_007 좌표 범위 밖/결측 %d건 — 공간 일치율·법정동 집계에서 제외", int(bad.sum()))
        st.loc[bad, ["lat", "lon"]] = float("nan")
    st["bjd_code"] = None
    if centroids is not None:
        ok = st["lat"].notna()
        codes, _ = nearest_region(st.loc[ok, "lat"], st.loc[ok, "lon"], centroids, centroid_max_km)
        st.loc[ok, "bjd_code"] = codes
    log.info("KEP_007: 설치장소 %d · 급속 %d대 · 완속 %d대 (⚠ 2019 기준 정적 스냅샷)",
             len(st), int(st["fast"].sum()), int(st["slow"].sum()))
    return st


def region_stock(stations):
    """법정동별 인프라 스톡. 설치장소 수·급속·완속 대수."""
    st = stations.dropna(subset=["bjd_code"])
    return st.groupby("bjd_code").agg(인프라_설치장소수=("place", "size"), 인프라_급속대수=("fast", "sum"),
                                      인프라_완속대수=("slow", "sum")).reset_index()
