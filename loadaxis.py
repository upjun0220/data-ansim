"""8단계 — 전력 축(부하 집중도) + 4사분면 처방.

집중도(r) = max_h Load(r, h) / Σ_h Load(r, h)      h: 시각(0~23)

- KEPCO 값은 1시간 구간 에너지(kWh)다. kWh ÷ 1h 는 그 구간의 **평균 kW** 이지 순간 최대전력이 아니다.
  여기서 Load(r, h) = Σ kWh(r, h) / 관측일수 = 시각 h 의 일평균 평균kW. 리포트에도 이 구분을 남긴다.
- 시간대구간(TIZO)이 아니라 1시간 단위로 계산한다. 구간 폭이 다르면(예: 3시간 구간 vs 6시간 구간)
  kWh 점유율이 폭에 비례해 부풀기 때문이다.

독립성 강제: 이 모듈의 부하 계산은 KEPCO_001/002 집계표만 받는다(SHC·CAN·CATE 피처를 받지 않는다).
  4사분면 결합 시 CATE 표가 부하 계열 피처로 추정됐으면(attrs.feature_names) 멈춘다.
단독 실행하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import log, mi_to_ym, ym_to_mi
from heterogeneity import assert_no_load_features

KEPCO_COLUMNS = {"bjd_code", "mi", "hour", "kwh", "n_days"}

QUADRANTS = {
    (True, False): ("① 접근성·수요 확인", "상권 효과 큼 · 충전 부하 분산 → 공공 접근성·수요 확인 후 검토"),
    (True, True): ("② 부하 완화 검토", "상권 효과 큼 · 충전 부하 집중 → ESS 시뮬레이션·현장 검토"),
    (False, False): ("③ 공공 필요 확인", "상권 효과 작음 · 충전 부하 분산 → 매출 효과와 별도로 공공 필요 확인"),
    (False, True): ("④ 부하 완화·공공 필요 검토", "상권 효과 작음 · 충전 부하 집중 → 투자 배제 없이 ESS·접근성 검토"),
}


# CATE 를 보고하지 않을 때(BLP 교정 미충족) 부하 집중도만으로 나누는 두 칸
QUADRANTS_LOAD_ONLY = {
    True: ("⑤ 부하 집중(CATE 미보고)", "CATE 미보고 · 충전 부하 집중 → ESS 시뮬레이션·현장 검토"),
    False: ("⑥ 부하 분산(CATE 미보고)", "CATE 미보고 · 충전 부하 분산 → 매출 효과와 별도로 공공 필요 확인"),
}


def compute_concentration(kepco_mh, period):
    missing = KEPCO_COLUMNS - set(kepco_mh.columns)
    if missing:
        raise ValueError(f"load_axis 는 KEPCO 월×시각 집계표만 받는다. 없는 열: {sorted(missing)}")
    p0, p1 = (ym_to_mi(v) for v in period)
    df = kepco_mh[kepco_mh["mi"].between(p0, p1)]
    if df.empty:
        raise ValueError(f"부하 집중도 기간 {period} 에 KEPCO 데이터가 없음")
    prof = df.groupby(["bjd_code", "hour"]).agg(kwh=("kwh", "sum"), n_days=("n_days", "sum")).reset_index()
    prof["avg_kw"] = prof["kwh"] / prof["n_days"].where(prof["n_days"] > 0)   # kWh/1h ÷ 일수 = 평균 kW
    g = prof.groupby("bjd_code")
    out = pd.DataFrame({
        "concentration": g["avg_kw"].max() / g["avg_kw"].sum(),
        "peak_hour": prof.loc[g["avg_kw"].idxmax(), ["bjd_code", "hour"]].set_index("bjd_code")["hour"],
        "peak_avg_kw": g["avg_kw"].max(),
        "daily_kwh": g["avg_kw"].sum(),
        "n_hours": g["hour"].nunique(),
    }).reset_index()
    short = out["n_hours"] < 24
    if short.any():
        log.warning("부하 집중도: 24개 시각이 다 없는 법정동 %d곳 — 집중도가 과대평가될 수 있음", int(short.sum()))
    log.info("8단계 부하 집중도(%s~%s): 법정동 %d · 중앙값 %.3f (균등분포면 %.3f)",
             mi_to_ym(p0), mi_to_ym(p1), len(out), out["concentration"].median(), 1 / 24)
    return out


def _cut(values, rule):
    if rule == "median":
        return float(np.nanmedian(values))
    if rule == "zero":
        return 0.0
    return float(rule)


def classify_quadrants(cate, concentration, params, reserved=None):
    """(CATE, 부하 집중도) → 4사분면. reserved: {bjd_code: 사유} — 유보 지역은 표시만 하고 제외하지 않는다."""
    assert_no_load_features(cate.attrs.get("feature_names", []))
    df = cate[["bjd_code", "cate", "W", "method"]].merge(concentration, on="bjd_code", how="inner")
    lost = set(cate["bjd_code"]) - set(df["bjd_code"])
    if lost:
        log.warning("4사분면: 부하 집중도가 없는 법정동 %d곳 제외", len(lost))
    conc_cut = _cut(df["concentration"], params["conc_cut"])
    hi_conc = df["concentration"] > conc_cut
    if df["cate"].isna().all():   # CATE 미보고 — NaN 을 "효과 작음"으로 오분류하지 않는다
        cate_cut = np.nan
        df["quadrant"] = [QUADRANTS_LOAD_ONLY[bool(b)][0] for b in hi_conc]
        df["처방"] = [QUADRANTS_LOAD_ONLY[bool(b)][1] for b in hi_conc]
    else:
        cate_cut = _cut(df["cate"], params["cate_cut"])
        hi_cate = df["cate"] > cate_cut
        df["quadrant"] = [QUADRANTS[(a, b)][0] for a, b in zip(hi_cate, hi_conc)]
        df["처방"] = [QUADRANTS[(a, b)][1] for a, b in zip(hi_cate, hi_conc)]
    reserved = reserved or {}
    df["유보사유"] = df["bjd_code"].map(reserved).fillna("")
    df["reserved"] = df["유보사유"] != ""
    df = df.sort_values(["quadrant", "cate"], ascending=[True, False]).reset_index(drop=True)
    log.info("8단계 4사분면: %s (CATE 기준 %.4f · 집중도 기준 %.3f · 유보 %d)",
             df["quadrant"].value_counts().sort_index().to_dict(), cate_cut, conc_cut, int(df["reserved"].sum()))
    return df, cate_cut, conc_cut
