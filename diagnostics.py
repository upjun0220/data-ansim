"""6.5단계 처치오염 진단 — T_r 전후 SHC001 가맹점 개설률 동반 급등.

충전 활성화와 같은 시기에 상권 자체가 커지고 있었다면(신규 가맹점이 한꺼번에 생김) 매출 변화를 충전 효과로
볼 수 없다. 이런 지역은 제외하지 않고 '유보' 플래그만 달아 CATE·처방 산출물에서 시각적으로 구분한다.

개설률(r, m) = 신규 등록 가맹점수 / 가동 가맹점수 (같은 달)
  - FRNC_STAT_CD 의 '등록' 행이 그 달 신규 등록(흐름)인지 등록 상태 전체(재고)인지 정의서만으로 불명.
    industry_codes.shc001.open_mode 로 분기한다.
      flow        — new_status_codes 행의 FRNC_CNT 를 신규로 본다
      stock_diff  — 재고의 전월 대비 증가분(음수는 0)을 신규로 본다
진단: 사후/사전 개설률 비율을 같은 달의 never-treated 중앙값 비율로 나눈 '초과 비율' ≥ excess_ratio 이고
      절대 증가폭 ≥ min_abs_increase 이면 유보.
단독 실행하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bjd_mapping import canonicalize, shc_bjd_code
from common import clean_label, log, mi_to_ym, parse_mi, read_columns, to_num


def load_shc001_monthly(path, columns, shc001_codes, chunksize=2_000_000, crosswalk=None):
    """SHC001 → 법정동×월 (new, closed, stock, open_rate, close_rate)."""
    new_codes = {str(c) for c in shc001_codes.get("new_status_codes", [])}
    close_codes = {str(c) for c in shc001_codes.get("close_status_codes", [])}
    stock_codes = {str(c) for c in shc001_codes.get("stock_status_codes", [])}
    oper_codes = {str(c) for c in shc001_codes.get("stock_oper_codes", [])}
    mode = shc001_codes.get("open_mode", "flow")
    if not stock_codes:
        raise ValueError("industry_codes.shc001.stock_status_codes 가 비어 있음 — 현장에서 FRNC_STAT_CD 값 확인 후 채울 것")
    parts = []
    for chunk in read_columns(path, columns["shc001"], "SHC001", chunksize=chunksize):
        status = clean_label(chunk["status"])
        oper = clean_label(chunk["oper"])
        cnt = to_num(chunk["cnt"]).fillna(0)
        stock_mask = status.isin(stock_codes) & (oper.isin(oper_codes) if oper_codes else True)
        df = pd.DataFrame({
            "bjd_code": canonicalize(shc_bjd_code(chunk["sido_cd"], chunk["sgg_cd"], chunk["umd_cd"]), crosswalk),
            "mi": parse_mi(chunk["ym"]),
            "new": cnt.where(status.isin(new_codes), 0),
            "closed": cnt.where(status.isin(close_codes), 0),
            "stock": cnt.where(stock_mask, 0),
        }).dropna(subset=["bjd_code", "mi"])
        parts.append(df.groupby(["bjd_code", "mi"])[["new", "closed", "stock"]].sum().reset_index())
    out = pd.concat(parts, ignore_index=True).groupby(["bjd_code", "mi"])[["new", "closed", "stock"]].sum().reset_index()
    out["mi"] = out["mi"].astype(int)
    out = out.sort_values(["bjd_code", "mi"]).reset_index(drop=True)
    if mode == "stock_diff":
        # 신규(총 개설) ≈ 재고 순증 + 해지. 해지 코드가 없으면 순감소분을 해지로 본다. 첫 달·월 공백은 계산 불가(NaN)
        diff = out.groupby("bjd_code")["stock"].diff()
        gap = out.groupby("bjd_code")["mi"].diff() != 1
        if not close_codes:
            out["closed"] = (-diff).clip(lower=0)
        out["new"] = (diff + out["closed"]).clip(lower=0).where(~gap)
        out["closed"] = out["closed"].where(~gap)
    stock = out["stock"].where(out["stock"] > 0)
    out["open_rate"] = out["new"] / stock
    out["close_rate"] = out["closed"] / stock
    log.info("SHC001: 법정동 %d · 월 %s~%s · 개설 방식 %s", out["bjd_code"].nunique(),
             mi_to_ym(out["mi"].min()), mi_to_ym(out["mi"].max()), mode)
    return out


def contamination_check(shc001m, activation, params):
    w = int(params["window"])
    rate = shc001m.pivot(index="bjd_code", columns="mi", values="open_rate")
    treated = activation[activation["status"] == "treated"]
    controls = [c for c in activation.loc[activation["status"] == "never_treated", "bjd_code"] if c in rate.index]

    def window_means(codes, g):
        pre_cols = [m for m in range(g - w, g) if m in rate.columns]
        post_cols = [m for m in range(g, g + w) if m in rate.columns]
        sub = rate.reindex(codes)
        pre = sub[pre_cols].mean(axis=1) if pre_cols else pd.Series(np.nan, index=sub.index)
        post = sub[post_cols].mean(axis=1) if post_cols else pd.Series(np.nan, index=sub.index)
        return pre, post

    rows = []
    for _, row in treated.iterrows():
        code, g = row["bjd_code"], int(row["T_r_mi"])
        pre, post = window_means([code], g)
        pre, post = float(pre.iloc[0]), float(post.iloc[0])
        cpre, cpost = window_means(controls, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_ratio = (cpost / cpre).replace([np.inf, -np.inf], np.nan).dropna()
        ctrl_ratio = float(c_ratio.median()) if len(c_ratio) else np.nan
        own_ratio = post / pre if pre and np.isfinite(pre) and pre > 0 else np.nan
        excess = own_ratio / ctrl_ratio if np.isfinite(own_ratio) and np.isfinite(ctrl_ratio) and ctrl_ratio > 0 else own_ratio
        if not np.isfinite(pre) or not np.isfinite(post):
            verdict = "판정불가(개설률 없음)"
        elif (np.isfinite(excess) and excess >= params["excess_ratio"]) and (post - pre) >= params["min_abs_increase"]:
            verdict = "유보(상권 자체 성장 가능성)"
        elif pre == 0 and post >= params["min_abs_increase"]:
            # pre=0 이면 자기비율(own_ratio)이 정의되지 않아 excess_ratio 조건을 계산할 수 없다(0으로 나눔).
            # 이 경우는 "초과비율 AND 증가폭" 두 조건이 아니라 증가폭 조건 하나로만 판단하는 별도 분기이므로,
            # 표에서 구분되도록 사유를 남긴다 — 소표본 노이즈로 오탐될 수 있어 현장에서 별도 확인이 필요하다.
            verdict = "유보(상권 자체 성장 가능성, 사전 개설률 0 — 초과비율 계산 불가)"
        else:
            verdict = "정상"
        rows.append({"bjd_code": code, "T_r": row["T_r"], "개설률_사전": pre, "개설률_사후": post,
                     "자기비율": own_ratio, "대조중앙비율": ctrl_ratio, "초과비율": excess, "판정": verdict})
    out = pd.DataFrame(rows)
    n_flag = int(out["판정"].str.startswith("유보").sum()) if len(out) else 0
    log.info("6.5단계 처치오염: 처치 %d 중 유보 %d%s", len(out), n_flag,
             f" — {out.loc[out['판정'].str.startswith('유보'), 'bjd_code'].tolist()}" if n_flag else "")
    return out
