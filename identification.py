"""4·5·6단계 — Y_ddd 축약 · 스태거드 이벤트 스터디 · 위약(대조 업종) 검정.

Y_ddd(r, m) = [log S(외지, 대기소비) − log S(거주자, 대기소비)] − [log S(외지, 대조) − log S(거주자, 대조)]

- 업종 코드(TOBU_BIDVS_CD/TOBU_MEDVS_CD)와 거주자 구간(IFW_DISTC_SECT_CD)은 industry_codes JSON 에서 읽는다.
- S 가 0 이거나 마스킹으로 결측인 (법정동, 월) 셀은 스킵하고 사유를 남긴다. use_log1p=True 면 log(S+1).
- 셀 결측과 '업종 부재'를 구분한다: 원천 행이 있는데 값이 전부 마스킹 → 결측, 원천 행 자체가 없음 → 0.

위약: 스펙 문구대로 '대기소비 자리에 대조 업종'을 넣으면 Y = [대조] − [대조] ≡ 0 이 되어 검정이 성립하지 않는다.
  그래서 대조 업종을 둘로 나눈다 — placebo 코드(예: 병원)를 대기소비 자리에, 나머지 대조(학원·전문서비스)를
  비교 자리에 넣은 같은 구조의 Y 를 쓴다. placebo 코드가 비어 있으면 대조 업종만의 외지−거주자 차분(DD)을 쓴다.

추정기: Callaway–Sant'Anna ATT(g,t) 를 numpy 로 직접 구현(외부 패키지 불필요) → k 기준 코호트 크기 가중 집계.
  표준오차는 법정동 단위 클러스터 부트스트랩. 대조군이 없거나 CS 가 실패하면
  지역 FE + 시점 FE + 상대시점 더미 회귀(2-way within 변환, 법정동 클러스터 SE)로 폴백한다.
  2기간 TWFE(단일 처치더미)는 음의 가중치 편향 때문에 쓰지 않는다.
단독 실행하지 않는다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bjd_mapping import canonicalize, shc_bjd_code
from common import clean_label, f_sf, filter_sido_code, log, mi_to_ym, norm_sf2, parse_mi, read_columns, to_num, wald_test

GROUPS = ("wait", "placebo", "control_ref")


# ---------------------------------------------------------------- 4단계: SHC002 스캔

def code_matcher(codes):
    """["FD", "RT:RT01"] → (bidvs, medvs) 시리즈를 받아 불리언을 돌려주는 함수."""
    bid_only = {str(c) for c in codes if ":" not in str(c)}
    pairs = {str(c) for c in codes if ":" in str(c)}

    def match(bidvs, medvs):
        m = bidvs.isin(bid_only)
        if pairs:
            m = m | (bidvs.fillna("") + ":" + medvs.fillna("")).isin(pairs)
        return m.fillna(False).astype(bool)
    return match


def _value_with_mask(raw, markers):
    txt = raw.fillna("").astype(str).str.strip()
    masked = (txt == "") | txt.isin(set(markers))
    return to_num(raw).where(~masked), masked


def scan_shc002(path, columns, industry, shc_params, snapshot_months=(), nrows=None, extra_stats=False, crosswalk=None):
    """SHC002 를 청크로 한 번만 읽어 (1) Y 계산용 셀 합계 (2) 이질성 변수 스냅샷 (3) 품질 통계를 만든다.

    3.13GB 를 두 번 읽지 않기 위해 한 패스에서 모두 집계한다.
    """
    metric = shc_params["metric"]
    need = ["ym", "sido_cd", "sgg_cd", "umd_cd", "bidvs", "medvs", "ifw_dist", metric]
    if extra_stats or snapshot_months:
        need += ["tizo", "sex", "age", "income"]
    mapping = {k: columns["shc002"][k] for k in dict.fromkeys(need)}
    is_wait = code_matcher(industry["wait"])
    is_ctrl = code_matcher(industry["control"])
    is_plc = code_matcher(industry["placebo"])
    resident = {str(c) for c in industry["resident_dist_codes"]}
    unknown = {str(c) for c in industry["unknown_dist_codes"]}
    snap = {int(parse_mi(pd.Series([m])).iloc[0]) for m in snapshot_months}

    cell_parts, snap_parts = [], {k: [] for k in ("group", "bidvs", "origin", "sex", "age", "income")}
    stats = {"rows": 0, "masked_rows": 0, "bad_region_rows": 0, "bad_month_rows": 0, "unknown_origin_rows": 0,
             "other_industry_rows": 0}
    tizo_vals, dist_vals = set(), set()
    reader = read_columns(path, mapping, "SHC002", chunksize=shc_params["chunksize"], nrows=nrows)
    for chunk in reader:
        stats["rows"] += len(chunk)
        chunk = filter_sido_code(chunk, "sido_cd", shc_params.get("sido"))
        if chunk.empty:
            continue
        bjd = canonicalize(shc_bjd_code(chunk["sido_cd"], chunk["sgg_cd"], chunk["umd_cd"]), crosswalk)
        mi = parse_mi(chunk["ym"])
        value, masked = _value_with_mask(chunk[metric], shc_params["masked_markers"])
        stats["masked_rows"] += int(masked.sum())
        bidvs, medvs = clean_label(chunk["bidvs"]), clean_label(chunk["medvs"])
        dist = clean_label(chunk["ifw_dist"])
        dist_vals.update(dist.dropna().unique().tolist())
        if "tizo" in chunk:
            tizo_vals.update(clean_label(chunk["tizo"]).dropna().unique().tolist())

        group = pd.Series("other", index=chunk.index, dtype="object")
        ctrl = is_ctrl(bidvs, medvs)
        group[ctrl] = "control_ref"
        group[ctrl & is_plc(bidvs, medvs)] = "placebo"
        group[is_wait(bidvs, medvs)] = "wait"
        origin = pd.Series("outsider", index=chunk.index, dtype="object")
        origin[dist.isin(resident)] = "resident"
        origin[dist.isna() | dist.isin(unknown)] = "unknown"

        base = pd.DataFrame({"bjd_code": bjd, "mi": mi, "group": group, "origin": origin,
                             "value": value, "valid": value.notna().astype(int), "masked": masked.astype(int)})
        bad_region, bad_month = base["bjd_code"].isna(), base["mi"].isna()
        stats["bad_region_rows"] += int(bad_region.sum())
        stats["bad_month_rows"] += int(bad_month.sum())
        stats["unknown_origin_rows"] += int((origin == "unknown").sum())
        stats["other_industry_rows"] += int((group == "other").sum())
        base = base[~bad_region & ~bad_month]

        use = base[(base["group"] != "other") & (base["origin"] != "unknown")]
        cell_parts.append(use.groupby(["bjd_code", "mi", "group", "origin"], sort=False)
                          .agg(S=("value", "sum"), n_valid=("valid", "sum"), n_rows=("value", "size"),
                               n_masked=("masked", "sum")).reset_index())
        if snap:
            in_snap = base["mi"].isin(snap)
            sb = base[in_snap]
            snap_parts["group"].append(sb.groupby(["bjd_code", "mi", "group"])["value"].sum().reset_index())
            snap_parts["origin"].append(sb.groupby(["bjd_code", "mi", "origin"])["value"].sum().reset_index())
            extra = pd.DataFrame({"bidvs": bidvs, "sex": clean_label(chunk["sex"]), "age": clean_label(chunk["age"]),
                                  "income": clean_label(chunk["income"])}).loc[sb.index]
            sbx = pd.concat([sb[["bjd_code", "mi", "value"]], extra], axis=1)
            for dim in ("bidvs", "sex", "age", "income"):
                snap_parts[dim].append(sbx.groupby(["bjd_code", "mi", dim])["value"].sum().reset_index())

    cells = pd.concat(cell_parts, ignore_index=True).groupby(["bjd_code", "mi", "group", "origin"])[
        ["S", "n_valid", "n_rows", "n_masked"]].sum().reset_index() if cell_parts else pd.DataFrame(
        columns=["bjd_code", "mi", "group", "origin", "S", "n_valid", "n_rows", "n_masked"])
    cells["mi"] = cells["mi"].astype(int)
    snapshot = {}
    for dim, parts in snap_parts.items():
        if parts:
            keys = ["bjd_code", "mi", dim]
            snapshot[dim] = pd.concat(parts, ignore_index=True).groupby(keys)["value"].sum().reset_index()
    stats["mask_rate"] = stats["masked_rows"] / stats["rows"] if stats["rows"] else float("nan")
    stats["tizo_values"] = sorted(tizo_vals)
    stats["dist_values"] = sorted(dist_vals)
    log.info("SHC002 스캔: %s행 · 마스킹 %.1f%% · 지역코드 불량 %s · 유입거리 미상 %s · 분석 외 업종 %s",
             f"{stats['rows']:,}", 100 * stats["mask_rate"], f"{stats['bad_region_rows']:,}",
             f"{stats['unknown_origin_rows']:,}", f"{stats['other_industry_rows']:,}")
    return {"cells": cells, "snapshot": snapshot, "stats": stats}


def _component(cells, groups, origin, max_masked_share):
    """(법정동, 월) 별 S 와 결측 여부. 원천 행 없음 → 0, 행은 있는데 값이 마스킹 → NaN."""
    sub = cells[cells["group"].isin(groups) & (cells["origin"] == origin)]
    g = sub.groupby(["bjd_code", "mi"])[["S", "n_valid", "n_rows", "n_masked"]].sum()
    masked_share = g["n_masked"] / g["n_rows"]
    s = g["S"].where((g["n_valid"] > 0) & (masked_share <= max_masked_share))
    return s


def build_ydd(cells, industry, use_log1p=False, max_masked_share=0.5):
    """셀 합계 → 패널(bjd_code, mi, y_main, y_placebo) + 스킵 목록."""
    idx = cells[["bjd_code", "mi"]].drop_duplicates().set_index(["bjd_code", "mi"]).index
    ctrl_groups = ["placebo", "control_ref"]
    comp = {
        "wait_out": _component(cells, ["wait"], "outsider", max_masked_share),
        "wait_res": _component(cells, ["wait"], "resident", max_masked_share),
        "ctrl_out": _component(cells, ctrl_groups, "outsider", max_masked_share),
        "ctrl_res": _component(cells, ctrl_groups, "resident", max_masked_share),
        "plc_out": _component(cells, ["placebo"], "outsider", max_masked_share),
        "plc_res": _component(cells, ["placebo"], "resident", max_masked_share),
        "ref_out": _component(cells, ["control_ref"], "outsider", max_masked_share),
        "ref_res": _component(cells, ["control_ref"], "resident", max_masked_share),
    }
    # 원천 행이 아예 없는 조합은 0 (업종 부재). 마스킹 결측(NaN)과 구분하기 위해 reindex 후 채운다.
    present = {}
    for name, s in comp.items():
        exists = s.index
        full = s.reindex(idx)
        present[name] = pd.Series(idx.isin(exists), index=idx)
        comp[name] = full.where(present[name], 0.0)

    def outcome(a_out, a_res, b_out, b_res, label):
        parts = {"a_out": comp[a_out], "a_res": comp[a_res]}
        if b_out:
            parts.update({"b_out": comp[b_out], "b_res": comp[b_res]})
        reason = pd.Series("", index=idx, dtype="object")
        names = {"a_out": a_out, "a_res": a_res, "b_out": b_out, "b_res": b_res}
        for key, s in parts.items():
            reason = reason.where(reason != "", np.where(s.isna(), f"마스킹/결측:{names[key]}", ""))
            if not use_log1p:
                reason = reason.where(reason != "", np.where(s.fillna(1) <= 0, f"0매출:{names[key]}", ""))
        f = np.log1p if use_log1p else np.log
        with np.errstate(divide="ignore", invalid="ignore"):
            y = f(parts["a_out"]) - f(parts["a_res"])
            if b_out:
                y = y - (f(parts["b_out"]) - f(parts["b_res"]))
        y = y.where(reason == "")
        skipped = reason[reason != ""].rename("사유").reset_index().assign(outcome=label)
        return y, skipped

    y_main, skip_main = outcome("wait_out", "wait_res", "ctrl_out", "ctrl_res", "main")
    if industry["placebo"]:
        y_plc, skip_plc = outcome("plc_out", "plc_res", "ref_out", "ref_res", "placebo")
        placebo_mode = "대조업종 분할 DDD(placebo vs 나머지 대조)"
    else:
        y_plc, skip_plc = outcome("ctrl_out", "ctrl_res", None, None, "placebo")
        placebo_mode = "대조업종 DD(외지−거주자)"
    panel = pd.DataFrame({"y_main": y_main, "y_placebo": y_plc}).reset_index()
    skipped = pd.concat([skip_main, skip_plc], ignore_index=True)
    for label, sk in (("main", skip_main), ("placebo", skip_plc)):
        if len(sk):
            log.warning("Y_ddd(%s) 스킵 %d/%d 셀 — 사유별 %s", label, len(sk), len(idx),
                        sk["사유"].str.split(":").str[0].value_counts().to_dict())
    log.info("4단계 Y_ddd 패널: 법정동 %d · 셀 %d · 위약 방식: %s%s", panel["bjd_code"].nunique(), len(panel),
             placebo_mode, " · log(S+1)" if use_log1p else "")
    panel.attrs["placebo_mode"] = placebo_mode
    return panel, skipped


# ---------------------------------------------------------------- 5단계: 이벤트 스터디

def _setup(panel, ycol, activation, units=None):
    cohorts = activation.loc[activation["status"] == "treated"].set_index("bjd_code")["T_r_mi"].astype(int)
    controls = activation.loc[activation["status"] == "never_treated", "bjd_code"]
    have = set(panel.loc[panel[ycol].notna(), "bjd_code"])
    cohorts = cohorts[cohorts.index.isin(have)]
    controls = sorted(set(controls) & have)
    if units is not None:
        cohorts = cohorts[cohorts.index.isin(units)]
        controls = [c for c in controls if c in units]
    return cohorts, controls


def cs_event_study(panel, ycol, activation, params, units=None):
    """Callaway–Sant'Anna(공변량 없음, 기준시점 g−1 고정) + 법정동 클러스터 부트스트랩.

    ATT(g,t) = E[Y_t − Y_{g−1} | 코호트 g] − E[Y_t − Y_{g−1} | 대조]
    θ(k) = Σ_g w_g ATT(g, g+k),  w_g = 해당 셀에 관측된 코호트 g 법정동 수
    부트스트랩은 법정동을 복원추출(다항 가중치)해 같은 식을 다시 계산한다. seed 가 같으면
    주 결과와 위약 결과가 같은 가중치 행렬을 써서 두 추정치의 차이 SE 를 정확히 낼 수 있다.
    """
    cohorts, controls = _setup(panel, ycol, activation, units)
    kmin, kmax = int(params["k_min"]), int(params["k_max"])
    if len(cohorts) == 0:
        raise ValueError("처치 법정동이 없음")
    ctrl_mode = params["control_group"]
    if ctrl_mode == "never_treated" and len(controls) < params["min_units_per_k"]:
        raise ValueError(f"never-treated 대조군 {len(controls)}곳 — CS 추정 불가")

    unit_ids = sorted(set(cohorts.index) | set(controls))
    wide = panel[panel["bjd_code"].isin(unit_ids)].pivot(index="bjd_code", columns="mi", values=ycol)
    months = list(range(int(wide.columns.min()), int(wide.columns.max()) + 1))
    wide = wide.reindex(index=unit_ids, columns=months)
    Y = wide.to_numpy(float)
    M = np.isfinite(Y)
    N, T = Y.shape
    g_arr = np.array([cohorts.get(u, np.inf) for u in unit_ids], dtype=float)
    col = {m: j for j, m in enumerate(months)}

    rng = np.random.default_rng(int(params["seed"]))
    B = int(params["n_boot"])
    W = np.vstack([np.ones((1, N)), rng.multinomial(N, np.full(N, 1.0 / N), size=B)]) if B > 0 else np.ones((1, N))

    ks = list(range(kmin, kmax + 1))
    num = {k: np.zeros(W.shape[0]) for k in ks}
    den = {k: np.zeros(W.shape[0]) for k in ks}
    n_units = {k: 0 for k in ks}
    attgt = []
    for g in sorted(set(cohorts.astype(int))):
        b = g - 1
        if b not in col:
            log.warning("코호트 %s: 기준월 %s 가 패널 밖 — 제외", mi_to_ym(g), mi_to_ym(b))
            continue
        jb = col[b]
        D = Y - Y[:, [jb]]
        Mb = M & M[:, [jb]]
        D0 = np.where(Mb, D, 0.0)
        in_g = g_arr == g
        for t in months:
            k = t - g
            if k < kmin or k > kmax or k == -1:
                continue
            jt = col[t]
            tr = in_g & Mb[:, jt]
            if ctrl_mode == "not_yet_treated":
                ct = (g_arr > max(t, b)) & Mb[:, jt]
            else:
                ct = np.isinf(g_arr) & Mb[:, jt]
            if tr.sum() == 0 or ct.sum() == 0:
                continue
            wt, wc = W[:, tr], W[:, ct]
            dt_, dc_ = wt.sum(1), wc.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                att = wt @ D0[tr, jt] / dt_ - wc @ D0[ct, jt] / dc_
            ok = np.isfinite(att) & (dt_ > 0)
            num[k] += np.where(ok, att * dt_, 0.0)
            den[k] += np.where(ok, dt_, 0.0)
            n_units[k] += int(tr.sum())
            attgt.append({"cohort": mi_to_ym(g), "t": mi_to_ym(t), "k": k, "att": float(att[0]),
                          "n_treated": int(tr.sum()), "n_control": int(ct.sum())})

    min_units = int(params["min_units_per_k"])
    boot = {}
    rows = []
    z = 1.959963984540054
    for k in ks:
        if k == -1:
            rows.append({"k": -1, "tau": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": np.nan, "n_units": np.nan,
                         "note": "기준"})
            continue
        if n_units[k] < min_units:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = num[k] / den[k]
        boot[k] = theta
        se = float(np.nanstd(theta[1:], ddof=1)) if B > 1 else np.nan
        est = float(theta[0])
        rows.append({"k": k, "tau": est, "se": se, "ci_lo": est - z * se, "ci_hi": est + z * se,
                     "p": norm_sf2(est / se) if se and np.isfinite(se) and se > 0 else np.nan,
                     "n_units": n_units[k], "note": ""})
    event = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if -1 not in boot and len(event) <= 1:
        raise ValueError("추정 가능한 k 가 없음(코호트별 관측 부족)")

    pre_ks = [k for k in boot if k <= -2]
    pre = _boot_wald(boot, pre_ks)
    post_ks = [k for k in boot if k >= 0]
    post = _boot_average(boot, post_ks, n_units)
    unit_pre = _unit_pretrend(Y, M, g_arr, unit_ids, col, kmin, params)
    k_avail = (int(event.loc[event["note"] != "기준", "k"].min()), int(event.loc[event["note"] != "기준", "k"].max())) \
        if (event["note"] != "기준").any() else (None, None)
    if not params.get("_quiet"):
        log.info("5단계 CS[%s] %s: 처치 %d · 대조 %d · 가용 k %s~%s · 사후 평균 %.4f (SE %.4f) · 사전추세 p=%.3f",
                 ycol, ctrl_mode, len(cohorts), len(controls), k_avail[0], k_avail[1], post["att"], post["se"], pre["p"])
    return {"estimator": "Callaway–Sant'Anna(내장)", "event": event, "att_gt": pd.DataFrame(attgt), "boot": boot,
            "pretrend": pre, "post": post, "unit_pretrend": unit_pre, "n_treated": len(cohorts),
            "n_control": len(controls), "k_available": k_avail, "unit_ids": unit_ids}


def _boot_wald(boot, ks):
    if not ks:
        return {"stat": np.nan, "df": 0, "p": np.nan, "ks": []}
    mat = np.vstack([boot[k] for k in ks])       # (K, B+1)
    draws = mat[:, 1:]
    draws = draws[:, np.isfinite(draws).all(axis=0)]
    if draws.shape[1] < 10:
        return {"stat": np.nan, "df": 0, "p": np.nan, "ks": ks}
    cov = np.atleast_2d(np.cov(draws))
    stat, df, p = wald_test(mat[:, 0], cov)
    return {"stat": stat, "df": df, "p": p, "ks": ks}


def _boot_average(boot, ks, n_units):
    if not ks:
        return {"att": np.nan, "se": np.nan, "p": np.nan, "draws": None}
    w = np.array([n_units[k] for k in ks], dtype=float)
    mat = np.vstack([boot[k] for k in ks])
    avg = (w[:, None] * mat).sum(0) / w.sum()
    se = float(np.nanstd(avg[1:], ddof=1)) if avg.shape[0] > 2 else np.nan
    return {"att": float(avg[0]), "se": se, "p": norm_sf2(avg[0] / se) if se and se > 0 else np.nan, "draws": avg}


def _unit_pretrend(Y, M, g_arr, unit_ids, col, kmin, params):
    """법정동별 사전추세 — 사전 월(k ≤ −2) 변화 벡터 d = (Y_t − Y_{g−1}) − 대조 평균의 호텔링 T² 예측검정.

    ⚠ [9/15 mock] 월별 z² 를 독립으로 더해 χ² 로 보면, 모든 사전 월이 같은 기준월 g−1 을 빼기 때문에
      z 들이 양의 상관을 가져 통계량이 부푼다(심은 사전추세 0 인데 처치 18곳 중 5곳 유보).
      대조군에서 공분산 S 를 추정해 T² = n/(n+1)·d'S⁻¹d 로 바꿨지만, χ² 근사로는 여전히 3곳이었다 —
      대조 10곳으로 4차원 S 를 추정하면 T² 평균이 χ²(4) 의 2배 이상이다.
      그래서 정확분포 T²·(n−p)/(p(n−1)) ~ F(p, n−p) 로 p값을 낸다.
    """
    ctrl = np.isinf(g_arr)
    rows = []
    for i, u in enumerate(unit_ids):
        g = g_arr[i]
        if np.isinf(g):
            continue
        b = int(g) - 1
        if b not in col or not M[i, col[b]]:
            rows.append({"bjd_code": u, "pre_n": 0, "stat": np.nan, "p": np.nan, "판정": "검정불가(기준월 결측)"})
            continue
        jb = col[b]
        cols = [col[int(g) + k] for k in range(kmin, -1) if int(g) + k in col and M[i, col[int(g) + k]]]
        ok = ctrl & M[:, jb]
        for jt in cols:
            ok &= M[:, jt]
        n_ctrl = int(ok.sum())
        if not cols or n_ctrl < len(cols) + 2:   # F(p, n−p) 에 n−p ≥ 2 필요
            rows.append({"bjd_code": u, "pre_n": len(cols), "stat": np.nan, "p": np.nan,
                         "판정": "검정불가(사전기간·대조군 부족)"})
            continue
        dc = Y[np.ix_(ok, cols)] - Y[ok, jb][:, None]
        d = (Y[i, cols] - Y[i, jb]) - dc.mean(axis=0)
        cov = np.atleast_2d(np.cov(dc, rowvar=False))
        rank = int(np.linalg.matrix_rank(cov))
        if rank == 0:
            rows.append({"bjd_code": u, "pre_n": len(cols), "stat": np.nan, "p": np.nan, "판정": "검정불가(대조 분산 0)"})
            continue
        stat = float(d @ np.linalg.pinv(cov) @ d) * n_ctrl / (n_ctrl + 1)
        zs = cols
        dof = n_ctrl - rank
        p = f_sf(stat * dof / (rank * (n_ctrl - 1)), rank, dof) if dof > 0 else float("nan")
        verdict = "유보(사전추세)" if p < params["alpha"] else "통과"
        rows.append({"bjd_code": u, "pre_n": len(zs), "stat": stat, "p": p, "판정": verdict})
    return pd.DataFrame(rows)


def _twoway_demean(values, unit, time, tol=1e-10, max_iter=1000):
    """불균형 패널 2-way within 변환(교대 사영). values: (n, p)."""
    x = values.astype(float).copy()
    u_codes = pd.factorize(unit)[0]
    t_codes = pd.factorize(time)[0]
    nu, nt = u_codes.max() + 1, t_codes.max() + 1
    cu = np.bincount(u_codes, minlength=nu)[:, None]
    ct = np.bincount(t_codes, minlength=nt)[:, None]
    for _ in range(max_iter):
        su = np.zeros((nu, x.shape[1]))
        np.add.at(su, u_codes, x)
        x -= (su / cu)[u_codes]
        st = np.zeros((nt, x.shape[1]))
        np.add.at(st, t_codes, x)
        shift = (st / ct)[t_codes]
        x -= shift
        if np.abs(shift).max() < tol:
            break
    return x


def regression_event_study(panel, ycol, activation, params, units=None):
    """폴백 — 지역 FE + 시점 FE + 상대시점 더미(k=−1 제외, 양 끝 구간은 binning). 법정동 클러스터 CR1 SE.

    never-treated 가 있으면 함께 넣어 시점 FE 식별을 돕고, 없으면(처치군만) 공선성 때문에 최소 k 더미도 뺀다.
    """
    cohorts, controls = _setup(panel, ycol, activation, units)
    if len(cohorts) == 0:
        raise ValueError("처치 법정동이 없음")
    kmin, kmax = int(params["k_min"]), int(params["k_max"])
    df = panel[panel["bjd_code"].isin(set(cohorts.index) | set(controls)) & panel[ycol].notna()].copy()
    df["g"] = df["bjd_code"].map(cohorts)
    df["k"] = (df["mi"] - df["g"]).clip(kmin, kmax)
    drop = {-1} | (set() if controls else {kmin})
    ks = [k for k in range(kmin, kmax + 1) if k not in drop]
    X = np.column_stack([(df["k"] == k).to_numpy(float) for k in ks])
    keep = X.sum(0) > 0
    ks = [k for k, kp in zip(ks, keep) if kp]
    X = X[:, keep]
    Z = _twoway_demean(np.column_stack([df[ycol].to_numpy(float), X]), df["bjd_code"].to_numpy(), df["mi"].to_numpy())
    yd, Xd = Z[:, 0], Z[:, 1:]
    var_ok = Xd.std(0) > 1e-12
    ks = [k for k, v in zip(ks, var_ok) if v]
    Xd = Xd[:, var_ok]
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    bread = np.linalg.pinv(Xd.T @ Xd)
    clusters = pd.factorize(df["bjd_code"])[0]
    G = clusters.max() + 1
    scores = np.zeros((G, Xd.shape[1]))
    np.add.at(scores, clusters, Xd * resid[:, None])
    V = bread @ (scores.T @ scores) @ bread * (G / max(G - 1, 1))
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = 1.959963984540054
    n_units = df[df["g"].notna()].groupby("k")["bjd_code"].nunique()
    rows = [{"k": -1, "tau": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": np.nan, "n_units": np.nan, "note": "기준"}]
    for j, k in enumerate(ks):
        note = "구간합(binned)" if k in (kmin, kmax) else ""
        rows.append({"k": k, "tau": beta[j], "se": se[j], "ci_lo": beta[j] - z * se[j], "ci_hi": beta[j] + z * se[j],
                     "p": norm_sf2(beta[j] / se[j]) if se[j] > 0 else np.nan, "n_units": int(n_units.get(k, 0)),
                     "note": note})
    event = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    event = event[(event["n_units"].fillna(np.inf) >= params["min_units_per_k"])].reset_index(drop=True)
    pre_idx = [j for j, k in enumerate(ks) if k <= -2]
    stat, dfree, p = wald_test(beta[pre_idx], V[np.ix_(pre_idx, pre_idx)]) if pre_idx else (np.nan, 0, np.nan)
    post_idx = [j for j, k in enumerate(ks) if k >= 0]
    if post_idx:
        w = np.array([n_units.get(ks[j], 0) for j in post_idx], dtype=float)
        w = w / w.sum() if w.sum() else np.full(len(post_idx), 1 / len(post_idx))
        att = float(w @ beta[post_idx])
        att_se = float(np.sqrt(w @ V[np.ix_(post_idx, post_idx)] @ w))
    else:
        att, att_se = np.nan, np.nan
    log.info("5단계 회귀[%s]: 처치 %d · 대조 %d · 사후 평균 %.4f (SE %.4f) · 사전추세 p=%.3f",
             ycol, len(cohorts), len(controls), att, att_se, p)
    return {"estimator": "상대시점 더미 + 2-way FE 회귀(폴백)", "event": event, "att_gt": pd.DataFrame(), "boot": None,
            "pretrend": {"stat": stat, "df": dfree, "p": p, "ks": [ks[j] for j in pre_idx]},
            "post": {"att": att, "se": att_se, "p": norm_sf2(att / att_se) if att_se and att_se > 0 else np.nan,
                     "draws": None},
            "unit_pretrend": pd.DataFrame(columns=["bjd_code", "pre_n", "stat", "p", "판정"]),
            "n_treated": len(cohorts), "n_control": len(controls),
            "k_available": (min(ks) if ks else None, max(ks) if ks else None), "unit_ids": None}


def run_event_study(panel, ycol, activation, params, units=None):
    """estimator=auto: CS 우선 → 실패(대조군 부족 등) 시 회귀 폴백. 반환에 사용 추정기와 폴백 사유를 남긴다."""
    choice = params["estimator"]
    if choice in ("auto", "cs"):
        try:
            res = cs_event_study(panel, ycol, activation, params, units)
            res["fallback_reason"] = ""
            return res
        except Exception as exc:
            if choice == "cs":
                raise
            log.warning("CS 추정 실패 → 회귀 폴백: %s", exc)
            reason = str(exc)
    else:
        reason = "설정으로 회귀 선택"
    res = regression_event_study(panel, ycol, activation, params, units)
    res["fallback_reason"] = reason
    return res


# ---------------------------------------------------------------- 6단계: 위약 할인

def discount_by_placebo(main, placebo):
    """주 결과에서 위약 효과를 빼서 보고한다(할인 전/후 모두 출력 — 숨기지 않는다).

    두 결과가 같은 부트스트랩 가중치(같은 seed·법정동 목록)로 나왔으면 차이의 SE 를 부트스트랩으로 직접 계산하고,
    아니면 독립 가정 sqrt(se² + se_plc²) 로 보수적으로 잡는다.
    """
    m = main["event"].set_index("k")
    p = placebo["event"].set_index("k")
    ks = sorted(set(m.index) & set(p.index))
    paired = (main.get("boot") is not None and placebo.get("boot") is not None
              and main.get("unit_ids") == placebo.get("unit_ids"))
    rows = []
    for k in ks:
        tau_adj = m.at[k, "tau"] - p.at[k, "tau"]
        if k == -1:
            se_adj, how = 0.0, "기준"
        elif paired and k in main["boot"] and k in placebo["boot"]:
            diff = main["boot"][k] - placebo["boot"][k]
            se_adj, how = float(np.nanstd(diff[1:], ddof=1)), "쌍 부트스트랩"
        else:
            se_adj, how = float(np.sqrt(m.at[k, "se"] ** 2 + p.at[k, "se"] ** 2)), "독립 가정"
        rows.append({"k": k, "tau_할인전": m.at[k, "tau"], "se_할인전": m.at[k, "se"],
                     "tau_위약": p.at[k, "tau"], "se_위약": p.at[k, "se"], "p_위약": p.at[k, "p"],
                     "tau_할인후": tau_adj, "se_할인후": se_adj,
                     "p_할인후": norm_sf2(tau_adj / se_adj) if se_adj > 0 else np.nan, "SE방식": how})
    table = pd.DataFrame(rows)
    mp, pp = main["post"], placebo["post"]
    if paired and mp.get("draws") is not None and pp.get("draws") is not None and len(mp["draws"]) == len(pp["draws"]):
        d = mp["draws"] - pp["draws"]
        post_se = float(np.nanstd(d[1:], ddof=1))
    else:
        post_se = float(np.sqrt(mp["se"] ** 2 + pp["se"] ** 2))
    post_adj = mp["att"] - pp["att"]
    summary = pd.DataFrame([
        {"구분": "주 결과(할인 전)", "사후평균": mp["att"], "SE": mp["se"], "p": mp["p"]},
        {"구분": "위약(대조 업종)", "사후평균": pp["att"], "SE": pp["se"], "p": pp["p"]},
        {"구분": "할인 후 = 주 − 위약", "사후평균": post_adj, "SE": post_se,
         "p": norm_sf2(post_adj / post_se) if post_se > 0 else np.nan},
    ])
    if np.isfinite(pp["p"]) and pp["p"] < 0.05:
        log.warning("⚠ 위약 효과가 0 과 유의하게 다름(p=%.3f, %.4f) — 주 결과를 그만큼 할인해 보고", pp["p"], pp["att"])
    log.info("6단계 위약 할인: 사후평균 %.4f → %.4f (위약 %.4f)", mp["att"], post_adj, pp["att"])
    return table, summary


def estimate_mde(panel, activation, params, repetitions=50, seed=42, activation_window=None,
                 bias_se_multiple=2.0):
    """never-treated 표본에 가짜 처치를 반복 배정해 CS 사후효과의 경험적 MDE를 구한다.

    실제 처치·대조 수를 유지하려고 never-treated 지역을 복원추출해 가상 표본을 만든다. 따라서
    never-treated 수가 실제 처치 수보다 적어도 현장 표본크기 가정을 그대로 평가할 수 있다.
    """
    repetitions = int(repetitions)
    if repetitions < 2:
        raise ValueError("MDE 반복 수는 2 이상이어야 함")
    treated = activation.loc[activation["status"] == "treated"]
    controls = activation.loc[activation["status"] == "never_treated", "bjd_code"].astype(str)
    n_treated, n_control = len(treated), len(controls)
    if n_treated < 1 or n_control < int(params["min_units_per_k"]):
        raise ValueError(f"MDE 표본 부족: 실제 처치 {n_treated} · never-treated {n_control}")
    eligible = sorted(set(controls) & set(panel["bjd_code"].astype(str)))
    if not eligible:
        raise ValueError("MDE에 쓸 never-treated 패널이 없음")
    cohorts = treated.get("T_r_mi", pd.Series(dtype=float)).dropna().astype(int).to_numpy()
    if len(cohorts) == 0:
        if not activation_window:
            raise ValueError("실제 처치 코호트와 activation.window가 모두 없음")
        from common import ym_to_mi
        lo, hi = (ym_to_mi(v) for v in activation_window)
        cohorts = np.arange(lo, hi + 1)

    rng = np.random.default_rng(int(seed))
    estimates = {"주 결과": [], "위약": [], "할인 후": []}
    local = dict(params, estimator="cs", n_boot=0, _quiet=True)
    started = time.perf_counter()
    for rep in range(repetitions):
        sampled = rng.choice(eligible, size=n_treated + n_control, replace=True)
        pieces = []
        ids = []
        for i, source in enumerate(sampled):
            clone = f"mde_{rep:03d}_{i:04d}"
            part = panel.loc[panel["bjd_code"].astype(str) == source].copy()
            part["bjd_code"] = clone
            pieces.append(part)
            ids.append(clone)
        fake_panel = pd.concat(pieces, ignore_index=True)
        fake_cohorts = rng.choice(cohorts, size=n_treated, replace=True)
        fake_activation = pd.DataFrame({
            "bjd_code": ids,
            "status": ["treated"] * n_treated + ["never_treated"] * n_control,
            "T_r_mi": np.r_[fake_cohorts, np.full(n_control, np.nan)],
        })
        main = cs_event_study(fake_panel, "y_main", fake_activation, local)
        placebo = cs_event_study(fake_panel, "y_placebo", fake_activation, local)
        estimates["주 결과"].append(main["post"]["att"])
        estimates["위약"].append(placebo["post"]["att"])
        estimates["할인 후"].append(main["post"]["att"] - placebo["post"]["att"])

    elapsed = time.perf_counter() - started
    rows = []
    for label, values in estimates.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError(f"MDE {label}: 유효 반복 {len(values)}회")
        mean, se = float(values.mean()), float(values.std(ddof=1))
        if abs(mean) > float(bias_se_multiple) * se:
            log.warning("MDE 추정기 편향 의심(%s): 가짜 효과 평균 %.4f · 경험 SE %.4f", label, mean, se)
        rows.append({"구분": label, "가짜효과평균": mean, "경험_SE": se, "MDE": 2.8 * se,
                     "반복수": len(values), "처치수_가정": n_treated, "대조수": n_control,
                     "처치수_근거": "실제 처치 수"})
    log.info("4.5 MDE: %d회 %.2f초 · 반복 1회당 %.3f초", repetitions, elapsed, elapsed / repetitions)
    return pd.DataFrame(rows)


def event_table(res):
    """반출용 τ(k) 표."""
    ev = res["event"].copy()
    ev.insert(0, "추정기", res["estimator"])
    return ev
