"""7단계 — 이질성 변수 + CATE(Causal Forest / 폴백 2×2 서브그룹).

Bad control 방지(이 모듈의 핵심 제약):
  SHC 기반 변수는 snapshot_months(기본 2025-01~02) 고정 스냅샷으로만 만든다. 스냅샷의 마지막 달이
  활성화 후보 창 시작(2025-03) 이상이면 ValueError 로 멈춘다. 연간 평균·처치 이후 월은 절대 쓰지 않는다.
금지 피처: 충전 집중도(피크 시간대 점유율) 등 부하 관련 정보 — 8단계 전력축(세로축)과 같은 개념이라
  CATE 피처에 넣으면 두 축이 모델 구조상 상관된다. 이름 패턴으로 검사해 들어오면 ValueError.

CATE 단위 결과: Δ_r = (T_r 이후 post_months 개월 평균 Y) − (T_r 이전 평균 Y).
  대조군은 처치 코호트 시점 분포로 가중한 같은 Δ 를 쓴다. W_r = 처치 여부.
추정: EconML CausalForestDML 이 있으면 사용, 없으면 estimate_cate_fallback()
  (업종구성 상/하위 × 외지유입 상/하위 2×2 셀별 처치−대조 평균차).
검증: 같은 교차검증 분할에서 상수 ATE / 2×2 / Causal Forest 를 변환결과(transformed outcome) 손실과
  GATES(예측 상위 절반 − 하위 절반의 실현 효과 차)로 비교한다 — '단순 서브그룹보다 실제로 나은지'.
단독 실행하지 않는다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import log, mi_to_ym, optional_import, ym_to_mi

# 부하·집중도 계열 이름. load_axis.py 에서만 쓰는 정보다.
FORBIDDEN_FEATURE = re.compile(r"(집중도|피크|부하|전력|kwh|kw$|_kw|peak|concentration|load)", re.IGNORECASE)


def assert_no_load_features(names):
    bad = [n for n in names if FORBIDDEN_FEATURE.search(str(n))]
    if bad:
        raise ValueError(f"CATE 피처에 부하/집중도 계열 변수 금지(8단계 전력축과 구조적 상관): {bad}")


def assert_snapshot_before(snapshot_months, window_start):
    last = max(ym_to_mi(m) for m in snapshot_months)
    if last >= ym_to_mi(window_start):
        raise ValueError(f"Bad control: 스냅샷 마지막 달 {mi_to_ym(last)} 이 활성화 후보 시작 {window_start} 이후다 — "
                         "처치 이전 고정 스냅샷만 허용")


def _shares(tab, dim, prefix, codes=None):
    total = tab.groupby("bjd_code")["value"].sum()
    wide = tab.groupby(["bjd_code", dim])["value"].sum().unstack(fill_value=0.0)
    if codes is not None:
        wide = wide.reindex(columns=codes, fill_value=0.0)
    share = wide.div(total.where(total > 0), axis=0)
    share.columns = [f"{prefix}_{c}" for c in share.columns]
    return share


def build_features(snapshot, shc001m, can_features, infra_stock, params, window_start):
    """법정동별 이질성 변수표. 모든 SHC 변수는 스냅샷 월만으로 계산한다."""
    snap_months = params["snapshot_months"]
    assert_snapshot_before(snap_months, window_start)
    snap_mi = {ym_to_mi(m) for m in snap_months}
    for dim, tab in snapshot.items():
        extra = set(tab["mi"].astype(int)) - snap_mi
        if extra:
            raise ValueError(f"스냅샷 표({dim})에 스냅샷 밖 월이 섞임: {[mi_to_ym(m) for m in sorted(extra)]}")
    if not snapshot:
        raise ValueError("SHC002 스냅샷이 비어 있음 — snapshot_months 기간 데이터가 있는지 확인")

    grp = snapshot["group"]
    total = grp.groupby("bjd_code")["value"].sum()
    feats = pd.DataFrame(index=total.index)
    feats["wait_share"] = grp[grp["group"] == "wait"].groupby("bjd_code")["value"].sum().reindex(total.index, fill_value=0) \
        / total.where(total > 0)
    org = snapshot["origin"]
    known = org[org["origin"] != "unknown"]
    feats["outsider_share"] = (known[known["origin"] == "outsider"].groupby("bjd_code")["value"].sum()
                               / known.groupby("bjd_code")["value"].sum()).reindex(total.index)
    feats["매출규모_log"] = np.log1p(total / len(snap_months))
    top_bidvs = snapshot["bidvs"].groupby("bidvs")["value"].sum().nlargest(params["top_k_bidvs"]).index.tolist()
    feats = feats.join(_shares(snapshot["bidvs"], "bidvs", "업종비", top_bidvs))
    for dim, prefix in (("sex", "성별비"), ("age", "연령비"), ("income", "소득비")):
        if dim in snapshot:
            feats = feats.join(_shares(snapshot[dim], dim, prefix))

    if shc001m is not None:
        s1 = shc001m[shc001m["mi"].isin(snap_mi)].groupby("bjd_code").agg(
            가맹점수_log=("stock", lambda v: np.log1p(v.mean())), 개설률=("open_rate", "mean"), 폐업률=("close_rate", "mean"))
        feats = feats.join(s1)
    if can_features is not None and len(can_features):
        feats = feats.join(can_features.set_index("bjd_code"))
    if infra_stock is not None and len(infra_stock):
        feats = feats.join(infra_stock.set_index("bjd_code"))
        for c in infra_stock.columns:
            if c != "bjd_code":
                feats[c] = feats[c].fillna(0.0)   # KEP_007 에 없는 법정동 = 설치 0 (2019 기준)

    feats.index.name = "bjd_code"
    assert_no_load_features(feats.columns)
    # 결측: 중앙값 대치 + 결측 지시변수(대치로 정보가 사라지지 않게)
    for c in list(feats.columns):
        miss = feats[c].isna()
        if miss.all():
            feats = feats.drop(columns=c)
            log.info("이질성 변수 %s: 전부 결측 → 제외", c)
        elif miss.any():
            feats[f"{c}_결측"] = miss.astype(float)
            feats[c] = feats[c].fillna(feats[c].median())
    log.info("7단계 이질성 변수: 법정동 %d × 변수 %d (스냅샷 %s)", len(feats), feats.shape[1], "~".join(snap_months))
    return feats.reset_index()


def region_outcomes(panel, ycol, activation, post_months):
    """법정동 단위 Δ = 사후 평균 − 사전 평균. 대조군은 코호트 시점 분포로 가중."""
    wide = panel.pivot(index="bjd_code", columns="mi", values=ycol)
    treated = activation[activation["status"] == "treated"].set_index("bjd_code")["T_r_mi"].astype(int)
    treated = treated[treated.index.isin(wide.index)]
    controls = [c for c in activation.loc[activation["status"] == "never_treated", "bjd_code"] if c in wide.index]

    def delta(codes, g):
        pre_cols = [m for m in wide.columns if m < g]
        post_cols = [m for m in wide.columns if g <= m < g + post_months]
        sub = wide.reindex(codes)
        return sub[post_cols].mean(axis=1) - sub[pre_cols].mean(axis=1)

    rows = [{"bjd_code": c, "W": 1, "delta": float(delta([c], g).iloc[0]), "T_r": mi_to_ym(g)} for c, g in treated.items()]
    weights = treated.value_counts(normalize=True)
    if controls:
        acc = pd.Series(0.0, index=controls)
        wsum = pd.Series(0.0, index=controls)
        for g, w in weights.items():
            d = delta(controls, int(g))
            ok = d.notna()
            acc[ok] += w * d[ok]
            wsum[ok] += w
        for c in controls:
            rows.append({"bjd_code": c, "W": 0, "delta": acc[c] / wsum[c] if wsum[c] > 0 else np.nan, "T_r": None})
    out = pd.DataFrame(rows).dropna(subset=["delta"])
    log.info("CATE 단위결과: 처치 %d · 대조 %d", int((out["W"] == 1).sum()), int((out["W"] == 0).sum()))
    return out


# ---------------------------------------------------------------- 추정기

def estimate_cate_fallback(train, test, split_features):
    """사전지정 서브그룹 폴백 — split_features 두 개의 중앙값(학습셋 기준) 상/하위 2×2 셀별 처치−대조 평균차.

    셀에 처치나 대조가 없으면 전체 ATE 로 대체하고 표시한다.
    """
    f1, f2 = split_features
    c1, c2 = train[f1].median(), train[f2].median()

    def cell(df):
        return cell_labels(df, f1, c1, f2, c2)

    tr = train.assign(cell=cell(train))
    ate = tr.loc[tr["W"] == 1, "delta"].mean() - tr.loc[tr["W"] == 0, "delta"].mean()
    rows = []
    for name in sorted({f"{f1}{a}×{f2}{b}" for a in ("상", "하") for b in ("상", "하")}):
        g = tr[tr["cell"] == name]
        t, c = g.loc[g["W"] == 1, "delta"], g.loc[g["W"] == 0, "delta"]
        if len(t) and len(c):
            tau = t.mean() - c.mean()
            se = np.sqrt((t.var(ddof=1) if len(t) > 1 else 0) / len(t) + (c.var(ddof=1) if len(c) > 1 else 0) / len(c))
            note = ""
        else:
            tau, se, note = ate, np.nan, "처치/대조 부재 → 전체ATE 대체"
        rows.append({"셀": name, "처치수": len(t), "대조수": len(c), "CATE": tau, "SE": se, "비고": note})
    table = pd.DataFrame(rows)
    lut = table.set_index("셀")["CATE"]
    pred = cell(test).map(lut).fillna(ate).to_numpy()
    return pred, table, {f1: c1, f2: c2}


def cell_labels(df, f1, c1, f2, c2):
    a = pd.Series(np.where(df[f1] >= c1, "상", "하"), index=df.index)
    b = pd.Series(np.where(df[f2] >= c2, "상", "하"), index=df.index)
    return f1 + a + "×" + f2 + b


def estimate_cate_forest(train, test, feature_cols, params, seed):
    econml_dml = optional_import("econml.dml")
    if econml_dml is None:
        raise ImportError("econml 없음")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    cfp = params["cf_params"]
    n_min_class = int(min((train["W"] == 1).sum(), (train["W"] == 0).sum()))
    est = econml_dml.CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed),
        discrete_treatment=True, cv=max(2, min(3, n_min_class)),
        n_estimators=cfp["n_estimators"], min_samples_leaf=cfp["min_samples_leaf"], max_depth=cfp["max_depth"],
        random_state=seed)
    est.fit(train["delta"].to_numpy(), train["W"].to_numpy(), X=train[feature_cols].to_numpy())
    pred = np.ravel(est.effect(test[feature_cols].to_numpy()))
    importance = pd.DataFrame({"변수": feature_cols, "중요도": est.feature_importances_}).sort_values("중요도", ascending=False)
    return pred, importance


def _folds(df, n_folds, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), dtype=int)
    for w in (0, 1):
        idx = np.flatnonzero(df["W"].to_numpy() == w)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def validate_cate(data, feature_cols, params, forest_ok):
    """같은 교차검증 분할로 상수 ATE / 2×2 / Causal Forest 비교."""
    n_t, n_c = int((data["W"] == 1).sum()), int((data["W"] == 0).sum())
    k = int(min(params["n_folds"], n_t, n_c))
    methods = {"상수 ATE": np.full(len(data), np.nan), "2×2 서브그룹": np.full(len(data), np.nan)}
    if forest_ok:
        methods["Causal Forest"] = np.full(len(data), np.nan)
    notes = {}
    if k < 2:
        return pd.DataFrame([{"방법": m, "비고": f"검증 불가(처치 {n_t}·대조 {n_c})"} for m in methods]), None
    fold = _folds(data, k, params["seed"])
    for f in range(k):
        tr, te = data[fold != f], data[fold == f]
        methods["상수 ATE"][fold == f] = tr.loc[tr["W"] == 1, "delta"].mean() - tr.loc[tr["W"] == 0, "delta"].mean()
        methods["2×2 서브그룹"][fold == f], _, _ = estimate_cate_fallback(tr, te, params["split_features"])
        if forest_ok:
            try:
                methods["Causal Forest"][fold == f], _ = estimate_cate_forest(tr, te, feature_cols, params, params["seed"])
            except Exception as exc:
                notes["Causal Forest"] = f"실패: {exc}"
                methods["Causal Forest"][fold == f] = np.nan
    y, w = data["delta"].to_numpy(), data["W"].to_numpy()
    e = w.mean()
    y_star = y * (w - e) / (e * (1 - e))
    rows = []
    for name, pred in methods.items():
        ok = np.isfinite(pred)
        if not ok.any():
            rows.append({"방법": name, "교차검증_fold": k, "변환결과_MSE": np.nan, "GATES_상위−하위": np.nan,
                         "비고": notes.get(name, "예측 없음")})
            continue
        mse = float(np.mean((y_star[ok] - pred[ok]) ** 2))
        hi = pred >= np.nanmedian(pred)
        gates = np.nan
        if np.ptp(pred[ok]) > 0:
            def eff(mask):
                t, c = y[mask & (w == 1)], y[mask & (w == 0)]
                return t.mean() - c.mean() if len(t) and len(c) else np.nan
            gates = eff(hi & ok) - eff(~hi & ok)
        rows.append({"방법": name, "교차검증_fold": k, "변환결과_MSE": mse, "GATES_상위−하위": gates,
                     "비고": notes.get(name, "상수 예측 → 순위 정보 없음" if np.ptp(pred[ok]) == 0 else "")})
    table = pd.DataFrame(rows)
    return table, methods


def run_heterogeneity(features, outcomes, params):
    data = outcomes.merge(features, on="bjd_code", how="inner")
    feature_cols = [c for c in features.columns if c != "bjd_code"]
    assert_no_load_features(feature_cols)
    for f in params["split_features"]:
        if f not in data:
            raise KeyError(f"폴백 분할 변수 {f} 가 이질성 변수에 없음")
    n_t = int((data["W"] == 1).sum())
    has_econml = optional_import("econml.dml") is not None
    forest_ok = has_econml and n_t >= params["min_treated_cf"]
    reason = "" if forest_ok else ("econml 미설치" if not has_econml else f"처치 {n_t} < {params['min_treated_cf']}")
    validation, _ = validate_cate(data, feature_cols, params, forest_ok)
    if not forest_ok:
        validation = pd.concat([validation, pd.DataFrame([{"방법": "Causal Forest", "비고": f"미실행: {reason}"}])],
                               ignore_index=True)

    primary = params.get("primary", "auto")
    use_forest = forest_ok and primary in ("auto", "forest")
    if use_forest and primary == "auto":
        v = validation.set_index("방법")
        if "Causal Forest" in v.index:
            cf_mse = v.at["Causal Forest", "변환결과_MSE"]
            if not np.isfinite(cf_mse):
                use_forest = False
                reason = "교차검증 실패(변환결과_MSE 계산 안 됨) — 2×2 서브그룹으로 대체"
            elif cf_mse > v.at["2×2 서브그룹", "변환결과_MSE"]:
                use_forest = False
                reason = "교차검증 손실이 2×2 서브그룹보다 큼"
    cells = None
    importance = None
    if use_forest:
        pred, importance = estimate_cate_forest(data, data, feature_cols, params, params["seed"])
        method = "Causal Forest(EconML)"
    else:
        pred, cells, cuts = estimate_cate_fallback(data, data, params["split_features"])
        method = "2×2 서브그룹(폴백)"
        # 대조군에도 셀 CATE 를 배정한다(아직 활성화 안 된 곳의 처방에 쓰기 위해)
    cate = data[["bjd_code", "W", "T_r", "delta"]].assign(cate=pred, method=method)
    if cells is not None:
        f1, f2 = params["split_features"]
        cate["셀"] = cell_labels(data, f1, cuts[f1], f2, cuts[f2]).to_numpy()
    log.info("7단계 CATE: %s%s", method, f" (Causal Forest 미사용 사유: {reason})" if not use_forest else "")
    cate.attrs["feature_names"] = feature_cols
    return {"cate": cate, "validation": validation, "cells": cells, "importance": importance,
            "method": method, "reason": reason}
