"""8-E 단계 — 재현 가능한 결합 점수. v11 기본은 급증위험·형평성 두 축(build_priority_v11), 세 축(안전·형평성·경제성)은 레거시 비교용."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

AXES = ("safety", "equity", "economy")


def normalize(values, reverse=False):
    s = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = np.isfinite(s)
    if not ok.any():
        return out
    lo, hi = s[ok].min(), s[ok].max()
    out.loc[ok] = 0.5 if np.isclose(lo, hi) else (s[ok] - lo) / (hi - lo)
    return 1 - out if reverse else out


def validate_weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("가중치는 안전·형평성·경제성 순의 유한한 비음수 3개 값이어야 함")
    return w / w.sum()


def ahp_consistency_ratio(matrix):
    a = np.asarray(matrix, dtype=float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("AHP 행렬은 양수 3x3이어야 함")
    if not np.allclose(np.diag(a), 1) or not np.allclose(a * a.T, 1, rtol=1e-5, atol=1e-8):
        raise ValueError("AHP 행렬은 대각선 1이고 상호 역수여야 함")
    eig = np.linalg.eigvals(a).real.max()
    return max(0.0, float((eig - 3) / 2 / 0.58))


def energy_axes(results, multiplier=1.2, new_chargers=3):
    required = {"bjd_code", "new_chargers", "multiplier", "mode", "baseline_exceedance_kwh",
                "energy_cost_difference_won", "plan_feasible", "forecast_reliable"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"ESS 결과 필수 열 없음: {sorted(missing)}")
    base = results[(results["new_chargers"] == new_chargers) & np.isclose(results["multiplier"], multiplier)
                   & (results["mode"] == "forecast")].copy()
    if base.empty:
        raise ValueError(f"기준 상한배율 {multiplier}의 forecast·N_new={new_chargers} 결과 없음")
    base["energy_cost_difference_won"] = pd.to_numeric(base["energy_cost_difference_won"], errors="coerce").clip(lower=0)
    return base.groupby("bjd_code", as_index=False).agg(
        safety_raw=("baseline_exceedance_kwh", "mean"),
        economy_raw=("energy_cost_difference_won", "mean"),
        forecast_confidence=("forecast_reliable", "mean"),
        plan_confidence=("plan_feasible", "mean"),
    )


def combine_scores(energy, access, weights=(1 / 3, 1 / 3, 1 / 3), can_confidence=None, min_axes=3):
    """축 결측 사유를 나누고 순위 대상(rank_eligible)을 표시한다.

    not_assessed_low_concentration: 8-B 평가 대상이 아님(07-1 부하 집중도 낮음) — 값이 "없는" 것이 아니라 "안 잰" 것.
    data_missing: 평가 대상인데 축 값이 없음. 둘 다 0점으로 채우지 않고(V9 3-2절) 재정규화만 유지한다.
    순위 대상은 유효 축이 min_axes개 이상인 동네뿐이다(기본 3 = 세 축 모두).
    """
    w = validate_weights(weights)
    df = energy.merge(access[["bjd_code", "access_2sfca", "access_data_present"]], on="bjd_code",
                      how="outer", indicator="_src")
    assessed = (df.pop("_src") != "right_only").to_numpy()
    df["safety_norm"] = normalize(df["safety_raw"])
    df["equity_norm"] = normalize(df["access_2sfca"], reverse=True)
    df["economy_norm"] = normalize(df["economy_raw"])
    values = df[[f"{a}_norm" for a in AXES]].to_numpy(float)
    valid = np.isfinite(values)
    denom = (valid * w).sum(axis=1)
    df["PriorityScore"] = np.divide(np.nansum(values * w, axis=1), denom,
                                    out=np.full(len(df), np.nan), where=denom > 0)
    df["missing_axis_count"] = (~valid).sum(axis=1)
    df["missing_axes"] = [",".join(a for a, ok in zip(AXES, row) if not ok) for row in valid]
    df["axes_present"] = valid.sum(axis=1)
    df["missing_reason"] = np.where(df["missing_axis_count"] == 0, "",
                                    np.where(assessed, "data_missing", "not_assessed_low_concentration"))
    df["rank_eligible"] = (df["axes_present"] >= int(min_axes)) & df["PriorityScore"].notna()
    confidence_parts = pd.DataFrame({
        "forecast": pd.to_numeric(df.get("forecast_confidence"), errors="coerce").fillna(0),
        "plan": pd.to_numeric(df.get("plan_confidence"), errors="coerce").fillna(0),
        "complete": 1 - df["missing_axis_count"] / len(AXES),
        "access_match": df.get("access_data_present", pd.Series(False, index=df.index)).fillna(False).astype(float),
    })
    if can_confidence is not None:
        confidence_parts["can"] = df["bjd_code"].map(can_confidence)
    df["confidence"] = confidence_parts.mean(axis=1, skipna=True)
    return df


def _weight_grid(center, delta):
    center = validate_weights(center)
    values = [{max(0.0, c + d) for d in (-delta, 0.0, delta)} for c in center]
    unique = {}
    for raw in itertools.product(*values):
        w = validate_weights(raw)
        unique[tuple(np.round(w, 8))] = w
    return list(unique.values())


def run_sensitivity(results, access, params):
    """상위군·강건 판정은 순위 대상(ranking_pool) 안에서만 계산한다."""
    min_axes = int(params.get("min_axes", 3))
    weights = params["weights"]
    multipliers = params.get("multipliers") or sorted(pd.to_numeric(results["multiplier"], errors="coerce").dropna().unique())
    runs = []
    for multiplier in multipliers:
        try:
            energy = energy_axes(results, float(multiplier), int(params["scenario_new_chargers"]))
        except ValueError:
            continue
        for w in _weight_grid(weights, float(params["weight_delta"])):
            score = combine_scores(energy, access, w, min_axes=min_axes)
            score = score.loc[score["rank_eligible"], ["bjd_code", "PriorityScore"]]
            if score.empty:
                continue
            n_top = max(1, int(np.ceil(len(score) * float(params["top_share"]))))
            score = score.sort_values(["PriorityScore", "bjd_code"], ascending=[False, True])
            top = set(score.head(n_top)["bjd_code"])
            runs.extend({"bjd_code": c, "top": c in top} for c in score["bjd_code"])
    if not runs:
        raise ValueError("민감도 조합을 계산할 수 없음")
    summary = pd.DataFrame(runs).groupby("bjd_code", as_index=False).agg(
        sensitivity_runs=("top", "size"), top_rate=("top", "mean"))
    summary["robust_top"] = summary["top_rate"] >= float(params["robust_share"])
    summary["boundary"] = summary["top_rate"].between(0, 1, inclusive="neither")
    return summary


def axes_correlation(df, threshold=0.9):
    """안전·경제성 축 원값의 상관. 둘 다 8-B의 같은 ESS 시뮬레이션에서 나와 사실상 한 축일 수 있다."""
    pair = df[["safety_raw", "economy_raw"]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(pair)
    if n < 3 or pair.nunique().min() < 2:
        return {"n": n, "pearson": np.nan, "spearman": np.nan, "axes_redundant": False}
    pearson = float(pair["safety_raw"].corr(pair["economy_raw"]))
    spearman = float(pair["safety_raw"].corr(pair["economy_raw"], method="spearman"))
    redundant = bool(max(abs(pearson), abs(spearman)) >= float(threshold))
    return {"n": n, "pearson": pearson, "spearman": spearman, "axes_redundant": redundant}


def build_priority(results, access, params, can_confidence=None):
    min_axes = int(params.get("min_axes", 3))
    base = combine_scores(energy_axes(results, float(params["base_multiplier"]),
                                      int(params["scenario_new_chargers"])), access,
                          params["weights"], can_confidence, min_axes=min_axes)
    sensitivity = run_sensitivity(results, access, params)
    out = base.merge(sensitivity, on="bjd_code", how="left")
    for col in ("robust_top", "boundary"):
        out[col] = out[col].fillna(False).astype(bool)
    out = out.sort_values(["rank_eligible", "PriorityScore", "bjd_code"],
                          ascending=[False, False, True]).reset_index(drop=True)
    out["rank"] = np.where(out["rank_eligible"], np.arange(1, len(out) + 1), np.nan)
    out.attrs["axes_correlation"] = axes_correlation(base, params.get("redundant_corr", 0.9))
    return out


def seasonal_coverage(hourly):
    dates = pd.to_datetime(hourly["date"], errors="coerce").dropna().dt.normalize().drop_duplicates()
    seasons = np.select([dates.dt.month.isin([3, 4, 5]), dates.dt.month.isin([6, 7, 8]),
                         dates.dt.month.isin([9, 10, 11])], ["봄", "여름", "가을"], default="겨울")
    counts = pd.Series(seasons).value_counts()
    present = set(counts[counts >= 28].index)
    required = {"봄", "여름", "가을", "겨울"}
    return {"season_count": len(present), "season_status": "검증 가능" if present == required else "보류",
            "missing_seasons": ",".join(sorted(required - present))}


# ================================================================ v11 두 축(급증위험·형평성)

AXES_V11 = ("risk", "equity")
LEGACY_AXES = list(AXES)


def validate_weights_v11(weights, axes=AXES_V11):
    """{"risk": w, "equity": w}(또는 같은 순서의 목록). 유한·음수 없음·합 1(허용오차 1e-6)이어야 한다(킬 14번)."""
    w = np.asarray([weights[a] for a in axes] if isinstance(weights, dict) else weights, dtype=float)
    if w.shape != (len(axes),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(f"가중치는 {list(axes)} 순의 유한한 비음수 {len(axes)}개 값이어야 함")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"가중치 합이 1이 아님: {w.sum():.6f}")
    return w


def risk_axis(risk, multiplier):
    """8-A s8a_risk 에서 기준 상한 배율의 위험 원값(증설 0대 — 증설 가정 ΔL 은 들어가지 않는다)."""
    r = risk[np.isclose(risk["multiplier"].astype(float), float(multiplier))].copy()
    if r.empty:
        raise ValueError(f"8-A 급증 위험에 상한 배율 {multiplier} 결과 없음")
    keep = ["bjd_code", "status", "surge_risk", "peak_ratio", "precision_ai", "recall_ai"] + \
        [c for c in r if c.startswith("surge_risk_with_new_")]
    r = r[[c for c in keep if c in r]].rename(columns={"status": "risk_status"})
    r["bjd_code"] = r["bjd_code"].astype(str)
    return r


def choose_risk_metric(axis, metric="auto", pool=None, zero_share=0.5):
    """auto: 순위 대상(pool) 중 zero_share 이상에서 surge_risk=0 이면 peak_ratio 로 바꾼다(킬 21번)."""
    if metric not in ("auto", "surge_risk", "peak_ratio"):
        raise ValueError(f"priority.risk_metric 은 auto|surge_risk|peak_ratio — 받은 값: {metric!r}")
    a = axis[axis["risk_status"] == "assessed"]
    if pool is not None:
        a = a[a["bjd_code"].isin(pool)]
    share = float((a["surge_risk"] == 0).mean()) if len(a) else np.nan
    if metric != "auto":
        return metric, share
    return ("peak_ratio" if np.isfinite(share) and share >= zero_share else "surge_risk"), share


def combine_two_axes(axis, access, weights, metric, min_axes=2, forecast_metrics=None):
    """두 축 정규화·가중합. 결측 축은 0점이 아니라 남은 가중치로 재정규화하고, 순위 대상은 유효 축 min_axes 개 이상.

    결측 사유: not_assessed_low_load(교정기간 최대 부하가 너무 작아 배율 상한이 무의미 — 값이 없는 것이 아니라
    재지 않은 것), data_missing(대상인데 값 없음). 신뢰도는 표시용이며 점수를 깎지 않는다(3-7절).
    """
    w = validate_weights_v11(weights)
    acc = access[["bjd_code", "access_2sfca", "access_data_present"]].assign(bjd_code=lambda d: d["bjd_code"].astype(str))
    df = axis.merge(acc, on="bjd_code", how="outer")
    df["risk_raw"] = pd.to_numeric(df[metric], errors="coerce").where(df["risk_status"] == "assessed")
    df["risk_norm"] = normalize(df["risk_raw"])
    df["equity_norm"] = normalize(df["access_2sfca"], reverse=True)
    values = df[["risk_norm", "equity_norm"]].to_numpy(float)
    valid = np.isfinite(values)
    denom = (valid * w).sum(axis=1)
    df["PriorityScore"] = np.divide(np.nansum(values * w, axis=1), denom, out=np.full(len(df), np.nan), where=denom > 0)
    df["axes_present"] = valid.sum(axis=1)
    df["missing_axes"] = [",".join(a for a, ok in zip(AXES_V11, row) if not ok) for row in valid]
    reason = np.where(valid.all(axis=1), "", "data_missing")
    low = (df["risk_status"] == "not_assessed_low_load").to_numpy()
    df["missing_reason"] = np.where(~valid[:, 0] & low, "not_assessed_low_load", reason)
    df["rank_eligible"] = (df["axes_present"] >= int(min_axes)) & df["PriorityScore"].notna()
    parts = {"complete": df["axes_present"] / len(AXES_V11),
             "access_match": df["access_data_present"].astype("boolean").fillna(False).astype(float)}
    if forecast_metrics is not None and len(forecast_metrics):
        fm = forecast_metrics[forecast_metrics["bjd_code"].astype(str) != ""]
        fm = fm[~fm.get("reference_only", pd.Series(False, index=fm.index)).astype(bool)]
        fm = fm.set_index(fm["bjd_code"].astype(str))
        parts["forecast"] = df["bjd_code"].map(fm["ai_beats_baseline"]).astype(float)
        parts["p90"] = df["bjd_code"].map(fm["p90_conservative_enough"]).astype(float)
    df["confidence"] = pd.DataFrame(parts).mean(axis=1, skipna=True)
    df["risk_metric"] = metric
    return df


def run_sensitivity_v11(risk, access, params, metric):
    """상한 배율 × w_risk ∈ [중심 ± weight_delta](7단계) 격자. 상위군·강건 판정은 순위 대상 안에서만."""
    center = validate_weights_v11(params["weights"])[0]
    delta = float(params["weight_delta"])
    grid = np.linspace(max(0.0, center - delta), min(1.0, center + delta), 7)
    runs = []
    for m in params["multipliers"]:
        try:
            axis = risk_axis(risk, m)
        except ValueError:
            continue
        for wr in grid:
            score = combine_two_axes(axis, access, [wr, 1 - wr], metric, int(params["min_axes"]))
            score = score.loc[score["rank_eligible"], ["bjd_code", "PriorityScore"]]
            if score.empty:
                continue
            n_top = max(1, int(np.ceil(len(score) * float(params["top_share"]))))
            top = set(score.sort_values(["PriorityScore", "bjd_code"], ascending=[False, True]).head(n_top)["bjd_code"])
            runs.extend({"bjd_code": c, "top": c in top} for c in score["bjd_code"])
    if not runs:
        raise ValueError("민감도 조합을 계산할 수 없음")
    summary = pd.DataFrame(runs).groupby("bjd_code", as_index=False).agg(
        sensitivity_runs=("top", "size"), top_rate=("top", "mean"))
    summary["robust_top"] = summary["top_rate"] >= float(params["robust_share"])
    summary["boundary"] = summary["top_rate"].between(0, 1, inclusive="neither")
    return summary


def display_columns(energy_results, params):
    """점수에 넣지 않는 표시용: 경제성(SMP 비용 차이), ESS 미적용 초과 kWh(기준 시나리오, 예측 운전)."""
    if energy_results is None:
        return None
    base = energy_results[(energy_results["mode"] == "forecast")
                          & (energy_results["new_chargers"] == int(params["scenario_new_chargers"]))
                          & np.isclose(energy_results["multiplier"], float(params["base_multiplier"]))]
    if base.empty:
        return None
    return base.groupby("bjd_code", as_index=False).agg(
        economy_smp_cost_diff_won=("energy_cost_difference_won", "mean"),
        ess_baseline_exceedance_kwh=("baseline_exceedance_kwh", "mean")).assign(bjd_code=lambda d: d["bjd_code"].astype(str))


def build_priority_v11(risk, access, params, energy_results=None, forecast_metrics=None):
    """v11 기본 결합 점수(급증위험·형평성). attrs: axes_correlation, risk_metric_used, share_zero_surge_risk."""
    min_axes = int(params["min_axes"])
    base_axis = risk_axis(risk, params["base_multiplier"])
    pool = set(base_axis.loc[base_axis["risk_status"] == "assessed", "bjd_code"]) & \
        set(access.loc[access["access_2sfca"].notna(), "bjd_code"].astype(str))
    metric, share = choose_risk_metric(base_axis, params["risk_metric"], pool)
    out = combine_two_axes(base_axis, access, params["weights"], metric, min_axes, forecast_metrics)
    out = out.merge(run_sensitivity_v11(risk, access, params, metric), on="bjd_code", how="left")
    for col in ("robust_top", "boundary"):
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    shown = display_columns(energy_results, params)
    if shown is not None:
        out = out.merge(shown, on="bjd_code", how="left")
    out = out.sort_values(["rank_eligible", "PriorityScore", "bjd_code"], ascending=[False, False, True]).reset_index(drop=True)
    out["rank"] = np.where(out["rank_eligible"], np.arange(1, len(out) + 1), np.nan)
    pair = out.loc[out["rank_eligible"], ["risk_norm", "equity_norm"]].dropna()
    corr = {"n": len(pair), "pearson": np.nan, "spearman": np.nan}
    if len(pair) >= 3 and pair.nunique().min() >= 2:
        corr.update(pearson=float(pair["risk_norm"].corr(pair["equity_norm"])),
                    spearman=float(pair["risk_norm"].corr(pair["equity_norm"], method="spearman")))
    out.attrs.update(axes_correlation=corr, risk_metric_used=metric, share_zero_surge_risk=share)
    return out
