"""8-E 단계 — 안전·형평성·운영비 절감 잠재력의 재현 가능한 결합 점수."""
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


def combine_scores(energy, access, weights=(1 / 3, 1 / 3, 1 / 3), can_confidence=None):
    w = validate_weights(weights)
    df = energy.merge(access[["bjd_code", "access_2sfca", "access_data_present"]], on="bjd_code", how="outer")
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
    weights = params["weights"]
    multipliers = params.get("multipliers") or sorted(pd.to_numeric(results["multiplier"], errors="coerce").dropna().unique())
    runs = []
    for multiplier in multipliers:
        try:
            energy = energy_axes(results, float(multiplier), int(params["scenario_new_chargers"]))
        except ValueError:
            continue
        for w in _weight_grid(weights, float(params["weight_delta"])):
            score = combine_scores(energy, access, w)[["bjd_code", "PriorityScore"]].dropna()
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


def build_priority(results, access, params, can_confidence=None):
    base = combine_scores(energy_axes(results, float(params["base_multiplier"]),
                                      int(params["scenario_new_chargers"])), access,
                          params["weights"], can_confidence)
    sensitivity = run_sensitivity(results, access, params)
    out = base.merge(sensitivity, on="bjd_code", how="left")
    return out.sort_values(["PriorityScore", "bjd_code"], ascending=[False, True]).reset_index(drop=True)


def seasonal_coverage(hourly):
    dates = pd.to_datetime(hourly["date"], errors="coerce").dropna().dt.normalize().drop_duplicates()
    seasons = np.select([dates.dt.month.isin([3, 4, 5]), dates.dt.month.isin([6, 7, 8]),
                         dates.dt.month.isin([9, 10, 11])], ["봄", "여름", "가을"], default="겨울")
    counts = pd.Series(seasons).value_counts()
    present = set(counts[counts >= 28].index)
    required = {"봄", "여름", "가을", "겨울"}
    return {"season_count": len(present), "season_status": "검증 가능" if present == required else "보류",
            "missing_seasons": ",".join(sorted(required - present))}
