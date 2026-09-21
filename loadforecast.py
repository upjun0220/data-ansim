"""8-A: 다음날 24시간 평균 부하 예측·검증·급증 위험. 실시간 제어가 아니다.

기준 모델(forecast_baseline): 같은 요일·시간 최근 W주 중앙값(V5). 비교 기준으로 남긴다.
AI(v11, run_ai_forecast): 서울 법정동 전체를 한 모델로 학습하는 그래디언트 부스팅 분위수 회귀(P50·P90).
  lightgbm → sklearn HistGradientBoostingRegressor(loss="quantile") → 기준 모델 순으로 폴백한다. 딥러닝 아님.
급증 위험(surge_risk, v11.3): 변압기 과부하가 아니라 "그 동네의 교정기간 최대 부하 × 배율을 넘는 급증"이다.
  점수·검증용은 증설 0대(관측 기반)이고, 증설 2/3/5대(ΔL 가정)는 참고 레이어(surge_risk_with_new_N)로만 낸다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import log


def daily_matrix(load_df, region):
    """날짜×0~23시. 없는 셀은 NaN으로 남기며 중복·음수는 거부한다."""
    g = load_df.loc[load_df["bjd_code"].astype(str) == str(region)].copy()
    if g.empty:
        raise ValueError(f"{region}: 시간별 부하 없음")
    g["date"] = pd.to_datetime(g["date"], errors="raise").dt.normalize()
    if g.duplicated(["date", "hour"]).any():
        raise ValueError("일자·시간 중복")
    if not g["hour"].isin(range(24)).all() or not np.isfinite(g["kw"]).all() or (g["kw"] < 0).any():
        raise ValueError("시간/부하 값 오류")
    return g.pivot(index="date", columns="hour", values="kw").reindex(columns=range(24)).sort_index()


def forecast_baseline(matrix, target_date, W=4, holidays=None):
    """날짜에서 7·w일을 뺀 뒤 동일 시간 열을 선택한다. 대상일 이후 값은 접근하지 않는다.

    holidays는 공식 달력을 확인한 {YYYY-MM-DD: 휴일유형} 선택 입력이다.
    동일 유형 휴일 표본이 W개 미만이면 예측을 실패 처리하며 임의 대체하지 않는다.
    """
    if not isinstance(W, int) or W < 1:
        raise ValueError("W는 양의 정수")
    day = pd.Timestamp(target_date).normalize()
    holiday_map = {pd.Timestamp(k).normalize(): v for k, v in (holidays or {}).items()}
    if day in holiday_map:
        dates = sorted((d for d, kind in holiday_map.items() if d < day and kind == holiday_map[day]),
                       reverse=True)[:W]
    else:
        dates = [day - pd.Timedelta(days=7 * w) for w in range(1, W + 1)]
    sample = matrix.reindex(dates)
    if len(dates) != W or sample.isna().any().any():
        raise ValueError(f"{day.date()}: 동일 요일·시간 {W}주 완전한 이력 부족")
    return sample.median(axis=0).rename("forecast_kw")


forecast_day = forecast_baseline   # V10 이름(8-B·테스트 호환)


def forecast_load(load_df, target_date, region, W=4, holidays=None):
    return forecast_baseline(daily_matrix(load_df, region), target_date, W, holidays)


def backtest_forecast(load_df, region, n_weeks_holdout=8, W=4, holidays=None, end_date=None):
    """동일 관측 셀에서 계절 중앙값과 전날 값을 비교. end_date까지의 자료만 평가한다."""
    matrix = daily_matrix(load_df, region)
    end = pd.Timestamp(end_date).normalize() if end_date is not None else matrix.index.max()
    if n_weeks_holdout < 1:
        raise ValueError("홀드아웃은 1주 이상")
    rows, skipped = [], 0
    for day in pd.date_range(end - pd.Timedelta(days=7 * n_weeks_holdout - 1), end):
        actual = matrix.reindex([day]).iloc[0]
        previous = matrix.reindex([day - pd.Timedelta(days=1)]).iloc[0]
        try:
            pred = forecast_day(matrix, day, W, holidays)
        except ValueError:
            skipped += 1
            continue
        if actual.isna().any() or previous.isna().any():
            skipped += 1
            continue
        growing = actual.sum() > matrix.reindex([day - pd.Timedelta(days=7 * w)
                                                for w in range(1, W + 1)]).sum(axis=1).median()
        peak = int(actual.idxmax())
        for hour in range(24):
            rows.append((day, hour, actual[hour] - pred[hour], abs(actual[hour] - previous[hour]),
                         bool(growing), hour == peak))
    details = pd.DataFrame(rows, columns=["date", "hour", "error", "persistence_error", "growing", "peak"])
    if details.empty:
        raise ValueError(f"{region}: 검증 가능한 완전한 날짜 0개")
    growth = details.loc[details["growing"], "error"]
    return {"bjd_code": str(region), "mae_naive_seasonal": details["error"].abs().mean(),
            "mae_persistence": details["persistence_error"].mean(), "bias_actual_minus_forecast": details["error"].mean(),
            "growth_bias": growth.mean(), "growth_n_obs": len(growth),
            "peak_mae": details.loc[details["peak"], "error"].abs().mean(),
            "n_obs": len(details), "n_days": details["date"].nunique(), "skipped_days": skipped,
            "calendar": "설정 달력 적용" if holidays else "공휴일 미보정", "validation_end": str(end.date())}


# ================================================================ v11 AI 분위수 예측

HOLIDAY_CODE = {"holiday": 1, "long_holiday": 2, "pre_post_holiday": 3}   # 평일·주말 = 0
BASE_FEATURES = ["hour", "dow", "holiday_type", "lag1_kw", "lag7_kw", "baseline_kw"]
REGION_FEATURES = ["ev_count", "chargers_assigned", "train_mean_kw"]


def _matrices(hourly):
    """법정동 → 날짜×0~23시 행렬(daily_matrix 규칙으로 중복·음수 거부)."""
    return {code: daily_matrix(hourly, code) for code in sorted(hourly["bjd_code"].astype(str).unique())}


def build_features(matrices, dates, holidays=None, W=4):
    """예측 대상일 dates 의 입력 특성(long 표). 모든 값은 d−1일 23시 이전 관측에서만 온다.

    lag1 = d−1일 같은 시간, lag7 = d−7일, baseline = forecast_baseline 과 같은 규칙(평일: 같은 요일 W주 중앙값이며
    W일 모두 24시간 완전할 때만, 휴일: 같은 유형 과거 휴일 W개). actual_kw 는 정답(특성 아님)이다.
    src_max_date 는 특성에 쓴 가장 늦은 관측일로, assert_no_leakage 가 예측일보다 앞인지 검사한다.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(sorted(set(dates)))).normalize()
    hmap = {pd.Timestamp(k).normalize(): v for k, v in (holidays or {}).items()}
    frames = []
    for code, m in matrices.items():
        full = m.reindex(pd.date_range(min(m.index.min(), idx.min() - pd.Timedelta(days=7 * W)), idx.max()))
        complete = full.notna().all(axis=1)
        lags = [full.shift(7 * w).reindex(idx) for w in range(1, W + 1)]
        ok = np.column_stack([complete.shift(7 * w, fill_value=False).reindex(idx).to_numpy(bool) for w in range(1, W + 1)]).all(axis=1)
        base = pd.DataFrame(np.median(np.stack([l.to_numpy(float) for l in lags]), axis=0), index=idx, columns=range(24))
        base[~ok] = np.nan
        for d in idx[idx.isin(list(hmap))]:
            try:
                base.loc[d] = forecast_baseline(m, d, W, holidays).to_numpy()
            except ValueError:
                base.loc[d] = np.nan
        parts = {"lag1_kw": full.shift(1).reindex(idx), "lag7_kw": full.shift(7).reindex(idx), "baseline_kw": base,
                 "actual_kw": full.reindex(idx)}
        long = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1)
        long.index = long.index.set_names(["date", "hour"])
        frames.append(long.reset_index().assign(bjd_code=code))
    out = pd.concat(frames, ignore_index=True)
    out["hour"] = out["hour"].astype(int)
    out["dow"] = out["date"].dt.dayofweek
    out["holiday_type"] = out["date"].map(lambda d: HOLIDAY_CODE.get(hmap.get(d), 0)).astype(int)
    out["src_max_date"] = out["date"] - pd.Timedelta(days=1)
    return out


def temperature_feature(weather, mode, station_map, cutoff="18:00"):
    """(bjd_code, date, hour) 기온 특성. forecast 는 마감 이전 발표분만(누설 검사 포함), observed 는 상한 참고용."""
    import weatherloader

    if mode == "none" or weather is None:
        return None
    if mode == "forecast":
        t = weatherloader.assert_forecast_before_cutoff(weatherloader.forecast_temps(weather, cutoff), cutoff)
    elif mode == "observed":
        t = weatherloader.observed_temps(weather).assign(fcst_issued_at=pd.NaT)
    else:
        raise ValueError(f"weather.mode 는 forecast|none|observed — 받은 값: {mode!r}")
    t = t.merge(station_map, on="station_or_grid")
    return pd.DataFrame({"bjd_code": t["bjd_code"].astype(str), "date": t["timestamp"].dt.normalize(),
                         "hour": t["timestamp"].dt.hour, "temp_c": t["temp_c"], "temp_issued_at": t["fcst_issued_at"]})


def assert_no_leakage(features, weather_mode="none", cutoff="18:00"):
    """예측일 d 의 입력이 d 이후 관측이거나, 예보 발표가 d−1일 cutoff 이후면 ValueError(킬 19번이 실패로 표시).

    observed 모드는 정의상 당일 실측 기온을 쓰는 상한 참고 모델이라 기온 검사에서 뺀다(발표 숫자 금지).
    """
    late = features["src_max_date"] >= features["date"]
    if late.any():
        raise ValueError(f"정보 누설: 예측일 이후 관측을 쓴 입력 {int(late.sum())}행")
    if weather_mode == "forecast" and "temp_issued_at" in features:
        hh, mm = (int(v) for v in str(cutoff).split(":"))
        limit = features["date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm)
        bad = features["temp_issued_at"].notna() & (features["temp_issued_at"] > limit)
        if bad.any():
            raise ValueError(f"정보 누설: 예보 발표 시각이 전날 {cutoff} 이후인 입력 {int(bad.sum())}행")
    return features


def _sklearn_quantile_supported():
    from sklearn.ensemble import HistGradientBoostingRegressor
    HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=2).fit(np.arange(10.0)[:, None], np.arange(10.0))
    return True


def select_model(requested="auto", random_state=42, max_iter=100):
    """(사용 모델 이름, factory(q, params) 또는 None). auto: lightgbm → sklearn(분위수 손실 지원 버전) → baseline."""
    order = {"auto": ["lightgbm", "sklearn"], "lightgbm": ["lightgbm"], "sklearn": ["sklearn"], "baseline": []}
    if requested not in order:
        raise ValueError(f"forecast.model 은 auto|lightgbm|sklearn|baseline — 받은 값: {requested!r}")
    for name in order[requested]:
        try:
            if name == "lightgbm":
                import lightgbm

                def factory(q, p):
                    return lightgbm.LGBMRegressor(objective="quantile", alpha=q, learning_rate=p["learning_rate"],
                                                  num_leaves=p["max_leaf_nodes"], min_child_samples=p["min_samples_leaf"],
                                                  n_estimators=max_iter, random_state=random_state, verbose=-1)
                return "lightgbm", factory
            _sklearn_quantile_supported()
            from sklearn.ensemble import HistGradientBoostingRegressor

            def factory(q, p):
                return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=p["learning_rate"],
                                                     max_leaf_nodes=p["max_leaf_nodes"],
                                                     min_samples_leaf=p["min_samples_leaf"], max_iter=max_iter,
                                                     early_stopping=False, random_state=random_state)
            return "sklearn", factory
        except Exception as exc:   # 미설치·분위수 손실 미지원
            log.warning("8-A %s 사용 불가(%s) — 다음 폴백", name, type(exc).__name__)
    return "baseline", None


def pinball(y, q_pred, tau):
    diff = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def fit_quantiles(train, features, quantiles, factory, grid, folds=3):
    """분위수마다 따로 학습. 하이퍼파라미터는 학습기간 안의 시간 순 확장창 교차검증(pinball)으로만 고른다."""
    import itertools

    dates = np.array(sorted(train["date"].unique()))
    blocks = np.array_split(dates, int(folds) + 1)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    models = {}
    for q in quantiles:
        scored = []
        for params in combos:
            losses = []
            for k in range(1, len(blocks)):
                tr = train[train["date"].isin(np.concatenate(blocks[:k]))]
                va = train[train["date"].isin(blocks[k])]
                if tr.empty or va.empty:
                    continue
                losses.append(pinball(va["actual_kw"], factory(q, params).fit(tr[features], tr["actual_kw"]).predict(va[features]), q))
            scored.append((np.mean(losses) if losses else np.inf, sorted(params.items())))
        best = dict(min(scored, key=lambda x: x[0])[1])
        models[q] = (factory(q, best).fit(train[features], train["actual_kw"]), best)
    return models


def predict_quantiles(models, frame, features):
    """q50·q90 예측. 분위수 교차를 막기 위해 q90 = max(q90, q50). 부하는 음수가 될 수 없어 0에서 자른다."""
    out = frame.copy()
    for q, (model, _) in models.items():
        out[f"q{int(round(q * 100))}"] = np.maximum(model.predict(frame[features]), 0.0)
    if {"q50", "q90"} <= set(out):
        out["q90"] = np.maximum(out["q90"], out["q50"])
    return out


def _metrics(g):
    """점 예측·분위수 지표. 세 예측(persistence·기준·AI)이 모두 있는 같은 셀에서만 비교한다."""
    c = g.dropna(subset=["actual_kw", "lag1_kw", "baseline_kw", "q50", "q90"])
    if c.empty:
        return None
    keys = ["bjd_code", "date"]
    day = c.groupby(keys)[["actual_kw", "baseline_kw"]].transform("sum")
    growing = day["actual_kw"] > day["baseline_kw"]          # 법정동·일 단위 증가일
    peak_idx = c.groupby(keys)["actual_kw"].idxmax()          # 법정동·일마다 실측 피크 시각
    peak = c.loc[peak_idx]
    return {"n_days": c["date"].nunique(), "n_obs": len(c),
            "mae_persistence": (c["actual_kw"] - c["lag1_kw"]).abs().mean(),
            "mae_baseline": (c["actual_kw"] - c["baseline_kw"]).abs().mean(),
            "mae_ai_p50": (c["actual_kw"] - c["q50"]).abs().mean(),
            "growth_bias_baseline": (c.loc[growing, "actual_kw"] - c.loc[growing, "baseline_kw"]).mean(),
            "growth_bias_ai": (c.loc[growing, "actual_kw"] - c.loc[growing, "q50"]).mean(),
            "peak_mae_baseline": (peak["actual_kw"] - peak["baseline_kw"]).abs().mean(),
            "peak_mae_ai": (peak["actual_kw"] - peak["q50"]).abs().mean(),
            "p90_coverage": float((c["actual_kw"] <= c["q90"]).mean()),
            "pinball_p50": pinball(c["actual_kw"], c["q50"], 0.5), "pinball_p90": pinball(c["actual_kw"], c["q90"], 0.9)}


def evaluate_forecasts(pred, model_used, weather_mode, min_coverage=0.8):
    """s8a_forecast_metrics — 전체 한 행(bjd_code 빈칸) + 법정동별. AI가 못 이긴 동네도 숨기지 않는다."""
    rows = [{"bjd_code": "", **(_metrics(pred) or {})}]
    rows += [{"bjd_code": code, **m} for code, g in pred.groupby("bjd_code") if (m := _metrics(g))]
    out = pd.DataFrame(rows)
    out["ai_beats_baseline"] = out["mae_ai_p50"] <= out["mae_baseline"]
    out["p90_conservative_enough"] = out["p90_coverage"] >= min_coverage
    out["model_used"], out["weather_mode"] = model_used, weather_mode
    return out


def ai_improvement(metrics, min_coverage=0.8):
    """킬 18번 — 전체 행(실측 기온 참고 모델 제외)에서 AI P50 MAE ≤ 기준 모델 MAE 이고 P90 적중률 ≥ min_coverage 인가.

    → (통과 여부, 근거 문자열). 지표가 없으면 (False, "AI 예측 미실행").
    """
    if metrics is None or metrics.empty or str(metrics["model_used"].iloc[0]).startswith("baseline"):
        return False, "AI 예측 미실행"
    ref = metrics["reference_only"].astype(bool) if "reference_only" in metrics \
        else pd.Series(False, index=metrics.index)
    overall = metrics["bjd_code"].isna() | metrics["bjd_code"].astype(str).isin(["", "nan"])
    m = metrics[overall & ~ref].iloc[0]
    ok = bool(m["mae_ai_p50"] <= m["mae_baseline"] and m["p90_coverage"] >= min_coverage)
    return ok, (f"AI P50 MAE {m['mae_ai_p50']:.3f} vs 기준 {m['mae_baseline']:.3f} · P90 적중률 {m['p90_coverage']:.1%}"
                f" (기준 {min_coverage:.0%})")


def surge_risk(pred, calib_peak, multipliers, added_loads=None, min_calib_peak_kw=1.0):
    """s8a_risk — 법정동 × 상한 배율. 목표 상한 = 교정기간 최대 부하 × 배율(8-B와 같은 policy_target).

    급증경보(r,d) = 1[max_h q90 > 상한], 급증위험 = 평가일 중 경보 비율(증설 0대, 점수용).
    실제급증(r,d) = 1[max_h 실측 > 상한]과 대조한 정밀도·재현율은 AI·기준 모델 경보가 모두 있고 실측이 24시간 완전한
    공통일에서만 계산한다(기준 모델이 휴일 표본 부족으로 실패한 날은 n_excluded_days 로 표시).
    surge_risk_with_new_N = max_h(q90 + ΔL_N) > 상한인 날 비율 — ΔL 가정값이라 참고용, 점수·검증에 쓰지 않는다.
    """
    from essoptimizer import policy_target

    day = pred.groupby(["bjd_code", "date"]).agg(
        ai_peak=("q90", "max"), base_peak=("baseline_kw", lambda s: s.max() if s.notna().all() else np.nan),
        actual_peak=("actual_kw", lambda s: s.max() if s.notna().all() else np.nan)).reset_index()
    curves = {k: g.sort_values("hour")["q90"].to_numpy() for k, g in pred.groupby(["bjd_code", "date"])}
    rows = []
    for code, g in day.groupby("bjd_code"):
        peak = float(calib_peak.get(code, np.nan))
        for m in multipliers:
            row = {"bjd_code": code, "multiplier": m, "calib_peak_kw": peak, "n_eval_days": len(g)}
            if not np.isfinite(peak) or peak < min_calib_peak_kw:
                rows.append({**row, "status": "not_assessed_low_load"})
                continue
            target = policy_target(peak, m)
            ai, base, act = g["ai_peak"] > target, g["base_peak"] > target, g["actual_peak"] > target
            common = g["ai_peak"].notna() & g["base_peak"].notna() & g["actual_peak"].notna()
            row.update(status="assessed", target_kw=target, surge_risk=float(ai.mean()),
                       peak_ratio=float(g["ai_peak"].mean() / peak), n_alert_days_ai=int(ai.sum()),
                       n_alert_days_base=int(base[g["base_peak"].notna()].sum()), n_actual_surge_days=int(act[g["actual_peak"].notna()].sum()),
                       n_common_days=int(common.sum()), n_excluded_days=int((~common).sum()))
            for name, alert in (("ai", ai), ("base", base)):
                tp = int((alert & act & common).sum())
                n_alert, n_act = int((alert & common).sum()), int((act & common).sum())
                row[f"tp_{name}"], row[f"alerts_{name}"], row[f"actual_{name}"] = tp, n_alert, n_act
                row[f"precision_{name}"] = tp / n_alert if n_alert else np.nan
                row[f"recall_{name}"] = tp / n_act if n_act else np.nan
            for n_new, delta in (added_loads or {}).items():
                row[f"surge_risk_with_new_{n_new}"] = float(np.mean(
                    [(curves[(code, d)] + delta).max() > target for d in g["date"]]))
            rows.append(row)
    return pd.DataFrame(rows)


def risk_summary(risk):
    """상한 배율별 합계 — 경보 정밀도·재현율은 법정동·일을 합친 값(micro). 발표 숫자 2·3의 재료."""
    rows = []
    for m, g in risk.groupby("multiplier"):
        a = g[g["status"] == "assessed"]
        row = {"multiplier": m, "n_assessed": len(a), "n_not_assessed_low_load": int((g["status"] != "assessed").sum()),
               "share_zero_surge_risk": float((a["surge_risk"] == 0).mean()) if len(a) else np.nan,
               "n_excluded_days": int(a["n_excluded_days"].sum()) if len(a) else 0}
        for name in ("ai", "base"):
            tp, al, ac = (int(a[f"{k}_{name}"].sum()) if len(a) else 0 for k in ("tp", "alerts", "actual"))
            row[f"precision_{name}"] = tp / al if al else np.nan
            row[f"recall_{name}"] = tp / ac if ac else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_ai_forecast(hourly, params, holidays=None, region_feats=None, weather=None, station_map=None,
                    weather_mode="none", cutoff="18:00"):
    """학습 → (평가 시작 전 holdout_weeks) 검증 → 평가일 예측. 세 기간은 시간 순으로 겹치지 않는다.

    params: energy 설정 + "forecast" 하위 설정. weather_mode 가 none 인데 실측 기온이 있으면 observed 참고 모델도
    학습해 지표만 나란히 낸다(발표 숫자·경보에는 쓰지 않음).
    """
    fp = params["forecast"]
    start = pd.Timestamp(params["evaluation_start"]).normalize()
    valid_start = start - pd.Timedelta(days=7 * int(params["holdout_weeks"]))
    train_start = valid_start - pd.Timedelta(days=int(fp["train_days"]))
    eval_days = pd.date_range(start, periods=int(params["evaluation_days"]), freq="D")
    matrices = _matrices(hourly)
    first = min(m.index.min() for m in matrices.values())
    train_days = pd.date_range(max(train_start, first + pd.Timedelta(days=7 * int(params["weeks"]))),
                               valid_start - pd.Timedelta(days=1))
    valid_days = pd.date_range(valid_start, start - pd.Timedelta(days=1))
    if len(train_days) < 14:
        raise ValueError(f"AI 학습일 {len(train_days)}일 — 14일 미만(시간별 이력 부족)")
    feats = build_features(matrices, train_days.append(valid_days).append(eval_days), holidays, int(params["weeks"]))
    train_mask = feats["date"].isin(train_days)
    feats = feats.merge(feats[train_mask].groupby("bjd_code")["actual_kw"].mean().rename("train_mean_kw"),
                        on="bjd_code", how="left")
    if region_feats is not None:
        cols = [c for c in ("ev_count", "chargers_assigned") if c in region_feats]
        feats = feats.merge(region_feats[["bjd_code"] + cols].assign(bjd_code=lambda d: d["bjd_code"].astype(str)),
                            on="bjd_code", how="left")
    name, factory = select_model(fp["model"], int(fp["random_state"]), int(fp["max_iter"]))
    modes = [weather_mode] + (["observed"] if weather_mode == "none" and weather is not None
                              and weather["temp_obs_c"].notna().any() else [])
    results = {}
    for mode in modes:
        frame = feats
        temp = temperature_feature(weather, mode, station_map, cutoff) if station_map is not None else None
        if temp is not None:
            frame = frame.merge(temp, on=["bjd_code", "date", "hour"], how="left")
        assert_no_leakage(frame, mode, cutoff)
        features = BASE_FEATURES + [c for c in REGION_FEATURES + ["temp_c"] if c in frame and frame[c].notna().any()]
        if factory is None:
            pred = frame.assign(q50=frame["baseline_kw"], q90=frame["baseline_kw"])
            chosen = {}
        else:
            train = frame[frame["date"].isin(train_days) & frame["actual_kw"].notna()]
            models = fit_quantiles(train, features, fp["quantiles"], factory, fp["grid"], fp["cv_folds"])
            pred = predict_quantiles(models, frame, features)
            chosen = {q: p for q, (_, p) in models.items()}
        label = name if factory else "baseline(AI 예측 미실행)"
        results[mode] = {"valid": pred[pred["date"].isin(valid_days)], "eval": pred[pred["date"].isin(eval_days)],
                         "features": features, "params": chosen}
        log.info("8-A AI %s · 기온 %s · 특성 %s · 학습 %d일 · 검증 %d일 · 평가 %d일", label, mode, features,
                 len(train_days), len(valid_days), len(eval_days))
    metrics = pd.concat([evaluate_forecasts(r["valid"], name if factory else "baseline(AI 예측 미실행)", mode,
                                            float(fp["min_p90_coverage"])).assign(reference_only=(mode == "observed"))
                         for mode, r in results.items()], ignore_index=True)
    main = results[weather_mode]
    return {"model_used": name if factory else "baseline(AI 예측 미실행)", "weather_mode": weather_mode,
            "metrics": metrics, "valid": main["valid"], "eval": main["eval"], "features": main["features"],
            "params": main["params"], "train_days": (str(train_days[0].date()), str(train_days[-1].date())),
            "valid_days": (str(valid_days[0].date()), str(valid_days[-1].date()))}
