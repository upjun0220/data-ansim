"""8-A+ (v12.1) AI 강화 — 기존 8-A AI 와 같은 학습·검증·평가 기간에서 나란히 돌려 비교표를 낸다(기존 숫자·점수는 그대로).

1차 실데이터에서 기존 AI(동네·시간별 kW 를 원래 크기로 한 모델에)는 기준 모델(4주 중앙값)은 이겼지만 어제 같은 시각 값
(persistence)을 못 이겼고, "24시간 중 한 시각이라도 P90 이 상한을 넘으면 경보"라 정밀도가 1%였다. 강화 내용:

1) 어제값 위 보정: 정답 = (실측 − 기준값) ÷ 동네 규모. 기준값 = 어제 같은 시각(없으면 최근 7일 같은 시각 평균 → 4주 중앙값).
   AI 는 기준값에서 얼마나 벗어날지만 배운다(지속성 잔차 보정). 동네 규모(학습기간 평균)로 나눠 큰 동네가 학습을 좌우하지 않게 한다.
2) 다음날 동네 최대부하(일 피크)를 직접 예측하고, 그 P90 으로 하루 한 번 급증 경보를 판단한다.
4) 특성: 1·2·7·14일 전 같은 시각, 최근 7일 같은 시각 평균, 4주 중앙값, 어제 일총량·일피크, 최근 7일 최대 일피크,
   추세(최근 7일 ÷ 그 전 28일), 휴일 유형, 같은 구 이웃 동네의 어제 같은 시각 평균, 기온 예보. 모두 d−1일까지만.
6) 구 합계·서울 합계 예측: 동네 합(그 시각 행이 없는 동네는 0으로 더함)을 같은 방식으로 예측한다(잡음이 작아 AI 효과가 잘 보인다).
7) 컨포멀 보정(검증기간 잔차로 P90 을 옮겨 평가기간 적중률을 목표에 맞춤) + 순열 중요도(무엇이 예측을 움직이나).
8) 충전 생활 유형: 학습기간 24시간 평균 모양(합 1)으로 K-평균 군집(비지도 학습).
모델은 센터에 있는 scikit-learn HistGradientBoostingRegressor(분위수 손실)뿐이다. 딥러닝 아님.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import loadforecast
from common import log

KW = ["lag1", "lag2", "lag7", "lag14", "roll7", "baseline", "yday_total", "yday_peak", "wk_peak"]
# 월·연중 위치는 넣지 않는다: 학습(약 반년) 뒤 검증·평가 기간의 달은 학습에 없던 값이라 나무 모델이 학습기간의 달별 우연한
# 차이를 그대로 옮긴다(가짜 자료에서 합계 예측이 어제값보다 60% 나빠졌다). 계절은 기온 예보·최근 값·추세로만 반영한다.
LIVING = ["lp_night_r", "lp_day_r", "lp_eve_r", "lp_mix"]   # 생활인구(13.csv, 선택) — 1주 전 값만(공개가 며칠 늦다)
FEATURES = [f"{c}_n" for c in KW] + ["ref_n", "group_lag1_n", "trend", "hour", "dow", "holiday_type", "temp_c"] + LIVING
PEAK_FEATURES = ["yday_peak_n", "wk_peak_n", "base_peak_n", "lag7_peak_n", "yday_total_n", "ref_n", "group_yday_peak_n",
                 "trend", "dow", "holiday_type", "temp_max"] + LIVING
LABELS = {"lag1_n": "어제 같은 시각", "lag2_n": "2일 전 같은 시각", "lag7_n": "1주 전 같은 시각", "lag14_n": "2주 전 같은 시각",
          "roll7_n": "최근 7일 같은 시각 평균", "baseline_n": "4주 중앙값(기준 모델)", "yday_total_n": "어제 일총량",
          "yday_peak_n": "어제 일피크", "wk_peak_n": "최근 7일 최대 일피크", "ref_n": "기준값", "group_lag1_n": "같은 구 이웃 어제",
          "trend": "추세(7일÷28일)", "hour": "시각", "dow": "요일", "holiday_type": "휴일 유형", "month": "월",
          "doy_sin": "연중 위치(sin)", "doy_cos": "연중 위치(cos)", "temp_c": "기온 예보",
          "lp_night_r": "생활인구 야간(1주 전·4주 대비)", "lp_day_r": "생활인구 주간(1주 전·4주 대비)",
          "lp_eve_r": "생활인구 저녁(1주 전·4주 대비)", "lp_mix": "생활인구 야간÷주간(1주 전)"}


def load_living(path):
    """13.csv(서울 생활인구 법정동·일: date, bjd_code, lp_night, lp_day, lp_eve) → 표."""
    d = pd.read_csv(path, dtype={"bjd_code": str}, encoding="utf-8-sig")
    return d.assign(date=pd.to_datetime(d["date"]).dt.normalize())


def living_features(living, key=lambda c: c):
    """(코드, 날짜) 생활인구 특성. 날짜 d 에는 d−7일 값만 쓴다(실제 공개가 며칠 늦어 전날 값은 운영에서 못 쓴다).
    비율 = d−7일 값 ÷ 그 전 28일(d−35~d−8) 평균 — 방학·연휴처럼 동네에 머무는 사람이 평소와 다른지를 본다."""
    bands = ["lp_night", "lp_day", "lp_eve"]
    lv = living.assign(code=living["bjd_code"].astype(str).map(key)).groupby(["code", "date"])[bands].sum()
    days = pd.date_range(lv.index.get_level_values("date").min(), lv.index.get_level_values("date").max() + pd.Timedelta(days=7))
    out = {}
    for b in bands:
        wide = lv[b].unstack("code").reindex(days)
        out[f"{b}_r"] = wide.shift(7) / wide.shift(8).rolling(28, min_periods=7).mean()
        out[b] = wide.shift(7)
    feats = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    feats.index = feats.index.set_names(["date", "bjd_code"])
    feats = feats.reset_index()
    feats["lp_mix"] = feats["lp_night"] / feats["lp_day"]
    return feats[["bjd_code", "date"] + LIVING].replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- 특성

def build_panel(matrices, dates, holidays=None, W=4, group_of=lambda c: str(c)[:5]):
    """(코드, 날짜, 시각) 특성 표. 8-A build_features(어제·1주 전·4주 중앙값) 위에 강화 특성을 붙인다. 모두 d−1일까지의 관측."""
    base = loadforecast.build_features(matrices, dates, holidays, W)
    idx = pd.DatetimeIndex(pd.to_datetime(sorted(set(dates)))).normalize()
    frames = []
    for code, m in matrices.items():
        full = m.reindex(pd.date_range(min(m.index.min(), idx.min() - pd.Timedelta(days=40)), idx.max()))
        complete = full.notna().all(axis=1)
        dtot, dpeak = full.sum(axis=1).where(complete), full.max(axis=1).where(complete)
        trend = dtot.shift(1).rolling(7, min_periods=3).mean() / dtot.shift(8).rolling(28, min_periods=7).mean()
        daily = pd.DataFrame({"yday_total_kw": dtot.shift(1), "yday_peak_kw": dpeak.shift(1),
                              # 규모 = 최근 28일(전날까지) 시간 평균·일피크 평균 — 부하가 늘어도 따라 움직인다
                              "level_kw": dtot.shift(1).rolling(28, min_periods=7).mean() / 24,
                              "level_peak_kw": dpeak.shift(1).rolling(28, min_periods=7).mean(),
                              "wk_peak_kw": dpeak.shift(1).rolling(7, min_periods=3).max(),
                              "trend": trend.replace([np.inf, -np.inf], np.nan)}).reindex(idx)
        hourly = {"lag2_kw": full.shift(2), "lag14_kw": full.shift(14),
                  "roll7_kw": full.shift(1).rolling(7, min_periods=3).mean()}
        long = pd.concat({k: v.reindex(idx).stack(future_stack=True) for k, v in hourly.items()}, axis=1)
        long.index = long.index.set_names(["date", "hour"])
        long = long.reset_index().merge(daily.rename_axis("date").reset_index(), on="date")
        frames.append(long.assign(bjd_code=code))
    extra = pd.concat(frames, ignore_index=True)
    extra["hour"] = extra["hour"].astype(int)
    out = base.merge(extra, on=["bjd_code", "date", "hour"], how="left")
    out["ref_kw"] = out["lag1_kw"].fillna(out["roll7_kw"]).fillna(out["baseline_kw"])
    out["group"] = out["bjd_code"].map(group_of)
    doy = out["date"].dt.dayofyear
    out["month"], out["doy_sin"], out["doy_cos"] = out["date"].dt.month, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    num = out.select_dtypes("float64").columns    # 서울 규모(약 300만 행)에서 메모리를 절반으로
    out[num] = out[num].astype("float32")
    return out


def _scale(frame, train_mask, col="actual_kw", level="level_kw"):
    """동네 규모 = 최근 28일 평균(전날까지, 날마다 움직임). 없으면 학습기간 평균 → 전체 중앙값.
    고정 규모(학습기간 평균)로 나누면 부하가 늘수록 검증기간 값이 모두 '평소보다 높게' 보여 모델이 하락을 예측한다
    (가짜 자료에서 합계 예측이 어제값보다 크게 나빠졌다). 규모로 나눠 큰 동네가 학습을 좌우하지 않게 한다."""
    s = frame[train_mask].groupby("bjd_code")[col].mean()
    s = s.where(s > 0)
    fallback = frame["bjd_code"].map(s).fillna(s.median() if s.notna().any() else 1.0)
    return frame[level].where(frame[level] > 0).fillna(fallback).astype(float)


def _normalize(frame, cols, scale):
    for c in cols:
        frame[f"{c}_n"] = frame[f"{c}_kw"] / scale
    return frame


# ---------------------------------------------------------------- 학습·예측

def _factory(fp):
    name, factory = loadforecast.select_model("sklearn", int(fp.get("random_state", 42)), int(fp["max_iter"]))
    if factory is None:
        raise RuntimeError("scikit-learn 분위수 회귀 사용 불가 — 8-A+ 생략")
    p = {k: fp[k] for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf")}
    return lambda q: factory(q, p)


def conformal_offset(residual, coverage):
    """분할 컨포멀: 검증기간 (실측 − P90) 잔차의 ⌈(n+1)·coverage⌉번째 값. P90 에 더하면 적중률이 coverage 에 맞춰진다."""
    e = np.sort(np.asarray(residual, float)[np.isfinite(residual)])
    if not len(e):
        return 0.0
    return float(e[min(int(np.ceil((len(e) + 1) * coverage)), len(e)) - 1])


def fit_residual(frame, features, target, ref, scale, train_mask, valid_mask, fp, prefix=""):
    """정답 (실측 − 기준값)/규모 를 P50·P90 으로 학습 → 예측(기준값 + 규모 × 잔차) + 컨포멀 P90. 모델(P50)도 돌려준다."""
    make = _factory(fp)
    train = frame[train_mask & frame[target].notna()]
    if len(train) < 100:
        raise ValueError(f"8-A+ 학습 행 {len(train)} — 100 미만")
    features = [f for f in features if train[f].notna().any()]   # 입력이 없는 특성(예: 기온 미사용)은 뺀다
    models = {q: make(q).fit(train[features], train[target]) for q in (0.5, 0.9)}
    out = frame.copy()
    q50 = np.maximum(out[ref] + scale * models[0.5].predict(out[features]), 0.0)
    q90 = np.maximum(out[ref] + scale * models[0.9].predict(out[features]), q50)
    actual = out[target.replace("_z", "")]   # z 정답 이름 = 실측 이름 + "_z"
    off = conformal_offset(((actual - q90) / scale)[valid_mask], float(fp["coverage"]))   # 규모로 나눈 잔차 기준
    out[f"{prefix}q50"], out[f"{prefix}q90"] = q50, q90
    out[f"{prefix}q90c"] = np.maximum(q90 + off * scale, q50)
    return out, (models[0.5], features), off


# ---------------------------------------------------------------- 지표

def _mae(a, p):
    return float(np.abs(np.asarray(a, float) - np.asarray(p, float)).mean())


def compare(frame, actual, preds, cover=None):
    """같은 셀(모든 예측이 있는 곳)에서 MAE·어제값 대비 개선률·P90 적중률. preds: {이름: 열}, 첫 항목 = 어제값."""
    c = frame.dropna(subset=[actual] + list(preds.values()))
    if c.empty:
        return {"n_obs": 0}
    row = {"n_obs": len(c), "n_codes": c["bjd_code"].nunique(), "n_days": c["date"].nunique()}
    maes = {name: _mae(c[actual], c[col]) for name, col in preds.items()}
    first = next(iter(maes.values()))
    for name, v in maes.items():
        row[f"mae_{name}"] = v
        row[f"skill_{name}"] = 1 - v / first if first else np.nan   # 어제값 대비 오차 감소율(+ 가 좋음)
    for name, col in (cover or {}).items():
        cc = frame.dropna(subset=[actual, col])
        row[f"p90cov_{name}"] = float((cc[actual] <= cc[col]).mean()) if len(cc) else np.nan
    return row


def peak_frame(panel):
    """시간 표 → (코드, 날짜) 일피크 표. 24시간이 모두 있는 날만 피크를 낸다."""
    keys = ["bjd_code", "date"]
    g = panel.groupby(keys)

    def full_max(col):
        return g[col].max().where(g[col].count() == 24)

    first = g[["yday_peak_kw", "wk_peak_kw", "yday_total_kw", "level_peak_kw", "trend", "dow", "holiday_type",
               "group"] + [c for c in LIVING if c in panel]].first()
    out = first.assign(actual_peak=full_max("actual_kw"), base_peak_kw=full_max("baseline_kw"),
                       lag7_peak_kw=full_max("lag7_kw"))
    if "temp_c" in panel:
        out["temp_max"] = g["temp_c"].max()
    for col in ("q50", "q90"):
        if col in panel:
            out[f"v1_{col}_peak"] = full_max(col)
    return out.reset_index()


def alert_table(peaks, calib_peak, multipliers, methods, min_calib_peak_kw=1.0):
    """상한 배율별 경보 정밀도·재현율 — 같은 법정동·일(모든 방법의 예측과 24시간 실측이 있는 날)에서 비교."""
    p = peaks.assign(calib=peaks["bjd_code"].map(calib_peak))
    p = p[p["calib"] >= min_calib_peak_kw].dropna(subset=["actual_peak"] + list(methods.values()))
    rows = []
    for m in multipliers:
        target = p["calib"] * float(m)
        act = p["actual_peak"] > target
        for name, col in methods.items():
            alert = p[col] > target
            tp, n_alert, n_act = int((alert & act).sum()), int(alert.sum()), int(act.sum())
            rows.append({"multiplier": m, "방법": name, "n_days": len(p), "실제급증": n_act, "경보": n_alert, "적중": tp,
                         "precision": tp / n_alert if n_alert else np.nan, "recall": tp / n_act if n_act else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 충전 생활 유형(비지도)

def charge_types(panel, train_mask, k=4, min_days=14, seed=0):
    """학습기간 24시간 평균 모양(합 1) K-평균 → (동네별 유형 번호 1..k, 유형 표, 유형별 평균 모양)."""
    from sklearn.cluster import KMeans

    t = panel[train_mask & panel["actual_kw"].notna()]
    days = t.groupby(["bjd_code", "date"])["actual_kw"].count()
    full = days[days == 24].reset_index()[["bjd_code", "date"]]
    t = t.merge(full, on=["bjd_code", "date"])
    ok = full.groupby("bjd_code").size()
    prof = t[t["bjd_code"].isin(ok[ok >= min_days].index)].pivot_table(index="bjd_code", columns="hour", values="actual_kw")
    prof = prof.div(prof.sum(axis=1), axis=0).dropna()
    if len(prof) < k * 3:
        raise ValueError(f"충전 유형: 모양을 만들 동네 {len(prof)}곳 — {k * 3} 미만")
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(prof.to_numpy())
    centers = pd.DataFrame(km.cluster_centers_, columns=range(24))
    night = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
    order = centers[night].sum(axis=1).sort_values(ascending=False).index   # 야간 비중 큰 유형부터 1번
    relabel = {old: new for new, old in enumerate(order, 1)}
    labels = pd.Series(km.labels_, index=prof.index).map(relabel)
    centers.index = centers.index.map(relabel)
    centers = centers.sort_index()

    def name(c):
        h = int(c.idxmax())
        kind = "야간형" if h in night else "아침형" if h <= 10 else "주간형" if h <= 16 else "저녁형"
        return f"{kind}(피크 {h}시)"

    table = pd.DataFrame({"유형": centers.index, "이름": [name(centers.loc[i]) for i in centers.index],
                          "동네 수": labels.value_counts().reindex(centers.index).fillna(0).astype(int).to_numpy(),
                          "야간(21~06시) 비중": centers[night].sum(axis=1).to_numpy(),
                          "주간(09~17시) 비중": centers[list(range(9, 18))].sum(axis=1).to_numpy()})
    return labels.astype(int), table, centers


def plot_types(centers, table, min_n):
    from outputs import new_figure, png_text

    fig = new_figure(8, 4)
    ax = fig.add_subplot(111)
    for i, row in table.iterrows():
        if row["동네 수"] >= min_n:   # 동네가 적은 유형의 평균 모양은 싣지 않는다(소표본)
            ax.plot(range(24), centers.loc[row["유형"]] * 100, marker="o", ms=3, label=png_text(f"{row['유형']} {row['이름']} · {row['동네 수']}곳"))
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel(png_text("시각"))
    ax.set_ylabel(png_text("하루 충전량 중 비중(%)"))
    ax.set_title(png_text("충전 생활 유형 — 동네별 24시간 충전 모양 군집(K-평균)"))
    ax.legend(fontsize=7)
    return fig


# ---------------------------------------------------------------- 전체

def aggregate_matrices(hourly, key=lambda c: str(c)[:5]):
    """동네 → 구(시군구 코드 5자리) 시간별 합. 그 시각 행이 없는 동네는 0으로 더한다(행 없음 = 사용 0 가정, 결과표에 적음)."""
    h = hourly.assign(bjd_code=hourly["bjd_code"].astype(str).map(key))
    h = h.groupby(["bjd_code", "date", "hour"], as_index=False)["kw"].sum()
    return loadforecast._matrices(h)


def _hourly_plus(matrices, days, params, temp, group_of, fp, living=None):
    train_days, valid_days, eval_days = days
    panel = build_panel(matrices, train_days.append(valid_days).append(eval_days), params["holidays"],
                        int(params["weeks"]), group_of)
    if temp is not None:
        panel = panel.merge(temp, on=["date", "hour"], how="left")
    else:
        panel["temp_c"] = np.nan
    if living is not None:
        panel = panel.merge(living, on=["bjd_code", "date"], how="left")
    else:
        panel[LIVING] = np.nan
    tr, va = panel["date"].isin(train_days), panel["date"].isin(valid_days)
    s = _scale(panel, tr)
    panel = _normalize(panel, KW + ["ref"], s)
    panel["group_lag1_n"] = panel.groupby(["group", "date", "hour"])["lag1_n"].transform("mean")
    panel["actual_kw_z"] = (panel["actual_kw"] - panel["ref_kw"]) / s
    panel, model, off = fit_residual(panel, FEATURES, "actual_kw_z", "ref_kw", s, tr, va, fp, prefix="plus_")
    return panel.drop(columns=["lag2_kw", "lag14_kw", "roll7_kw"]), model, off, s


def run_ai_plus(hourly, params, v1=None, calib_peak=None, temp=None, visible=None, living=None):
    """8-A+ 전체. params = energy 설정 + forecast + forecast_plus. v1 = 기존 AI 예측(valid·eval 합친 표, q50·q90).
    temp = (date, hour, temp_c) 기온 예보(마감 이전 발표분) 또는 None. visible = PNG 에 실을 법정동(소표본 제외).
    living = load_living(13.csv) 생활인구 또는 None(그 특성만 빠진다)."""
    fp = params["forecast_plus"]
    matrices = loadforecast._matrices(hourly)
    days = loadforecast.ai_periods(params, matrices)
    train_days, valid_days, eval_days = days
    panel, model, off, s = _hourly_plus(matrices, days, params, temp, lambda c: str(c)[:5], fp,
                                        living_features(living) if living is not None else None)
    if v1 is not None and len(v1):
        panel = panel.merge(v1[["bjd_code", "date", "hour", "q50", "q90"]].assign(bjd_code=lambda d: d["bjd_code"].astype(str)),
                            on=["bjd_code", "date", "hour"], how="left")
    else:
        panel["q50"], panel["q90"] = np.nan, np.nan
    has_v1 = bool(panel["q50"].notna().any())    # 기존 AI 가 없으면 그 열만 빼고 비교
    preds = {"persistence": "lag1_kw", "baseline": "baseline_kw", **({"ai_v1": "q50"} if has_v1 else {}), "ai_plus": "plus_q50"}
    cover = {**({"ai_v1": "q90"} if has_v1 else {}), "ai_plus": "plus_q90", "ai_plus_conformal": "plus_q90c"}
    shown = panel if visible is None else panel[panel["bjd_code"].isin(visible)]
    hourly_rows = []
    for period, ds in (("검증", valid_days), ("평가", eval_days)):
        for scope, frame in (("전체", panel), ("PNG", shown)):
            f = frame[frame["date"].isin(ds)]
            hourly_rows.append({"기간": period, "범위": scope, **compare(f, "actual_kw", preds, cover)})
    hourly_tbl = pd.DataFrame(hourly_rows)

    # 2) 일피크
    tr, va = panel["date"].isin(train_days), panel["date"].isin(valid_days)
    peaks = peak_frame(panel)
    ptr, pva = peaks["date"].isin(train_days), peaks["date"].isin(valid_days)
    sp = _scale(peaks, ptr, "actual_peak", "level_peak_kw")
    peaks = _normalize(peaks, ["yday_peak", "wk_peak", "base_peak", "lag7_peak", "yday_total"], sp)
    peaks["ref_kw"] = peaks["yday_peak_kw"].fillna(peaks["wk_peak_kw"]).fillna(peaks["base_peak_kw"])
    peaks["ref_n"] = peaks["ref_kw"] / sp
    peaks["group_yday_peak_n"] = peaks.groupby(["group", "date"])["yday_peak_n"].transform("mean")
    if "temp_max" not in peaks:
        peaks["temp_max"] = np.nan
    peaks["actual_peak_z"] = (peaks["actual_peak"] - peaks["ref_kw"]) / sp
    peaks, _, peak_off = fit_residual(peaks, PEAK_FEATURES, "actual_peak_z", "ref_kw", sp, ptr, pva, fp, prefix="peak_")
    ppreds = {"persistence": "yday_peak_kw", "baseline": "base_peak_kw", **({"ai_v1": "v1_q50_peak"} if has_v1 else {}),
              "ai_plus": "peak_q50"}
    pcover = {**({"ai_v1": "v1_q90_peak"} if has_v1 else {}), "ai_plus": "peak_q90", "ai_plus_conformal": "peak_q90c"}
    pshown = peaks if visible is None else peaks[peaks["bjd_code"].isin(visible)]
    peak_tbl = pd.DataFrame([{"기간": period, "범위": scope, **compare(f[f["date"].isin(ds)], "actual_peak", ppreds, pcover)}
                             for period, ds in (("검증", valid_days), ("평가", eval_days))
                             for scope, f in (("전체", peaks), ("PNG", pshown))])
    alerts = None
    if calib_peak is not None:
        ev = peaks[peaks["date"].isin(eval_days)]
        methods = {"어제 피크": "yday_peak_kw", "기준 모델": "base_peak_kw",
                   **({"AI(기존, 시각별 P90)": "v1_q90_peak"} if has_v1 else {}), "AI+(일피크 P90·보정)": "peak_q90c"}
        alerts = {scope: alert_table(f, calib_peak, params["multipliers"], methods,
                                     float(params.get("min_calib_peak_kw", 1.0)))
                  for scope, f in (("전체", ev), ("PNG", ev if visible is None else ev[ev["bjd_code"].isin(visible)]))}

    # 6) 구·서울 합계
    agg = aggregate_matrices(hourly)
    gpanel, _, _, _ = _hourly_plus(agg, days, params, temp, lambda c: "11", fp,
                                   living_features(living, lambda c: str(c)[:5]) if living is not None else None)
    agg_rows = []
    for period, ds in (("검증", valid_days), ("평가", eval_days)):
        g = gpanel[gpanel["date"].isin(ds)]
        agg_rows.append({"단위": "구", "기간": period, **compare(g, "actual_kw", {"persistence": "lag1_kw", "baseline": "baseline_kw",
                                                                                 "ai_plus": "plus_q50"})})
        seoul = g.dropna(subset=["actual_kw", "lag1_kw", "baseline_kw", "plus_q50"]).groupby(["date", "hour"])[
            ["actual_kw", "lag1_kw", "baseline_kw", "plus_q50"]].sum().reset_index().assign(bjd_code="11")
        agg_rows.append({"단위": "서울 합계", "기간": period, **compare(seoul, "actual_kw", {"persistence": "lag1_kw",
                                                                                      "baseline": "baseline_kw",
                                                                                      "ai_plus": "plus_q50"})})
    agg_tbl = pd.DataFrame(agg_rows)

    # 7) 순열 중요도(검증기간 표본)
    from sklearn.inspection import permutation_importance

    vs = panel[va & panel["actual_kw_z"].notna()]
    vs = vs.sample(min(len(vs), int(fp["importance_rows"])), random_state=0)
    model, used = model
    imp = permutation_importance(model, vs[used], vs["actual_kw_z"], n_repeats=3, random_state=0,
                                 scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"특성": [LABELS[f] for f in used], "중요도": imp.importances_mean}).sort_values(
        "중요도", ascending=False)
    importance["비중"] = importance["중요도"].clip(lower=0) / importance["중요도"].clip(lower=0).sum()

    # 8) 충전 생활 유형
    try:
        types, type_tbl, centers = charge_types(panel, tr, int(fp["n_types"]))
    except ValueError as exc:
        log.warning("8-A+ 충전 유형 생략: %s", exc)
        types, type_tbl, centers = pd.Series(dtype=int), pd.DataFrame(), pd.DataFrame()

    ev_h = hourly_tbl[(hourly_tbl["기간"] == "평가") & (hourly_tbl["범위"] == "전체")].iloc[0]
    ev_p = peak_tbl[(peak_tbl["기간"] == "평가") & (peak_tbl["범위"] == "전체")].iloc[0]
    ev_s = agg_tbl[(agg_tbl["단위"] == "서울 합계") & (agg_tbl["기간"] == "평가")].iloc[0]
    note = (f"평가기간 어제값 대비 오차 감소: 시간별 AI+ {ev_h.get('skill_ai_plus', np.nan):+.1%}(기존 AI {ev_h.get('skill_ai_v1', np.nan):+.1%}) · "
            f"일피크 AI+ {ev_p.get('skill_ai_plus', np.nan):+.1%} · 서울 합계 AI+ {ev_s.get('skill_ai_plus', np.nan):+.1%} · "
            f"P90 적중률 보정 후 {ev_h.get('p90cov_ai_plus_conformal', np.nan):.1%}")
    log.info("8-A+ %s", note)
    return {"hourly": hourly_tbl, "peak": peak_tbl, "alerts": alerts, "agg": agg_tbl, "importance": importance,
            "types": types, "type_table": type_tbl, "type_centers": centers, "note": note,
            "conformal_offset": {"hourly": off, "peak": peak_off}}

# ---------------------------------------------------------------- 사전 등록 대표 지표(결과를 보기 전에 고정, 2026-10-07)

PREREG_KEEP = 0.05   # 대표 지표가 이 값 이상이면 "AI가 어제값보다 낫다"를 본문 주장으로 쓴다


def preregistered(res, multiplier=1.2):
    """발표 숫자 B 를 결과를 보기 전에 정해 둔 지표·판정 기준으로만 낸다(여러 비교표 중 잘 나온 것을 고르지 않기 위해).

    대표: 평가기간 일피크 예측 오차(MAE)의 어제값 대비 감소율, 소표본 제외 법정동(PNG 범위).
    판정: ≥ +5% → 본문 주장 유지 / 0~5% → 보조 근거로만 / < 0 → AI 주장 철회(제목을 '어디부터 볼까'로).
    보조: 서울 합계 시간별 감소율, 일피크 P90 보정 후 적중률(목표 90%±5%p), 상한 1.2배 경보 재현율·정밀도."""
    def row(tbl, **where):
        t = tbl
        for k, v in where.items():
            t = t[t[k] == v]
        return t.iloc[0] if len(t) else pd.Series(dtype=float)

    peak = row(res["peak"], 기간="평가", 범위="PNG")
    seoul = row(res["agg"], 단위="서울 합계", 기간="평가")
    skill = float(peak.get("skill_ai_plus", np.nan))
    verdict = ("판정 불가(값 없음)" if not np.isfinite(skill) else "본문 주장 유지" if skill >= PREREG_KEEP
               else "보조 근거로만" if skill >= 0 else "AI 주장 철회 — 제목 변경")
    cov = float(peak.get("p90cov_ai_plus_conformal", np.nan))
    rows = [{"구분": "대표(숫자 B)", "지표": "평가기간 일피크 오차의 어제값 대비 감소율(AI+, 소표본 제외)", "값": skill,
             "비교": float(peak.get("skill_ai_v1", np.nan)), "판정 기준": "≥ +5% 유지 · 0~5% 보조 · < 0 철회", "판정": verdict},
            {"구분": "보조", "지표": "평가기간 서울 합계 시간별 오차의 어제값 대비 감소율(AI+)", "값": float(seoul.get("skill_ai_plus", np.nan)),
             "비교": np.nan, "판정 기준": "보고만", "판정": "—"},
            {"구분": "보조", "지표": "평가기간 일피크 P90 보정 후 적중률(AI+)", "값": cov, "비교": float(peak.get("p90cov_ai_v1", np.nan)),
             "판정 기준": "85~95%면 보정 성공", "판정": "—" if not np.isfinite(cov) else "보정 성공" if 0.85 <= cov <= 0.95 else "보정 실패"}]
    if res.get("alerts") is not None:
        a = res["alerts"]["PNG"]
        a = a[np.isclose(a["multiplier"].astype(float), multiplier)].set_index("방법")
        for name in ("AI+(일피크 P90·보정)", "기준 모델"):
            if name in a.index:
                rows.append({"구분": "보조", "지표": f"상한 {multiplier}배 경보 재현율 — {name}", "값": float(a.loc[name, "recall"]),
                             "비교": float(a.loc[name, "precision"]), "판정 기준": "보고만(비교 = 정밀도)", "판정": "—"})
    return pd.DataFrame(rows)
