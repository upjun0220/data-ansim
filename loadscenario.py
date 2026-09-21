"""8-F(확장안) — 2028·2030 충전 부하 시나리오. 시나리오(예측 아님), 충전기 추가 설치 없음 가정.

1) 보급: 행정동별 전기차 등록 월별 이력(OA-21236 형식) → 행정동→법정동 대응표(가중치)로 법정동에 배분.
   g(r) = 기준월 이전 36개월 연평균 증가율. EV_Y(r) = EV_now(r) × (1 + k·g(r))^(Y − Y0), Y − Y0 는 기준월 말부터
   Y년 말까지 월 수 ÷ 12(기준월 2026-06 → 2030년 말 4.5년). 저 k=0.5, 중 k=1.0, 고 k=k_goal —
   서울 합계 증가 배율이 전국 목표 배율 m_goal = goal_national / national_base 와 같아지는 k(이분법). "서울의 전국 비중
   유지" 가정과 같으며 OA-21236 서울 합계와 국토부 서울 값의 정의 차이에 영향받지 않는다.
2) 탄력성 β: 동네 간 log(대표 부하) ~ log(전기차 대수) OLS, 부트스트랩 95% 구간. 구간이 0을 포함하거나 β가
   beta_bounds 밖이면 β=1 로 두고 0.8·1.2 민감도만 낸다.
3) L_Y(r,h) = L_typ(r,h) × (EV_Y/EV_now)^β. 8-A 트리 모델로 외삽하지 않는다(학습 범위 밖에서 평평해짐).
4) 판정: 현재 상한 P_target(r) = 교정기간 최대 × 배율(8-A·8-B와 같은 policy_target) 초과 동네 수, 8-B 후보 격자로 본
   필요 ESS 용량 변화, 지표 1(충전기 1기당 전기차, 충전기 수는 현재 유지)의 미래값.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bjdmapping import canonicalize
from common import clean_code, log, read_columns, to_num

SCENARIO_TITLE = "시나리오(예측 아님), 충전기 추가 설치 없음 가정"


def _ym(s):
    digits = s.astype(str).str.replace(r"\D", "", regex=True).str[:6]
    return digits.str[:4] + "-" + digits.str[4:6]


def load_ev_history(path, columns, fuel_value="전기", prefix=None):
    """→ (hdong_code, ym 'YYYY-MM', ev). 연료 열이 있으면 fuel_value 만, 코드 앞 2자리가 prefix 인 행정동만."""
    df = read_columns(path, columns, "전기차등록이력", required=["ym", "code", "count"])
    if "fuel" in df and df["fuel"].notna().any():
        df = df[df["fuel"].astype(str).str.strip() == fuel_value]
    out = pd.DataFrame({"hdong_code": clean_code(df["code"], 10), "ym": _ym(df["ym"]), "ev": to_num(df["count"])}).dropna()
    if prefix:
        out = out[out["hdong_code"].str.startswith(str(prefix))]
    out = out.groupby(["hdong_code", "ym"], as_index=False)["ev"].sum()
    log.info("전기차 등록 이력: 행정동 %d · %s~%s", out["hdong_code"].nunique(), out["ym"].min(), out["ym"].max())
    return out


def load_hdong_bjd(path, columns):
    """행정동→법정동 대응표. 가중치(그 행정동 전기차 중 법정동 몫)가 없으면 같은 몫으로 나누고 경고한다."""
    df = read_columns(path, columns, "행정동법정동대응", required=["hdong", "bjd"])
    out = pd.DataFrame({"hdong_code": clean_code(df["hdong"], 10), "bjd_code": clean_code(df["bjd"], 10)})
    if "weight" in df and df["weight"].notna().any():
        out["weight"] = to_num(df["weight"])
    else:
        log.warning("행정동→법정동 대응표에 가중치 없음 — 행정동 안 법정동에 같은 몫으로 배분(가정)")
        out["weight"] = 1.0
    out = out.dropna()
    out["weight"] = out["weight"] / out.groupby("hdong_code")["weight"].transform("sum")
    return out


def map_to_bjd(history, mapping, crosswalk=None, base_month=None):
    """행정동 이력 → 법정동 이력. 매핑 실패율 = 기준월(없으면 마지막 달) 전기차 중 대응표에 없는 행정동 몫."""
    month = base_month or history["ym"].max()
    at = history[history["ym"] == month]
    fail = float(at.loc[~at["hdong_code"].isin(mapping["hdong_code"]), "ev"].sum() / at["ev"].sum()) \
        if at["ev"].sum() > 0 else np.nan
    m = history.merge(mapping, on="hdong_code")
    m["bjd_code"] = canonicalize(m["bjd_code"], crosswalk)
    out = m.assign(ev=m["ev"] * m["weight"]).groupby(["bjd_code", "ym"], as_index=False)["ev"].sum()
    return out.rename(columns={"ev": "ev_count"}), fail


def years_ahead(base_month, year):
    """기준월 말 → year 년 말까지 연수(월 단위). 2026-06 → 2030: 4.5."""
    y, m = (int(v) for v in str(base_month).split("-"))
    return (int(year) * 12 + 12 - (y * 12 + m)) / 12


def growth_rate(history, base_month, months=36):
    """법정동별 연평균 증가율. 36개월 전 값이 없거나 0이면 서울 합계 증가율로 대신하고 표시한다."""
    y, m = (int(v) for v in base_month.split("-"))
    idx = y * 12 + m - 1 - int(months)
    past = f"{idx // 12:04d}-{idx % 12 + 1:02d}"
    now = history[history["ym"] == base_month].set_index("bjd_code")["ev_count"]
    then = history[history["ym"] == past].set_index("bjd_code")["ev_count"].reindex(now.index)
    if now.empty:
        raise ValueError(f"기준월 {base_month} 전기차 등록 자료 없음")
    total = (now.sum() / then.sum()) ** (12 / months) - 1 if then.sum() > 0 else np.nan
    g = (now / then.where(then > 0)) ** (12 / months) - 1
    fallback = g.isna()
    g = g.fillna(total)
    if fallback.any():
        log.warning("증가율: %d곳은 %s 값이 없어 서울 합계 증가율 %.3f 사용", int(fallback.sum()), past, total)
    return pd.DataFrame({"ev_now": now, "g": g, "g_fallback": fallback})


def project_ev(ev_now, g, k, t):
    return ev_now * np.power(np.maximum(1 + k * g, 0.0), t)


def solve_k_goal(ev_now, g, t, m_goal, tol=1e-9, k_max=1000.0):
    """Σ EV_2030 / Σ EV_now = m_goal 인 k(이분법). 증가율이 모두 0 이하이거나 해가 없으면 None."""
    if not np.isfinite(m_goal) or m_goal <= 1 or not (np.asarray(g) > 0).any():
        return None
    total = float(np.sum(ev_now))
    f = lambda k: float(np.sum(project_ev(ev_now, g, k, t))) / total - m_goal   # noqa: E731
    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2
        if hi > k_max:
            return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def estimate_beta(rep_load, ev_now, bounds=(0.3, 2.0), n_boot=999, seed=42):
    """log(대표 부하) ~ log(전기차 대수) OLS 기울기와 부트스트랩 95% 구간. 불안정하면 β=1(민감도 0.8·1.2)."""
    df = pd.DataFrame({"y": rep_load, "x": ev_now}).dropna()
    df = df[(df["y"] > 0) & (df["x"] > 0)]
    x, y = np.log(df["x"].to_numpy()), np.log(df["y"].to_numpy())
    out = {"n": len(df), "beta_ols": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    if len(df) >= 5 and np.ptp(x) > 0:
        out["beta_ols"] = float(np.polyfit(x, y, 1)[0])
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(int(n_boot)):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) > 0:
                boots.append(np.polyfit(x[i], y[i], 1)[0])
        out["ci_lo"], out["ci_hi"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    b = out["beta_ols"]
    stable = np.isfinite(b) and out["ci_lo"] > 0 and bounds[0] <= b <= bounds[1]
    reason = "" if stable else ("표본 부족(5곳 미만)" if len(df) < 5 else "구간이 0 포함" if not out["ci_lo"] > 0
                                else f"β 가 설정 범위 {list(bounds)} 밖")
    out.update(beta=b if stable else 1.0, fallback=not stable, reason=reason,
               sensitivity=[] if stable else [0.8, 1.2])
    log.info("8-F β=%.3f (OLS %.3f, 95%% [%.3f, %.3f], n=%d)%s", out["beta"], b, out["ci_lo"], out["ci_hi"], len(df),
             f" — 불안정: {reason} → β=1, 0.8·1.2 민감도" if not stable else "")
    return out


def typical_curves(eval_hourly, evening=range(17, 23), quantile=0.9):
    """법정동별 대표 곡선 L_typ: 평가기간 평일 중 저녁 피크가 P90(최근접 순위)인 날의 24시간 곡선."""
    d = eval_hourly.assign(date=pd.to_datetime(eval_hourly["date"]).dt.normalize())
    d = d[d["date"].dt.dayofweek < 5]
    wide = d.pivot_table(index=["bjd_code", "date"], columns="hour", values="kw").reindex(columns=range(24)).dropna()
    evening_peak = wide[list(evening)].max(axis=1)
    rows = {}
    for code, s in evening_peak.groupby(level="bjd_code"):
        s = s.sort_values()
        pick = s.index[min(len(s) - 1, int(np.ceil(quantile * len(s))) - 1)]
        rows[code] = wide.loc[pick].to_numpy(float)
    return rows


def run_scenarios_8f(ev_hist, curves, calib_peak, access, params, energy_params, base_multiplier):
    """→ (s8f_scenario, s8f_scenario_region, s8f_beta, notes)."""
    from essoptimizer import policy_target, size_ess

    sp = params
    base = growth_rate(ev_hist, sp["base_month"], int(sp["growth_months"]))
    codes = sorted(set(base.index) & set(curves) & set(calib_peak.index))
    if len(codes) < 2:
        raise ValueError(f"전기차 이력·대표 부하·교정기간 최대가 모두 있는 법정동 {len(codes)}곳 — 시나리오 불가")
    base = base.loc[codes]
    notes = []
    seoul_now = float(ev_hist.loc[ev_hist["ym"] == sp["base_month"], "ev_count"].sum())
    if sp.get("seoul_base") and abs(seoul_now / float(sp["seoul_base"]) - 1) > 0.05:
        msg = (f"기준월 전기차 합계 {seoul_now:,.0f}대가 scenario.seoul_base {float(sp['seoul_base']):,.0f}대와 "
               f"{abs(seoul_now / float(sp['seoul_base']) - 1):.1%} 다름 — 정의 차이 가능성(계산은 계속, seoul_base는 점검용)")
        log.warning("8-F %s", msg)
        notes.append(msg)
    rep = pd.Series({c: curves[c].max() for c in codes})
    beta = estimate_beta(rep, base["ev_now"], tuple(sp["beta_bounds"]), int(sp["n_boot"]), int(sp["seed"]))
    k = dict(sp["k"])
    m_goal = float(sp["goal_national"]) / float(sp["national_base"]) \
        if sp.get("goal_national") and sp.get("national_base") else np.nan
    t2030 = years_ahead(sp["base_month"], 2030)
    k_goal = solve_k_goal(base["ev_now"].to_numpy(), base["g"].to_numpy(), t2030, m_goal)
    if k_goal is None:
        msg = "고 시나리오 생략 — goal_national·national_base 없음" if not np.isfinite(m_goal) \
            else f"고 시나리오 생략 — m_goal {m_goal:.3f} 를 맞추는 k 없음(증가율 0 이하)"
        log.warning("8-F %s", msg)
        notes.append(msg)
    else:
        k["고"] = k_goal
    betas = [("main", beta["beta"])] + [(f"sens_{b}", b) for b in beta["sensitivity"]]
    chargers = access.set_index(access["bjd_code"].astype(str))["chargers_assigned"] if access is not None \
        and "chargers_assigned" in access else None
    ep = energy_params
    region_rows = []
    for year in sp["years"]:
        t = years_ahead(sp["base_month"], year)
        for scen, kv in k.items():
            ev_y = project_ev(base["ev_now"], base["g"], kv, t)
            for case, b in betas:
                ratio = (ev_y / base["ev_now"]).pow(b)
                for code in codes:
                    now_curve, peak = curves[code], float(calib_peak[code])
                    future = now_curve * float(ratio[code])
                    row = {"year": year, "scenario": scen, "k": kv, "beta_case": case, "beta": b, "bjd_code": code,
                           "ev_now": base.at[code, "ev_now"], "ev_Y": ev_y[code], "g": base.at[code, "g"],
                           "load_peak_now_kw": now_curve.max(), "load_peak_Y_kw": future.max()}
                    for m in ep["multipliers"]:
                        row[f"over_limit_{m}"] = bool(future.max() > policy_target(peak, m))
                    if case == "main":
                        target = policy_target(peak, base_multiplier)
                        for label, curve in (("now", now_curve), ("Y", future)):
                            size = size_ess([curve], target, ep["ess_params"], ep["candidates"], ep["solver"])
                            row[f"ess_kwh_{label}"] = size["params"]["energy_kwh"] if size["feasible"] else np.nan
                    if chargers is not None and chargers.get(code, 0) > 0:
                        row["ev_per_charger_Y"] = ev_y[code] / chargers[code]
                    region_rows.append(row)
    region = pd.DataFrame(region_rows)
    if (region["scenario"] == "고").any():
        region["over_only_high"] = False
        for (year, case), g in region.groupby(["year", "beta_case"]):
            mid = set(g.loc[(g["scenario"] == "중") & g[f"over_limit_{base_multiplier}"], "bjd_code"])
            sel = (region["year"] == year) & (region["beta_case"] == case) & (region["scenario"] == "고")
            region.loc[sel, "over_only_high"] = region.loc[sel, f"over_limit_{base_multiplier}"] & \
                ~region.loc[sel, "bjd_code"].isin(mid)
    scenario = summarize_scenarios(region, ep["multipliers"], base_multiplier)
    beta_table = pd.DataFrame([{**{k2: v for k2, v in beta.items() if k2 != "sensitivity"},
                                "sensitivity": ",".join(map(str, beta["sensitivity"])), "k_goal": k_goal,
                                "m_goal": m_goal, "years_ahead_2030": t2030, "seoul_ev_base_month": seoul_now}])
    return scenario, region, beta_table, notes


def summarize_scenarios(region, multipliers, base_multiplier):
    """법정동 표 → 연도 × 시나리오 × β경우 × 상한 배율 요약. PNG용은 소표본 법정동을 뺀 region 으로 다시 부른다."""
    from headline import ev_per_charger_multiple

    rows = []
    for (year, scen, case), g in region.groupby(["year", "scenario", "beta_case"], sort=False):
        mid = region[(region["year"] == year) & (region["scenario"] == "중") & (region["beta_case"] == case)]
        for m in multipliers:
            col = f"over_limit_{m}"
            only_high = int((g.set_index("bjd_code")[col] & ~mid.set_index("bjd_code")[col]).sum()) \
                if scen == "고" and len(mid) else 0
            row = {"year": year, "scenario": scen, "k": g["k"].iloc[0], "beta_case": case, "beta": g["beta"].iloc[0],
                   "multiplier": m, "n_regions": len(g), "ev_total_now": g["ev_now"].sum(), "ev_total_Y": g["ev_Y"].sum(),
                   "n_over_limit": int(g[col].sum()), "n_over_only_high": only_high}
            if case == "main" and np.isclose(m, base_multiplier):
                row.update(ess_candidate_median_kwh_now=g["ess_kwh_now"].median(),
                           ess_candidate_median_kwh_Y=g["ess_kwh_Y"].median(),
                           n_ess_infeasible_Y=int(g["ess_kwh_Y"].isna().sum()))
            if "ev_per_charger_Y" in g:
                row["ev_per_charger_multiple_Y"] = ev_per_charger_multiple(
                    g.rename(columns={"ev_per_charger_Y": "ev_per_charger"}))[0]
            rows.append(row)
    return pd.DataFrame(rows)
