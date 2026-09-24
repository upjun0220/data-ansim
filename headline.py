"""9단계 발표 3숫자 — V9 9절 "1지도 3숫자". 각 숫자 옆에 가정을 함께 적는다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_per_charger_multiple(access, share=0.10):
    """숫자 1 — 충전기 1기당 EV 수의 상위 10% 평균 ÷ 하위 10% 평균, 대상 수. 충전기 0기·EV 0대인 동네는 뺀다."""
    if "ev_per_charger" not in access:
        return np.nan, 0
    v = pd.to_numeric(access["ev_per_charger"], errors="coerce")
    v = v[np.isfinite(v) & (v > 0)].sort_values()
    if len(v) < 2:
        return np.nan, len(v)
    k = max(1, int(np.ceil(len(v) * share)))
    return float(v.tail(k).mean() / v.head(k).mean()), len(v)


def _energy_at(results, n_new, multiplier, codes):
    g = results[(results["mode"] == "forecast") & (results["new_chargers"] == n_new)
                & np.isclose(results["multiplier"], multiplier)]
    if codes is not None:
        g = g[g["bjd_code"].astype(str).isin(codes)]
    if g.empty:
        return None
    return {"before": g["baseline_exceedance_kwh"].mean(), "after": g["exceedance_kwh"].mean(),
            "cost": g["energy_cost_difference_won"].mean(), "n": g["bjd_code"].nunique()}


def headline_table(access, results, params, utilization_scale, codes=None):
    """codes 를 주면 그 법정동만 집계한다(PNG 는 소표본 법정동을 뺀 목록을 넘긴다)."""
    n_new = int(params["scenario_new_chargers"])
    base_m = float(params["base_multiplier"])
    rows = []
    if access is not None:
        multiple, n = ev_per_charger_multiple(access)
        rows.append({"번호": "숫자 1", "내용": "충전기 1기당 전기차 대수의 지역 간 배수(상위10%÷하위10% 평균)",
                     "값": multiple, "범위": "",
                     "가정": f"충전소를 법정동 중심점 최근접에 배정(폴리곤 공간조인 아님) · 대상 {n}곳(충전기·EV 0 제외)"})
        km = pd.to_numeric(access.get("nearest_charger_km"), errors="coerce").dropna()
        if len(km):
            rows.append({"번호": "보조", "내용": "최근접 공용충전기 거리 중앙값 / 90퍼센타일(km)", "값": float(km.median()),
                         "범위": f"p90 {km.quantile(0.9):.2f}", "가정": "법정동 중심점 기준 근사(격자점 아님)"})
    if results is not None:
        assume = f"가정: 증설 {n_new}대, 상한 배율 {base_m}, 이용률 배율 {utilization_scale}"
        at = _energy_at(results, n_new, base_m, codes)
        multipliers = sorted(float(m) for m in pd.to_numeric(results["multiplier"], errors="coerce").dropna().unique())
        ranged = {m: _energy_at(results, n_new, m, codes) for m in multipliers}
        ranged = {m: v for m, v in ranged.items() if v}
        if at:
            def span(key):
                vals = [v[key] for v in ranged.values() if np.isfinite(v[key])]
                return f"상한 {min(ranged)}~{max(ranged)}: {min(vals):.1f}~{max(vals):.1f}" if vals else ""
            n_regions = f" · {at['n']}곳 · 지역·일 평균"
            rows.append({"번호": "숫자 2", "내용": "ESS 미적용 평균 상한 초과 kWh", "값": at["before"],
                         "범위": span("before"), "가정": assume + n_regions})
            rows.append({"번호": "숫자 3", "내용": "ESS 적용 후 잔여 초과 kWh", "값": at["after"],
                         "범위": span("after"), "가정": assume + n_regions})
            rows.append({"번호": "숫자 3", "내용": "SMP 기준 비용 차이(원, ESS 적용 시)", "값": at["cost"],
                         "범위": span("cost"), "가정": assume + n_regions + " · SMP 없으면 빈 값"})
    if not rows:
        raise ValueError("8-C 접근성과 8-B ESS 결과가 모두 없음")
    return pd.DataFrame(rows)

# ================================================================ v11 발표 숫자(1지도 3숫자)

def headline_table_v11(access, risk=None, effect=None, scenario_region=None, ai=None, params=None, codes=None):
    """v11.3 s9_headline. 각 행에 가정(증설 대수·상한 배율·이용률 배율·기온 모드·사용 모델)을 찍는다.

    숫자 1: 충전기 1기당 전기차 배수(V10 그대로).
    숫자 2: 증설 0대 급증 경보 정밀도·재현율(AI 대 기준 모델) → AI 효과(ESS 초과 kWh 감소율) 평균·범위·대상 수.
            AI 가 실행되지 않았거나 킬 18번 기준을 못 넘으면 "미실행"/"AI 개선 없음"을 적는다.
    숫자 3: (a) 현재 증설 0대 급증위험 ≥ risk_threshold 인 동네 수(상한 base_multiplier),
            (b) 2030 중 시나리오에서 현재 상한을 넘는 동네 수(괄호에 고 시나리오). 둘 다 충전기 추가 없음 기준.
    codes 를 주면 그 법정동만 집계한다(PNG 는 소표본 법정동을 뺀 목록을 넘긴다).
    """
    from loadforecast import ai_improvement

    p = params
    base_m, n_new = float(p["base_multiplier"]), int(p["scenario_new_chargers"])
    model = ai["model_used"] if ai else "미실행"
    wmode = ai["weather_mode"] if ai else "-"
    common = f"기온 {wmode} · 모델 {model}"
    keep = (lambda df: df if codes is None or df is None else df[df["bjd_code"].astype(str).isin(codes)])
    rows = []
    if access is not None:
        a = keep(access)
        multiple, n = ev_per_charger_multiple(a)
        rows.append({"번호": "숫자 1", "내용": "충전기 1기당 전기차 대수의 지역 간 배수(상위10%÷하위10% 평균)", "값": multiple,
                     "비교": np.nan, "범위": "", "가정": f"법정동 중심점 최근접 배정(폴리곤 공간조인 아님) · 대상 {n}곳"})
    ok, why = ai_improvement(ai["metrics"] if ai else None, float(p.get("min_p90_coverage", 0.8)))
    status = "" if ok else (" · 미실행" if not ai or model.startswith("baseline") else " · AI 개선 없음(킬 18)")
    if risk is not None:
        r = keep(risk)
        r = r[np.isclose(r["multiplier"].astype(float), base_m) & (r["status"] == "assessed")]
        for label, key in (("정밀도", "precision"), ("재현율", "recall")):
            tp = r["tp_ai"].sum()
            denom_ai = r["alerts_ai" if key == "precision" else "actual_ai"].sum()
            denom_b = r["alerts_base" if key == "precision" else "actual_base"].sum()
            rows.append({"번호": "숫자 2", "내용": f"증설 0대 급증 경보 {label} — AI(값) 대 기준 모델(비교)",
                         "값": tp / denom_ai if denom_ai else np.nan,
                         "비교": r["tp_base"].sum() / denom_b if denom_b else np.nan, "범위": "",
                         "가정": f"증설 0대 · 상한 {base_m}배 · 공통일 법정동·일 합산 · {common} · {len(r)}곳{status}"})
    if effect is not None and len(effect):
        e = keep(effect)
        e = e[e["new_chargers"] == n_new]
        at = e[np.isclose(e["multiplier"].astype(float), base_m)]
        spans = [g["ai_effect"].mean() for _, g in e.groupby("multiplier") if g["ai_effect"].notna().any()]
        rows.append({"번호": "숫자 2", "내용": "AI 효과 — ESS 초과 kWh 감소율(1 − P90/기준 모델), 법정동 평균",
                     "값": at["ai_effect"].mean(), "비교": np.nan,
                     "범위": f"상한 1.1~1.3: {min(spans):.2f}~{max(spans):.2f}" if spans else "",
                     "가정": f"증설 {n_new}대 · 상한 {base_m}배 · 이용률 배율 {p['utilization_scale']} · {common} · "
                             f"대상 {int(at['ai_effect_eligible'].sum())}/{len(at)}곳(기준 모델 초과 0 제외){status}"})
    elif ai is not None:
        rows.append({"번호": "숫자 2", "내용": "AI 효과 — ESS 초과 kWh 감소율", "값": np.nan, "비교": np.nan, "범위": "",
                     "가정": f"8-B 비교 운전 없음 · {common}{status}"})
    if risk is not None:
        r = keep(risk)
        r = r[np.isclose(r["multiplier"].astype(float), base_m) & (r["status"] == "assessed")]
        th = float(p["risk_threshold"])
        rows.append({"번호": "숫자 3(a)", "내용": f"현재 급증위험 ≥ {th:g}인 동네 수(평가일 중 경보일 비율)",
                     "값": int((r["surge_risk"] >= th).sum()), "비교": np.nan, "범위": f"평가 {len(r)}곳 중",
                     "가정": f"충전기 추가 없음(증설 0대) · 상한 {base_m}배 · 평가기간 예측·관측 기준 · {common}"})
    if scenario_region is not None and len(scenario_region):
        s = keep(scenario_region)
        s = s[(s["year"] == 2030) & (s["beta_case"] == "main")]
        col = f"over_limit_{base_m}"
        mid, high = s[s["scenario"] == "중"], s[s["scenario"] == "고"]
        rows.append({"번호": "숫자 3(b)", "내용": "2030 중 시나리오(추세 연장)에서 현재 상한을 넘는 동네 수 (괄호: 고 시나리오)",
                     "값": int(mid[col].sum()) if len(mid) else np.nan, "비교": int(high[col].sum()) if len(high) else np.nan,
                     "범위": f"({int(high[col].sum())})" if len(high) else "(고 시나리오 생략)",
                     "가정": f"시나리오(예측 아님) · 충전기 추가 설치 없음 · 상한 {base_m}배 · β={s['beta'].iloc[0]:.2f} · "
                             f"{len(mid)}곳"})
    if not rows:
        raise ValueError("발표 숫자 재료(8-C·8-A·8-B·8-F) 없음")
    out = pd.DataFrame(rows)
    out.attrs["ai_check"] = why
    return out


# ================================================================ 발표 본문용: 발견 문장·새 3숫자·동네 카드

def _pick(head, contains):
    rows = head[head["내용"].astype(str).str.contains(contains, regex=False)]
    return (float(rows["값"].iloc[0]), float(rows["비교"].iloc[0])) if len(rows) else (np.nan, np.nan)


def story_table(head, sim=None, can_check=None):
    """s9_story — 발표 본문의 '발견 한 문장'과 새 3숫자(격차·예측·개입). ESS·2030 은 s9_headline(부록)에 남긴다.

    head: headline_table_v11 결과. sim: 8-G 충전기 추가 실험 표(첫 행 = 우선순위 전략, 둘째 = 비교 전략).
    can_check: 8-C CAN 원정 충전 검증 표(attrs low·high·rho) — 있으면 '고유 데이터 발견' 행을 더한다.
    문구는 초안이며 팀이 확정한다. 값이 없으면 '미산출'로 적고 지어내지 않는다.
    """
    gap = head.loc[head["번호"] == "숫자 1", "값"]
    gap = float(gap.iloc[0]) if len(gap) else np.nan
    p_ai, p_b = _pick(head, "정밀도")
    r_ai, r_b = _pick(head, "재현율")
    rows = []
    if np.isfinite(r_ai) and np.isfinite(r_b):
        if r_ai > r_b:
            sentence = (f"전날 AI 경보가 실제 급증일의 {r_ai:.0%}를 미리 잡아 기준 모델({r_b:.0%})보다 "
                        f"{(r_ai - r_b) * 100:.0f}%p 높았다")
        else:
            sentence = f"AI 경보 재현율 {r_ai:.0%}로 기준 모델({r_b:.0%})을 넘지 못했다 — 그대로 보고"
    else:
        sentence = "AI 경보 재현율 미산출 — 평가기간에 실제 급증일이 없거나 8-A 미실행(s8a_risk 확인)"
    col = [c for c in (sim.columns if sim is not None else []) if c.endswith("개선율") and c.startswith("하위")]
    gain = gain_b = np.nan
    if col and len(sim) >= 1:
        gain = float(sim[col[0]].iloc[0])
        gain_b = float(sim[col[0]].iloc[1]) if len(sim) >= 2 else np.nan
        if np.isfinite(gain):
            sentence += (f". 이 순위대로 충전기를 더하면 접근성 하위 동네가 {gain:+.0%} 개선된다"
                         + (f"(전기차 많은 순 {gain_b:+.0%})" if np.isfinite(gain_b) else ""))
        elif "접근성 0 동네(후)" in sim:   # 하위 동네가 모두 0 이면 개선율을 못 구한다 → 0 동네 수로 말한다
            z0, z1 = int(sim["접근성 0 동네(전)"].iloc[0]), int(sim["접근성 0 동네(후)"].iloc[0])
            sentence += (f". 이 순위대로 충전기를 더하면 접근성 0 동네가 {z0}→{z1}곳으로 준다"
                         + (f"(전기차 많은 순 {int(sim['접근성 0 동네(후)'].iloc[1])}곳)" if len(sim) >= 2 else ""))
    rows.append({"구분": "발견 문장(초안)", "내용": sentence, "값": np.nan, "비교": np.nan})
    rows.append({"구분": "숫자 A 격차", "내용": "충전기 1기당 전기차 수 — 상위 10% 동네가 하위 10%의 몇 배",
                 "값": gap, "비교": np.nan})
    rows.append({"구분": "숫자 B 예측", "내용": "전날 AI 급증 경보 재현율(값) 대 기준 모델(비교)", "값": r_ai, "비교": r_b})
    rows.append({"구분": "숫자 B 예측", "내용": "전날 AI 급증 경보 정밀도(값) 대 기준 모델(비교)", "값": p_ai, "비교": p_b})
    if sim is not None and len(sim):
        rows.append({"구분": "숫자 C 개입", "내용": f"{sim['전략'].iloc[0]}에 충전기 추가 시 {col[0] if col else '개선율'}"
                                                 f"(값) 대 비교 전략(비교)", "값": gain, "비교": gain_b})
    else:
        rows.append({"구분": "숫자 C 개입", "내용": "충전기 추가 가정 실험 미산출(8-G)", "값": np.nan, "비교": np.nan})
    if can_check is not None and "low" in can_check.attrs:
        k = can_check.attrs
        rho = f"순위 상관 {k['rho']:.2f}" if np.isfinite(k["rho"]) else "순위 상관 계산 불가"
        rows.append({"구분": "고유 데이터 발견(CAN)",
                     "내용": f"접근성 하위 동네에 사는 차량의 원정 충전(거주지 밖) 비율(값) 대 나머지 동네(비교) · "
                            f"{rho} · {k['n']}개 동네",
                     "값": k["low"], "비교": k["high"]})
    else:
        rows.append({"구분": "고유 데이터 발견(CAN)", "내용": "원정 충전 검증 미산출 — CAN 개별 차량 판정·거주 차량 수 필요(8-C)",
                     "값": np.nan, "비교": np.nan})
    return pd.DataFrame(rows)


def recommend(risk_norm, equity_norm, cut=0.5):
    """규칙 기반 권고 초안. 실제 설치·점검 결정이 아니다."""
    high_r, high_e = bool(risk_norm >= cut), bool(equity_norm >= cut)
    if high_r and high_e:
        return "부하 현장 점검 + 공용 충전기 추가 검토"
    if high_r:
        return "배전 부하 현장 점검(한전 협의)"
    if high_e:
        return "공용 충전기 추가 검토"
    return "관찰 유지"


def region_cards(priority, risk, access, master, params, codes=None):
    """s9_region_cards — 우선순위 상위 card_top_n 곳의 동네 카드. codes 를 주면 그 법정동만(PNG 소표본 제외)."""
    base_m = float(params["base_multiplier"])
    p = priority[priority["rank_eligible"].astype(bool)].copy()
    p["bjd_code"] = p["bjd_code"].astype(str)
    p["접근성 순위(낮은 순)"] = p["access_2sfca"].rank(method="min").astype("Int64")
    if codes is not None:
        p = p[p["bjd_code"].isin(set(map(str, codes)))]
    p = p.sort_values("rank").head(int(params["card_top_n"]))
    out = p[["rank", "bjd_code"]].rename(columns={"rank": "순위"}).astype({"순위": "Int64"})
    names = master.drop_duplicates("bjd_code").set_index("bjd_code")["region_name"] if master is not None else {}
    out["법정동"] = p["bjd_code"].map(names)
    out["결합점수"] = p["PriorityScore"].to_numpy()
    out["급증위험"] = p["risk_raw"].to_numpy()
    if risk is not None:
        r = risk[np.isclose(risk["multiplier"].astype(float), base_m)].copy()
        r["bjd_code"] = r["bjd_code"].astype(str)
        r = r.set_index("bjd_code")
        hit = p["bjd_code"].map(lambda c: f"{int(r.at[c, 'tp_ai'])}/{int(r.at[c, 'actual_ai'])}일"
                                if c in r.index and r.at[c, "actual_ai"] > 0 else "실제 급증 없음")
        out["AI 경보 적중(급증일)"] = hit.to_numpy()
    out["접근성 순위(낮은 순)"] = [f"{v}/{len(priority[priority['rank_eligible'].astype(bool)])}"
                               for v in p["접근성 순위(낮은 순)"]]
    if access is not None:
        a = access.assign(bjd_code=access["bjd_code"].astype(str)).set_index("bjd_code")
        per, n_ch = p["bjd_code"].map(a["ev_per_charger"]), p["bjd_code"].map(a["chargers_assigned"])
        out["충전기 1기당 EV"] = [f"{v:.1f}" if pd.notna(v) else "충전기 없음" if n == 0 else "—"
                              for v, n in zip(per, n_ch)]
    out["강건 상위"] = p["robust_top"].to_numpy() if "robust_top" in p else np.nan
    out["권고(초안)"] = [recommend(rn, en) for rn, en in zip(p["risk_norm"], p["equity_norm"])]
    return out.reset_index(drop=True)
