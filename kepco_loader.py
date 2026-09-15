"""1·2단계 — KEPCO 충전 시계열 집계 + 변화점 탐지(활성화 시점 T_r).

1단계 산출: 지역(텍스트)×월×시각(0~23) 집계표 — kWh 합, 관측 일수, 고객호수 합.
  시간대구간(TIZO) 은 이 표에서 파생한다. 부하 집중도(8단계)는 시각 단위 그대로 쓴다.
2단계 산출: 법정동별 활성화 시점 T_r · 상태(treated / never_treated / 제외 사유).

KEPCO 원천 형태가 현장 확인 전까지 불명이라 세 가지를 모두 받는다(params.kepco.layout).
  long  — 한 행 = 한 시각. 시각은 '조회기간'(YYYYMMDDHH…) 또는 columns.kepco.hour 열에서 읽는다.
  wide  — 한 행 = 하루, '시간대별 사용량' 접두 열이 시각별로 여러 개(열 이름 끝 숫자가 시각).
  auto  — 헤더에 '시간대별 사용량' 접두 열이 2개 이상이면 wide.
조회기간이 월(YYYYMM)뿐이면 관측 일수를 그 달의 일수로 잡는다(평균 kW 계산이 틀어지지 않게).

⚠ KEPCO_001 과 002 는 포함/배타 관계가 불명이다. 이 모듈은 둘을 절대 합산하지 않는다.
단독 실행하지 않는다.
"""
from __future__ import annotations

import calendar
import re

import numpy as np
import pandas as pd

from bjd_mapping import canonicalize, match_regions, sido_short
from common import log, mi_to_ym, norm_text, optional_import, read_columns, read_header, to_num, ym_to_mi

KEYS = ["sido", "sigungu", "emd"]


# ---------------------------------------------------------------- 1단계: 로드·집계

def _layout(header, cols, layout):
    prefix = cols["kwh"]
    matches = [c for c in header if str(c).strip().startswith(prefix)]
    if layout == "auto":
        return "wide" if len(matches) >= 2 else "long"
    return layout


def _iter_long(path, cols, params, source):
    mapping = {k: cols[k] for k in ("period", "sido", "sigungu", "emd", "customers", "kwh")}
    if cols.get("hour"):
        mapping["hour"] = cols["hour"]
    for chunk in read_columns(path, mapping, source, chunksize=params["chunksize"]):
        yield chunk


def _iter_wide(path, cols, params, source, header, enc, sep):
    prefix = cols["kwh"]
    hour_re = re.compile(params["wide_hour_regex"])
    hour_cols = {}
    for c in header:
        name = str(c).strip()
        if name.startswith(prefix):
            m = hour_re.search(name[len(prefix):])
            if m:
                hour_cols[c] = int(m.group(1))
    if len(hour_cols) < 2:
        raise KeyError(f"[{source}] wide 형식인데 시간대 열을 못 찾음(접두 {prefix!r}, 정규식 {params['wide_hour_regex']!r})")
    base = {k: cols[k] for k in ("period", "sido", "sigungu", "emd", "customers")}
    from common import resolve_columns

    colmap = resolve_columns(header, base, source)
    rename = {orig: key for key, orig in colmap.items()}
    log.info("[%s] wide 형식: 시간대 열 %d개", source, len(hour_cols))
    reader = pd.read_csv(path, encoding=enc, sep=sep, dtype=str, chunksize=params["chunksize"],
                         usecols=list(rename) + list(hour_cols))
    for chunk in reader:
        chunk = chunk.rename(columns=rename)
        long = chunk.melt(id_vars=list(base), value_vars=list(hour_cols), var_name="_col", value_name="kwh")
        long["hour"] = long["_col"].map(hour_cols).astype(str)
        yield long.drop(columns="_col")


def _prepare_chunk(chunk, params, source):
    digits = chunk["period"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    year = pd.to_numeric(digits.str[:4], errors="coerce")
    month = pd.to_numeric(digits.str[4:6], errors="coerce")
    mi = year * 12 + month - 1
    if "hour" in chunk:
        hour = to_num(chunk["hour"].astype(str).str.replace(r"\D", "", regex=True))
    else:
        if not (digits.str.len() >= 10).any():
            raise ValueError(f"[{source}] 시각을 찾을 수 없음: 조회기간 예 {chunk['period'].head(3).tolist()} — "
                             "조회기간에 시각이 없으면 columns.kepco.hour 를 지정하거나 layout='wide' 로 둘 것")
        hour = pd.to_numeric(digits.str[8:10].where(digits.str.len() >= 10), errors="coerce")
    has_day = digits.str.len() >= 8
    dim = pd.Series([calendar.monthrange(int(y), int(m))[1] if pd.notna(y) and pd.notna(m) else np.nan
                     for y, m in zip(year, month)], index=chunk.index) if (~has_day).any() else 1.0
    days = pd.Series(1.0, index=chunk.index).where(has_day, dim)

    out = pd.DataFrame({
        "sido": sido_short(chunk["sido"]), "sigungu": norm_text(chunk["sigungu"]), "emd": norm_text(chunk["emd"]),
        "date8": digits.str[:8], "mi": mi, "hour": hour,
        "kwh": to_num(chunk["kwh"]), "cust": to_num(chunk["customers"]), "days": days,
    })
    if params.get("sido"):
        out = out[out["sido"].isin(set(sido_short(pd.Series(params["sido"]))))]
    if params.get("date_start"):
        out = out[out["date8"] >= str(params["date_start"]).replace("-", "")[:8]]
    if params.get("date_end"):
        out = out[out["date8"] <= str(params["date_end"]).replace("-", "")[:8]]
    bad = out["mi"].isna() | out["hour"].isna()
    if bad.any():
        log.warning("[%s] 조회기간/시각 해석 실패 %d행 제외", source, int(bad.sum()))
        out = out[~bad]
    out = out[out["kwh"].notna()]
    out["days"] = out["days"].astype(float)
    return out


def load_kepco_monthly_hour(path, columns, params, source="KEPCO_001", max_chunks=None):
    """KEPCO 원천 → 지역×월×시각 집계. 반환 열: sido sigungu emd mi hour kwh n_days cust_sum."""
    header, enc, sep = read_header(path)
    layout = _layout(header, columns, params["layout"])
    it = _iter_wide(path, columns, params, source, header, enc, sep) if layout == "wide" \
        else _iter_long(path, columns, params, source)
    parts = []
    n_rows = 0
    for i, chunk in enumerate(it):
        if max_chunks and i >= max_chunks:
            log.info("[%s] max_chunks=%d 에서 중단(표본 집계)", source, max_chunks)
            break
        prep = _prepare_chunk(chunk, params, source)
        n_rows += len(prep)
        parts.append(prep.groupby(KEYS + ["mi", "hour"], observed=True, sort=False)
                     .agg(kwh=("kwh", "sum"), n_days=("days", "sum"), cust_sum=("cust", "sum")).reset_index())
    if not parts:
        raise ValueError(f"[{source}] 읽은 행이 없음: {path}")
    agg = pd.concat(parts, ignore_index=True).groupby(KEYS + ["mi", "hour"], observed=True, sort=True)[
        ["kwh", "n_days", "cust_sum"]].sum().reset_index()
    agg["mi"] = agg["mi"].astype(int)
    agg["hour"] = agg["hour"].astype(int)
    if params.get("hour_base") == "auto" and agg["hour"].max() == 24 and agg["hour"].min() >= 1:
        log.info("[%s] 시각 표기 1~24 → 0~23 으로 당김(24시 = 23~24시 구간)", source)
        agg["hour"] -= 1
    bad_hour = ~agg["hour"].between(0, 23)
    if bad_hour.any():
        log.warning("[%s] 0~23 밖 시각 %d건 제외: %s", source, int(bad_hour.sum()), sorted(agg.loc[bad_hour, "hour"].unique())[:10])
        agg = agg[~bad_hour]
    log.info("[%s] %s행 → 지역 %d · 월 %s~%s · 집계 %d행", source, f"{n_rows:,}",
             agg[KEYS].drop_duplicates().shape[0], mi_to_ym(agg["mi"].min()), mi_to_ym(agg["mi"].max()), len(agg))
    return agg


def attach_bjd(agg, master, fail_warn_rate, crosswalk=None):
    """고유 지역만 매칭한 뒤 붙인다(수천만 행에 문자열 연산을 반복하지 않기 위해).

    crosswalk 가 있으면 기간 중 개편 전/후 명칭으로 매칭된 코드를 기준코드 하나로 묶는다.
    """
    regions = agg.groupby(KEYS, observed=True)["kwh"].sum().reset_index()
    matched, rate_table, unmatched, fail_rate = match_regions(regions, master, weight_col="kwh",
                                                              fail_warn_rate=fail_warn_rate)
    matched["bjd_code"] = canonicalize(matched["bjd_code"], crosswalk)
    out = agg.merge(matched[KEYS + ["bjd_code", "region_name"]], on=KEYS, how="left")
    out = out[out["bjd_code"].notna()]
    # 서로 다른 텍스트가 같은 법정동으로 모인 경우('가람제1동'·'가람1동', 개편 전/후 명칭) 합친다
    out = out.groupby(["bjd_code", "mi", "hour"], sort=True).agg(
        kwh=("kwh", "sum"), n_days=("n_days", "max"), cust_sum=("cust_sum", "sum"), region_name=("region_name", "first")
    ).reset_index()
    return out, rate_table, unmatched, fail_rate


def daily_series(mh):
    """법정동×월 일평균 충전량(kWh/일) = Σ_시각 (kWh / 관측일수). 월 길이·결측일 차이를 없앤다."""
    tmp = mh.assign(kwh_per_day=mh["kwh"] / mh["n_days"].where(mh["n_days"] > 0))
    return tmp.groupby(["bjd_code", "mi"])["kwh_per_day"].sum(min_count=1).reset_index()


def band_series(mh, bands):
    """시간대구간(TIZO) 별 월 일평균 충전량. bands: {"코드": [시각,...]}."""
    hour_to_band = {int(h): str(code) for code, hours in bands.items() for h in hours}
    tmp = mh.assign(band=mh["hour"].map(hour_to_band), kwh_per_day=mh["kwh"] / mh["n_days"].where(mh["n_days"] > 0))
    missing = tmp["band"].isna()
    if missing.any():
        log.warning("시간대구간 매핑에 없는 시각 %s — 구간 집계에서 제외", sorted(tmp.loc[missing, "hour"].unique()))
    return tmp[~missing].groupby(["bjd_code", "mi", "band"])["kwh_per_day"].sum(min_count=1).reset_index()


# ---------------------------------------------------------------- 2단계: 변화점

def _l2_cost_fn(y):
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(a, b):  # 구간 [a, b)
        n = b - a
        s = cs[b] - cs[a]
        return (cs2[b] - cs2[a]) - s * s / n
    return cost


def pelt_builtin(y, pen, min_size=2):
    """벌점 최적분할(평균 이동, L2 비용) — ruptures.Pelt(model='l2') 와 같은 목적함수의 정확해.

    PELT 의 가지치기는 속도용일 뿐 해는 같다. 월 시계열(n≈36)이라 가지치기 없이 O(n²)로 푼다.
    반환: 새 구간이 시작하는 인덱스 목록(0 과 n 제외).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    cost = _l2_cost_fn(y)
    best = np.full(n + 1, np.inf)
    best[0] = -pen
    prev = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        for s in range(0, t - min_size + 1):
            if not np.isfinite(best[s]):
                continue  # s 에서 끝나는 앞 구간이 min_size 를 못 채움
            v = best[s] + cost(s, t) + pen
            if v < best[t]:
                best[t], prev[t] = v, s
    cps, t = [], n
    while t > 0:
        s = prev[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def _changepoints(y, pen, min_size, method):
    rpt = optional_import("ruptures") if method in ("auto", "ruptures") else None
    if method == "ruptures" and rpt is None:
        raise ImportError("activation.method='ruptures' 인데 ruptures 가 설치돼 있지 않음")
    if rpt is not None:
        bkps = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(np.asarray(y).reshape(-1, 1)).predict(pen=pen)
        return [b for b in bkps if b < len(y)], "ruptures.Pelt"
    return pelt_builtin(y, pen, min_size), "builtin.PELT"


def _ratio_at(level, idx, r):
    before = level[max(0, idx - r):idx]
    after = level[idx:idx + r]
    before, after = before[np.isfinite(before)], after[np.isfinite(after)]
    if len(before) == 0 or len(after) == 0 or before.mean() <= 0:
        return np.nan
    return after.mean() / before.mean()


def detect_activation(daily, params):
    """법정동별 활성화 시점 T_r.

    변화점(PELT)마다 전후 ratio_months 개월 평균 비율을 계산해, min_ratio 이상 '상승' 변화만 활성화로 본다.
      treated          — 첫 활성화가 window 안, lookback~window 시작 사이에는 활성화 없음
      excluded_prior   — lookback 이후 window 이전에 이미 활성화(처치 시점이 SHC 사전기간과 겹침)
      excluded_late    — window 이후에만 활성화
      never_treated    — lookback 이후 활성화 없음(대조군)
      insufficient     — 관측 월 부족
    전후 비율 단순법(킬 크라이테리아 2번 방식)의 첫 후보 월 T_simple 도 함께 남겨 두 방법의 일치 여부를 본다.
    """
    w0, w1 = (ym_to_mi(v) for v in params["window"])
    lookback = ym_to_mi(params["control_lookback_start"])
    r = int(params["ratio_months"])
    rows = []
    method_used = None
    for code, g in daily.groupby("bjd_code"):
        g = g.set_index("mi")["kwh_per_day"].sort_index()
        full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
        level = full.to_numpy(float)
        months = full.index.to_numpy()
        row = {"bjd_code": code, "n_months": int(np.isfinite(level).sum())}
        if row["n_months"] < params["min_months"]:
            rows.append({**row, "status": "insufficient"})
            continue
        y = np.log1p(pd.Series(level).interpolate(limit_direction="both").to_numpy())
        diffs = np.diff(y)
        sigma2 = max((np.median(np.abs(diffs - np.median(diffs))) / 0.6745) ** 2 / 2.0, 1e-4)
        pen = params["penalty_mult"] * sigma2 * np.log(len(y))
        cps, method_used = _changepoints(y, pen, int(params["min_size"]), params["method"])
        ups = {}
        for c in cps:
            # ⚠ [9/15 mock] 잡음이 있으면 PELT 변화점이 실제 계단에서 1~2개월 어긋난다(2025-04 활성화가 2025-02 로
            #   잡혀 excluded_prior 로 빠짐). ±2개월 안에서 전후 평균비가 최대인 달로 보정한다.
            best = None
            for c2 in range(max(1, c - 2), min(len(level) - 1, c + 2) + 1):
                ratio = _ratio_at(level, c2, r)
                if np.isfinite(ratio) and (best is None or ratio > best[1]):
                    best = (c2, ratio)
            if best and best[1] >= params["min_ratio"]:
                ups[int(months[best[0]])] = float(best[1])
        # ⚠ [9/15 mock] 계단 3~4개월 앞의 가짜 변화점이 보정되면 계단 2개월 전(비율≈1.33)에 걸려,
        #   진짜 계단(비율≈2.0)보다 먼저 T_r 로 채택됐다(18곳 중 5곳이 2개월 이르게 잡힘).
        #   ratio_months−1 개월 안에 붙은 상승 변화는 하나로 묶고 비율 최대인 달만 남긴다.
        merged = []
        for month, ratio in sorted(ups.items()):
            if merged and month - merged[-1][-1][0] <= r - 1:
                merged[-1].append((month, ratio))
            else:
                merged.append([(month, ratio)])
        ups = [max(group, key=lambda u: u[1]) for group in merged]
        prior = [u for u in ups if lookback <= u[0] < w0]
        inwin = [u for u in ups if w0 <= u[0] <= w1]
        late = [u for u in ups if u[0] > w1]
        best = None
        for m in range(w0, w1 + 1):
            if m in full.index:
                ratio = _ratio_at(level, int(np.where(months == m)[0][0]), r)
                if np.isfinite(ratio) and (best is None or ratio > best[1]):
                    best = (m, ratio)
        simple = best[0] if best and best[1] >= params["min_ratio"] else None
        if prior:
            status, t_r = "excluded_prior", prior[0]
        elif inwin:
            status, t_r = "treated", inwin[0]
        elif late:
            status, t_r = "excluded_late", late[0]
        else:
            status, t_r = "never_treated", None
        rows.append({**row, "status": status, "n_changepoints": len(cps),
                     "T_r_mi": t_r[0] if t_r else np.nan, "T_r": mi_to_ym(t_r[0]) if t_r else None,
                     "ratio": t_r[1] if t_r else np.nan,
                     "T_simple": mi_to_ym(simple) if simple is not None else None,
                     "agree_simple": (simple is not None and t_r is not None and abs(simple - t_r[0]) <= 1)
                     if status == "treated" else None})
    out = pd.DataFrame(rows)
    counts = out["status"].value_counts().to_dict()
    log.info("2단계 변화점(%s): %s", method_used, counts)
    treated = out[out["status"] == "treated"]
    if len(treated):
        log.info("  처치 코호트: %s", treated["T_r"].value_counts().sort_index().to_dict())
        log.info("  단순 전후비율법과 ±1개월 일치: %d/%d", int(treated["agree_simple"].sum()), len(treated))
    out.attrs["method"] = method_used
    return out


def simple_candidates(daily, window, min_ratio, ratio_months):
    """킬 크라이테리아 2번 — window 월 중 전후 평균비율 ≥ min_ratio 인 고유 법정동 집합(중복 집계 금지)."""
    w0, w1 = (ym_to_mi(v) for v in window)
    found = set()
    for code, g in daily.groupby("bjd_code"):
        s = g.set_index("mi")["kwh_per_day"].sort_index()
        full = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
        level = full.to_numpy(float)
        for m in range(w0, w1 + 1):
            if m in full.index:
                ratio = _ratio_at(level, m - int(full.index.min()), ratio_months)
                if np.isfinite(ratio) and ratio >= min_ratio:
                    found.add(code)
                    break
    return found
