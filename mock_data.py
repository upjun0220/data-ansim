"""mock/synthetic 데이터 생성 — 스펙 §2 스키마와 같은 컬럼 구조.

⚠ 지역명·코드·수치는 전부 가상이다. 로더와 파이프라인 동작 검증용이며 분석 결론에 쓰지 않는다.

실행(PowerShell, 저장소 루트):
    .venv\\Scripts\\python.exe mock_data.py --out data/mock

심어 둔 정답(테스트가 확인한다):
  - 법정동 30곳(서울 3개 자치구 이름·코드, 동 이름은 가상). 처치 18곳(2025-03~07 스태거드) · never-treated 10 · 창 이전 활성화 1 · 창 이후 1
  - 처치 효과: 대기소비 업종의 외지인 매출만 exp(τ_r) 배. τ_r = 0.05 + 0.30·[업종구성 상위 & 외지유입 상위]
  - 위약 업종(MD)과 거주자 매출에는 효과 없음 → 위약 τ ≈ 0
  - 처치오염 2곳: T_r 부터 가맹점 신규 개설 급증 + 전 업종·전 유입 매출 1.3배(Y_ddd 에서는 상쇄)
  - v11: 신한카드(SHC) 파일은 --with-shc(상권 단계 회귀용)에서만 만든다. --energy 는 기온(실측·예보)·공휴일 달력·
    기온·요일·휴일에 반응하는 일별 충전 부하를 함께 만든다. mock 에서 AI가 기준 모델을 이겨도 성능 근거가 아니다.
  - KEPCO: '가람제1동' 처럼 '제' 표기 차이(정규화로 매칭), 동명이인 '중앙동'(구가 달라 3단 매칭), 마스터에 없는 '행복동'(미매칭)
  - CAN: 개별 차량(individual) 파일과 모델 코드(model) 파일 두 개
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SIDO_CODE, SIDO_FULL, SIDO_SHORT = "11", "서울특별시", "서울"
SIGUNGU = [("110", "종로구"), ("140", "중구"), ("170", "용산구")]   # 서울 자치구 코드·이름(법정동 이름은 가상)
EMD_NAMES = {
    "110": ["가람1동", "가람2동", "중앙동", "새솔동", "한울동", "별빛동", "누리동", "꽃내동", "솔밭동", "미리내동"],
    "140": ["나래1동", "나래2동", "중앙동", "은빛동", "푸른동", "햇살동", "초록동", "바람동", "구름동", "하늘동"],
    "170": ["다솜1동", "다솜2동", "노을동", "샘물동", "들꽃동", "산마루동", "달빛동", "새터동", "물빛동", "보람동"],
}
WAIT_MEDVS = [("FD", "FD01"), ("FD", "FD02"), ("RT", "RT01"), ("RT", "RT02")]
OTHER_MEDVS = [("MD", "MD01"), ("ED", "ED01"), ("PS", "PS01"), ("ET", "ET01")]
DIST_CODES = ["1", "2", "3", "4", "9"]
COHORTS = ["2025-03"] * 4 + ["2025-04"] * 4 + ["2025-05"] * 4 + ["2025-06"] * 3 + ["2025-07"] * 3


def _mi(ym):
    y, m = ym.split("-")
    return int(y) * 12 + int(m) - 1


def build_regions(rng):
    rows = []
    for sgg, sgg_name in SIGUNGU:
        for i, emd in enumerate(EMD_NAMES[sgg]):
            code = f"{SIDO_CODE}{sgg}{101 + i:03d}00"
            kepco_emd = re.sub(r"^(.+?)(\d)동$", r"\1제\2동", emd)   # '가람1동' → '가람제1동'
            rows.append({"bjd_code": code, "sgg": sgg, "sigungu": sgg_name, "emd": emd, "kepco_emd": kepco_emd})
    reg = pd.DataFrame(rows)
    n = len(reg)
    order = rng.permutation(n)
    role = np.array(["never_treated"] * n, dtype=object)
    act = np.array([None] * n, dtype=object)
    for j, idx in enumerate(order[:len(COHORTS)]):
        role[idx], act[idx] = "treated", COHORTS[j]
    role[order[18]], act[order[18]] = "excluded_prior", "2024-10"
    role[order[19]], act[order[19]] = "excluded_late", "2025-11"
    reg["role"], reg["activation"] = role, act
    reg["wait_share"] = rng.uniform(0.3, 0.7, n)
    reg["outsider_share"] = rng.uniform(0.2, 0.6, n)
    reg["tau"] = np.where(reg["role"] == "treated",
                          0.05 + 0.30 * ((reg["wait_share"] > 0.5) & (reg["outsider_share"] > 0.4)), 0.0)
    treated = reg[reg["role"] == "treated"]
    contaminated = [treated[treated["activation"] == "2025-04"].index[0], treated[treated["activation"] == "2025-05"].index[0]]
    reg["contaminated"] = reg.index.isin(contaminated)
    reg["grid_lat"] = 37.48 + 0.012 * (np.arange(n) // 6)
    reg["grid_lon"] = 126.95 + 0.015 * (np.arange(n) % 6)
    reg["kwh_level"] = rng.uniform(200, 800, n)
    reg["peakiness"] = rng.uniform(0, 1, n)
    reg["cpo_share"] = rng.uniform(0.3, 0.6, n)
    reg["sales_size"] = rng.uniform(5e4, 2e5, n)
    return reg


def write_master(reg, path):
    lines = [f"{SIDO_CODE}00000000\t{SIDO_FULL}\t존재"]
    for sgg, name in SIGUNGU:
        lines.append(f"{SIDO_CODE}{sgg}00000\t{SIDO_FULL} {name}\t존재")
    for _, r in reg.iterrows():
        lines.append(f"{r['bjd_code']}\t{SIDO_FULL} {r['sigungu']} {r['emd']}\t존재")
        lines.append(f"{r['bjd_code'][:8]}01\t{SIDO_FULL} {r['sigungu']} {r['emd']} 가상리\t존재")  # 리 단위(무시 대상)
    lines.append(f"{SIDO_CODE}170199" + "00\t" + f"{SIDO_FULL} 용산구 옛터동\t폐지")
    Path(path).write_text("법정동코드\t법정동명\t폐지여부\n" + "\n".join(lines) + "\n", encoding="cp949")


def write_centroids(reg, path):
    pd.DataFrame({"법정동코드": reg["bjd_code"], "위도": reg["grid_lat"].round(6), "경도": reg["grid_lon"].round(6)}) \
        .to_csv(path, index=False, encoding="utf-8-sig")


def write_kepco(reg, rng, path001, path002, daily_period=None, day_factor=None):
    """day_factor: 일별 배수(날짜 인덱스 Series) — daily_period 에서만 쓴다(기온·요일·휴일·추세 반응)."""
    months = np.arange(_mi("2023-01"), _mi("2025-12") + 1)
    days = np.array([1, 8, 15, 22])
    hours = np.arange(24)
    extra = pd.DataFrame([{"sigungu": "용산구", "kepco_emd": "행복동", "kwh_level": 30.0, "peakiness": 0.5,
                           "activation": None, "cpo_share": 0.4}])
    regs = pd.concat([reg, extra], ignore_index=True)
    frames001, frames002 = [], []
    for _, r in regs.iterrows():
        t = months - months[0]
        season = 1 + 0.08 * np.sin(2 * np.pi * (months % 12) / 12)
        level = r["kwh_level"] * (1 + 0.003 * t) * season * rng.lognormal(0, 0.04, len(months))
        if isinstance(r["activation"], str):   # concat 후 None 이 NaN(float)로 바뀐다
            level = level * np.where(months >= _mi(r["activation"]), 2.0, 1.0)
        flat = 0.6 + 0.4 * np.exp(-((hours - 19) / 3.0) ** 2)
        peaky = 0.2 + 1.8 * np.exp(-((hours - 19) / 1.5) ** 2)
        prof = (1 - r["peakiness"]) * flat / flat.sum() + r["peakiness"] * peaky / peaky.sum()
        M, D, H = np.meshgrid(np.arange(len(months)), days, hours, indexing="ij")
        M, D, H = M.ravel(), D.ravel(), H.ravel()
        if daily_period is not None:
            dates = pd.date_range(*daily_period, freq="D")
            M = np.repeat((dates.year * 12 + dates.month - 1 - months[0]).to_numpy(), 24)
            D, H = np.repeat(dates.day.to_numpy(), 24), np.tile(hours, len(dates))
            if (M < 0).any() or (M >= len(months)).any():
                raise ValueError("mock 일별 기간은 2023~2025 안이어야 함")
        kwh = level[M] * prof[H] * rng.lognormal(0, 0.08, len(M))
        if daily_period is not None and day_factor is not None:
            kwh = kwh * np.repeat(day_factor.reindex(dates).to_numpy(float), 24)
        mi = months[M]
        period = ((mi // 12) * 1_000_000 + (mi % 12 + 1) * 10_000 + D * 100 + H).astype(str)
        cust = np.maximum(1, np.round(kwh / 7)).astype(int)
        base = {"조회기간": period, "시도": SIDO_SHORT, "시군구": r["sigungu"], "읍면동": r["kepco_emd"]}
        frames001.append(pd.DataFrame({**base, "고객호수": cust, "시간대별 사용량(kWh)": np.round(kwh, 3)}))
        keep = rng.random(len(M)) > 0.02
        share = r["cpo_share"]
        frames002.append(pd.DataFrame({k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in base.items()}).assign(
            고객호수=np.floor(cust[keep] * share).astype(int), **{"시간대별 사용량(kWh)": np.round(kwh[keep] * share, 3)}))
    pd.concat(frames001).to_csv(path001, index=False, encoding="cp949")
    pd.concat(frames002).to_csv(path002, index=False, encoding="cp949")


def write_shc(reg, rng, path001, path002, cell_noise=0.02):
    months = np.arange(_mi("2025-01"), _mi("2025-12") + 1)
    medvs = WAIT_MEDVS + OTHER_MEDVS
    rows002 = []
    rows001 = []
    for _, r in reg.iterrows():
        g = _mi(r["activation"]) if isinstance(r["activation"], str) else None
        region_shock = rng.lognormal(0, 0.03, len(months))
        for ci, (bid, med) in enumerate(medvs):
            is_wait = ci < len(WAIT_MEDVS)
            ind_w = r["wait_share"] / 4 if is_wait else (1 - r["wait_share"]) / 4
            # SHC001: 가맹점 재고 흐름(신규 개설 Poisson, 해지 1%)
            stock = int(rng.integers(30, 80))
            for mj, m in enumerate(months):
                opened = rng.poisson(stock * 0.02)
                if r["contaminated"] and g is not None and g <= m <= g + 1:
                    opened += int(stock * 0.15)
                closed = rng.binomial(stock, 0.01)
                stock = stock + opened - closed
                ym = f"{m // 12}{m % 12 + 1:02d}"
                code = r["bjd_code"]
                for status, oper, cnt in (("1", "1", stock), ("1", "0", int(rng.integers(0, 3))), ("2", "0", closed)):
                    rows001.append((ym, SIDO_CODE, SIDO_CODE + r["sgg"], code, bid, med, status, oper, cnt))
                for d in DIST_CODES:
                    ow = (1 - r["outsider_share"]) * 0.95 if d == "1" else (0.05 if d == "9" else r["outsider_share"] * 0.95 / 3)
                    s = r["sales_size"] * ind_w * ow * (1 + 0.05 * np.sin(2 * np.pi * m / 12)) * region_shock[mj] \
                        * rng.lognormal(0, cell_noise)
                    if g is not None and m >= g:
                        if r["contaminated"]:
                            s *= 1.3
                        if r["role"] == "treated" and is_wait and d in ("2", "3", "4"):
                            s *= np.exp(r["tau"])
                    split = rng.dirichlet(np.full(4, 2.0))
                    smallest = int(split.argmin())
                    for pi, part in enumerate(split):
                        amt = max(1, int(round(s * part)))
                        # 마스킹은 소액 셀에서 주로 생긴다 — 가장 작은 조각만 25% 확률로 가림(행 기준 ≈6%)
                        masked = pi == smallest and rng.random() < 0.25
                        rows002.append((ym, SIDO_CODE, SIDO_CODE + r["sgg"], code, bid, med, "1",
                                        str(rng.integers(1, 3)), str(rng.integers(1, 7)), rng.choice(["M", "F"]),
                                        str(rng.choice([20, 30, 40, 50, 60])), "1", str(rng.integers(1, 4)),
                                        SIDO_CODE + r["sgg"], d,
                                        "*" if masked else str(max(1, amt // 15)), "*" if masked else str(amt)))
    pd.DataFrame(rows001, columns=["STD_YM", "WIAR_SIDO_CD", "SGNG_CD", "UMD_CD", "TOBU_BIDVS_CD", "TOBU_MEDVS_CD",
                                   "FRNC_STAT_CD", "OPER_CD", "FRNC_CNT"]).to_csv(path001, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows002, columns=["STD_YM", "WIAR_SIDO_CD", "SGNG_CD", "UMD_CD", "TOBU_BIDVS_CD", "TOBU_MEDVS_CD",
                                   "PERS_CORP_CLCD", "PHOLI_CLCD", "TIZO_CLCD", "SEX_CLCD", "N10_UNIT_AGE_CLCD",
                                   "HSH_LFTM_MAIN_PRE_CD", "EST_INCM_SECT_CD", "IFW_AREA_CD", "IFW_DISTC_SECT_CD",
                                   "PAYM_NOCA", "SALE_AMT"]).to_csv(path002, index=False, encoding="utf-8-sig")


def write_kep007(reg, rng, path, n_sites=60):
    idx = rng.integers(0, len(reg), n_sites)
    r = reg.iloc[idx].reset_index(drop=True)
    df = pd.DataFrame({
        "WIAR_SIDO_CD": SIDO_CODE, "WIAR_SIDO_NM": SIDO_FULL, "SGNG_CD": r["sgg"], "SGNG_NM": r["sigungu"],
        "INST_PLC_NM": [f"{e} 공영주차장 {i}" for i, e in enumerate(r["emd"])],
        "ADDR": [f"{SIDO_FULL} {s} {e} {i + 1}" for i, (s, e) in enumerate(zip(r["sigungu"], r["emd"]))],
        "QCK_CHNG_PRE_NOEQ": rng.integers(0, 5, n_sites), "SLW_CHNG_PRE_NOEQ": rng.integers(0, 11, n_sites),
        "LTD": (r["grid_lat"] + rng.uniform(-0.003, 0.003, n_sites)).round(6),
        "LNGT": (r["grid_lon"] + rng.uniform(-0.003, 0.003, n_sites)).round(6),
        "SPRT_CAKI_NM": "아이오닉5 외",
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_access_data(reg, stations, station_path, ev_path):
    """V9 접근성 단계용 공개자료 형태의 합성 입력. 실제 행정동 자료가 아니다."""
    public = pd.DataFrame({
        "위도": stations["LTD"], "경도": stations["LNGT"],
        "충전기수": stations["QCK_CHNG_PRE_NOEQ"] + stations["SLW_CHNG_PRE_NOEQ"],
    })
    public.loc[public["충전기수"] <= 0, "충전기수"] = 1
    public.to_csv(station_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"법정동코드": reg["bjd_code"],
                  "전기차등록대수": 30 + (np.arange(len(reg)) * 37) % 470}).to_csv(
                      ev_path, index=False, encoding="utf-8-sig")


def write_ev_history(reg, rng, history_path, mapping_path):
    """8-F 입력(합성): 행정동별 연료별 등록 월별 이력(2015-01~2026-08)과 행정동→법정동 대응표.

    2026-06 전기차 수는 write_access_data 의 법정동 등록대수와 같게 두고, 연 10~40% 증가로 거꾸로 만든다.
    심어 둔 것: 첫 두 법정동은 한 행정동(가중치 0.6/0.4)에서 나뉘고, 대응표에 없는 행정동 1곳(매핑 실패).
    """
    ev_bjd = pd.Series(30 + (np.arange(len(reg)) * 37) % 470, index=reg["bjd_code"].to_numpy(), dtype=float)
    growth = rng.uniform(0.10, 0.40, len(reg))
    hdong = [f"{c[:5]}5{i:02d}00" for i, c in enumerate(reg["bjd_code"])]
    hdong[1] = hdong[0]                                           # 행정동 1개 → 법정동 2개
    mapping = pd.DataFrame({"행정동코드": hdong, "법정동코드": reg["bjd_code"], "가중치": 1.0})
    mapping.loc[[0, 1], "가중치"] = [0.6, 0.4]
    months = pd.period_range("2015-01", "2026-08", freq="M")
    t = (months - pd.Period("2026-06", freq="M")).map(lambda x: x.n).to_numpy() / 12
    rows = []
    level = {}
    for i, h in enumerate(hdong):
        level.setdefault(h, []).append((ev_bjd.iloc[i], growth[i]))
    level["1117059900"] = [(25.0, 0.2)]                           # 대응표에 없는 행정동
    for h, parts in level.items():
        now = sum(p[0] for p in parts)
        g = np.average([p[1] for p in parts], weights=[p[0] for p in parts])
        ev = np.round(now * (1 + g) ** t)
        for ym, v in zip(months.strftime("%Y%m"), ev):
            rows.append((ym, h, "전기", int(v)))
            rows.append((ym, h, "휘발유", 5000))
    pd.DataFrame(rows, columns=["기준년월", "행정동코드", "연료", "대수"]).to_csv(history_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")


def write_can(reg, stations, rng, path, mode, n_vehicles=40):
    """mode='individual': 차량 40대 × 세션 40 (2022-11~2025-06). mode='model': 같은 차량들을 모델 코드 2개로 합친 15일치."""
    if mode == "individual":
        start, span_days, n_sessions = pd.Timestamp("2022-11-01"), 970, 40
    else:
        start, span_days, n_sessions = pd.Timestamp("2025-01-01"), 15, 6
    home_regions = rng.choice(len(reg), 12, replace=False)
    recs = []
    for v in range(n_vehicles):
        vid = f"VH{v:04d}" if mode == "individual" else f"MDL_{'AB'[v % 2]}"
        home = reg.iloc[home_regions[v % 12]]
        home_lat = home["grid_lat"] + rng.uniform(-0.002, 0.002)
        home_lon = home["grid_lon"] + rng.uniform(-0.002, 0.002)
        day_offsets = np.sort(rng.choice(span_days, n_sessions, replace=False))
        soc = rng.uniform(40, 80)
        for day in day_offsets:
            is_home = rng.random() < 0.6
            if is_home:
                lat, lon = home_lat, home_lon
                t0 = start + pd.Timedelta(days=int(day), hours=int(rng.integers(19, 23)), minutes=int(rng.integers(0, 60)))
                dur, rate = rng.uniform(90, 300), 0.2
            else:
                st = stations.iloc[int(rng.integers(len(stations)))]
                lat, lon = st["LTD"] + rng.uniform(-0.0002, 0.0002), st["LNGT"] + rng.uniform(-0.0002, 0.0002)
                t0 = start + pd.Timedelta(days=int(day), hours=int(rng.integers(9, 20)), minutes=int(rng.integers(0, 60)))
                dur, rate = float(np.clip(rng.normal(30, 7), 15, 60)), 1.0
            t0 = t0 + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            soc = float(np.clip(soc - rng.uniform(10, 30), 10, 70))
            heading = rng.uniform(0, 2 * np.pi)
            for step in (3, 2, 1):   # 도착 전 주행(30km/h, 5분 간격)
                d_deg = 2.5 * step / 111.0
                recs.append((vid, t0 - pd.Timedelta(minutes=5 * step), "ON", "0", "D", soc + step, lat + d_deg * np.sin(heading),
                             lon + d_deg * np.cos(heading) / 0.8, 30.0))
            n_steps = int(dur // 5)
            for s in range(n_steps + 1):
                soc_now = min(100.0, soc + rate * 5 * s)
                recs.append((vid, t0 + pd.Timedelta(minutes=5 * s), "OFF", "1", "P", soc_now,
                             lat + rng.uniform(-0.00005, 0.00005), lon + rng.uniform(-0.00005, 0.00005), 0.0))
            soc = min(100.0, soc + rate * 5 * n_steps)
            end = t0 + pd.Timedelta(minutes=5 * n_steps)
            for step in (1, 2, 3):
                d_deg = 2.5 * step / 111.0
                recs.append((vid, end + pd.Timedelta(minutes=5 * step), "ON", "0", "D", soc - step, lat - d_deg * np.sin(heading),
                             lon - d_deg * np.cos(heading) / 0.8, 30.0))
            soc -= 3
    df = pd.DataFrame(recs, columns=["차종_식별번호", "발생시간", "시동상태", "충전중여부(CHARGERCONNECTION)", "기어상태",
                                     "배터리상태_SOC", "위도", "경도", "속도"])
    df = df.sort_values(["차종_식별번호", "발생시간"], kind="mergesort")
    n = len(df)
    out = pd.DataFrame({
        "발생시간": df["발생시간"].dt.strftime("%Y-%m-%d %H:%M:%S"), "시동상태": df["시동상태"],
        "충전중여부(CHARGERCONNECTION)": df["충전중여부(CHARGERCONNECTION)"], "기어상태": df["기어상태"],
        "배터리상태_SOC": df["배터리상태_SOC"].round(1), "배터리상태_SOH": 97.5, "배터리상태_C_V": 3.8,
        "속도": df["속도"], "엔진회전수": 0, "12V배터리전압": 12.6,
        "충전량": np.round(rng.uniform(0, 1, n), 2), "주행가능거리": np.round(df["배터리상태_SOC"] * 4.5, 0),
        "위도": df["위도"].round(6), "경도": df["경도"].round(6), "차종_식별번호": df["차종_식별번호"],
    })
    out.to_csv(path, index=False, encoding="utf-8-sig")


def generate(out_dir, seed=42, cell_noise=0.02, with_shc=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reg = build_regions(rng)
    write_master(reg, out / "bjd_master.txt")
    write_centroids(reg, out / "bjd_centroids.csv")
    write_kepco(reg, rng, out / "kepco_001.csv", out / "kepco_002.csv")
    if with_shc:
        write_shc(reg, rng, out / "shc001.csv", out / "shc002.csv", cell_noise)
    stations = write_kep007(reg, rng, out / "kep007.csv")
    write_access_data(reg, stations, out / "access_stations.csv", out / "ev_registration.csv")
    write_ev_history(reg, np.random.default_rng(seed + 1), out / "ev_history.csv", out / "hdong_bjd.csv")
    write_can(reg, stations, rng, out / "can_m_individual.csv", "individual")
    write_can(reg, stations, rng, out / "can_m_model.csv", "model")
    truth = reg[["bjd_code", "sigungu", "emd", "role", "activation", "wait_share", "outsider_share", "tau", "contaminated"]]
    truth.to_csv(out / "_truth.csv", index=False, encoding="utf-8-sig")
    return truth


MOCK_HOLIDAYS = {  # 2025년 5~12월 공휴일(mock 전용 입력 — 관보·특일 정보 API로 대조하지 않았다)
    "2025-05-05": "어린이날·부처님오신날", "2025-05-06": "대체공휴일", "2025-06-03": "대통령선거일",
    "2025-06-06": "현충일", "2025-08-15": "광복절",
    "2025-10-03": "개천절", "2025-10-05": "추석 연휴", "2025-10-06": "추석", "2025-10-07": "추석 연휴",
    "2025-10-08": "대체공휴일", "2025-10-09": "한글날", "2025-12-25": "기독탄신일",
}
HOLIDAY_FACTOR = {"holiday": 0.8, "long_holiday": 0.75, "pre_post_holiday": 1.1}


def write_weather(rng, dates, path, station="108"):
    """서울 1개 지점의 시간별 실측 기온과 과거 예보(합성). 예보는 전날 17시(마감 전)·20시(마감 후) 두 번 발표한다.

    20시 발표분은 오차가 더 작지만 마감(18시) 뒤라 모델이 쓰면 안 되는 값이다 — 발표 시각 검사 시험용.
    """
    doy = dates.dayofyear.to_numpy()
    daily = 13 + 14 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 2.5, len(dates))
    ts = pd.date_range(dates[0], dates[-1] + pd.Timedelta(hours=23), freq="h")
    obs = np.repeat(daily, 24) + 4 * np.sin(2 * np.pi * (ts.hour.to_numpy() - 9) / 24) + rng.normal(0, 0.5, len(ts))
    frames = []
    for issued_hour, noise in ((17, 1.5), (20, 0.8)):
        frames.append(pd.DataFrame({
            "timestamp": ts, "station_or_grid": station, "temp_obs_c": obs.round(1),
            "temp_fcst_c": (obs + rng.normal(0, noise, len(ts))).round(1),
            "fcst_issued_at": ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=issued_hour)}))
    pd.concat(frames).sort_values(["timestamp", "fcst_issued_at"]).to_csv(path, index=False, encoding="utf-8-sig")
    return pd.Series(daily, index=dates)


def generate_energy(out_dir, seed=42):
    """기존 mock 파일을 건드리지 않는 연속 일별 8-A/8-B 시험 자료. 실측이 아니다.

    기간은 2025-05~12(AI 학습·8주 검증·28일 평가를 채우도록). 부하는 기온(12°C 아래로 1°C당 +1.2%, 26°C 위로
    1°C당 +0.8%)·주말·휴일유형·완만한 증가 추세(하루 0.15%)에 반응한다. 이 반응은 mock 에
    심어 둔 것이라 AI가 기준 모델을 이겨도 성능 근거가 아니다.
    """
    from weatherloader import classify_holidays, dump_holidays

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reg = build_regions(rng)
    # 작은 공용 거점 가정: 증설 2~5대가 상한을 넘는 경우까지 시험한다. 현장 규모 추정 아님.
    reg["kwh_level"] *= 0.1
    period = ("2025-05-01", "2025-12-31")
    dates = pd.date_range(*period, freq="D")
    daily_temp = write_weather(rng, dates, out / "weather_mock.csv")
    types = classify_holidays(MOCK_HOLIDAYS)
    (out / "holidays.yaml").write_text(dump_holidays({
        d: {"type": t, "public_holiday": d in MOCK_HOLIDAYS, "name": MOCK_HOLIDAYS.get(d, "연휴 전날·다음날"),
            "source": "mock(관보·특일 정보 API 대조 안 함)"} for d, t in types.items()}), encoding="utf-8")
    kind = pd.Series(types).rename(index=pd.Timestamp).reindex(dates)
    factor = ((1 + 0.012 * np.maximum(0, 12 - daily_temp) + 0.008 * np.maximum(0, daily_temp - 26))
              * np.select([dates.dayofweek == 5, dates.dayofweek == 6], [0.9, 0.85], 1.0)
              * kind.map(HOLIDAY_FACTOR).fillna(1.0)
              * (1 + 0.0015 * np.arange(len(dates))))
    write_kepco(reg, rng, out / "kepco_001_hourly.csv", out / "kepco_002_hourly.csv",
                daily_period=period, day_factor=factor)
    ts = pd.date_range(period[0], period[1] + " 23:00", freq="h")
    smp = 100 + 70 * np.exp(-((ts.hour.to_numpy() - 19) / 3) ** 2)
    pd.DataFrame({"timestamp": ts, "smp": smp}).to_csv(out / "smp_mock.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="충전 리플맵 mock 데이터 생성")
    parser.add_argument("--out", default="data/mock")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--energy", action="store_true", help="8-A/8-B 연속 일별 합성 자료·기온·공휴일도 생성")
    parser.add_argument("--with-shc", action="store_true", help="신한카드 SHC001·002 도 생성(상권 단계 회귀용, v11 기본 비활성)")
    args = parser.parse_args()
    truth = generate(args.out, args.seed, with_shc=args.with_shc)
    if args.energy:
        generate_energy(args.out, args.seed)
    print(truth["role"].value_counts().to_string())
    print("생성 완료:", Path(args.out).resolve())


if __name__ == "__main__":
    main()
