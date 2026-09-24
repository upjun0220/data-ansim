"""충전 리플맵 — 0~9단계 end-to-end 실행(v11: 서울 법정동, 상권 단계 기본 비활성).

실행(로컬, PowerShell, 저장소 루트):
    .venv\\Scripts\\python.exe mock_data.py --out data/mock
    .venv\\Scripts\\python.exe pipeline.py --config config/mock.json
현장(안심구역 JupyterLab):
    from pipeline import run
    summary = run("config/field.json")

단계 격리 원칙
  - 필수: 0·1(정합·집계). 상권 단계(stages.commerce=true)일 때만 2(변화점) · 4(Y_ddd) · 5(이벤트 스터디) · 6(위약)도
    필수이며, 기본(false)은 2·4~7단계를 건너뛰고 실행 요약에 "상권 단계 비활성(v11)"을 남긴다.
  - 격리: 3(CAN · KEP_007) · 6.5(처치오염) · 7(CATE) · 8(처방) · 8-C(접근성) · 8-E(결합점수)
    — 실패해도 나머지는 계속 돈다.
    CAN 이 실패하면 'CAN 처리 생략됨' 로그와 함께 3단계 표만 비워 둔다.
산출물은 out_dir/png(반출용) · out_dir/csv(현장 작업용), 로그는 out_dir/pipeline.log.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import bjdmapping
import canloader
import diagnostics
import equityaccess
import headline
import heterogeneity
import identification
import kep007loader
import kepcoloader
import loadaxis
import loadforecast
import loadscenario
import essoptimizer
import priorityscore
import weatherloader
from common import keep_region, log, mi_to_ym, setup_logging, ym_to_mi
from config import deep_merge, load_config, resolve_region, select_kepco_source
from outputs import (OutputWriter, plot_activation_examples, plot_bar, plot_event_study, plot_histogram,
                     plot_quadrants)

CRITICAL = True
ISOLATED = False


class _StopAfterStage1(Exception):
    """0·1단계 산출물을 정상 저장한 뒤 후속 단계를 건너뛰기 위한 내부 신호."""


class StageRunner:
    def __init__(self):
        self.rows = []

    def run(self, key, name, fn, critical):
        t0 = time.time()
        log.info("=" * 10 + " %s단계 %s " + "=" * 10, key, name)
        try:
            result = fn()
        except Exception as exc:
            elapsed = time.time() - t0
            msg = f"{type(exc).__name__}: {exc}"
            self.rows.append({"단계": key, "이름": name, "상태": "실패" if critical else "생략", "소요초": elapsed, "비고": msg})
            if critical:
                log.error("%s단계 실패 — 파이프라인 중단: %s\n%s", key, msg, traceback.format_exc())
                raise
            log.warning("%s단계 %s 생략됨: %s", key, name, msg)
            log.debug(traceback.format_exc())
            return None
        self.rows.append({"단계": key, "이름": name, "상태": "완료", "소요초": time.time() - t0, "비고": ""})
        return result

    def table(self):
        return pd.DataFrame(self.rows)


def _ym_col(df, col="mi", name="월"):
    out = df.copy()
    out.insert(1 if "bjd_code" in out.columns else 0, name, out[col].map(mi_to_ym))
    return out.drop(columns=col)


def _energy_load_window(params, train_days=0):
    """검증·교정·평가(+ AI 학습 train_days)에 필요한 최소 시간별 원자료 기간만 계산한다."""
    if not params.get("evaluation_start"):
        raise ValueError("energy.enabled=true인데 energy.evaluation_start가 비어 있음 — 현장 평가 시작일을 설정할 것")
    start = pd.Timestamp(params["evaluation_start"]).normalize()
    validation_days = max(7 * int(params["holdout_weeks"]), int(params["min_validation_days"]))
    history_days = max(int(params["calibration_days"]), int(train_days) + validation_days + 7 * int(params["weeks"]))
    date_start = start - pd.Timedelta(days=history_days)
    date_end = start + pd.Timedelta(days=int(params["evaluation_days"]) - 1)
    log.info("8-A 시간별 원자료 필터: %s~%s", date_start.date(), date_end.date())
    return date_start, date_end


def _warn_energy_calendar(params):
    """연말·설 평가는 공식 휴일 달력 확인이 필요함을 알린다(연휴 날짜 자체는 설정에서만 읽는다)."""
    if not params.get("evaluation_start"):
        raise ValueError("energy.enabled=true인데 energy.evaluation_start가 비어 있음 — 현장 평가 시작일을 설정할 것")
    start = pd.Timestamp(params["evaluation_start"]).normalize()
    end = start + pd.Timedelta(days=int(params["evaluation_days"]) - 1)
    days = pd.date_range(start, end)
    holidays = params.get("holidays") or {}
    if not holidays:
        log.warning("8-A holidays가 비어 있음 — 공식 공휴일·설 연휴 포함 여부를 확인할 것")
    if any((d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 1) for d in days):
        log.warning("8-A 평가 구간이 12/24~1/1에 걸침 — 연말 특수기간 사용 여부를 확인할 것")
    lunar = [pd.Timestamp(d).normalize() for d, label in holidays.items() if "설" in str(label)]
    if any(start - pd.Timedelta(days=1) <= d <= end + pd.Timedelta(days=1) for d in lunar):
        log.warning("8-A 평가 구간이 설정된 설 연휴 전후에 걸침 — 대표성 확인 필요")


def _kepco_export(df, customer_min, min_count, value_cols):
    """내부 CSV에는 원값과 억제 표시를 남기고, 반출 PNG에서만 법정동 수치를 가린다."""
    csv = df.copy()
    if "bjd_code" not in csv:
        return csv, csv.copy()
    small = csv["bjd_code"].astype(str).map(customer_min).fillna(0) < int(min_count)
    csv["소표본억제"] = small
    png = csv.copy()
    for col in set(value_cols) & set(png.columns):
        png.loc[small, col] = np.nan
    if small.any():
        log.info("KEPCO 소표본 PNG 억제: %d행 · 최소 고객호수 < %d", int(small.sum()), min_count)
    return csv, png


def _period_customer_count(df, cust_col, basis, group_cols=("bjd_code",)):
    """법정동(×월) 대표 고객호수 — suppress_basis 설정에 따라 최댓값(기본) 또는 최솟값.

    최댓값을 기본으로 하는 이유: 서로 다른 시각의 고객 수는 서로 다른 하한이라, 새벽처럼
    저조한 시간대 하나로 법정동 전체를 가리지 않기 위함(R1, 2026-09-19 리뷰).
    """
    if basis not in ("max", "min"):
        raise ValueError(f"params.kepco.suppress_basis 는 'max' 또는 'min' — 받은 값: {basis!r}")
    return df.groupby(list(group_cols))[cust_col].agg(basis)


def _exclude_small_cells(df, rep, min_count, label):
    """(법정동,월) 대표 고객호수 rep 기준 소표본 셀을 제외한 부분집합 — 전국 합계 PNG 전용(R2).

    행을 가리는 방식(_kepco_export)이 아니라 전국 합계에 기여하는 것 자체를 막는다.
    법정동 열이 없는 전국 합계표(s1)는 나머지 값을 빼는 차감 역산으로 소표본 법정동이
    재식별될 수 있기 때문이다.
    """
    idx = pd.MultiIndex.from_frame(df[["bjd_code", "mi"]])
    small = pd.Series(rep.reindex(idx).fillna(0).to_numpy() < min_count, index=df.index)
    if small.any():
        log.info("%s PNG 억제: 법정동×월 %d칸 제외 · 대표 고객호수 < %d", label, int(small.sum()), min_count)
    return df[~small]


def _exclude_small_activation(act, rep, min_count):
    """활성화 시점 예시 그림 후보에서 소표본 법정동을 제외한다(R2). rep: bjd_code 인덱스 대표 고객호수."""
    small = act["bjd_code"].astype(str).map(rep).fillna(0) < min_count
    return act[~small], int((small & (act["status"] == "treated")).sum())


def channel_share(mh001, mh002):
    """KEPCO_002/001 비율(법정동×월×시각 셀). 비율만 보며 두 원천을 합산하지 않는다."""
    keys = ["bjd_code", "mi", "hour"]
    m = mh002[keys + ["kwh"]].merge(mh001[keys + ["kwh"]], on=keys, suffixes=("_002", "_001"))
    m = m[m["kwh_001"] > 0].assign(ratio=lambda d: d["kwh_002"] / d["kwh_001"])
    if m.empty:
        raise ValueError("KEPCO_001·002 가 겹치는 법정동×월×시각 셀 없음")
    return m


def channel_share_table(cells):
    """시간대별 002/001 비율 분포."""
    g = cells.groupby("hour")["ratio"]
    return pd.DataFrame({"셀수": g.size(), "비율_p10": g.quantile(0.1), "비율_중앙값": g.median(),
                         "비율_p90": g.quantile(0.9), "002>001_셀비율": g.apply(lambda r: float((r > 1.0001).mean()))}
                        ).reset_index().rename(columns={"hour": "시각"})


def run(config_path, params_override=None, paths_override=None, stop_after_stage1=False):
    cfg = load_config(config_path)
    if params_override:
        cfg["params"] = resolve_region(deep_merge(cfg["params"], params_override))
    bjdmapping.set_analysis_level(cfg["params"]["analysis_level"])
    if paths_override:
        cfg["paths"].update({k: (str(Path(v).resolve()) if v else None) for k, v in paths_override.items()})
    P, C, paths, ind = cfg["params"], cfg["columns"], cfg["paths"], cfg["industry"]
    commerce = bool(P["stages"]["commerce"])
    prefix = P["region"]["sido_prefix"]
    out_dir = Path(paths["out_dir"])
    setup_logging(out_dir)
    writer = OutputWriter(out_dir, **P["outputs"], font_path=paths["font"])
    runner = StageRunner()
    S = {}
    log.info("설정: %s", cfg["config_path"])

    try:
        # ------------------------------------------------ 0·1단계
        def stage01():
            master = keep_region(bjdmapping.load_bjd_master(paths["bjd_master"], C["bjd"]), prefix)
            S["crosswalk"] = bjdmapping.load_crosswalk(paths["bjd_crosswalk"], C["crosswalk"]) \
                if paths["bjd_crosswalk"] else None
            # 상권 단계를 끄면 부하 분석의 주 입력은 KEPCO_001(kepco.source 무시). 001이 없을 때만 002 잠정.
            src, preliminary = select_kepco_source(paths, P)
            if preliminary:
                log.warning("KEPCO_002 잠정 분석 — 정식 결과는 KEPCO_001 확보 후 다시 실행")
            raw = kepcoloader.load_kepco_monthly_hour(paths[f"kepco_{src}"], C["kepco" if src == "001" else "kepco_cpo"],
                                                       P["kepco"], f"KEPCO_{src}")
            if commerce:
                last = kepcoloader.last_month(raw)
                act_p = P["activation"]
                short = kepcoloader.window_shortfall(last, act_p["window"], act_p["ratio_months"])
                log.info("KEPCO_%s 마지막 수록 달 %s · 활성화 창 %s · 사후 %d개월을 못 채우는 후보 달 %d개",
                         src, mi_to_ym(last), "~".join(act_p["window"]), act_p["ratio_months"], short)
                if act_p.get("clip_to_data") and last < ym_to_mi(act_p["window"][1]):
                    act_p["window"] = kepcoloader.clip_window(act_p["window"], last)
                    log.warning("activation.clip_to_data: 창 끝을 수록 마지막 달로 줄임 → %s", "~".join(act_p["window"]))
            mh, rate, unmatched, fail = kepcoloader.attach_bjd(raw, master, P["bjd"]["fail_warn_rate"], S["crosswalk"])
            writer.table(rate, "s0_bjd_match_rate", "0단계 법정동 매칭률 (시도+시군구+읍면동 3단 매칭)",
                         note=f"실패율 {fail:.1%} — {P['bjd']['fail_warn_rate']:.0%} 이상이면 행정동 기준 의심" +
                              (" · KEPCO_002 잠정 결과" if preliminary else ""))
            unmatched_png = pd.DataFrame([{"항목": "미매칭 지역 수", "값": len(unmatched),
                                           "비고": "상세 목록은 센터 내부 CSV에서만 검토"}])
            writer.table(unmatched, "s0_bjd_unmatched", "0단계 미매칭 목록 (사람 검토 필요)",
                         png_df=unmatched_png,
                         note="CSV는 센터 내부 검토용. PNG에는 원천 지역명을 싣지 않음")
            daily = kepcoloader.daily_series(mh)
            monthly = daily.groupby("mi").agg(법정동수=("bjd_code", "nunique"),
                                              일평균충전량_합_kWh=("kwh_per_day", "sum")).reset_index()
            # R2: 전국 합계표는 bjd_code 열이 없어 _kepco_export 행 마스킹이 못 걸린다.
            # 대신 소표본 법정동의 기여분 자체를 PNG 집계에서 빼, 다른 표와의 차감 역산을 막는다.
            basis = P["kepco"].get("suppress_basis", "max")
            min_count = P["can"]["min_cell_count"]
            cell_rep = _period_customer_count(mh, "cust_min", basis, ("bjd_code", "mi"))
            daily_visible = _exclude_small_cells(daily, cell_rep, min_count, "s1_kepco_monthly")
            monthly_png = (daily_visible.groupby("mi").agg(
                법정동수=("bjd_code", "nunique"), 일평균충전량_합_kWh=("kwh_per_day", "sum"))
                .reindex(monthly["mi"]).fillna(0).reset_index())
            writer.table(_ym_col(monthly), "s1_kepco_monthly", f"1단계 KEPCO_{src} 월별 집계 (법정동 합)", digits=1,
                         png_df=_ym_col(monthly_png),
                         note=f"PNG는 그 달 대표 고객호수(시간별 {basis}) < {min_count}인 법정동의 기여분 제외" +
                              (" · KEPCO_002 잠정 결과, 001 확보 후 재실행" if preliminary else ""))
            if ind.get("tizo_bands"):
                band = kepcoloader.band_series(mh, ind["tizo_bands"])
                wide = band.groupby(["mi", "band"])["kwh_per_day"].sum().unstack(fill_value=0)
                share = wide.div(wide.sum(axis=1), axis=0).add_prefix("구간비_").reset_index()
                band_visible = _exclude_small_cells(band, cell_rep, min_count, "s1_kepco_band_share")
                wide_png = (band_visible.groupby(["mi", "band"])["kwh_per_day"].sum()
                            .unstack(fill_value=0).reindex(index=wide.index, columns=wide.columns, fill_value=0))
                share_png = wide_png.div(wide_png.sum(axis=1), axis=0).add_prefix("구간비_").reset_index()
                writer.table(_ym_col(share), "s1_kepco_band_share", "1단계 시간대구간(TIZO)별 충전량 비중",
                             png_df=_ym_col(share_png),
                             note=f"PNG는 그 달 대표 고객호수(시간별 {basis}) < {min_count}인 법정동의 기여분 제외")
            S.update(master=master, mh=mh, mh_source=src, daily=daily, fail_rate=fail, cell_rep=cell_rep,
                     preliminary_source=preliminary)
        runner.run("0·1", "지역키 정합 + KEPCO 시계열 집계", stage01, CRITICAL)
        if stop_after_stage1:
            log.info("현장 1차 실행: 0·1단계 완료 후 정상 종료")
            raise _StopAfterStage1

        def load_mh(src):
            """정합이 끝난 KEPCO 월×시각 표. 0·1단계가 읽은 원천이면 다시 읽지 않는다."""
            if S["mh_source"] == src:
                return S["mh"]
            raw = kepcoloader.load_kepco_monthly_hour(paths[f"kepco_{src}"], C["kepco" if src == "001" else "kepco_cpo"],
                                                       P["kepco"], f"KEPCO_{src}")
            return kepcoloader.attach_bjd(raw, S["master"], P["bjd"]["fail_warn_rate"], S.get("crosswalk"))[0]

        # ------------------------------------------------ 1-C단계 (격리): KEPCO_002 채널 비교(v11, 합산 금지)
        if paths["kepco_002"]:
            def stage1c():
                mh001 = load_mh("001")
                cells = channel_share(mh001, load_mh("002"))
                min_count = P["can"]["min_cell_count"]
                rep001 = _period_customer_count(mh001, "cust_min", P["kepco"].get("suppress_basis", "max"),
                                                ("bjd_code", "mi"))
                visible = _exclude_small_cells(cells, rep001, min_count, "s1_channel_share")
                writer.table(channel_share_table(cells), "s1_channel_share",
                             "1-C KEPCO_002/001 시간대별 비율 분포 (합산하지 않음)", digits=3,
                             png_df=channel_share_table(visible) if len(visible) else channel_share_table(cells).iloc[0:0],
                             note=f"법정동×월 셀 기준. 002>001 셀이 많으면 포함관계(002⊆001) 의심. "
                                  f"PNG는 KEPCO_001 대표 고객호수 < {min_count}인 셀 제외")
            runner.run("1-C", "KEPCO_002 채널 비교", stage1c, ISOLATED)

        # ------------------------------------------------ 2단계
        def stage2():
            act = kepcoloader.detect_activation(S["daily"], P["activation"])
            cols = ["bjd_code", "status", "T_r", "ratio", "T_simple", "agree_simple", "n_changepoints", "n_months"]
            writer.table(act[[c for c in cols if c in act]], "s2_activation",
                         f"2단계 {kepcoloader.activation_label(P['kepco']['source'], P['activation'].get('source_label'))} 시점 T_r "
                         f"({act.attrs.get('method')}, 창 {'~'.join(P['activation']['window'])})")
            counts = act["status"].value_counts().rename_axis("상태").reset_index(name="법정동수")
            writer.table(counts, "s2_status_counts", "2단계 처치 상태별 법정동 수")
            if (act["status"] == "treated").any():
                # R2: 그림은 법정동명·수치가 그대로 드러나므로 예시 후보에서 소표본 법정동을 뺀다.
                basis = P["kepco"].get("suppress_basis", "max")
                rep = _period_customer_count(S["mh"], "cust_min", basis)
                eligible, n_excluded = _exclude_small_activation(act, rep, P["can"]["min_cell_count"])
                if n_excluded:
                    log.info("s2_activation_examples: 소표본 법정동 %d곳 예시에서 제외", n_excluded)
                if (eligible["status"] == "treated").any():
                    writer.figure(plot_activation_examples(S["daily"], eligible), "s2_activation_examples")
                else:
                    log.warning("s2_activation_examples: 소표본 제외 후 남은 처치 지역 없음 — 그림 생략")
            S["activation"] = act
        if commerce:
            runner.run("2", "변화점 탐지", stage2, CRITICAL)

        # ------------------------------------------------ 3단계 (격리)
        def stage3_centroids():
            if not paths["emd_centroids"]:
                raise FileNotFoundError("paths.emd_centroids 없음 — CAN·KEP_007 좌표를 법정동으로 보낼 수 없음")
            cent = keep_region(bjdmapping.load_emd_centroids(paths["emd_centroids"], C["centroid"]), prefix)
            cent["bjd_code"] = bjdmapping.canonicalize(cent["bjd_code"], S.get("crosswalk"))
            # 시군구 모드는 읍면동 중심점을 다 남긴다: 좌표→가장 가까운 읍면동→그 시군구 코드로 보내려는 것이다.
            S["centroids"] = cent if bjdmapping.ANALYSIS_LEVEL == "sigungu" else cent.drop_duplicates("bjd_code")
        runner.run("3", "법정동 중심점", stage3_centroids, ISOLATED)

        def stage3_kep007():
            if not paths["kep007"]:
                raise FileNotFoundError("paths.kep007 없음")
            st = kep007loader.load_kep007(paths["kep007"], C, P["kep007"], S.get("centroids"),
                                           P["can"]["centroid_max_km"], P["can"]["lat_range"], P["can"]["lon_range"])
            stock = kep007loader.region_stock(st)
            writer.table(stock, "s3_kep007_stock", "3단계 KEP_007 법정동별 인프라 스톡 (2019 기준 정적 스냅샷)", digits=0)
            S.update(stations=st, infra_stock=stock)
        runner.run("3", "KEP_007 설치현황", stage3_kep007, ISOLATED)

        def stage3_can():
            if not P["can"]["enabled"] or not paths["can_m"]:
                raise RuntimeError("CAN 비활성 또는 paths.can_m 없음")
            # 교차검증용 KEPCO 는 s1 PNG 와 같은 기준으로 소표본 법정동×월을 뺀다.
            kepco_visible = _exclude_small_cells(S["mh"], S["cell_rep"], P["can"]["min_cell_count"], "s3_can_kepco") \
                if "mh" in S else None
            res = canloader.run_can_stage(paths["can_m"], C, P["can"], S.get("centroids"), S.get("stations"),
                                           S.get("activation"), P["heterogeneity"]["can_feature_before"],
                                           kepco_mh=kepco_visible)
            writer.table(res["evidence"], "s3_can_join_key", f"3단계 CAN 식별번호 검증 → {res['verdict']} · 경로 {res['path']}")
            writer.table(res["duration_summary"], "s3_can_session_summary", "3단계 충전 세션 길이 요약 (20~40분 체류 전제 검증)")
            writer.table(res["duration_table"], "s3_can_session_dist", "3단계 충전 세션 길이 분포")
            writer.figure(plot_histogram(res["duration_table"], f"충전 세션 길이 분포 — 경로 {res['path']}", "세션 길이"),
                          "s3_can_session_hist")
            if "duration_by_type" in res:
                writer.table(res["duration_by_type"], "s3_can_session_by_type", "3단계 거점성/공용 세션 길이 비교 (경로 A)")
            title = "3단계 법정동별 충전 거점성 (경로 A)" if res["verdict"] == "individual" \
                else "3단계 법정동별 충전 위치 밀집도 (경로 B — 정황상 보조 근거, 개별 차량 식별 아님)"
            writer.table(res["region_table_export"], "s3_can_region", title,
                         note=f"소표본(<{P['can']['min_cell_count']}) 억제 '—'")
            writer.table(res["spatial"], "s3_can_spatial_agreement", "3단계 CAN 충전 위치 ↔ KEP_007 공간 일치율")
            if res["treated_excl_base"] is not None:
                writer.table(res["treated_excl_base"], "s3_treated_excl_base", "3단계 거점성 제외 버전 처치 지역 목록")
            if res.get("kepco_check") is not None:
                writer.table(res["kepco_check"], "s3_can_kepco_check", "3단계 CAN × KEPCO_001 교차검증 (겹치는 기간)",
                             note=f"KEPCO 소표본 법정동×월 제외 · 공간 비교는 CAN 세션 {P['can']['min_cell_count']}건 이상 법정동만")
                if res["kepco_shape"] is not None:
                    writer.table(res["kepco_shape"], "s3_can_kepco_shape",
                                 "3단계 시간대별 비중 — CAN 충전 점유 vs KEPCO 사용량 (서울 전체)", digits=4)
            min_n = P["can"]["min_cell_count"]
            writer.table(res["away_summary"], "s3_can_away_summary", "3단계 CAN 원정 충전 요약 (거주지 밖 충전)",
                         note="거주지는 밤 주차 위치로 추정(센터 내부 계산). 차량 단위 결과는 저장·반출하지 않음")
            if res["away_region"] is not None:
                writer.table(res["away_region"], "s3_can_away_region", "3단계 거주 법정동별 원정 충전",
                             digits=3, png_df=res["away_region_export"], note=f"PNG 는 거주 추정 차량 {min_n}대 미만 법정동 '—'")
            writer.table(res["flex_summary"], "s3_can_flex_summary", "3단계 피크 시간대 충전의 시작 SOC — 시간 이동 여지")
            writer.table(res["flex_bands"], "s3_can_flex_bands", "3단계 피크 시간대 충전 시작 SOC 분포", digits=3,
                         note=f"세션 {min_n}건 미만 구간 '—'")
            S["can"] = res
        can_name = "CAN 처치 정제" if commerce else "CAN 세션 모양·반복 위치"
        if runner.run("3", can_name, stage3_can, ISOLATED) is None and "can" not in S:
            reason = runner.rows[-1]["비고"]
            log.warning("CAN 처리 생략됨 — 3단계 CAN 리포트 섹션은 비워 둔다 (%s)", reason)
            skipped = pd.DataFrame([{"항목": "CAN 처리", "상태": "생략됨", "사유": reason}])
            writer.table(skipped, "s3_can_skipped", "3단계 CAN 처리 생략됨",
                         png_df=skipped.assign(사유=str(reason).split(":")[0] + " — 상세는 내부 CSV"))

        # ------------------------------------------------ 4단계
        def stage4():
            scan = identification.scan_shc002(paths["shc002"], C, ind, P["shc"],
                                              snapshot_months=P["heterogeneity"]["snapshot_months"],
                                              crosswalk=S.get("crosswalk"))
            panel, skipped = identification.build_ydd(scan["cells"], ind, P["identification"]["use_log1p"])
            writer.table(_ym_col(panel), "s4_ydd_panel", f"4단계 Y_ddd 패널 (위약: {panel.attrs['placebo_mode']})")
            if len(skipped):
                summary = skipped.assign(사유유형=skipped["사유"].str.split(":").str[0]).groupby(
                    ["outcome", "사유유형"]).size().reset_index(name="셀수")
            else:
                summary = pd.DataFrame(columns=["outcome", "사유유형", "셀수"])
            writer.table(summary, "s4_skipped_summary", "4단계 Y_ddd 스킵 셀 사유별 집계")
            writer.table(_ym_col(skipped) if len(skipped) else skipped, "s4_skipped_cells", "4단계 Y_ddd 스킵 셀 목록")
            st = scan["stats"]
            quality = pd.DataFrame([{"항목": k, "값": (", ".join(map(str, v)) if isinstance(v, list) else v)}
                                    for k, v in st.items()])
            writer.table(quality, "s4_shc002_quality", "4단계 SHC002 품질 통계")
            S.update(scan=scan, panel=panel)
        if commerce:
            runner.run("4", "Outcome 축약(Y_ddd)", stage4, CRITICAL)

        # ------------------------------------------------ 4.5단계 (격리)
        def stage45():
            mp = P["mde"]
            mde = identification.estimate_mde(
                S["panel"], S["activation"], P["identification"], mp["repetitions"], mp["seed"],
                P["activation"]["window"], mp["bias_se_multiple"])
            writer.table(mde, "s45_mde", "4.5단계 검출 가능 최소효과(MDE)",
                         note="MDE=2.8×경험 SE. 처치 수는 실제 처치 수를 가정")
            S["mde"] = mde
        if commerce:
            runner.run("4.5", "검출 가능 최소효과(MDE)", stage45, ISOLATED)

        # ------------------------------------------------ 5단계
        def stage5():
            panel = S["panel"]
            units = set(panel.loc[panel["y_main"].notna(), "bjd_code"]) & set(panel.loc[panel["y_placebo"].notna(), "bjd_code"])
            ip = P["identification"]
            main = identification.run_event_study(panel, "y_main", S["activation"], ip, units)
            writer.table(identification.event_table(main), "s5_event_main", "5단계 이벤트 스터디 tau(k) — 주 결과",
                         note=f"추정량: 충전 활동 활성화 시점의 외지 대기소비 매출 변화(충전소 설치 효과가 아님) · 처치 {main['n_treated']} · 대조 {main['n_control']} · {ip['control_group']}"
                              + (f" · 폴백 사유: {main['fallback_reason']}" if main["fallback_reason"] else ""))
            if len(main["att_gt"]):
                writer.table(main["att_gt"], "s5_att_gt", "5단계 ATT(g,t) 원표")
            pre = main["pretrend"]
            writer.table(pd.DataFrame([{"검정": "사전추세 결합 Wald (k<=-2)", "통계량": pre["stat"], "자유도": pre["df"],
                                        "p": pre["p"], "판정": ("유의 — 평행추세 의심" if pre["p"] < ip["alpha"] else "기각 못함")
                                        if np.isfinite(pre["p"]) else "검정불가(사전기간 부족)"}]),
                         "s5_pretrend_overall", "5단계 사전 추세 검정")
            unit_pre = main["unit_pretrend"]
            writer.table(unit_pre, "s5_pretrend_region", "5단계 법정동별 사전 추세 (표시만 · 제외하지 않음, 지역별 검정은 검정력이 낮음)")
            flagged = set(unit_pre.loc[unit_pre["판정"].str.startswith("유보"), "bjd_code"]) if len(unit_pre) else set()
            if ip.get("pretrend_action", "flag") != "flag":
                # 사전추세 검정으로 표본을 고르면 이후 추론이 왜곡된다(Roth 2022). 지역별 검정은 검정력도 낮아 표시만 한다.
                log.warning("params.identification.pretrend_action=%r 는 지원하지 않음 — 지역 제외 옵션은 삭제됨. flag(표시만)로 진행",
                            ip["pretrend_action"])
            if main["estimator"].startswith("Callaway"):
                try:
                    reg = identification.regression_event_study(panel, "y_main", S["activation"], ip, units)
                    writer.table(identification.event_table(reg), "s5_event_regression", "5단계 강건성 — 상대시점 더미 회귀")
                    S["regression"] = reg
                except Exception as exc:
                    log.warning("강건성 회귀 실패(주 결과에는 영향 없음): %s", exc)
            S.update(main=main, units=units, pretrend_flagged=flagged)
        if commerce:
            runner.run("5", "이벤트 스터디", stage5, CRITICAL)

        # ------------------------------------------------ 6단계
        def stage6():
            main = S["main"]
            ip = dict(P["identification"])
            ip["estimator"] = "cs" if main["estimator"].startswith("Callaway") else "regression"
            plc = identification.run_event_study(S["panel"], "y_placebo", S["activation"], ip, S["units"])
            by_k, post = identification.discount_by_placebo(main, plc)
            writer.table(identification.event_table(plc), "s6_event_placebo", "6단계 위약(대조 업종) tau(k)")
            writer.table(by_k, "s6_discount_by_k", "6단계 위약 할인 — k별 할인 전/후 (둘 다 보고)")
            writer.table(post, "s6_post_summary", "6단계 사후 평균효과 — 할인 전 · 위약 · 할인 후",
                         note="판정 기준 = 위약이 0과 구분되는가(p<0.05이면 주 결과 해석 유보). 할인 후 값은 참고이며 CI 는 부트스트랩 백분위수")
            adj = by_k.rename(columns={"tau_할인후": "tau"}).assign(
                ci_lo=lambda d: d["tau"] - 1.96 * d["se_할인후"], ci_hi=lambda d: d["tau"] + 1.96 * d["se_할인후"])
            writer.figure(plot_event_study({"주 결과(할인 전)": main["event"], "위약(대조 업종)": plc["event"],
                                            "할인 후": adj[["k", "tau", "ci_lo", "ci_hi"]]},
                                           f"5·6단계 이벤트 스터디 — {main['estimator']}"), "s5_s6_event_study")
            S.update(placebo=plc, post_summary=post)
        if commerce:
            runner.run("6", "위약 검정", stage6, CRITICAL)

        # ------------------------------------------------ 6.5단계 (격리)
        def stage65():
            if not paths["shc001"]:
                raise FileNotFoundError("paths.shc001 없음")
            m = diagnostics.load_shc001_monthly(paths["shc001"], C, ind["shc001"], P["shc"]["chunksize"],
                                                crosswalk=S.get("crosswalk"), sido=P["shc"].get("sido"))
            cont = diagnostics.contamination_check(m, S["activation"], P["diagnostics"])
            writer.table(cont, "s65_contamination", "6.5단계 처치오염 진단 — T_r 전후 가맹점 개설률",
                         note="유보 = 상권 자체 성장 가능성. 제외하지 않고 CATE·처방에 표시만 한다")
            S.update(shc001m=m, contamination=cont)
        if commerce:
            runner.run("6.5", "처치오염 진단", stage65, ISOLATED)

        reserved = _reserved(S)

        # ------------------------------------------------ 7단계 (격리)
        def stage7():
            hp = P["heterogeneity"]
            can_feats = S["can"]["features"] if "can" in S else None
            feats = heterogeneity.build_features(S["scan"]["snapshot"], S.get("shc001m"), can_feats, S.get("infra_stock"),
                                                 hp, P["activation"]["window"][0])
            outcomes = heterogeneity.region_outcomes(S["panel"], "y_main", S["activation"], hp["post_months"])
            het = heterogeneity.run_heterogeneity(feats, outcomes, hp)
            writer.table(feats, "s7_features", f"7단계 이질성 변수 (스냅샷 {'~'.join(hp['snapshot_months'])} 고정)")
            cate = het["cate"].assign(유보사유=lambda d: d["bjd_code"].map(reserved).fillna(""))
            notes = [f"Causal Forest 미사용 사유: {het['reason']}"] if het["reason"] else []
            if not het["reportable"]:
                notes.append(f"CATE 미보고 — {het['report_reason']}")
            writer.table(cate, "s7_cate_region", f"7단계 법정동별 CATE — {het['method']}", note=" · ".join(notes))
            writer.table(het["validation"], "s7_cate_validation", "7단계 CATE 검증 — 같은 교차검증 분할에서 방법 비교",
                         note=f"변환결과 MSE 낮을수록 · GATES 클수록 좋음 · BLP 기울기>0 이고 p<={P['heterogeneity']['blp_alpha']}여야 CATE 보고")
            if het["importance"] is not None:
                writer.table(het["importance"], "s7_importance", "7단계 Causal Forest 변수 중요도")
            if het["cells"] is not None:
                writer.table(het["cells"], "s7_subgroup_cells", "7단계 사전지정 2x2 서브그룹 CATE")
            S["het"] = het
        if commerce:
            runner.run("7", "CATE", stage7, ISOLATED)
        else:
            runner.rows.append({"단계": "2·4~7", "이름": "상권 단계(T_r·Y_ddd·이벤트 스터디·위약·처치오염·CATE)",
                                "상태": "비활성", "소요초": 0.0, "비고": "상권 단계 비활성(v11)"})
            log.info("상권 단계 비활성(v11) — 2·4·4.5·5·6·6.5·7단계 건너뜀")

        # ------------------------------------------------ 8단계 (격리)
        def stage8():
            mh001 = load_mh("001")
            conc = loadaxis.compute_concentration(mh001, P["load_axis"]["period"])
            p0, p1 = (ym_to_mi(v) for v in P["load_axis"]["period"])
            # R1: 최솟값 대신 기간 내 시간별 고객호수의 최댓값을 기본으로 쓴다(suppress_basis).
            customer_min = _period_customer_count(
                mh001.loc[mh001["mi"].between(p0, p1)], "cust_min", P["kepco"].get("suppress_basis", "max"))
            conc_csv, conc_png = _kepco_export(conc, customer_min, P["can"]["min_cell_count"],
                                               ["concentration", "peak_avg_kw", "daily_kwh"])
            writer.table(conc_csv, "s8_load_concentration", "8단계 부하 집중도",
                         png_df=conc_png,
                         note="avg_kw = 1시간 kWh ÷ 1h 의 일평균 = 구간 평균 kW (순간 최대전력 아님)")
            types, conc_cut = loadaxis.classify_load(conc, P["load_axis"])
            types_csv, types_png = _kepco_export(types, customer_min, P["can"]["min_cell_count"],
                                                 ["concentration", "peak_avg_kw", "daily_kwh"])
            writer.table(types_csv, "s8_load_type", "8단계 부하 집중도 구분과 공공 검토 유형", png_df=types_png,
                         note=f"집중도 기준 {conc_cut:.3f}({P['load_axis']['conc_cut']}). 집중도는 점검 신호이며 배전망 위험도가 아님")
            type_counts = types.groupby("load_group").size().rename("법정동수").reset_index()
            writer.table(type_counts, "s8_load_type_counts", "8단계 부하 집중도 구분별 법정동 수")
            writer.figure(plot_bar(type_counts, "load_group", "법정동수", "부하 집중도 구분별 법정동 수", "법정동 수"),
                          "s8_load_type_bar")
            S["load_types"] = types
            if not commerce:
                return
            if "het" not in S:
                raise RuntimeError("7단계 CATE 결과가 없어 4사분면 처방을 만들 수 없음")
            quad, cc, kc = loadaxis.classify_quadrants(S["het"]["cate"], conc, P["load_axis"], reserved)
            export = quad.drop(columns=["reserved"])
            export_csv, export_png = _kepco_export(
                export, customer_min, P["can"]["min_cell_count"],
                ["concentration", "peak_avg_kw", "daily_kwh"])
            writer.table(export_csv, "s8_quadrants", "8단계 4사분면 처방 표", png_df=export_png,
                         note="유보사유가 있는 지역은 해석 유보" + (" · CATE 미보고: 부하 집중도만으로 분류" if not quad["cate"].notna().any() else ""))
            counts = quad.groupby("quadrant").agg(법정동수=("bjd_code", "size"), 유보수=("reserved", "sum")).reset_index()
            writer.table(counts, "s8_quadrant_counts", "8단계 사분면별 법정동 수")
            if quad["cate"].notna().any():
                writer.figure(plot_quadrants(quad, cc, kc), "s8_quadrants_plot")
            writer.figure(plot_bar(counts, "quadrant", "법정동수", "사분면별 법정동 수", "법정동 수"), "s8_quadrant_counts_bar")
            S["quadrants"] = quad
        runner.run("8", "전력축·처방" if commerce else "전력축·공공 검토 유형", stage8, ISOLATED)

        # ------------------------------------------------ 8-C단계 (격리)
        if P["priority"]["enabled"]:
            def stage8c():
                if bjdmapping.ANALYSIS_LEVEL == "sigungu":
                    raise RuntimeError("analysis_level=sigungu: 300/500/800m 2SFCA와 중심점 근사는 시군구 규모에서 의미가 없어 생략(8-E도 생략)")
                from_history = not paths["ev_registration"] and paths["ev_history"] and paths["hdong_bjd"]
                if not paths["access_stations"] or not (paths["ev_registration"] or from_history):
                    raise FileNotFoundError("paths.access_stations 또는 전기차 등록(ev_registration, 또는 ev_history+hdong_bjd) 없음")
                if "centroids" not in S:
                    raise RuntimeError("법정동 중심점 없음 — 2SFCA 계산 불가")
                ev_counts = None
                if from_history:
                    # v11: 반입 파일을 줄이려고 8-F 등록 이력의 기준월 값을 행정동→법정동 대응표로 배분해 쓴다.
                    sp = P["scenario"]
                    hist = loadscenario.load_ev_history(paths["ev_history"], C["ev_history"], sp["fuel_value"], prefix)
                    ev, _ = loadscenario.map_to_bjd(hist, loadscenario.load_hdong_bjd(paths["hdong_bjd"], C["hdong_bjd"]),
                                                    S.get("crosswalk"), sp["base_month"])
                    ev_counts = ev.loc[ev["ym"] == sp["base_month"], ["bjd_code", "ev_count"]]
                    log.info("8-C 전기차 대수: 등록 이력 %s 값(행정동→법정동 배분) %d곳", sp["base_month"], len(ev_counts))
                stations, points = equityaccess.load_access_inputs(
                    paths["access_stations"], paths["ev_registration"], C, S["centroids"], S.get("crosswalk"), ev_counts)
                access = equityaccess.compute_2sfca(
                    stations, points, P["priority"]["radii_m"], P["priority"]["default_radius_m"])
                access = equityaccess.add_access_indicators(access, stations, points)
                writer.table(access, "s8c_accessibility", "8-C 법정동별 2SFCA 접근성과 형평성 부족도",
                             note="2SFCA가 낮을수록 equity_need_norm은 높음. 지표 1·2는 법정동 중심점 근사(격자점·폴리곤 아님). 실제 취약계층 규모가 아님")
                S.update(access=access, access_match_rate=float(points["ev_count"].notna().mean()),
                         access_inputs=(stations, points))
            runner.run("8-C", "충전 접근성 2SFCA", stage8c, ISOLATED)

            def stage8c_can():
                """고유 데이터(CAN)의 원정 충전으로 형평성 축(2SFCA)을 검증한다."""
                if "access" not in S or S.get("can", {}).get("away_region") is None:
                    raise RuntimeError("8-C 접근성 또는 CAN 원정 충전(개별 차량 판정 필요) 없음")
                check = canloader.away_access_check(S["can"]["away_region"], S["access"], P["can"]["min_cell_count"])
                writer.table(check, "s8c_can_away_check", "8-C CAN 원정 충전 ↔ 2SFCA 접근성 검증",
                             note=f"거주 추정 차량 {P['can']['min_cell_count']}대 이상 법정동만. 기간이 달라 방향 확인용")
                S["can_away_check"] = check
            runner.run("8-C", "CAN 원정 충전 ↔ 접근성 검증", stage8c_can, ISOLATED)

        if P["energy"]["enabled"]:
            def stage_external():
                """공휴일 달력을 8-A 달력에 넣고 기온 입력을 점검한다. 예보는 전날 마감 이전 발표분만 남긴다."""
                wp = P["weather"]
                holidays = weatherloader.load_holidays(paths["holidays"], P["holidays"])
                if holidays:
                    P["energy"]["holidays"] = {**holidays, **(P["energy"]["holidays"] or {})}
                kinds = pd.Series(holidays, dtype=object).value_counts()
                rows = [{"항목": "공휴일 달력", "값": f"{len(holidays)}일" if holidays else "없음 — 공휴일 미보정",
                         "비고": ", ".join(f"{k} {v}" for k, v in kinds.items())}]
                mode = wp["mode"]
                if paths["weather"]:
                    weather = weatherloader.load_weather(paths["weather"])
                    fcst = weatherloader.forecast_temps(weather, wp["fcst_cutoff"])
                    n_fcst = int(weather["temp_fcst_c"].notna().sum())
                    in_time = int((weather["temp_fcst_c"].notna() & (weather["fcst_issued_at"] <= weatherloader.fcst_cutoff(
                        weather["timestamp"], wp["fcst_cutoff"]))).sum())
                    span = f"{weather['timestamp'].min():%Y-%m-%d}~{weather['timestamp'].max():%Y-%m-%d}"
                    rows += [{"항목": "기온 지점 수", "값": weather["station_or_grid"].nunique(), "비고": "서울 공통 기온에 가까움"},
                             {"항목": "기온 기간", "값": span, "비고": "결측 시간 보간 안 함"},
                             {"항목": "예보 사용 가능 시각 수", "값": len(fcst),
                              "비고": f"전날 {wp['fcst_cutoff']} 이전 발표분 중 최신. 마감 뒤 발표 {n_fcst - in_time}행 제외"}]
                    if mode == "forecast" and fcst.empty:
                        log.warning("기온 예보 없음 — weather.mode forecast → none 폴백")
                        mode = "none"
                    S["weather"] = weather
                elif mode != "none":
                    log.warning("paths.weather 없음 — weather.mode %s → none 폴백", mode)
                    mode = "none"
                rows.append({"항목": "기온 모드", "값": mode, "비고": "observed(실측)는 상한 참고용 — 발표 숫자에 쓰지 않음"})
                S["weather_mode"] = mode
                writer.table(pd.DataFrame(rows), "s0_external_inputs", "외부 입력 점검 — 공휴일 달력·기온")
            runner.run("0-E", "외부 입력(공휴일·기온)", stage_external, ISOLATED)

            # 8-A/8-B는 일자 보존 원자료를 별도로 읽는다. 기존 월 집계를 일별로 복제하지 않는다.
            def stage8a():
                ep = P["energy"]
                _warn_energy_calendar(ep)
                date_start, date_end = _energy_load_window(ep, P["forecast"]["train_days"])
                hourly_params = dict(P["kepco"], date_start=str(date_start.date()), date_end=str(date_end.date()))
                hourly = kepcoloader.load_kepco_hourly(paths["kepco_hourly"] or paths["kepco_001"],
                                                       C["kepco"], hourly_params, S["master"], S.get("crosswalk"))
                # 공학 평가 대상도 평가 시작 전 교정자료로 고정한다(CATE 크기로 제외하지 않음).
                start = pd.Timestamp(ep["evaluation_start"])
                prior = hourly[hourly["date"].between(start - pd.Timedelta(days=ep["calibration_days"]),
                                                    start - pd.Timedelta(days=1))]
                complete = []
                for code in prior["bjd_code"].unique():
                    matrix = loadforecast.daily_matrix(prior, code)
                    if len(matrix) == ep["calibration_days"] and not matrix.isna().any().any():
                        profile = matrix.mean(axis=0)
                        complete.append({"bjd_code": code, "concentration": profile.max() / profile.sum()
                                         if profile.sum() > 0 else 0.0})
                    else:
                        log.warning("8-A %s: 교정기간 결측 — 대상선정 보류", code)
                conc = pd.DataFrame(complete, columns=["bjd_code", "concentration"])
                cutoff = loadaxis._cut(conc["concentration"], P["load_axis"]["conc_cut"]) if len(conc) else np.nan
                regions = conc.loc[conc["concentration"] > cutoff, "bjd_code"].tolist()
                if not regions:
                    raise ValueError("고집중도 지역 없음 — 대상 기준 확인 필요")
                rows = []
                for code in regions:
                    try:
                        rows.append(loadforecast.backtest_forecast(
                            hourly, code, ep["holdout_weeks"], ep["weeks"], ep["holidays"],
                            end_date=pd.Timestamp(ep["evaluation_start"]) - pd.Timedelta(days=1)))
                    except ValueError as exc:
                        log.warning("8-A %s 검증 실패: %s", code, exc)
                        rows.append({"bjd_code": code, "error": str(exc)})
                validation = pd.DataFrame(rows)
                # R1: 최솟값 대신 기간 내 시간별 고객호수의 최댓값을 기본으로 쓴다(suppress_basis).
                customer_min = _period_customer_count(hourly, "cust", P["kepco"].get("suppress_basis", "max"))
                validation_csv, validation_png = _kepco_export(
                    validation, customer_min, P["can"]["min_cell_count"],
                    ["mae_naive_seasonal", "mae_persistence", "bias_actual_minus_forecast", "growth_bias", "peak_mae"])
                writer.table(validation_csv, "s8a_validation", "8-A 과거 일별 부하예측 검증",
                             png_df=validation_png,
                             note="증가 지역 bias 양수=하향 추정. 검증 종료는 ESS 평가 시작 이전")
                S["energy_input"] = (hourly, regions, validation)
                S["energy_customer_min"] = customer_min
            runner.run("8-A", "기준 모델(4주 중앙값) 검증", stage8a, ISOLATED)

            def stage8a_ai():
                """v11 AI 분위수 예측(모든 서울 법정동) + 증설 0대 급증 위험. 8-C 동네 특성이 없으면 빼고 학습한다."""
                if "energy_input" not in S:
                    raise RuntimeError("8-A 시간별 원자료 없음")
                ep, hourly = P["energy"], S["energy_input"][0]
                mode = S.get("weather_mode", "none")
                weather = S.get("weather")
                station_map = None
                if weather is not None:
                    if "centroids" in S:
                        station_map = weatherloader.assign_stations(weather, S["centroids"])
                    elif weather["station_or_grid"].nunique() == 1:
                        station_map = pd.DataFrame({"bjd_code": hourly["bjd_code"].astype(str).unique(),
                                                    "station_or_grid": weather["station_or_grid"].iloc[0]})
                if station_map is None and mode != "none":
                    log.warning("기온 지점을 법정동에 배정할 수 없음 — weather.mode %s → none", mode)
                    mode = "none"
                if "access" not in S:
                    log.warning("8-C 결과 없음 — 8-A는 동네 특성(전기차·충전기 수) 없이 학습")
                ai = loadforecast.run_ai_forecast(hourly, dict(ep, forecast=P["forecast"]), ep["holidays"],
                                                  S.get("access"), weather, station_map, mode,
                                                  P["weather"]["fcst_cutoff"])
                cm, min_count = S["energy_customer_min"], P["can"]["min_cell_count"]
                visible = set(cm[cm >= int(min_count)].index.astype(str))
                m_csv = ai["metrics"].assign(소표본억제=lambda d: d["bjd_code"].ne("") & ~d["bjd_code"].isin(visible))
                # R2: 전체 행도 소표본 법정동 기여분을 뺀 검증 셀로 다시 계산해 PNG에 싣는다(차감 역산 방지).
                shown = ai["valid"][ai["valid"]["bjd_code"].isin(visible)]
                m_png = loadforecast.evaluate_forecasts(shown, ai["model_used"], ai["weather_mode"],
                                                        P["forecast"]["min_p90_coverage"]) if len(shown) else m_csv.iloc[0:0]
                writer.table(m_csv, "s8a_forecast_metrics", f"8-A 예측 검증 — persistence·기준 모델·AI({ai['model_used']})",
                             png_df=m_png, digits=3,
                             note=f"검증 {ai['valid_days'][0]}~{ai['valid_days'][1]}(평가 시작 전), 학습 {ai['train_days'][0]}~"
                                  f"{ai['train_days'][1]}. 기온 {ai['weather_mode']}. bjd_code 빈칸=전체. "
                                  "AI가 못 이긴 동네도 표시(ai_beats_baseline=False). reference_only=실측 기온 상한 참고")
                start = pd.Timestamp(ep["evaluation_start"]).normalize()
                calib = hourly[hourly["date"].between(start - pd.Timedelta(days=int(ep["calibration_days"])),
                                                      start - pd.Timedelta(days=1))]
                calib_peak = calib.groupby(calib["bjd_code"].astype(str))["kw"].max()
                u, _ = essoptimizer.utilization_profile(ep, S.get("can", {}).get("charging_shape"))
                added = {n: essoptimizer.compute_added_load(u, n, ep["charger_kw"]) for n in ep["new_chargers"] if n > 0}
                risk = loadforecast.surge_risk(ai["eval"], calib_peak, ep["multipliers"], added,
                                               P["risk"]["min_calib_peak_kw"])
                risk_values = [c for c in risk if c.endswith("_kw")]
                r_csv, r_png = _kepco_export(risk, cm, min_count, risk_values)
                title = "8-A 급증 위험 — 교정기간 최대 부하 × 배율을 넘는 급증(변압기 과부하 아님)"
                writer.table(r_csv, "s8a_risk", title, png_df=r_png, digits=3,
                             note="surge_risk=증설 0대 P90 경보일 비율(점수용). 정밀도·재현율은 AI·기준 모델이 모두 있는 공통일만. "
                                  "surge_risk_with_new_N=증설 가정(ΔL) 참고 레이어 — 점수·검증 제외")
                summary = loadforecast.risk_summary(risk)
                writer.table(summary, "s8a_risk_summary", "8-A 급증 경보 정밀도·재현율(AI 대 기준 모델, 상한 배율별)",
                             png_df=loadforecast.risk_summary(risk[risk["bjd_code"].isin(visible)]), digits=3,
                             note="법정동·일 합산(micro). PNG는 소표본 법정동 제외. 기준 모델 실패일은 비교에서 제외(n_excluded_days)")
                S.update(ai=ai, risk=risk, risk_summary=summary, calib_peak=calib_peak)
            runner.run("8-A", "AI 분위수 예측·급증 위험", stage8a_ai, ISOLATED)
            if runner.rows and runner.rows[-1]["이름"] == "AI 분위수 예측·급증 위험" and "ai" in S:
                ok, why = loadforecast.ai_improvement(S["ai"]["metrics"], P["forecast"]["min_p90_coverage"])
                runner.rows[-1]["비고"] = (f"사용 모델 {S['ai']['model_used']} · 기온 {S['ai']['weather_mode']} · "
                                          f"{'AI 개선' if ok else 'AI 개선 없음(킬 18)'}: {why} · "
                                          "mock(합성) 자료에서 AI가 기준 모델을 이겨도 성능 근거가 아님")

            def stage8b():
                if "energy_input" not in S:
                    raise RuntimeError("8-A 입력 없음 — oracle로 예측 성과를 대체하지 않음")
                try:
                    smp = essoptimizer.load_smp(paths["smp"])
                except (OSError, ValueError, KeyError) as exc:
                    log.warning("SMP 로드 실패 — 피크 목적함수: %s", exc)
                    smp = None
                hourly, regions, validation = S["energy_input"]
                shape = S.get("can", {}).get("charging_shape")
                ai_pred = S["ai"]["eval"] if "ai" in S and S["ai"]["model_used"] in ("lightgbm", "sklearn") else None
                results, schedules = essoptimizer.run_scenarios(hourly, regions, dict(P["energy"], ess=P["ess"]),
                                                                validation, smp, shape, ai_pred)
                evidence = None
                if paths["public_evidence"]:
                    evidence = json.loads(Path(paths["public_evidence"]).read_text(encoding="utf-8"))
                priority = essoptimizer.public_priority(results, evidence, P["energy"]["priority_multiplier"])
                note = "정책 상한·후보 용량·SMP 비용 차이 시뮬레이션. 정전 예방/교체비 절감 실증 아님"
                customer_min = S["energy_customer_min"]
                result_values = [c for c in results if c.endswith(("_kw", "_kwh", "_won"))]
                results_csv, results_png = _kepco_export(results, customer_min, P["can"]["min_cell_count"], result_values)
                writer.table(results_csv, "s8b_scenarios", "8-B ESS 시나리오 전체 결과", note=note,
                             png_df=results_png)
                schedule_values = [c for c in schedules if c.endswith(("_kw", "_kwh"))]
                schedules_csv, schedules_png = _kepco_export(
                    schedules, customer_min, P["can"]["min_cell_count"], schedule_values)
                writer.table(schedules_csv, "s8b_schedules", "8-B 계획 및 실측 재현 스케줄",
                             png_df=schedules_png, note="법정동·시간 집계. 개별 차량 자료 없음")
                summary = results.groupby(["mode", "forecast_input", "new_chargers", "multiplier"]).agg(
                    계획가능비율=("plan_feasible", "mean"), 평균초과_kW=("exceedance_kw", "mean"),
                    평균비용차이_원=("energy_cost_difference_won", "mean"), 평균종단잔량오차_kWh=("terminal_error_kwh", "mean")).reset_index()
                writer.table(summary, "s8b_comparison", "8-B oracle / 예측 비교 (동일 후보 용량)", note=note)
                priority_csv, priority_png = _kepco_export(
                    priority, customer_min, P["can"]["min_cell_count"],
                    ["baseline_exceedance_kwh", "residual_exceedance_kwh", "peak_reduction_kw"])
                writer.table(priority_csv, "s9_public_review", "9단계 공공 인프라 잠정 검토표",
                             png_df=priority_png,
                             note="부하 점검 순서이며 공공 투자 확정 순위 아님. 형평성 미확보는 별도 표시")
                effect = essoptimizer.ai_effect(results)
                if len(effect):
                    min_count = P["can"]["min_cell_count"]
                    visible = set(customer_min[customer_min >= int(min_count)].index.astype(str))
                    e_csv, e_png = _kepco_export(effect, customer_min, min_count,
                                                 [c for c in effect if c.startswith(("exceedance_kwh", "over_discharge"))])
                    writer.table(e_csv, "s8b_ai_effect", "8-B AI 효과 — 같은 ESS 용량, 예측 입력만 바꿔 실측에 재현",
                                 png_df=e_png, digits=3,
                                 note="ai_effect = 1 − 초과kWh(AI P90)/초과kWh(기준 모델). 모든 입력의 계획이 있는 공통일만 합산. "
                                      "기준 모델 초과 0이면 대상 제외(ai_effect_eligible=False). 음수여도 그대로 보고")
                    effect_summary = essoptimizer.ai_effect_summary(effect)
                    writer.table(effect_summary, "s8b_ai_effect_summary", "8-B AI 효과 요약(상한 배율·증설 대수별)",
                                 png_df=essoptimizer.ai_effect_summary(effect[effect["bjd_code"].isin(visible)]), digits=3,
                                 note="과잉 방전 kWh = 실측 기준으로 필요 없었던 방전. PNG는 소표본 법정동 제외")
                    S.update(ai_effect=effect, ai_effect_summary=effect_summary)
                S.update(energy_results=results, energy_schedules=schedules, public_priority=priority)
            if P["stages"].get("ess", True):
                runner.run("8-B", "ESS 시나리오 및 공공 검토", stage8b, ISOLATED)

        # ------------------------------------------------ 8-E단계 (격리)
        if P["priority"]["enabled"]:
            def stage8e():
                pp = P["priority"]
                legacy = list(pp["axes"]) == priorityscore.LEGACY_AXES
                if "access" not in S:
                    raise RuntimeError("8-C 접근성 결과 없음")
                if legacy:
                    if "energy_results" not in S:
                        raise RuntimeError("레거시 세 축: 8-B ESS 결과 없음")
                    lp = dict(pp, weights=pp["legacy_weights"], min_axes=pp["legacy_min_axes"])
                    result = priorityscore.build_priority(S["energy_results"], S["access"], lp)
                else:
                    if "risk" not in S:
                        raise RuntimeError("8-A 급증 위험 없음 — 위험 축을 만들 수 없음")
                    if list(pp["axes"]) != list(priorityscore.AXES_V11):
                        raise ValueError(f"priority.axes 는 {list(priorityscore.AXES_V11)} 또는 레거시 {priorityscore.LEGACY_AXES}")
                    result = priorityscore.build_priority_v11(S["risk"], S["access"], pp, S.get("energy_results"),
                                                              S["ai"]["metrics"] if "ai" in S else None)
                attrs = dict(result.attrs)   # merge 뒤에는 attrs가 사라지므로 먼저 꺼낸다
                corr = attrs.get("axes_correlation", {})
                # R3: 8-A 평가창 필터가 걸린 S["energy_input"] 대신, period 열만 가볍게 스캔한
                # 원천 전체 기간의 관측 일자로 계절 커버리지를 판정한다(시간별 원자료 전체를 다시 올리지 않음).
                observed_dates = kepcoloader.scan_observed_dates(
                    paths["kepco_hourly"] or paths["kepco_001"], C["kepco"], P["kepco"])
                coverage = priorityscore.seasonal_coverage(observed_dates)
                result["season_status"] = coverage["season_status"]
                result["missing_seasons"] = coverage["missing_seasons"]
                if "het" in S:
                    cate = S["het"]["cate"][["bjd_code", "cate"]].rename(columns={"cate": "cate_reference"})
                    result = result.merge(cate, on="bjd_code", how="left")
                pool = result[result["rank_eligible"]]
                rest = result[~result["rank_eligible"]]
                axes_label = "안전·경제성" if legacy else "급증위험·형평성"
                log.info("8-E 순위 대상 %d곳 · 순위 밖 %d곳(min_axes=%s) · %s 상관 피어슨 %.4f 스피어만 %.4f",
                         len(pool), len(rest), pp["legacy_min_axes"] if legacy else pp["min_axes"], axes_label,
                         corr.get("pearson", np.nan), corr.get("spearman", np.nan))
                note = ""
                if legacy and corr.get("axes_redundant"):
                    note = "안전·경제성 축 상관 ≥ %.2f: 세 축이 아니라 사실상 두 축" % pp["redundant_corr"]
                    log.warning("8-E %s (피어슨 %.4f, 스피어만 %.4f)", note, corr["pearson"], corr["spearman"])
                if not legacy and attrs.get("risk_metric_used") == "peak_ratio" and pp["risk_metric"] == "auto":
                    note = (f"순위 대상 {attrs['share_zero_surge_risk']:.0%}에서 급증위험 0 → 위험 축을 피크비율로 대체(킬 21번)")
                    log.warning("8-E %s", note)
                cm, min_count = S.get("energy_customer_min", pd.Series(dtype=float)), P["can"]["min_cell_count"]
                score_values = [c for c in result if c != "bjd_code" and pd.api.types.is_numeric_dtype(result[c])
                                and not pd.api.types.is_bool_dtype(result[c])]
                result_csv, result_png = _kepco_export(result, cm, min_count, score_values)
                ranked = result_csv["rank_eligible"].astype(bool)
                writer.table(result_csv[ranked], "s8e_priority", "8-E 공공 인프라 현장 검토 우선순위(순위 대상만)",
                             png_df=result_png[ranked],
                             note=("경제성=운영비 절감 잠재력. CATE=부가 편익 참고값" if legacy else
                                   f"점수 = {pp['weights']} × (급증위험[{attrs.get('risk_metric_used')}, 증설 0대]·형평성). "
                                   "경제성·ESS 미적용 초과 kWh·증설 시나리오 급증위험은 표시용(점수 제외)")
                                  + ". 실제 설치 지점 선정 결과가 아님")
                if not ranked.all():
                    cols = [c for c in ("bjd_code", "access_2sfca", "equity_norm", "risk_status", "PriorityScore",
                                        "axes_present", "missing_axes", "missing_reason", "confidence", "소표본억제")
                            if c in result_csv]
                    writer.table(result_csv.loc[~ranked, cols], "s8e_not_ranked", "순위 밖 — 축 결측(0점 아님)",
                                 png_df=result_png.loc[~ranked, cols],
                                 note="유효 축이 min_axes 미만인 동네. 결측 축은 0점으로 채우지 않으며 결합점수는 참고값")
                if not legacy:
                    for axis, col, asc, title in (("risk", "risk_raw", False, "급증위험"), ("equity", "access_2sfca", True, "형평성(접근성 낮은 순)")):
                        keep = [c for c in ("bjd_code", col, f"{axis}_norm", "소표본억제") if c in result_csv]
                        order = result_csv[keep].dropna(subset=[col]).sort_values([col, "bjd_code"], ascending=[asc, True])
                        png = result_png.loc[order.index, keep]
                        writer.table(order.assign(축별순위=np.arange(1, len(order) + 1)), f"s8e_rank_{axis}",
                                     f"8-E 축별 순위표 — {title}", png_df=png.assign(축별순위=np.arange(1, len(png) + 1)))
                summary = pd.DataFrame([
                    {"항목": "모드", "값": "레거시 세 축" if legacy else "두 축(급증위험·형평성)"},
                    {"항목": "순위 대상 수(ranking_pool)", "값": len(pool)},
                    {"항목": "순위 밖 수", "값": len(rest)},
                    {"항목": "순위 대상 비율", "값": len(pool) / max(1, len(result))},
                    {"항목": "min_axes", "값": pp["legacy_min_axes"] if legacy else pp["min_axes"]},
                    {"항목": f"{axes_label} 피어슨 상관", "값": corr.get("pearson", np.nan)},
                    {"항목": f"{axes_label} 스피어만 상관", "값": corr.get("spearman", np.nan)},
                ] + ([{"항목": "axes_redundant", "값": bool(corr.get("axes_redundant", False))}] if legacy else [
                    {"항목": "위험 축 지표", "값": attrs.get("risk_metric_used")},
                    {"항목": "순위 대상 중 급증위험 0 비율", "값": attrs.get("share_zero_surge_risk")},
                ]))
                writer.table(summary, "s8e_priority_summary", "8-E 순위 풀·축 상관 요약")
                S.update(priority_score=result, season_coverage=coverage, priority_note=note, priority_attrs=attrs)
            runner.run("8-E", "투자 검토 결합점수", stage8e, ISOLATED)
            if runner.rows and runner.rows[-1]["단계"] == "8-E" and runner.rows[-1]["상태"] == "완료":
                runner.rows[-1]["비고"] = S.get("priority_note", "")

            # ------------------------------------------------ 8-G (격리): 충전기 추가 가정 실험
            def stage8g():
                if "priority_score" not in S or "access_inputs" not in S:
                    raise RuntimeError("8-E 순위 또는 8-C 접근성 입력 없음")
                pp = P["priority"]
                n = int(pp["sim_top_n"])
                stations, points = S["access_inputs"]
                ranked = S["priority_score"]
                ranked = ranked[ranked["rank_eligible"].astype(bool)].sort_values("rank")
                by_ev = points.dropna(subset=["ev_count"]).sort_values(["ev_count", "bjd_code"], ascending=[False, True])
                strategies = {f"우선순위 상위 {n}곳": ranked["bjd_code"].astype(str).head(n).tolist(),
                              f"전기차 등록 상위 {n}곳(비교)": by_ev["bjd_code"].astype(str).head(n).tolist()}
                sim = equityaccess.simulate_added_chargers(stations, points, strategies, int(pp["sim_chargers_each"]),
                                                           pp["radii_m"], pp["default_radius_m"])
                writer.table(sim, "s8g_charger_sim", "8-G 충전기 추가 가정 실험 — 같은 기수를 어디에 더하면 격차가 더 줄어드나",
                             digits=3, note=f"대상 동네 중심점에 {pp['sim_chargers_each']}기씩 가상 추가 후 2SFCA({pp['default_radius_m']}m) 재계산. "
                                            "입지 최적화·실제 설치 계획이 아닌 가정 실험")
                S["charger_sim"] = sim
            runner.run("8-G", "충전기 추가 가정 실험", stage8g, ISOLATED)

        # ------------------------------------------------ 8-F단계 (격리·선택): 2028·2030 충전 부하 시나리오
        if paths["ev_history"] and P["energy"]["enabled"] and P["stages"].get("scenario", True):
            def stage8f():
                if not paths["hdong_bjd"]:
                    raise FileNotFoundError("paths.hdong_bjd 없음 — 행정동→법정동 대응표가 있어야 등록 이력을 법정동에 배분")
                if "energy_input" not in S or "calib_peak" not in S:
                    raise RuntimeError("8-A 결과(평가기간 부하·교정기간 최대) 없음")
                sp, ep = P["scenario"], P["energy"]
                hist = loadscenario.load_ev_history(paths["ev_history"], C["ev_history"], sp["fuel_value"], prefix)
                mapping = loadscenario.load_hdong_bjd(paths["hdong_bjd"], C["hdong_bjd"])
                ev, fail = loadscenario.map_to_bjd(hist, mapping, S.get("crosswalk"), sp["base_month"])
                hourly = S["energy_input"][0]
                start = pd.Timestamp(ep["evaluation_start"]).normalize()
                window = hourly[hourly["date"].between(start, start + pd.Timedelta(days=int(ep["evaluation_days"]) - 1))]
                curves = loadscenario.typical_curves(window.assign(bjd_code=window["bjd_code"].astype(str)))
                base_m = float(P["priority"]["base_multiplier"])
                scen, region, beta, notes = loadscenario.run_scenarios_8f(
                    ev, curves, S["calib_peak"], S.get("access"), sp, ep, base_m)
                cm, min_count = S["energy_customer_min"], P["can"]["min_cell_count"]
                visible = set(cm[cm >= int(min_count)].index.astype(str))
                title = loadscenario.SCENARIO_TITLE
                writer.table(scen, "s8f_scenario", f"8-F 2028·2030 충전 부하 — {title}", digits=2,
                             png_df=loadscenario.summarize_scenarios(region[region["bjd_code"].isin(visible)],
                                                                     ep["multipliers"], base_m),
                             note=f"현재 상한(교정기간 최대 × 배율) 초과 동네 수. 매핑 실패율 {fail:.1%}. "
                                  f"β={beta.at[0, 'beta']:.2f}{'(불안정 → 1, 0.8·1.2 민감도)' if beta.at[0, 'fallback'] else ''}. "
                                  "PNG는 소표본 법정동 제외" + (" · " + " · ".join(notes) if notes else ""))
                load_cols = [c for c in region if c.endswith("_kw") or c.startswith("ess_kwh")]
                r_csv, r_png = _kepco_export(region, cm, min_count, load_cols)
                writer.table(r_csv, "s8f_scenario_region", f"8-F 법정동별 시나리오 — {title}", png_df=r_png, digits=2)
                writer.table(beta.assign(map_fail_rate=fail), "s8f_beta",
                             "8-F 탄력성 β(동네 간 로그-로그, 부트스트랩 95%)와 k_goal", digits=4)
                mid = region[(region["year"] == 2030) & (region["scenario"] == "중") & (region["beta_case"] == "main")
                             & region["bjd_code"].isin(visible)].copy()
                if len(mid):
                    mid["부하_상한비"] = mid["load_peak_Y_kw"] / S["calib_peak"].reindex(mid["bjd_code"]).to_numpy() / base_m
                    writer.figure(plot_bar(mid.sort_values("부하_상한비", ascending=False), "bjd_code", "부하_상한비",
                                           f"2030 중 시나리오 피크 ÷ 현재 상한({base_m}배) — {title}",
                                           "피크 / 상한 (1 초과 = 상한 초과)"), "s8f_scenario_2030_mid")
                S.update(scenario=scen, scenario_region=region, scenario_beta=beta, scenario_map_fail=fail)
            runner.run("8-F", "충전 부하 시나리오(확장안)", stage8f, ISOLATED)

        # ------------------------------------------------ 9-H단계 (격리): 발표 3숫자
        if P["priority"]["enabled"] or P["energy"]["enabled"]:
            def stage9h():
                access, results = S.get("access"), S.get("energy_results")
                cm = S.get("energy_customer_min")
                visible = set(cm[cm >= int(P["can"]["min_cell_count"])].index.astype(str)) if cm is not None else None
                if list(P["priority"]["axes"]) == priorityscore.LEGACY_AXES:
                    args = (P["priority"], P["energy"]["utilization_scale"])
                    table = headline.headline_table(access, results, *args)
                    build = (lambda codes: headline.headline_table(access, results, *args, codes=codes))
                    note = "정책 상한 시뮬레이션 결과이며 실증 효과 아님(레거시 세 축 정의)"
                else:
                    hp = dict(P["priority"], utilization_scale=P["energy"]["utilization_scale"],
                              risk_threshold=P["headline"]["risk_threshold"],
                              min_p90_coverage=P["forecast"]["min_p90_coverage"])
                    parts = (access, S.get("risk"), S.get("ai_effect"), S.get("scenario_region"), S.get("ai"), hp)
                    table = headline.headline_table_v11(*parts)
                    build = (lambda codes: headline.headline_table_v11(*parts, codes=codes))
                    note = ("숫자 3(a)는 평가기간 예측·관측, (b)는 보급 증가 시나리오(예측 아님) — 둘 다 충전기 추가 없음. "
                            f"AI 점검: {table.attrs.get('ai_check', '')}. 헤드라인 문구는 팀이 확정")
                png = table
                if visible is not None:
                    # R2: PNG 집계에서는 소표본 법정동의 기여분을 뺀다(CSV 는 전체).
                    try:
                        png = build(visible)
                    except ValueError:
                        png = table.iloc[0:0]
                writer.table(table, "s9_headline", "9단계 발표 3숫자 (가정 병기)", digits=2, png_df=png,
                             note=note + ". PNG 는 소표본 법정동 기여분 제외")
                if list(P["priority"]["axes"]) != priorityscore.LEGACY_AXES:
                    sim = S.get("charger_sim")
                    can_check = S.get("can_away_check")
                    story = headline.story_table(table, sim, can_check)
                    writer.table(story, "s9_story", "9단계 발표 본문 — 발견 문장(초안)과 새 3숫자 (ESS·2030은 s9_headline 부록)",
                                 digits=3, png_df=headline.story_table(png, sim, can_check) if len(png) else story.iloc[0:0],
                                 note="문구는 팀이 확정. PNG 는 소표본 법정동 기여분 제외")
                    if "priority_score" in S:
                        card_args = (S["priority_score"], S.get("risk"), access, S.get("master"), P["priority"])
                        cards = headline.region_cards(*card_args)
                        writer.table(cards, "s9_region_cards", f"9단계 동네 카드 — 우선순위 상위 {P['priority']['card_top_n']}곳",
                                     digits=3, png_df=headline.region_cards(*card_args, codes=visible) if visible is not None else cards,
                                     note="권고는 규칙 기반 초안(급증위험·형평성 정규화 0.5 기준). 실제 설치·점검 결정이 아님. PNG 는 소표본 법정동 제외")
            runner.run("9-H", "발표 3숫자", stage9h, ISOLATED)
    except _StopAfterStage1:
        pass
    finally:
        # ------------------------------------------------ 9단계: 요약·목록 (실패해도 남긴다)
        stages = runner.table()
        stages["사용폰트"] = writer.font_label
        stages["analysis_level"] = P["analysis_level"]
        stages["min_cell_count"] = P["output"]["min_cell_count"]
        stages["suppress_basis"] = P["kepco"]["suppress_basis"]
        stages["분석지역"] = f"{P['region']['sido_name'] or '전체'}({prefix or '-'})"
        stages["지역단위수"] = S["mh"]["bjd_code"].nunique() if "mh" in S else np.nan   # sigungu 모드면 자치구 수
        stages["상권단계"] = "활성" if commerce else "비활성(v11)"
        # PNG 에는 센터 경로·원본 값이 섞일 수 있는 오류 전문과 절대경로를 싣지 않는다(전문은 내부 CSV).
        failed = stages["상태"].isin(["실패", "생략"]) & stages["비고"].notna()
        stages_png = stages.assign(비고=stages["비고"].where(
            ~failed, stages["비고"].astype(str).str.split(":").str[0] + " — 상세는 내부 CSV"))
        writer.table(stages, "s0_run_summary", "실행 요약 — 단계별 상태", digits=1, png_df=stages_png)
        manifest = writer.manifest_table()
        manifest_png = manifest.assign(경로=[Path(p).name for p in manifest["경로"]])
        writer.table(manifest, "s9_manifest", "9단계 산출물 목록 (PNG = 반출 후보 · CSV = 현장 작업용)",
                     png_df=manifest_png)
        log.info("완료: %s", stages[["단계", "상태"]].to_dict("records"))

    status = "완료" if stages["상태"].isin(["완료", "비활성"]).all() else "부분완료(격리 단계 생략)"
    return {"status": status, "stages": stages, "out_dir": str(out_dir), "state": S}


def run_day1(config_path, params_override=None, paths_override=None):
    """현장 첫 방문용: 법정동 정합과 KEPCO 월별 집계까지만 저장하고 정상 종료한다."""
    return run(config_path, params_override=params_override, paths_override=paths_override, stop_after_stage1=True)


def _reserved(S):
    """유보 사유 모음 — 처치오염 · 사전추세 · CAN 거점성 다수. 제외하지 않고 표시만 한다."""
    reasons = {}

    def add(codes, text):
        for c in codes:
            reasons[c] = f"{reasons[c]} · {text}" if c in reasons else text
    if "contamination" in S and len(S["contamination"]):
        add(S["contamination"].loc[S["contamination"]["판정"].str.startswith("유보"), "bjd_code"], "처치오염")
    add(S.get("pretrend_flagged", set()), "사전추세")
    if "can" in S and "변화점_신뢰도" in S["can"].get("region_table", ()):
        reg = S["can"]["region_table"]
        add(reg.loc[reg["변화점_신뢰도"].str.startswith("낮음"), "bjd_code"], "거점성 충전 다수")
    return reasons


def main():
    parser = argparse.ArgumentParser(description="충전 리플맵 4.2 파이프라인")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(result["stages"].to_string(index=False))
    print("상태:", result["status"], "· 산출물:", result["out_dir"])


if __name__ == "__main__":
    main()
