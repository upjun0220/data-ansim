"""충전 리플맵 4.2 — 0~9단계 end-to-end 실행.

실행(로컬, PowerShell, 저장소 루트):
    .venv\\Scripts\\python.exe mock_data.py --out data/mock
    .venv\\Scripts\\python.exe pipeline.py --config config/mock.json
현장(안심구역 JupyterLab):
    from pipeline import run
    summary = run("config/field.json")

단계 격리 원칙
  - 필수: 0·1(정합·집계) · 2(변화점) · 4(Y_ddd) · 5(이벤트 스터디) · 6(위약) — 실패하면 중단한다.
  - 격리: 3(CAN · KEP_007) · 6.5(처치오염) · 7(CATE) · 8(처방) — 실패해도 나머지는 계속 돈다.
    CAN 이 실패하면 'CAN 처리 생략됨' 로그와 함께 3단계 표만 비워 둔다.
산출물은 out_dir/png(반출용) · out_dir/csv(현장 작업용), 로그는 out_dir/pipeline.log.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import bjd_mapping
import can_loader
import diagnostics
import heterogeneity
import identification
import kep007_loader
import kepco_loader
import load_axis
from common import log, mi_to_ym, setup_logging
from config import deep_merge, load_config
from outputs import (OutputWriter, plot_activation_examples, plot_bar, plot_event_study, plot_histogram,
                     plot_quadrants)

CRITICAL = True
ISOLATED = False


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


def run(config_path, params_override=None, paths_override=None):
    cfg = load_config(config_path)
    if params_override:
        cfg["params"] = deep_merge(cfg["params"], params_override)
    if paths_override:
        cfg["paths"].update({k: (str(Path(v).resolve()) if v else None) for k, v in paths_override.items()})
    P, C, paths, ind = cfg["params"], cfg["columns"], cfg["paths"], cfg["industry"]
    out_dir = Path(paths["out_dir"])
    setup_logging(out_dir)
    writer = OutputWriter(out_dir, **P["outputs"])
    runner = StageRunner()
    S = {}
    log.info("설정: %s", cfg["config_path"])

    try:
        # ------------------------------------------------ 0·1단계
        def stage01():
            master = bjd_mapping.load_bjd_master(paths["bjd_master"], C["bjd"])
            src = str(P["kepco"]["source"])
            if src not in ("001", "002"):
                raise ValueError("params.kepco.source 는 '001' 또는 '002' — 둘을 합산하지 않는다")
            raw = kepco_loader.load_kepco_monthly_hour(paths[f"kepco_{src}"], C["kepco" if src == "001" else "kepco_cpo"],
                                                       P["kepco"], f"KEPCO_{src}")
            mh, rate, unmatched, fail = kepco_loader.attach_bjd(raw, master, P["bjd"]["fail_warn_rate"])
            writer.table(rate, "s0_bjd_match_rate", "0단계 법정동 매칭률 (시도+시군구+읍면동 3단 매칭)",
                         note=f"실패율 {fail:.1%} — {P['bjd']['fail_warn_rate']:.0%} 이상이면 행정동 기준 의심")
            writer.table(unmatched, "s0_bjd_unmatched", "0단계 미매칭 목록 (사람 검토 필요)")
            daily = kepco_loader.daily_series(mh)
            monthly = daily.groupby("mi").agg(법정동수=("bjd_code", "nunique"),
                                              일평균충전량_합_kWh=("kwh_per_day", "sum")).reset_index()
            writer.table(_ym_col(monthly), "s1_kepco_monthly", f"1단계 KEPCO_{src} 월별 집계 (법정동 합)", digits=1)
            if ind.get("tizo_bands"):
                band = kepco_loader.band_series(mh, ind["tizo_bands"])
                wide = band.groupby(["mi", "band"])["kwh_per_day"].sum().unstack(fill_value=0)
                share = wide.div(wide.sum(axis=1), axis=0).add_prefix("구간비_").reset_index()
                writer.table(_ym_col(share), "s1_kepco_band_share", "1단계 시간대구간(TIZO)별 충전량 비중")
            S.update(master=master, mh=mh, daily=daily, fail_rate=fail)
        runner.run("0·1", "지역키 정합 + KEPCO 시계열 집계", stage01, CRITICAL)

        # ------------------------------------------------ 2단계
        def stage2():
            act = kepco_loader.detect_activation(S["daily"], P["activation"])
            cols = ["bjd_code", "status", "T_r", "ratio", "T_simple", "agree_simple", "n_changepoints", "n_months"]
            writer.table(act[[c for c in cols if c in act]], "s2_activation",
                         f"2단계 활성화 시점 T_r ({act.attrs.get('method')}, 창 {'~'.join(P['activation']['window'])})")
            counts = act["status"].value_counts().rename_axis("상태").reset_index(name="법정동수")
            writer.table(counts, "s2_status_counts", "2단계 처치 상태별 법정동 수")
            if (act["status"] == "treated").any():
                writer.figure(plot_activation_examples(S["daily"], act), "s2_activation_examples")
            S["activation"] = act
        runner.run("2", "변화점 탐지", stage2, CRITICAL)

        # ------------------------------------------------ 3단계 (격리)
        def stage3_centroids():
            if not paths["emd_centroids"]:
                raise FileNotFoundError("paths.emd_centroids 없음 — CAN·KEP_007 좌표를 법정동으로 보낼 수 없음")
            S["centroids"] = bjd_mapping.load_emd_centroids(paths["emd_centroids"], C["centroid"])
        runner.run("3", "법정동 중심점", stage3_centroids, ISOLATED)

        def stage3_kep007():
            if not paths["kep007"]:
                raise FileNotFoundError("paths.kep007 없음")
            st = kep007_loader.load_kep007(paths["kep007"], C, P["kep007"], S.get("centroids"),
                                           P["can"]["centroid_max_km"], P["can"]["lat_range"], P["can"]["lon_range"])
            stock = kep007_loader.region_stock(st)
            writer.table(stock, "s3_kep007_stock", "3단계 KEP_007 법정동별 인프라 스톡 (2019 기준 정적 스냅샷)", digits=0)
            S.update(stations=st, infra_stock=stock)
        runner.run("3", "KEP_007 설치현황", stage3_kep007, ISOLATED)

        def stage3_can():
            if not P["can"]["enabled"] or not paths["can_m"]:
                raise RuntimeError("CAN 비활성 또는 paths.can_m 없음")
            res = can_loader.run_can_stage(paths["can_m"], C, P["can"], S.get("centroids"), S.get("stations"),
                                           S["activation"], P["heterogeneity"]["can_feature_before"])
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
            S["can"] = res
        if runner.run("3", "CAN 처치 정제", stage3_can, ISOLATED) is None and "can" not in S:
            reason = runner.rows[-1]["비고"]
            log.warning("CAN 처리 생략됨 — 3단계 CAN 리포트 섹션은 비워 둔다 (%s)", reason)
            writer.table(pd.DataFrame([{"항목": "CAN 처리", "상태": "생략됨", "사유": reason}]), "s3_can_skipped",
                         "3단계 CAN 처리 생략됨")

        # ------------------------------------------------ 4단계
        def stage4():
            scan = identification.scan_shc002(paths["shc002"], C, ind, P["shc"],
                                              snapshot_months=P["heterogeneity"]["snapshot_months"])
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
        runner.run("4", "Outcome 축약(Y_ddd)", stage4, CRITICAL)

        # ------------------------------------------------ 5단계
        def stage5():
            panel = S["panel"]
            units = set(panel.loc[panel["y_main"].notna(), "bjd_code"]) & set(panel.loc[panel["y_placebo"].notna(), "bjd_code"])
            ip = P["identification"]
            main = identification.run_event_study(panel, "y_main", S["activation"], ip, units)
            writer.table(identification.event_table(main), "s5_event_main", "5단계 이벤트 스터디 tau(k) — 주 결과",
                         note=f"처치 {main['n_treated']} · 대조 {main['n_control']} · {ip['control_group']}"
                              + (f" · 폴백 사유: {main['fallback_reason']}" if main["fallback_reason"] else ""))
            if len(main["att_gt"]):
                writer.table(main["att_gt"], "s5_att_gt", "5단계 ATT(g,t) 원표")
            pre = main["pretrend"]
            writer.table(pd.DataFrame([{"검정": "사전추세 결합 Wald (k<=-2)", "통계량": pre["stat"], "자유도": pre["df"],
                                        "p": pre["p"], "판정": ("유의 — 평행추세 의심" if pre["p"] < ip["alpha"] else "기각 못함")
                                        if np.isfinite(pre["p"]) else "검정불가(사전기간 부족)"}]),
                         "s5_pretrend_overall", "5단계 사전 추세 검정")
            unit_pre = main["unit_pretrend"]
            writer.table(unit_pre, "s5_pretrend_region", "5단계 법정동별 사전 추세 (유보 표시)")
            flagged = set(unit_pre.loc[unit_pre["판정"].str.startswith("유보"), "bjd_code"]) if len(unit_pre) else set()
            if ip["pretrend_action"] == "exclude" and flagged:
                log.info("사전추세 유보 %d곳 제외 후 재추정", len(flagged))
                main = identification.run_event_study(panel, "y_main", S["activation"], ip, units - flagged)
                writer.table(identification.event_table(main), "s5_event_main_excl_pretrend",
                             "5단계 tau(k) — 사전추세 유보 지역 제외")
                units = units - flagged
            if main["estimator"].startswith("Callaway"):
                try:
                    reg = identification.regression_event_study(panel, "y_main", S["activation"], ip, units)
                    writer.table(identification.event_table(reg), "s5_event_regression", "5단계 강건성 — 상대시점 더미 회귀")
                    S["regression"] = reg
                except Exception as exc:
                    log.warning("강건성 회귀 실패(주 결과에는 영향 없음): %s", exc)
            S.update(main=main, units=units, pretrend_flagged=flagged)
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
            writer.table(post, "s6_post_summary", "6단계 사후 평균효과 — 할인 전 · 위약 · 할인 후")
            adj = by_k.rename(columns={"tau_할인후": "tau"}).assign(
                ci_lo=lambda d: d["tau"] - 1.96 * d["se_할인후"], ci_hi=lambda d: d["tau"] + 1.96 * d["se_할인후"])
            writer.figure(plot_event_study({"주 결과(할인 전)": main["event"], "위약(대조 업종)": plc["event"],
                                            "할인 후": adj[["k", "tau", "ci_lo", "ci_hi"]]},
                                           f"5·6단계 이벤트 스터디 — {main['estimator']}"), "s5_s6_event_study")
            S.update(placebo=plc, post_summary=post)
        runner.run("6", "위약 검정", stage6, CRITICAL)

        # ------------------------------------------------ 6.5단계 (격리)
        def stage65():
            if not paths["shc001"]:
                raise FileNotFoundError("paths.shc001 없음")
            m = diagnostics.load_shc001_monthly(paths["shc001"], C, ind["shc001"], P["shc"]["chunksize"])
            cont = diagnostics.contamination_check(m, S["activation"], P["diagnostics"])
            writer.table(cont, "s65_contamination", "6.5단계 처치오염 진단 — T_r 전후 가맹점 개설률",
                         note="유보 = 상권 자체 성장 가능성. 제외하지 않고 CATE·처방에 표시만 한다")
            S.update(shc001m=m, contamination=cont)
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
            writer.table(cate, "s7_cate_region", f"7단계 법정동별 CATE — {het['method']}",
                         note=f"Causal Forest 미사용 사유: {het['reason']}" if het["reason"] else "")
            writer.table(het["validation"], "s7_cate_validation", "7단계 CATE 검증 — 같은 교차검증 분할에서 방법 비교",
                         note="변환결과 MSE 낮을수록 · GATES 상위−하위 클수록 좋음")
            if het["importance"] is not None:
                writer.table(het["importance"], "s7_importance", "7단계 Causal Forest 변수 중요도")
            if het["cells"] is not None:
                writer.table(het["cells"], "s7_subgroup_cells", "7단계 사전지정 2x2 서브그룹 CATE")
            S["het"] = het
        runner.run("7", "CATE", stage7, ISOLATED)

        # ------------------------------------------------ 8단계 (격리)
        def stage8():
            if "het" not in S:
                raise RuntimeError("7단계 CATE 결과가 없어 처방을 만들 수 없음")
            conc = load_axis.compute_concentration(S["mh"], P["load_axis"]["period"])
            writer.table(conc, "s8_load_concentration", "8단계 부하 집중도",
                         note="avg_kw = 1시간 kWh ÷ 1h 의 일평균 = 구간 평균 kW (순간 최대전력 아님)")
            quad, cc, kc = load_axis.classify_quadrants(S["het"]["cate"], conc, P["load_axis"], reserved)
            export = quad.drop(columns=["reserved"])
            writer.table(export, "s8_quadrants", "8단계 4사분면 처방 표", note="유보사유가 있는 지역은 해석 유보")
            counts = quad.groupby("quadrant").agg(법정동수=("bjd_code", "size"), 유보수=("reserved", "sum")).reset_index()
            writer.table(counts, "s8_quadrant_counts", "8단계 사분면별 법정동 수")
            writer.figure(plot_quadrants(quad, cc, kc), "s8_quadrants_plot")
            writer.figure(plot_bar(counts, "quadrant", "법정동수", "사분면별 법정동 수", "법정동 수"), "s8_quadrant_counts_bar")
            S["quadrants"] = quad
        runner.run("8", "전력축·처방", stage8, ISOLATED)
    finally:
        # ------------------------------------------------ 9단계: 요약·목록 (실패해도 남긴다)
        stages = runner.table()
        writer.table(stages, "s0_run_summary", "실행 요약 — 단계별 상태", digits=1)
        writer.table(writer.manifest_table(), "s9_manifest", "9단계 산출물 목록 (PNG = 반출용 · CSV = 현장 작업용)")
        log.info("완료: %s", stages[["단계", "상태"]].to_dict("records"))

    status = "완료" if (stages["상태"] == "완료").all() else "부분완료(격리 단계 생략)"
    return {"status": status, "stages": stages, "out_dir": str(out_dir), "state": S}


def _reserved(S):
    """유보 사유 모음 — 처치오염 · 사전추세 · CAN 거점성 다수. 제외하지 않고 표시만 한다."""
    reasons = {}

    def add(codes, text):
        for c in codes:
            reasons[c] = f"{reasons[c]} · {text}" if c in reasons else text
    if "contamination" in S and len(S["contamination"]):
        add(S["contamination"].loc[S["contamination"]["판정"].str.startswith("유보"), "bjd_code"], "처치오염")
    add(S.get("pretrend_flagged", set()), "사전추세")
    if "can" in S and S["can"]["verdict"] == "individual":
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
