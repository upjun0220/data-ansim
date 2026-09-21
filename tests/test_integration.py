"""통합 테스트 — mock 데이터로 0~9단계 end-to-end + 킬 크라이테리아 + 격리·가드 동작.

v11: 이 파일의 mock_env 는 상권 단계 회귀용이다(SHC 생성 + stages.commerce=true). 상권 단계 기본 비활성 경로는
tests/test_v11.py 가 검증한다. 상권 전용 테스트에는 commerce 마커를 붙였다(pytest -m "not commerce" 로 뺄 수 있음).

실행(PowerShell, 저장소 루트):
    .venv\\Scripts\\python.exe -m pytest -q tests
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import canloader  # noqa: E402
import heterogeneity  # noqa: E402
import identification  # noqa: E402
import kepcoloader  # noqa: E402
import loadaxis  # noqa: E402
import mock_data  # noqa: E402
from bjdmapping import load_bjd_master, match_regions, shc_bjd_code  # noqa: E402
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS  # noqa: E402
from killcriteria import run_checks  # noqa: E402
from outputs import OutputWriter  # noqa: E402
from pipeline import run  # noqa: E402

STAGE_OUTPUTS = {
    "0": ["s0_bjd_match_rate", "s0_bjd_unmatched"],
    "1": ["s1_kepco_monthly", "s1_kepco_band_share"],
    "2": ["s2_activation", "s2_status_counts"],
    "3": ["s3_kep007_stock", "s3_can_join_key", "s3_can_session_summary", "s3_can_session_dist", "s3_can_region",
          "s3_can_spatial_agreement", "s3_treated_excl_base"],
    "4": ["s4_ydd_panel", "s4_skipped_summary", "s4_shc002_quality"],
    "4.5": ["s45_mde"],
    "5": ["s5_event_main", "s5_pretrend_overall", "s5_pretrend_region", "s5_event_regression"],
    "6": ["s6_event_placebo", "s6_discount_by_k", "s6_post_summary"],
    "6.5": ["s65_contamination"],
    "7": ["s7_features", "s7_cate_region", "s7_cate_validation", "s7_subgroup_cells"],
    "8": ["s8_load_concentration", "s8_quadrants", "s8_quadrant_counts"],
    "9": ["s0_run_summary", "s9_manifest"],
}
FIGURES = ["s2_activation_examples", "s3_can_session_hist", "s5_s6_event_study", "s8_quadrants_plot"]


@pytest.fixture(scope="session")
def mock_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("ripplemap")
    data = base / "data"
    truth = mock_data.generate(data, seed=42, with_shc=True)
    cfg = {
        "paths": {
            "bjd_master": str(data / "bjd_master.txt"), "emd_centroids": str(data / "bjd_centroids.csv"),
            "kepco_001": str(data / "kepco_001.csv"), "kepco_002": str(data / "kepco_002.csv"),
            "shc001": str(data / "shc001.csv"), "shc002": str(data / "shc002.csv"),
            "can_m": str(data / "can_m_individual.csv"), "kep007": str(data / "kep007.csv"),
            "industry_codes": str(ROOT / "config" / "industry_codes_mock.json"), "out_dir": str(base / "out"),
        },
        "params": {"stages": {"commerce": True}, "identification": {"n_boot": 199},
                   "kill": {"sample_rows": 200000, "compare_rows": 200000}},
    }
    path = base / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return {"base": base, "data": data, "config": path, "truth": truth}


@pytest.fixture(scope="session")
def pipeline_result(mock_env):
    return run(mock_env["config"])


# ---------------------------------------------------------------- end-to-end

@pytest.mark.commerce
def test_all_stages_complete(pipeline_result):
    stages = pipeline_result["stages"]
    assert (stages["상태"] == "완료").all(), stages.to_string()
    assert pipeline_result["status"] == "완료"


@pytest.mark.commerce
def test_energy_environment_check_is_added_when_enabled(mock_env):
    table = run_checks(mock_env["config"], params_override={"energy": {"enabled": True}}, write=False)
    assert table["번호"].tolist() == list(range(1, 12)) + [16, 17, 18, 19]   # v11: 17~19 AI·기온 항목
    assert table.set_index("번호").at[9, "판정"] == "통과"


@pytest.mark.commerce
def test_every_stage_writes_csv_and_png(pipeline_result):
    out = Path(pipeline_result["out_dir"])
    missing = []
    for names in STAGE_OUTPUTS.values():
        for name in names:
            for kind in ("csv", "png"):
                if not (out / kind / f"{name}.{kind}").exists():
                    missing.append(f"{kind}/{name}")
    for name in FIGURES:
        if not (out / "png" / f"{name}.png").exists():
            missing.append(f"png/{name}")
    assert not missing, missing


@pytest.mark.commerce
def test_activation_recovers_truth(pipeline_result, mock_env):
    act = pipeline_result["state"]["activation"]
    truth = mock_env["truth"].set_index("bjd_code")
    treated = act[act["status"] == "treated"].set_index("bjd_code")
    true_treated = truth[truth["role"] == "treated"]
    hit = sum(1 for code, row in true_treated.iterrows()
              if code in treated.index and abs(treated.at[code, "T_r_mi"] - mock_data._mi(row["activation"])) <= 1)
    assert hit >= 16, f"처치 시점 복원 {hit}/18"
    never = set(act.loc[act["status"] == "never_treated", "bjd_code"])
    assert never <= set(truth.index[truth["role"] == "never_treated"]) | set(), "never-treated 오분류"
    assert set(act.loc[act["status"] == "excluded_prior", "bjd_code"]) == set(truth.index[truth["role"] == "excluded_prior"])


@pytest.mark.commerce
def test_event_study_effect_and_placebo(pipeline_result):
    post = pipeline_result["state"]["post_summary"].set_index("구분")
    main_att = post.at["주 결과(할인 전)", "사후평균"]
    plc_att = post.at["위약(대조 업종)", "사후평균"]
    assert main_att > 0.05 and post.at["주 결과(할인 전)", "p"] < 0.05, post
    assert abs(plc_att) < 0.05, post
    adj = post.at["할인 후 = 주 − 위약", "사후평균"]
    assert np.isclose(adj, main_att - plc_att)
    assert pipeline_result["state"]["main"]["estimator"].startswith("Callaway")


@pytest.mark.commerce
def test_contamination_flags_planted_regions(pipeline_result, mock_env):
    cont = pipeline_result["state"]["contamination"]
    flagged = set(cont.loc[cont["판정"].str.startswith("유보"), "bjd_code"])
    planted = set(mock_env["truth"].loc[mock_env["truth"]["contaminated"], "bjd_code"])
    assert planted <= flagged, (planted, flagged)
    assert len(flagged - planted) <= 2


@pytest.mark.commerce
def test_cate_fallback_and_quadrants(pipeline_result):
    het = pipeline_result["state"]["het"]
    assert "2×2" in het["method"]            # econml 없는 환경 → 폴백
    val = het["validation"].set_index("방법")
    assert {"상수 ATE", "2×2 서브그룹", "Causal Forest"} <= set(val.index)
    quad = pipeline_result["state"]["quadrants"]
    assert quad["quadrant"].nunique() >= 3
    assert quad["reserved"].any()


@pytest.mark.commerce
def test_can_path_a(pipeline_result):
    can = pipeline_result["state"]["can"]
    assert can["verdict"] == "individual"
    assert can["treated_excl_base"] is not None
    summary = can["duration_summary"].iloc[0]
    assert summary["세션수"] > 500


def test_no_gps_in_exports(pipeline_result):
    from outputs import GPS_COLUMN
    csvs = list((Path(pipeline_result["out_dir"]) / "csv").glob("*.csv"))
    assert csvs
    for csv in csvs:
        cols = pd.read_csv(csv, nrows=0, encoding="utf-8-sig").columns
        assert not [c for c in cols if GPS_COLUMN.search(str(c))], csv


# ---------------------------------------------------------------- 킬 크라이테리아

@pytest.mark.commerce
def test_kill_criteria(mock_env):
    table = run_checks(mock_env["config"])
    assert list(table["번호"]) == list(range(1, 9)) + [10, 11, 16]
    assert set(table["판정"]) <= {"통과", "경고", "실패", "오류"}
    assert not (table["판정"] == "오류").any(), table.to_string()
    v = table.set_index("번호")["판정"]
    assert v[2] == "경고"          # 후보 18곳: 10 이상 30 미만 → Causal Forest 포기 경고
    assert v[3] == "통과" and v[4] == "통과" and v[5] == "통과" and v[6] == "통과"
    assert v[7] == "통과"          # mock 002 ⊆ 001
    assert v[8] == "통과"          # individual
    out = Path(json.loads(Path(mock_env["config"]).read_text(encoding="utf-8"))["paths"]["out_dir"]) / "kill_criteria"
    assert (out / "png" / "k_kill_criteria.png").exists()


@pytest.mark.commerce
def test_kill_criteria_model_can_and_high_fail_rate(mock_env, tmp_path):
    # 마스터에서 종로구 법정동을 모두 지우면 KEPCO 매핑 실패율 ≈ 1/3 → 실패
    src = mock_env["data"] / "bjd_master.txt"
    lines = src.read_text(encoding="cp949").splitlines()
    cut = tmp_path / "master_cut.txt"
    cut.write_text("\n".join(l for l in lines if not l.startswith("11110")) + "\n", encoding="cp949")
    table = run_checks(mock_env["config"], paths_override={
        "bjd_master": str(cut), "can_m": str(mock_env["data"] / "can_m_model.csv"), "out_dir": str(tmp_path / "out")})
    v = table.set_index("번호")["판정"]
    assert v[5] == "실패"
    assert v[8] == "경고" and "model" in table.set_index("번호").at[8, "근거"]


@pytest.mark.commerce
def test_check10_uses_configured_kepco_source(mock_env, tmp_path):
    """R4: check10 은 항상 KEPCO_001이 아니라 P["kepco"]["source"]에 맞는 파일을 읽어야 한다."""
    missing = tmp_path / "missing_kepco_002.csv"
    table = run_checks(mock_env["config"], params_override={"kepco": {"source": "002"}},
                       paths_override={"kepco_002": str(missing)}, write=False)
    v = table.set_index("번호")["판정"]
    assert v[10] == "오류", table.set_index("번호").at[10, "근거"]


@pytest.mark.commerce
def test_check10_reports_sample_shortage_as_warning_not_error(mock_env, monkeypatch):
    """R4: estimate_mde 의 표본 부족 ValueError 는 '오류'가 아니라 '경고'로 내려야 한다."""
    def boom(*args, **kwargs):
        raise ValueError("MDE 표본 부족: 실제 처치 1 · never-treated 0")
    monkeypatch.setattr(identification, "estimate_mde", boom)
    table = run_checks(mock_env["config"], write=False)
    row = table.set_index("번호").loc[10]
    assert row["판정"] == "경고"
    assert "표본 기반 참고값" in row["근거"] and "pipeline 4.5" in row["근거"] and "표본 부족" in row["근거"]


# ---------------------------------------------------------------- 격리·분기

@pytest.mark.commerce
def test_can_failure_is_isolated(mock_env, tmp_path):
    broken = tmp_path / "broken_can.csv"
    broken.write_text("아무컬럼\n1\n", encoding="utf-8")
    res = run(mock_env["config"], params_override={"identification": {"n_boot": 49}},
              paths_override={"can_m": str(broken), "out_dir": str(tmp_path / "out")})
    stages = res["stages"]
    can_row = stages[stages["이름"] == "CAN 처치 정제"].iloc[0]
    assert can_row["상태"] == "생략"
    others = stages[stages["이름"] != "CAN 처치 정제"]
    assert (others["상태"] == "완료").all(), stages.to_string()
    assert (tmp_path / "out" / "png" / "s3_can_skipped.png").exists()
    assert (tmp_path / "out" / "png" / "s8_quadrants.png").exists()


def test_can_model_path_b(mock_env):
    params = DEFAULT_PARAMS["can"]
    can = canloader.load_can_m(mock_env["data"] / "can_m_model.csv", DEFAULT_COLUMNS, params)
    verdict, _ = canloader.verify_join_key(can, params)
    assert verdict == "model"
    from bjdmapping import load_emd_centroids
    cent = load_emd_centroids(mock_env["data"] / "bjd_centroids.csv", DEFAULT_COLUMNS["centroid"])
    act = pd.DataFrame({"bjd_code": [], "status": [], "T_r": []})
    res = canloader.run_can_stage(mock_env["data"] / "can_m_model.csv", DEFAULT_COLUMNS, params, cent, None, act, "2025-03")
    assert res["path"].startswith("B") and res["treated_excl_base"] is None
    assert "밀집도_일평균세션" in res["region_table"].columns


def test_can_small_sample_unknown(mock_env):
    params = DEFAULT_PARAMS["can"]
    can = canloader.load_can_m(mock_env["data"] / "can_m_individual.csv", DEFAULT_COLUMNS, params, nrows=30)
    assert canloader.verify_join_key(can, params)[0] == "unknown"


@pytest.mark.commerce
def test_regression_fallback_positive(pipeline_result):
    S = pipeline_result["state"]
    params = dict(DEFAULT_PARAMS["identification"], estimator="regression")
    import identification
    res = identification.run_event_study(S["panel"], "y_main", S["activation"], params)
    assert res["estimator"].startswith("상대시점") and res["post"]["att"] > 0.05


# ---------------------------------------------------------------- 가드

def test_bad_control_guard():
    with pytest.raises(ValueError, match="Bad control"):
        heterogeneity.assert_snapshot_before(["2025-02", "2025-03"], "2025-03")
    heterogeneity.assert_snapshot_before(["2025-01", "2025-02"], "2025-03")


def test_load_feature_forbidden():
    with pytest.raises(ValueError, match="부하"):
        heterogeneity.assert_no_load_features(["wait_share", "피크시간_점유율"])
    cate = pd.DataFrame({"bjd_code": ["1"], "cate": [0.1], "W": [1], "method": ["x"]})
    cate.attrs["feature_names"] = ["concentration"]
    conc = pd.DataFrame({"bjd_code": ["1"], "concentration": [0.1]})
    with pytest.raises(ValueError):
        loadaxis.classify_quadrants(cate, conc, DEFAULT_PARAMS["load_axis"])


def test_load_axis_rejects_non_kepco_input():
    with pytest.raises(ValueError, match="KEPCO"):
        loadaxis.compute_concentration(pd.DataFrame({"bjd_code": ["1"], "wait_share": [0.3]}), ["2025-01", "2025-12"])


def test_gps_guard(tmp_path):
    w = OutputWriter(tmp_path)
    with pytest.raises(ValueError, match="좌표"):
        w.table(pd.DataFrame({"bjd_code": ["1"], "위도": [37.5]}), "bad", "bad")


def test_bjd_matching_rules(mock_env, caplog):
    master = load_bjd_master(mock_env["data"] / "bjd_master.txt", DEFAULT_COLUMNS["bjd"])
    assert len(master) == 30                              # 리 단위·폐지 코드 제외
    regions = pd.DataFrame({
        "sido": ["서울", "서울특별시", "서울", "서울", "서울"],
        "sigungu": ["종로구", "중구", "종로구", "용산구", "용산구"],
        "emd": ["가람제1동", "중앙동", "중앙동", "행복동", "중앙동"],   # 마지막: 용산구에는 중앙동 없음 → 단독 매칭 금지
    })
    out, rate, unmatched, fail = match_regions(regions, master)
    codes = out["bjd_code"].tolist()
    assert codes[0] == "1111010100" and out.at[0, "match_method"] == "정규화일치"
    assert codes[1] == "1114010300" and codes[2] == "1111010300"   # 동명이인 → 구로 구분
    assert pd.isna(codes[3]) and pd.isna(codes[4])
    assert len(unmatched) == 2 and fail == pytest.approx(0.4)
    with caplog.at_level(logging.WARNING, logger="ripplemap"):
        match_regions(regions, master, fail_warn_rate=0.30)
    assert any("행정동 기준" in r.message for r in caplog.records)


def test_crosswalk_load_and_canonicalize(tmp_path):
    from bjdmapping import canonicalize, load_crosswalk
    p = tmp_path / "cw.csv"
    p.write_text("코드,기준코드\n1211010100,4611010100\n4611010100,4611010100\n", encoding="utf-8-sig")
    cw = load_crosswalk(p, DEFAULT_COLUMNS["crosswalk"])
    assert cw == {"1211010100": "4611010100"}
    s = canonicalize(pd.Series(["1211010100", "1111010100", None], dtype=object), cw)
    assert s.tolist()[:2] == ["4611010100", "1111010100"] and pd.isna(s.iloc[2])
    chained = tmp_path / "chain.csv"
    chained.write_text("코드,기준코드\n1111010100,2222010100\n2222010100,3333010100\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="연쇄"):
        load_crosswalk(chained, DEFAULT_COLUMNS["crosswalk"])


def test_attach_bjd_merges_renamed_region(tmp_path):
    """개편 전/후 명칭(전라남도 목포시 ↔ 전남광주통합특별시 목포시)이 기준코드 하나로 합쳐진다."""
    master_path = tmp_path / "master.txt"
    master_path.write_text(
        "법정동코드\t법정동명\t폐지여부\n"
        "4600000000\t전라남도\t존재\n4611000000\t전라남도 목포시\t존재\n4611010100\t전라남도 목포시 용당동\t존재\n"
        "1200000000\t전남광주통합특별시\t존재\n1211000000\t전남광주통합특별시 목포시\t존재\n"
        "1211010100\t전남광주통합특별시 목포시 용당동\t존재\n", encoding="utf-8-sig")
    master = load_bjd_master(master_path, DEFAULT_COLUMNS["bjd"])
    agg = pd.DataFrame({"sido": ["전라남도", "전남광주통합특별시"], "sigungu": ["목포시", "목포시"],
                        "emd": ["용당동", "용당동"], "mi": [24299, 24318], "hour": [0, 0],
                        "kwh": [10.0, 12.0], "n_days": [1.0, 1.0], "cust_sum": [1.0, 1.0]})
    out, _, unmatched, fail = kepcoloader.attach_bjd(agg, master, 0.3, {"1211010100": "4611010100"})
    assert fail == 0 and unmatched.empty
    assert out["bjd_code"].unique().tolist() == ["4611010100"] and len(out) == 2


def test_shc_code_variants():
    s = shc_bjd_code(pd.Series(["11", "11", "11", "11"]), pd.Series(["11710", "710", "11710", ""]),
                     pd.Series(["1171010100", "101", "101", "11710101"]))
    assert s.tolist() == ["1171010100"] * 4


def test_pvalue_functions_match_tables():
    from common import chi2_sf, f_sf
    assert f_sf(4.0, 1, 10) == pytest.approx(0.0734, abs=1e-4)
    assert f_sf(4.534, 4, 6) == pytest.approx(0.05, abs=2e-4)
    assert f_sf(3.098, 3, 20) == pytest.approx(0.05, abs=2e-4)
    assert chi2_sf(9.488, 4) == pytest.approx(0.05, abs=2e-4)


def test_unit_pretrend_power_and_size():
    """심은 사전추세는 잡고(검정력), 추세 없는 지역은 거의 안 잡는다(크기)."""
    import identification
    rng = np.random.default_rng(1)
    months = list(range(24300, 24312))
    codes = [f"C{i:02d}" for i in range(30)]
    rows = []
    for c in codes:
        for m in months:
            y = rng.normal(0, 0.05)
            if c == "C00":
                y += 0.15 * (m - 24300)
            rows.append((c, m, y))
    panel = pd.DataFrame(rows, columns=["bjd_code", "mi", "y"])
    act = pd.DataFrame({"bjd_code": codes, "status": ["treated"] * 10 + ["never_treated"] * 20,
                        "T_r_mi": [24306] * 10 + [np.nan] * 20})
    res = identification.cs_event_study(panel, "y", act, dict(DEFAULT_PARAMS["identification"], n_boot=49))
    verdict = res["unit_pretrend"].set_index("bjd_code")["판정"]
    assert verdict["C00"].startswith("유보")
    assert verdict.drop("C00").str.startswith("유보").sum() <= 1


def test_pelt_builtin_finds_step():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 0.05, 20), rng.normal(1, 0.05, 16)])
    assert kepcoloader.pelt_builtin(y, pen=0.05 * np.log(36), min_size=2) == [20]


def test_sigungu_mode_matches_admin_dong_names_and_folds_codes():
    from bjdmapping import canonicalize, make_key, match_regions, set_analysis_level
    master = pd.DataFrame({"bjd_code": ["1168010100", "1168010300"], "sido": ["서울", "서울"],
                           "sigungu": ["강남구", "강남구"], "emd": ["역삼동", "개포동"]})
    master["key_raw"] = make_key(master["sido"], master["sigungu"], master["emd"])
    master["key_norm"] = make_key(master["sido"], master["sigungu"], master["emd"], normalized=True)
    master["region_name"] = "서울 강남구 " + master["emd"]
    regions = pd.DataFrame({"sido": ["서울"], "sigungu": ["강남구"], "emd": ["역삼1동"], "kwh": [1.0]})  # 행정동 표기
    assert pd.isna(match_regions(regions, master)[0].at[0, "bjd_code"])       # 법정동 모드는 못 맞춘다
    try:
        set_analysis_level("sigungu")
        out, _rate, _unmatched, fail = match_regions(regions, master, weight_col="kwh")
        assert out.at[0, "bjd_code"] == "1168000000" and fail == 0 and out.at[0, "emd"] == "역삼1동"
        folded = canonicalize(pd.Series(["1168010100", None], dtype=object), None)
        assert folded.iloc[0] == "1168000000" and pd.isna(folded.iloc[1])
    finally:
        set_analysis_level("emd")
    with pytest.raises(ValueError, match="analysis_level"):
        set_analysis_level("dong")