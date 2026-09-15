"""파이프라인 설정 — 데이터 경로 · 컬럼명 · 파라미터 · 업종코드 매핑을 코드에서 분리한다.

현장(안심구역)에서는 config/field_template.json 과 config/industry_codes_template.json 을 복사해
경로와 코드값만 채워 쓴다.
- JSON 안의 상대경로는 JSON 파일 위치 기준으로 해석한다.
- "_" 로 시작하는 키는 설명용이라 무시한다.
- JSON 에 적지 않은 항목은 아래 DEFAULT_* 값을 쓴다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

# 실제 컬럼명(스펙 §2 확정분). 헤더가 다르면 JSON 의 columns 에서 그 항목만 덮어쓴다. 접두 일치도 허용.
DEFAULT_COLUMNS = {
    # KEPCO_001/002 — 6개 컬럼. 시간대 정보가 '조회기간'에 들어 있는지(YYYYMMDDHH), 별도 열인지,
    # 시간대별 가로 열인지 현장 확인 전까지 불명 → params.kepco.layout 과 columns.hour 로 분기한다.
    "kepco": {"period": "조회기간", "sido": "시도", "sigungu": "시군구", "emd": "읍면동",
              "customers": "고객호수", "kwh": "시간대별 사용량", "hour": ""},
    "kepco_cpo": {"period": "조회기간", "sido": "시도", "sigungu": "시군구", "emd": "읍면동",
                  "customers": "고객호수", "kwh": "시간대별 사용량", "hour": ""},
    "bjd": {"code": "법정동코드", "name": "법정동명", "status": "폐지여부", "sido": "", "sigungu": "", "emd": ""},
    # 법정동 중심점(QGIS 에서 법정동 경계 → 중심점 → CSV). CAN·KEP_007 좌표를 법정동으로 보낼 때만 쓴다.
    "centroid": {"code": "법정동코드", "lat": "위도", "lon": "경도"},
    "shc001": {"ym": "STD_YM", "sido_cd": "WIAR_SIDO_CD", "sgg_cd": "SGNG_CD", "umd_cd": "UMD_CD",
               "bidvs": "TOBU_BIDVS_CD", "medvs": "TOBU_MEDVS_CD", "status": "FRNC_STAT_CD",
               "oper": "OPER_CD", "cnt": "FRNC_CNT"},
    "shc002": {"ym": "STD_YM", "sido_cd": "WIAR_SIDO_CD", "sgg_cd": "SGNG_CD", "umd_cd": "UMD_CD",
               "bidvs": "TOBU_BIDVS_CD", "medvs": "TOBU_MEDVS_CD", "pers_corp": "PERS_CORP_CLCD",
               "pholi": "PHOLI_CLCD", "tizo": "TIZO_CLCD", "sex": "SEX_CLCD", "age": "N10_UNIT_AGE_CLCD",
               "lifecycle": "HSH_LFTM_MAIN_PRE_CD", "income": "EST_INCM_SECT_CD",
               "ifw_area": "IFW_AREA_CD", "ifw_dist": "IFW_DISTC_SECT_CD",
               "n_pay": "PAYM_NOCA", "sales": "SALE_AMT"},
    "can_m": {"time": "발생시간", "vehicle_id": "차종_식별번호", "charging": "충전중여부",
              "soc": "배터리상태_SOC", "lat": "위도", "lon": "경도"},
    "kep007": {"sido_cd": "WIAR_SIDO_CD", "sido_nm": "WIAR_SIDO_NM", "sgg_cd": "SGNG_CD", "sgg_nm": "SGNG_NM",
               "place": "INST_PLC_NM", "addr": "ADDR", "fast": "QCK_CHNG_PRE_NOEQ",
               "slow": "SLW_CHNG_PRE_NOEQ", "lat": "LTD", "lon": "LNGT", "car": "SPRT_CAKI_NM"},
}

DEFAULT_PARAMS = {
    "kepco": {
        "source": "001",                  # 변화점 탐지에 쓸 원천. 001/002 는 포함관계 불명 → 합산 금지
        "layout": "auto",                 # auto | long | wide
        "wide_hour_regex": r"(\d{1,2})\D*$",
        "hour_base": "auto",              # auto: 1~24 표기면 0~23 으로 당긴다
        "date_start": None, "date_end": None,
        "sido": None,
        "chunksize": 2_000_000,
    },
    "bjd": {"fail_warn_rate": 0.30},
    "activation": {
        "window": ["2025-03", "2025-09"],  # 최초 활성화 후보 기간(SHC 2025.01~12 안에서 사전 2개월 확보)
        "min_ratio": 1.3,                  # 전후 평균 비율 문턱
        "ratio_months": 3,                 # 전후 비교 개월 수(킬 크라이테리아 2번과 같은 방식)
        "method": "auto",                  # auto(ruptures 있으면 PELT) | ruptures | builtin
        "penalty_mult": 3.0,               # 벌점 = mult × 잡음분산 × log(n)
        "min_size": 2,
        "min_months": 12,
        "control_lookback_start": "2024-07",  # 이 시점 이후 상승 변화가 없어야 never-treated
    },
    "shc": {"chunksize": 2_000_000, "metric": "sales", "masked_markers": ["*", "-"]},
    "identification": {
        "use_log1p": False,                # True: log(S+1) — 마스킹으로 인한 결측 폭 완화용
        "k_min": -5, "k_max": 5,
        "min_units_per_k": 2,
        "control_group": "never_treated",  # never_treated | not_yet_treated
        "estimator": "auto",               # auto(CS→실패 시 회귀) | cs | regression
        "n_boot": 499, "seed": 42, "alpha": 0.05,
        "pretrend_action": "flag",         # flag(유보 표시) | exclude(표본 제외 후 재추정)
    },
    # ⚠ [9/15 mock] 초과비율 1.5 · 증가폭 0.5%p 는 월 개설 Poisson 잡음만으로 처치 17곳 중 3곳을 오탐했다.
    #   실데이터 개설률 분포를 보고 조정할 것
    "diagnostics": {"window": 2, "excess_ratio": 2.0, "min_abs_increase": 0.02},
    "heterogeneity": {
        "snapshot_months": ["2025-01", "2025-02"],
        "post_months": 3,
        "top_k_bidvs": 5,
        "min_treated_cf": 30,
        "n_folds": 5, "seed": 42,
        "cf_params": {"n_estimators": 400, "min_samples_leaf": 3, "max_depth": None},
        "split_features": ["wait_share", "outsider_share"],  # 폴백 2×2 서브그룹 축
        "can_feature_before": "2025-03",   # CAN 세션 특성은 이 달 이전 기록만(분석기간 오염 방지)
    },
    "can": {
        "enabled": True,
        "lat_range": [33.0, 39.0], "lon_range": [124.0, 132.0],
        "charging_true_values": ["1", "Y", "TRUE", "충전중", "CONNECTED"],
        "chunksize": 1_000_000,
        # verify_join_key
        "min_ids": 3,
        "min_rows_per_id": 20,
        "pair_max_gap_min": 10.0,          # 이 간격 이하 연속 레코드만 이상치 판정에 쓴다
        "max_speed_kmh": 200.0,            # 이보다 빠른 이동 = 다른 차량이 섞인 것
        "max_soc_jump": 15.0,              # 짧은 간격에서 이보다 큰 SOC 변화 = 다른 차량
        "anomaly_individual_max": 0.02,
        "anomaly_model_min": 0.10,
        # 세션
        "session_max_gap_min": 15.0,
        "session_min_min": 3.0, "session_max_min": 24 * 60.0,
        "geo_cell_m": 200.0,               # 경로B 세션 근사: 식별번호 + 위치격자
        # 거점성 클러스터링(경로A)
        "dbscan_eps_m": 150.0, "dbscan_min_samples": 3,
        "repeat_min_days": 3,
        "base_share_flag": 0.5,            # 법정동 세션 중 거점성 비율이 이 이상이면 변화점 신뢰도 낮음
        "centroid_max_km": 3.0,
        "station_match_m": 300.0,          # KEP_007 설치장소와의 공간 일치 반경
        "min_cell_count": 3,               # 산출물 소표본 억제
    },
    "kep007": {"sido_code_remap": {"42": "51", "45": "52"}},
    "load_axis": {"period": ["2025-01", "2025-12"], "cate_cut": "median", "conc_cut": "median"},
    "kill": {"sample_rows": 2_000_000, "compare_rows": 1_000_000, "kepco_max_chunks": None,
             "cf_min_regions": 30, "es_min_regions": 10, "min_tizo": 4, "mask_max_rate": 0.30},
    "outputs": {"dpi": 150, "png_max_rows": 30},
}

DEFAULT_INDUSTRY = {
    # 코드 목록. "A" 는 TOBU_BIDVS_CD 일치, "A:01" 은 대분류 A + 중분류 01 일치.
    "wait": [],        # 대기소비 가능: 카페·편의점·음식점·소매
    "control": [],     # 대기소비 불가능: 병원·학원·전문서비스
    "placebo": [],     # 위약 검정용 — control 의 부분집합. 나머지 control 이 위약의 비교 업종이 된다
    "resident_dist_codes": [],   # IFW_DISTC_SECT_CD 중 '거주자'로 볼 가장 가까운 구간
    "unknown_dist_codes": [],    # 유입거리 미상 — Y 계산에서 제외
    # TIZO_CLCD 코드 → 포함 시각(0~23). 비워 두면 시간대구간 산출물을 만들지 않는다.
    "tizo_bands": {},
    # SHC001 가맹점 상태 코드(6.5단계). FRNC_STAT_CD '등록'이 흐름(신규)인지 재고인지 현장 확인 후 open_mode 결정
    "shc001": {"open_mode": "stock_diff", "new_status_codes": [], "close_status_codes": [],
               "stock_status_codes": [], "stock_oper_codes": []},
    "labels": {},
}

PATH_KEYS = ("bjd_master", "emd_centroids", "kepco_001", "kepco_002", "shc001", "shc002",
             "can_m", "kep007", "industry_codes", "out_dir")


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    return obj


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_industry_codes(path, required=True):
    """required=False: 킬 크라이테리아처럼 코드값을 아직 모르는 첫 방문에도 설정을 읽을 수 있게 한다."""
    if not path:
        if not required:
            return copy.deepcopy(DEFAULT_INDUSTRY)
        raise ValueError("paths.industry_codes 가 필요하다(업종·유입거리 코드 매핑 — 하드코딩 금지)")
    raw = _strip_comments(json.loads(Path(path).read_text(encoding="utf-8")))
    ind = deep_merge(DEFAULT_INDUSTRY, raw)
    missing = [k for k in ("wait", "control", "resident_dist_codes") if not ind[k]]
    if missing and required:
        raise ValueError(f"업종코드 매핑 {path} 에 비어 있는 항목: {missing} — 현장에서 실제 코드값을 채울 것")
    extra = set(ind["placebo"]) - set(ind["control"])
    if extra:
        raise ValueError(f"placebo 는 control 의 부분집합이어야 한다. control 에 없는 코드: {sorted(extra)}")
    overlap = set(ind["wait"]) & set(ind["control"])
    if overlap:
        raise ValueError(f"wait 와 control 에 동시에 들어간 코드: {sorted(overlap)}")
    return ind


def load_config(path, require_industry=True):
    path = Path(path).resolve()
    raw = _strip_comments(json.loads(path.read_text(encoding="utf-8")))
    base = path.parent
    given = raw.get("paths", {})
    unknown = set(given) - set(PATH_KEYS)
    if unknown:
        raise KeyError(f"알 수 없는 paths 키: {sorted(unknown)} (허용: {PATH_KEYS})")
    paths = {key: (str((base / given[key]).resolve()) if given.get(key) else None) for key in PATH_KEYS}
    if not paths["out_dir"]:
        raise ValueError("paths.out_dir 는 필수")
    return {
        "config_path": str(path),
        "paths": paths,
        "columns": deep_merge(DEFAULT_COLUMNS, raw.get("columns")),
        "params": deep_merge(DEFAULT_PARAMS, raw.get("params")),
        "industry": load_industry_codes(paths["industry_codes"], required=require_industry),
    }
