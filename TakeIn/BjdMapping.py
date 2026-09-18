"""0단계 지역키 정합 — KEPCO 텍스트 키(시도/시군구/읍면동) ↔ 법정동코드(10자리) ↔ SHC 코드.

입력: 팀이 별도 반입하는 법정동코드 마스터(공개 데이터). 경로는 인자로 받는다(하드코딩 금지).
  - 형식 1(기본): 법정동코드 · 법정동명("경기도 수원시 장안구 파장동") · 폐지여부
    (행정표준코드관리시스템 '법정동코드 전체자료'. 탭/쉼표, CP949/UTF-8 자동 판별)
  - 형식 2: 법정동코드 + 시도/시군구/읍면동 텍스트 열 → columns.bjd 의 sido/sigungu/emd 지정

매칭 규칙(스펙 §3):
  1. 시도+시군구+읍면동 3단 전체로만 매칭한다. 읍면동명 단독 매칭은 하지 않는다(동명이인 방지).
  2. 1차 원문(공백만 제거) 일치 → 2차 정규화('제1동'→'1동', 가운뎃점·괄호 제거) 일치.
  3. 정규화로도 못 푼 건은 미매칭 목록으로 남겨 사람이 검토한다.
  4. 매칭률표와 미매칭 목록을 항상 반환한다. 실패율이 문턱(기본 30%) 이상이면 '행정동 기준일 수 있음' 경고.
단독 실행하지 않는다.
"""
from __future__ import annotations

import re

import pandas as pd

from common import clean_code, log, norm_text, nospace, read_columns

_SIDO_FULL_TO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
    # ⚠ [9/16 확인] 2026-06-30 광주(29)·전남(46) → 전남광주통합특별시(12). 분석기간 데이터는 옛 명칭일 가능성이 높다
    "전남광주통합특별시": "전남광주",
}
SIDO_LOOKUP = dict(_SIDO_FULL_TO_SHORT)
for _short in set(_SIDO_FULL_TO_SHORT.values()):
    SIDO_LOOKUP[_short] = _short
for _short in ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"):
    SIDO_LOOKUP[_short + "시"] = _short


def sido_short(s):
    """'서울특별시'/'서울시'/'서울' → '서울', '강원도'/'강원특별자치도' → '강원'."""
    key = nospace(s)
    return key.map(SIDO_LOOKUP).fillna(key)


_RE_JE_NUM = re.compile(r"제(?=\d)")
_RE_PUNCT = re.compile(r"[·ㆍ.\-_()\[\]]")


def normalize_name(s):
    """표기 차이 완화 — 공백 제거, '제1동'→'1동', 가운뎃점·마침표·괄호 제거.

    ⚠ 숫자 앞 '제' 만 지운다. '제기동'·'제주' 처럼 이름 자체의 '제'는 건드리지 않는다.
    """
    txt = nospace(s)
    txt = txt.map(lambda v: _RE_PUNCT.sub("", _RE_JE_NUM.sub("", v)))
    return txt


def make_key(sido, sigungu, emd, normalized=False):
    """매칭 키 '시도약칭|시군구|읍면동'. 세종은 시군구가 없으므로 비운다."""
    sd = sido_short(sido)
    f = normalize_name if normalized else nospace
    sg = f(sigungu).where(sd != "세종", "")
    return sd + "|" + sg + "|" + f(emd)


def load_bjd_master(path, columns):
    """마스터 → 읍면동 단위 표(bjd_code, sido, sigungu, emd, key_raw, key_norm)."""
    cols = dict(columns)
    text_mode = all(cols.get(k) for k in ("sido", "sigungu", "emd"))
    if not text_mode:
        cols = {k: v for k, v in cols.items() if k not in ("sido", "sigungu", "emd")}
    required = ["code", "sido", "sigungu", "emd"] if text_mode else ["code", "name"]
    df = read_columns(path, cols, "법정동코드마스터", required=required)
    df["code"] = clean_code(df["code"], 10)
    df = df.dropna(subset=["code"])
    if "status" in df:
        status = norm_text(df["status"])
        dropped = int((status == "폐지").sum())
        if dropped:
            log.info("법정동 마스터: 폐지 코드 %d개 제외", dropped)
        df = df[status != "폐지"]

    code = df["code"]
    # 법정동코드 = 시도2 · 시군구3 · 읍면동3 · 리2. 읍면동 단위는 리 '00' 이고 읍면동 ≠ '000'
    is_emd = (code.str[8:] == "00") & (code.str[5:8] != "000")
    if text_mode:
        emd = df[is_emd].copy()
        for c in ("sido", "sigungu", "emd"):
            emd[c] = norm_text(emd[c])
    else:
        name = norm_text(df["name"])
        is_sido = code.str[2:] == "00000000"
        sido_full = dict(zip(code[is_sido].str[:2], name[is_sido]))
        is_sgg = (code.str[5:] == "00000") & (code.str[2:5] != "000")
        sgg_full = dict(zip(code[is_sgg].str[:5], name[is_sgg]))
        rows = []
        for c, full in zip(code[is_emd], name[is_emd]):
            sd = sido_full.get(c[:2], full.split(" ")[0])
            sg_full = sgg_full.get(c[:5], "")
            sg = sg_full[len(sd):].strip() if sg_full.startswith(sd) else sg_full
            em = full[len(sg_full):].strip() if sg_full and full.startswith(sg_full) else full.split(" ")[-1]
            rows.append((c, sd, sg, em))
        emd = pd.DataFrame(rows, columns=["code", "sido", "sigungu", "emd"])

    emd = emd.rename(columns={"code": "bjd_code"})[["bjd_code", "sido", "sigungu", "emd"]].reset_index(drop=True)
    emd["key_raw"] = make_key(emd["sido"], emd["sigungu"], emd["emd"])
    emd["key_norm"] = make_key(emd["sido"], emd["sigungu"], emd["emd"], normalized=True)
    emd["region_name"] = (emd["sido"] + " " + emd["sigungu"] + " " + emd["emd"]).str.replace(r"\s+", " ", regex=True).str.strip()
    for key in ("key_raw", "key_norm"):
        dup = emd[emd[key].duplicated(keep=False)]
        if not dup.empty:
            log.warning("법정동 마스터: 같은 %s 에 코드가 여러 개 %d건 — 이 키는 매칭하지 않고 미매칭으로 남긴다. 예: %s",
                        key, len(dup), dup[[key, "bjd_code"]].head(5).to_dict("records"))
    log.info("법정동 마스터: 읍면동 %d개", len(emd))
    return emd


def _unique_lookup(master, key):
    """중복 키는 어느 코드인지 확정할 수 없으므로 룩업에서 뺀다(임의로 첫 코드를 쓰지 않는다)."""
    counts = master[key].value_counts()
    uniq = master[master[key].isin(counts[counts == 1].index)]
    return dict(zip(uniq[key], uniq["bjd_code"]))


def match_regions(regions, master, weight_col=None, fail_warn_rate=0.30):
    """sido/sigungu/emd 텍스트 열을 가진 '고유 지역' 표에 bjd_code 를 붙인다.

    반환: (지역표 + bjd_code/match_method, 매칭률표, 미매칭 목록)
    """
    out = regions.copy()
    out["bjd_code"] = make_key(out["sido"], out["sigungu"], out["emd"]).map(_unique_lookup(master, "key_raw"))
    out["match_method"] = out["bjd_code"].notna().map({True: "원문일치", False: "미매칭"})
    miss = out["bjd_code"].isna()
    if miss.any():
        norm = make_key(out.loc[miss, "sido"], out.loc[miss, "sigungu"], out.loc[miss, "emd"], normalized=True)
        found = norm.map(_unique_lookup(master, "key_norm"))
        hit = found.notna()
        out.loc[found[hit].index, "bjd_code"] = found[hit]
        out.loc[found[hit].index, "match_method"] = "정규화일치"

    names = master.drop_duplicates("bjd_code").set_index("bjd_code")["region_name"]
    out["region_name"] = out["bjd_code"].map(names)
    raw_name = (norm_text(out["sido"]) + " " + norm_text(out["sigungu"]) + " " + norm_text(out["emd"])).str.strip()
    out["region_name"] = out["region_name"].fillna(raw_name)

    n = len(out)
    n_raw = int((out["match_method"] == "원문일치").sum())
    n_norm = int((out["match_method"] == "정규화일치").sum())
    n_miss = n - n_raw - n_norm
    fail_rate = n_miss / n if n else float("nan")
    rows = [
        {"항목": "KEPCO 고유 지역 수", "값": n, "비율": 1.0 if n else float("nan")},
        {"항목": "원문 3단 일치", "값": n_raw, "비율": n_raw / n if n else float("nan")},
        {"항목": "정규화 3단 일치", "값": n_norm, "비율": n_norm / n if n else float("nan")},
        {"항목": "미매칭(사람 검토)", "값": n_miss, "비율": fail_rate},
    ]
    if weight_col and weight_col in out:
        total = out[weight_col].sum()
        share = out.loc[out["bjd_code"].isna(), weight_col].sum() / total if total else float("nan")
        rows.append({"항목": f"미매칭 {weight_col} 비중", "값": float(out.loc[out['bjd_code'].isna(), weight_col].sum()),
                     "비율": share})
    rate_table = pd.DataFrame(rows)

    unmatched = out[out["bjd_code"].isna()][["sido", "sigungu", "emd"] + ([weight_col] if weight_col in out else [])]
    unmatched = unmatched.assign(정규화키=make_key(unmatched["sido"], unmatched["sigungu"], unmatched["emd"], normalized=True))
    if weight_col in unmatched:
        unmatched = unmatched.sort_values(weight_col, ascending=False)

    if n_miss:
        log.warning("법정동 매칭 실패 %d/%d (%.1f%%). 예: %s", n_miss, n, 100 * fail_rate,
                    unmatched[["sido", "sigungu", "emd"]].head(10).agg(" ".join, axis=1).tolist())
    if n and fail_rate >= fail_warn_rate:
        log.warning("⚠ 법정동 매칭 실패율 %.1f%% ≥ %.0f%% — 한전 '읍면동'이 법정동이 아니라 행정동 기준일 수 있음 "
                    "(킬 크라이테리아 5번). 매핑표(행정동↔법정동) 확보 전까지 결과 해석 유보",
                    100 * fail_rate, 100 * fail_warn_rate)
    else:
        log.info("법정동 매칭: 원문 %d · 정규화 %d · 미매칭 %d (실패율 %.1f%%)", n_raw, n_norm, n_miss, 100 * fail_rate)
    return out, rate_table, unmatched.reset_index(drop=True), fail_rate


def shc_bjd_code(sido_cd, sgg_cd, umd_cd):
    """SHC001/002 의 WIAR_SIDO_CD + SGNG_CD + UMD_CD → 법정동코드 10자리.

    세 컬럼 모두 VARCHAR2(10) 이라 저장 형태가 불명이다. 아래를 모두 받는다.
      - UMD_CD 가 10자리(법정동코드 전체)          → 그대로
      - UMD_CD 8자리(시도+시군구+읍면동)           → + '00'
      - SGNG_CD 5자리(시도+시군구) + UMD_CD 3자리  → SGNG_CD + UMD_CD + '00'
      - 시도 2 + 시군구 3 + 읍면동 3               → 이어 붙이고 + '00'
    """
    sd = clean_code(sido_cd).fillna("")
    sg = clean_code(sgg_cd).fillna("")
    um = clean_code(umd_cd).fillna("")
    out = pd.Series(pd.NA, index=um.index, dtype="object")
    full10 = um.str.len() == 10
    out[full10] = um[full10]
    len8 = um.str.len() == 8
    out[len8] = um[len8] + "00"
    rest = ~(full10 | len8) & (um.str.len() > 0)
    um3 = um.str.zfill(3).str[-3:]
    sg5 = sg.where(sg.str.len() >= 5, sd.str.zfill(2) + sg.str.zfill(3)).str[-5:]
    out[rest] = sg5[rest] + um3[rest] + "00"
    bad = out.isna() | (out.astype(str).str.len() != 10)
    return out.where(~bad, pd.NA)


def load_crosswalk(path, columns):
    """법정동 코드대응표 → {코드: 기준코드}.

    분석기간 중 개편된 법정동(예: 2023-12 부천 3구 신설)은 같은 동이 기간에 따라 다른 코드·명칭으로 나온다.
    한 기준코드로 묶지 않으면 KEPCO 시계열이 두 법정동으로 쪼개지고 SHC 와 조인이 끊긴다.
    """
    df = read_columns(path, columns, "법정동코드대응")
    code, canon = clean_code(df["code"], 10), clean_code(df["canonical"], 10)
    ok = code.notna() & canon.notna() & (code != canon)
    code_ok, canon_ok = code[ok], canon[ok]
    conflicting = sorted({c for c in code_ok[code_ok.duplicated(keep=False)]
                           if canon_ok[code_ok == c].nunique() > 1})
    if conflicting:
        raise ValueError(f"코드대응표에 같은 코드가 서로 다른 기준코드로 중복 매핑됨: {conflicting[:5]} — 대응표를 정리할 것")
    mapping = dict(zip(code_ok, canon_ok))
    chained = sorted(c for c in set(mapping.values()) if c in mapping)
    if chained:
        raise ValueError(f"코드대응이 연쇄됨(기준코드가 다시 다른 코드로 대응): {chained[:5]} — 대응표를 한 단계로 정리할 것")
    log.info("법정동 코드대응: %d개 코드 → 기준코드", len(mapping))
    return mapping


def canonicalize(codes, mapping):
    """법정동코드 시리즈를 기준코드로 바꾼다. 대응표에 없는 코드·결측은 그대로."""
    if not mapping:
        return codes
    return codes.map(mapping).fillna(codes)


def load_emd_centroids(path, columns):
    """법정동 중심점(법정동코드, 위도, 경도). QGIS 에서 경계 폴리곤 → 중심점 → CSV 로 내보낸 것."""
    from common import to_num

    df = read_columns(path, columns, "법정동중심점")
    out = pd.DataFrame({"bjd_code": clean_code(df["code"], 10), "lat": to_num(df["lat"]), "lon": to_num(df["lon"])})
    out = out.dropna()
    log.info("법정동 중심점 %d개", len(out))
    return out
