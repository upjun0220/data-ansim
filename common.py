"""공용 유틸 — CSV 로딩(인코딩·구분자 자동 판별, 컬럼명 해석), 정규화, 월 인덱스, 공간 근사, p값.

실행 환경: 현장(안심구역) JupyterLab / 로컬 개발(PowerShell) 공통.
외부 의존성: pandas · numpy · scikit-learn 만 사용한다. scipy 도 없다고 가정하고 p값은 math 로 직접 계산한다.
단독 실행하지 않는다.
"""
from __future__ import annotations

import codecs
import importlib
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER_NAME = "ripplemap"
log = logging.getLogger(LOGGER_NAME)
EARTH_RADIUS_KM = 6371.0088


def filter_sido_code(chunk, col, sidos):
    """시도 코드 앞 2자리가 sidos 에 든 행만 남긴다. 청크마다 불러 메모리를 줄인다(sidos 가 비면 그대로)."""
    if not sidos:
        return chunk
    keep = {str(s)[:2] for s in sidos}
    return chunk[chunk[col].astype(str).str.replace(r"\D", "", regex=True).str[:2].isin(keep)]


def keep_region(df, prefix, col="bjd_code"):
    """코드 앞 2자리가 region.sido_prefix 인 행만(v11.2 서울 한정). prefix 가 비면 그대로."""
    if not prefix:
        return df
    return df[df[col].astype(str).str.startswith(str(prefix))]


def setup_logging(out_dir=None, level=logging.INFO):
    """콘솔 + out_dir/pipeline.log. 여러 번 호출해도 핸들러가 중복되지 않는다."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(out_dir) / "pipeline.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def optional_import(name):
    """현장 PC 에 없을 수 있는 패키지. 없으면 None — 호출부가 폴백 경로로 간다."""
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError 외에 의존 패키지 버전 충돌도 '없음'으로 본다
        return None


# ---------------------------------------------------------------- CSV 로딩

def detect_encoding(path, sample_bytes=4_000_000):
    """UTF-8(BOM 포함) → CP949 순으로 판별. 공공데이터 원본은 CP949 인 경우가 많다."""
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    for enc in ("utf-8-sig", "cp949"):
        try:
            codecs.getincrementaldecoder(enc)().decode(raw, final=False)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 판별 실패(UTF-8/CP949 아님): {path}")


def detect_sep(path, encoding):
    """첫 줄 기준 탭 / 파이프 / 쉼표 중 가장 많은 것."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        first = f.readline()
    counts = {"\t": first.count("\t"), "|": first.count("|"), ",": first.count(",")}
    return max(counts, key=counts.get) if max(counts.values()) else ","


def read_header(path):
    enc = detect_encoding(path)
    sep = detect_sep(path, enc)
    return list(pd.read_csv(path, encoding=enc, sep=sep, nrows=0).columns), enc, sep


def resolve_columns(columns, mapping, source, required=None):
    """내부 키 → 실제 컬럼명. 정확 일치 → 접두 일치('충전중여부' → '충전중여부(CHARGERCONNECTION)') 순.

    못 찾으면 실제 컬럼 목록을 붙여 KeyError — 현장에서 config 의 columns 만 고치면 되도록.
    """
    required = set(mapping) if required is None else set(required)
    stripped = {str(c).strip(): c for c in columns}
    found, missing = {}, []
    for key, name in mapping.items():
        if not name:
            if key in required:
                missing.append(f"{key}=(미지정)")
            continue
        if name in stripped:
            found[key] = stripped[name]
            continue
        candidates = [orig for s, orig in stripped.items() if s.startswith(name)]
        if len(candidates) == 1:
            found[key] = candidates[0]
        elif key in required:
            hint = f" (접두 일치 후보 {len(candidates)}개)" if candidates else ""
            missing.append(f"{key}={name!r}{hint}")
    if missing:
        raise KeyError(f"[{source}] 컬럼을 찾지 못함: {missing} / 실제 컬럼: {list(columns)} "
                       f"→ config 의 columns 항목에서 이름을 맞춰주세요")
    return found


def read_columns(path, mapping, source, required=None, chunksize=None, nrows=None):
    """필요한 컬럼만 문자열로 읽어 내부 키 이름으로 바꾼다. chunksize 를 주면 청크 반복자."""
    path = str(path)
    header, enc, sep = read_header(path)
    colmap = resolve_columns(header, mapping, source, required)
    rename = {orig: key for key, orig in colmap.items()}
    kwargs = dict(encoding=enc, sep=sep, usecols=list(rename), dtype=str, nrows=nrows)
    log.info("[%s] 로드: %s (인코딩 %s, 구분자 %r%s)", source, Path(path).name, enc, sep,
             f", 앞 {nrows:,}행" if nrows else "")
    if chunksize:
        return (chunk.rename(columns=rename) for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs))
    return pd.read_csv(path, **kwargs).rename(columns=rename)


# ---------------------------------------------------------------- 텍스트·숫자·코드

def norm_text(s):
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def nospace(s):
    return s.fillna("").astype(str).str.replace(r"\s+", "", regex=True)


def to_num(s):
    """천단위 쉼표 허용 숫자 변환. 실패(마스킹 '*' 등)는 NaN.

    ⚠ 반드시 float64 로 돌려준다 — nullable Float64 가 섞이면 pd.NA 때문에 numpy 불리언 인덱싱이 깨진다
      (charging-divide 9/14 mock 실행에서 확인한 pandas 3 동작).
    """
    num = pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return num.astype("float64")


def clean_code(s, width=None):
    """숫자만 남긴다. '301.0' 처럼 실수로 저장된 코드도 처리. width 를 주면 왼쪽 0 채움."""
    txt = s.fillna("").astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    txt = txt.str.replace(r"\D", "", regex=True)
    if width:
        txt = txt.where(txt == "", txt.str.zfill(width))
    return txt.replace("", np.nan)


def clean_label(s):
    """분류코드(업종·유입거리 등) — 영문자가 섞일 수 있으므로 공백·'.0' 만 정리한다."""
    return s.fillna("").astype(str).str.strip().str.replace(r"\.0+$", "", regex=True).replace("", np.nan)


# ---------------------------------------------------------------- 월 인덱스
# 월은 정수 mi = 연*12 + (월-1) 로 다룬다. 사람이 읽는 표기는 'YYYY-MM'.

def ym_to_mi(label):
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if len(digits) < 6:
        raise ValueError(f"연월 형식 아님: {label!r}")
    return int(digits[:4]) * 12 + int(digits[4:6]) - 1


def mi_to_ym(mi):
    mi = int(mi)
    return f"{mi // 12:04d}-{mi % 12 + 1:02d}"


def parse_mi(s):
    """'202503' · '2025-03' · '20250301' · '2025.03' 문자열 시리즈 → 월 인덱스(float, 실패 NaN)."""
    digits = s.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    ok = digits.str.len() >= 6
    year = pd.to_numeric(digits.str[:4].where(ok), errors="coerce")
    month = pd.to_numeric(digits.str[4:6].where(ok), errors="coerce")
    month = month.where((month >= 1) & (month <= 12))
    return (year * 12 + month - 1).astype("float64")


# ---------------------------------------------------------------- 공간

def nearest_region(lat, lon, centroids, max_km):
    """좌표 → 가장 가까운 법정동 중심점의 코드. max_km 초과는 None.

    ⚠ 폴리곤 포함 판정이 아니라 중심점 근사다. 현장에서 QGIS 공간조인 결과가 있으면 그것이 정확하다.
    """
    from sklearn.neighbors import BallTree

    cent = centroids.dropna(subset=["lat", "lon"])
    if cent.empty:
        raise ValueError("법정동 중심점 좌표가 비어 있음")
    tree = BallTree(np.radians(cent[["lat", "lon"]].to_numpy(float)), metric="haversine")
    pts = np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])
    ok = np.isfinite(pts).all(axis=1)
    codes = np.full(len(pts), None, dtype=object)
    dist = np.full(len(pts), np.nan)
    if ok.any():
        d, i = tree.query(np.radians(pts[ok]), k=1)
        d_km = d[:, 0] * EARTH_RADIUS_KM
        found = cent["bjd_code"].to_numpy()[i[:, 0]].astype(object)
        found[d_km > max_km] = None
        codes[ok] = found
        dist[ok] = d_km
    return codes, dist


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------- p값 (scipy 없이)

def norm_sf2(z):
    """양측 정규 p값."""
    if not np.isfinite(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


def _gammaincc(a, x):
    """정규화 상부 불완전감마 Q(a, x). Numerical Recipes 급수/연분수."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x, df):
    if not np.isfinite(x) or df <= 0:
        return float("nan")
    return _gammaincc(df / 2.0, x / 2.0)


def _betacf(a, b, x):
    """정규화 불완전베타의 연분수 부분(Numerical Recipes betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc_reg(a, b, x):
    """정규화 불완전베타 I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(f, d1, d2):
    """F(d1, d2) 상측 확률. P(F > f) = I_{d2/(d2+d1·f)}(d2/2, d1/2)."""
    if not np.isfinite(f) or d1 <= 0 or d2 <= 0:
        return float("nan")
    if f <= 0:
        return 1.0
    return betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def wald_test(beta, cov):
    """H0: beta = 0 결합검정. 공분산이 특이하면 유사역행렬 + 유효 자유도(rank)."""
    beta = np.asarray(beta, dtype=float)
    cov = np.asarray(cov, dtype=float)
    ok = np.isfinite(beta)
    if ok.sum() == 0:
        return float("nan"), 0, float("nan")
    beta, cov = beta[ok], cov[np.ix_(ok, ok)]
    rank = int(np.linalg.matrix_rank(cov)) if cov.size else 0
    if rank == 0:
        return float("nan"), 0, float("nan")
    stat = float(beta @ np.linalg.pinv(cov) @ beta)
    return stat, rank, chi2_sf(stat, rank)
