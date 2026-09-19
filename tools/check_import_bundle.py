"""반입 묶음 점검 — 코드 zip 1 + 데이터 CSV + 폰트가 반입 규정(최대 10개·총 50MB·CSV만)에 맞는지 표로 보인다.

사용: python tools/check_import_bundle.py <폴더 또는 파일> [...]   (위반이 있으면 종료코드 1)
"""
from __future__ import annotations

import sys
from pathlib import Path

MAX_FILES, MAX_BYTES = 10, 50 * 1024 * 1024
FONT_EXT = {".ttf", ".otf", ".ttc"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    return files


def kind(f):
    ext = f.suffix.lower()
    return "code" if ext == ".zip" else "data" if ext == ".csv" else "font" if ext in FONT_EXT else "other"


def check(files):
    """(행 목록, 위반 목록). 폰트는 파일 수·용량에 포함하되 없어도 위반은 아니다(현장 서버 폰트 사용 가능)."""
    rows = [(f.name, kind(f), f.stat().st_size) for f in files]
    total = sum(r[2] for r in rows)
    kinds = [r[1] for r in rows]
    problems = []
    if len(rows) > MAX_FILES:
        problems.append(f"파일 {len(rows)}개 > 최대 {MAX_FILES}개")
    if total > MAX_BYTES:
        problems.append(f"총 {total / 1024 ** 2:.1f}MB > 최대 {MAX_BYTES // 1024 ** 2}MB")
    if kinds.count("code") != 1:
        problems.append(f"코드 zip {kinds.count('code')}개(정확히 1개여야 함)")
    problems += [f"허용되지 않는 형식: {n}" for n, k, _ in rows if k == "other"]
    return rows, problems


def main(argv):
    if not argv:
        sys.exit(__doc__)
    rows, problems = check(collect(argv))
    print(f"{'파일':<40}{'구분':<8}{'MB':>8}")
    for name, k, size in rows:
        print(f"{name:<40}{k:<8}{size / 1024 ** 2:>8.2f}")
    kinds = [r[1] for r in rows]
    print(f"\n합계 {len(rows)}개 · {sum(r[2] for r in rows) / 1024 ** 2:.1f}MB · 코드 {kinds.count('code')} · 데이터 {kinds.count('data')}"
          f" · 폰트 {kinds.count('font')}" + ("" if "font" in kinds else " (폰트 없음 — 서버 폰트 확인)"))
    print("판정: " + ("통과" if not problems else "위반 — " + "; ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main(sys.argv[1:])