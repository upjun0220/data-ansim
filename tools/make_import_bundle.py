"""반입 폴더 만들기 — 코드 파일을 하나씩 따로 반입한다(한 폴더에 평평하게 놓이는 구성).

반입 포털은 .py·.csv·.txt·폰트만 받고(zip·json·md 불가) 파일명은 영문·숫자만 된다. 그래서
  - 분석 모듈 20개는 원래 이름 그대로(.py),
  - 설정 템플릿(json)·공휴일 달력(yaml)·README(md)는 내용 그대로 .txt 로 바꿔 담는다. 코드는 확장자가 아니라
    내용(JSON)으로 읽으므로 그대로 동작한다.
  - 올린 파일은 센터에서 한 폴더에 평평하게 놓이므로, 경로가 같은 폴더를 가리키는 fieldtemplate.txt 를 따로 만든다.
  - checksums.txt 에 파일마다 SHA-256 을 적어, 센터에서 일부만 옛 버전이거나 전송 중 깨졌는지 확인한다.
데이터 파일은 이전 반입 이름과 겹치지 않게 숫자로 매긴다(IMPORT_NAMES — 2~9.csv, 10.ttf).

사용: python tools/make_import_bundle.py [출력폴더]        (기본 dist/import)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["bjdmapping", "canloader", "common", "config", "diagnostics", "equityaccess", "essoptimizer", "headline",
           "heterogeneity", "identification", "kep007loader", "kepcoloader", "killcriteria", "loadaxis", "loadforecast",
           "outputs", "pipeline", "priorityscore", "weatherloader", "loadscenario"]
# 원본 → 반입 이름(.txt). fieldtemplate.txt 는 flat_template() 이 한 폴더용으로 만든다.
TEXT_FILES = {"config/holidays.yaml": "holidays.txt", "requirements.txt": "requirements.txt", "README.md": "README.txt"}
CHECKSUMS = "checksums.txt"

# 데이터 반입 이름(영문·숫자만, 이전 반입분과 겹치지 않게 숫자). 키 = 반입 이름, 값 = 내용.
IMPORT_NAMES = {
    "2.csv": "법정동코드 마스터(build_reference_files.py 의 bjdmaster.csv)",
    "3.csv": "법정동 코드대응(bjdcrosswalk.csv)",
    "4.csv": "법정동 중심점(bjdcentroids.csv)",
    "5.csv": "SMP(timestamp,smp)",
    "6.csv": "공용 충전소 위치(위도·경도·충전기수)",
    "7.csv": "행정동별 연료별 자동차 등록 월별 이력(OA-21236)",
    "8.csv": "행정동→법정동 대응표(행정동코드·법정동코드·가중치)",
    "9.csv": "기온 실측·과거 예보(timestamp,station_or_grid,temp_obs_c,temp_fcst_c,fcst_issued_at)",
    "10.ttf": "한글 폰트(서버에 없을 때만)",
}
# 한 폴더 구성에서 설정 경로 → 반입 이름. KEPCO·CAN 은 센터가 주는 원자료라 센터에서 경로를 적는다.
FLAT_PATHS = {"bjd_master": "2.csv", "bjd_crosswalk": "3.csv", "emd_centroids": "4.csv", "smp": "5.csv",
              "access_stations": "6.csv", "ev_history": "7.csv", "hdong_bjd": "8.csv", "weather": "9.csv",
              "holidays": "holidays.txt", "out_dir": "out", "font": None, "ev_registration": None}


def flat_template():
    """config/fieldtemplate.json → 반입 파일이 모두 같은 폴더에 있을 때의 경로로 바꾼 설정 원본."""
    tpl = json.loads((ROOT / "config" / "fieldtemplate.json").read_text(encoding="utf-8"))
    paths = tpl["paths"]
    for key, value in FLAT_PATHS.items():
        paths[key] = value
    for key in ("kepco_001", "kepco_002", "kepco_hourly", "can_m"):
        paths[key] = "센터경로를적을것_" + key
    paths["_반입파일명"] = "반입 파일이 모두 이 설정 파일과 같은 폴더에 있다고 보고 경로를 적었다. 폰트를 가져왔으면 font 에 \"10.ttf\""
    tpl["_설명"] = ("센터용 설정 원본 — field.txt 로 복사해 KEPCO·CAN 경로(센터가 알려 준 실제 경로)와 "
                   "params.energy.evaluation_start 를 채운다. 확장자는 .txt 지만 내용은 JSON 이다.")
    return json.dumps(tpl, ensure_ascii=False, indent=2) + "\n"


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for m in MODULES:
        shutil.copyfile(ROOT / f"{m}.py", out / f"{m}.py")
        names.append(f"{m}.py")
    for src, dst in TEXT_FILES.items():
        (out / dst).write_text((ROOT / src).read_text(encoding="utf-8"), encoding="utf-8")
        names.append(dst)
    (out / "fieldtemplate.txt").write_text(flat_template(), encoding="utf-8")
    names.append("fieldtemplate.txt")
    lines = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in sorted(names)]
    (out / CHECKSUMS).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, names


def verify_checksums(folder):
    """센터에서도 쓸 수 있는 확인: checksums.txt 에 적힌 파일이 모두 있고 해시가 맞는지. → 문제 목록."""
    folder = Path(folder)
    problems = []
    for line in (folder / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = folder / name
        if not path.exists():
            problems.append(f"없음: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"해시 불일치: {name}")
    return problems


def verify(folder):
    """새 폴더에 복사해 해시·모듈 import·설정 읽기를 확인한다."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "dataansim11"
        shutil.copytree(folder, copy)
        problems = verify_checksums(copy)
        if problems:
            raise SystemExit("; ".join(problems))
        code = ("import " + ",".join(MODULES) + "; from config import load_config; "
                "c = load_config('fieldtemplate.txt'); from weatherloader import load_holidays; "
                "print('import ok:', %d, 'modules · 공휴일', len(load_holidays(c['paths']['holidays'])), '일')" % len(MODULES))
        subprocess.run([sys.executable, "-B", "-c", code], cwd=copy, check=True)


if __name__ == "__main__":
    folder, names = build(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist" / "import")
    print(f"만듦: {folder} · 코드·설정 {len(names)}개 + {CHECKSUMS}")
    verify(folder)
