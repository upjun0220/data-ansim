"""반입용 단일 파일 만들기 — 반입 가능한 형식(.py·.csv·.txt·폰트)에 맞춰 분석 코드 전체를 dataansimbundle.py 한 파일에 담는다.

zip·json·md 는 반입할 수 없어서, 코드(.py 18개)·설정 템플릿(json)·README·requirements 의 원문을 이 파일 안에 그대로 넣는다.
센터에서 `%run dataansimbundle.py` 를 실행하면 dataansim/ 폴더에 원래 파일들이 풀린다(SHA-256 으로 검증).

사용: python tools/make_import_bundle.py [출력폴더]        (기본 dist/import)
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["bjdmapping", "canloader", "common", "config", "diagnostics", "equityaccess", "essoptimizer", "headline",
           "heterogeneity", "identification", "kep007loader", "kepcoloader", "killcriteria", "loadaxis", "loadforecast",
           "outputs", "pipeline", "priorityscore"]
FILES = [f"{m}.py" for m in MODULES] + ["config/fieldtemplate.json", "config/industrycodestemplate.json",
                                         "requirements.txt", "README.md"]

HEADER = '''"""dataansimbundle.py — 충전 리플맵 v10 반입용 단일 파일.

반입 가능한 형식(.py·.csv·.txt·폰트) 때문에 분석 코드 전체를 이 파일 하나에 원문 그대로 담았다. 아래 FILES 에 각 파일의
내용이 그대로 있어 읽어서 검토할 수 있다(인코딩·압축 없음).

사용(센터 JupyterLab):
    %run dataansimbundle.py            현재 폴더에 dataansim/ 를 만든다
    python dataansimbundle.py [대상폴더] [--force]

동작: FILES 의 내용을 대상 폴더에 파일로 쓰고 SHA-256(HASHES)으로 검증한다. 그 밖의 일(네트워크·실행·삭제)은 하지 않는다.
이미 다른 내용의 파일이 있으면 건너뛰고 알린다(덮어쓰려면 --force).
"""
import hashlib
import sys
from pathlib import Path

'''

FOOTER = '''

def main(dest="dataansim", force=False):
    root = Path(dest)
    wrote, skipped = [], []
    for name, text in FILES.items():
        path = root / name
        if path.exists() and path.read_text(encoding="utf-8") != text and not force:
            skipped.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\\n")
        wrote.append(name)
    bad = [n for n in FILES if n not in skipped
           and hashlib.sha256((root / n).read_bytes()).hexdigest() != HASHES[n]]
    print(f"{root.resolve()} 에 {len(wrote)}개 파일을 씀 · 건너뜀 {len(skipped)}개 · 해시 불일치 {len(bad)}개")
    for n in skipped:
        print("  건너뜀(이미 다른 내용이 있음):", n)
    for n in bad:
        print("  해시 불일치:", n)
    return not bad


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "dataansim", "--force" in sys.argv)
'''


def build(out_dir):
    entries, hashes = [], {}
    for rel in FILES:
        text = (ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")
        assert "'''" not in text, f"{rel}: ''' 가 들어 있어 원문 그대로 담을 수 없음"
        assert not text.endswith("\\"), rel
        hashes[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entries.append(f"    {rel!r}: r'''{text}''',")
    body = HEADER + "FILES = {\n" + "\n".join(entries) + "\n}\n\nHASHES = {\n" + \
        "\n".join(f"    {k!r}: {v!r}," for k, v in hashes.items()) + "\n}\n" + FOOTER
    ast.parse(body)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "dataansimbundle.py"
    target.write_text(body, encoding="utf-8", newline="\n")
    return target


def verify(target):
    """새 폴더에 풀어서 해시를 확인하고 모든 모듈을 import 해 본다."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(target), str(Path(tmp) / "dataansim")], check=True)
        code = "import " + ",".join(MODULES) + "; print('import ok:', len(MODULES), 'modules')"
        subprocess.run([sys.executable, "-B", "-c", code.replace("len(MODULES)", str(len(MODULES)))],
                       cwd=Path(tmp) / "dataansim", check=True)


if __name__ == "__main__":
    target = build(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist" / "import")
    print("만듦:", target, f"{target.stat().st_size / 1024:.0f}KB")
    verify(target)
