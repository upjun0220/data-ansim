"""9단계 반출 — 모든 표는 CSV + PNG 를 동시에 만든다(스펙 §9).

  out_dir/csv/<이름>.csv  — 현장 작업용. 반출 대상 아님.
  out_dir/png/<이름>.png  — 같은 표·차트를 이미지로. 실제 반출용.

GPS 규칙: 위도/경도/좌표 컬럼이 든 표는 CSV·PNG 모두 저장을 거부한다(ValueError).
CAN 산출물은 법정동 집계만 넘어오도록 호출부에서 만들고, 여기서 한 번 더 막는다.
그림(figure)은 호출부가 좌표를 그리지 않는다는 전제 — 산점도는 법정동 단위 통계값끼리만 그린다.

pyplot 을 쓰지 않는다(Figure + Agg 캔버스) — JupyterLab 의 inline 백엔드 상태를 건드리지 않기 위해서.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from common import log

KOREAN_FONTS = ("Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic",
                "Noto Sans CJK KR", "Noto Sans KR", "UnDotum")
GPS_COLUMN = re.compile(r"(위도|경도|좌표|gps|latitude|longitude)", re.IGNORECASE)
GPS_SHORT_TOKENS = {"lat", "lon", "lng", "ltd", "lngt"}
_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z]+")
_GLYPH_FALLBACK = str.maketrans({"−": "-", "≈": "~", "≥": ">=", "≤": "<=", "τ": "tau", "×": "x"})


def setup_korean_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return name
    log.warning("한글 폰트를 찾지 못함 — PNG 의 한글이 네모로 깨질 수 있음 (후보: %s)", ", ".join(KOREAN_FONTS))
    return None


def new_figure(width=8.0, height=4.5):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def png_text(text):
    return str(text).translate(_GLYPH_FALLBACK)


def _text_units(text):
    return sum(2 if ord(ch) >= 0x1100 else 1 for ch in text)


def _fmt(value, digits):
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "예" if value else "아니오"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return str(value)


def suppress_small(df, count_col, min_count, cols=None):
    """소표본 억제 — count_col < min_count 인 행의 수치를 NaN('—')으로 가린다. 개체 식별 방지."""
    out = df.copy()
    small = out[count_col] < min_count
    targets = cols or [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in targets:
        out[c] = out[c].astype("float64")
        out.loc[small, c] = np.nan
    if small.any():
        log.info("소표본 억제: %d행(%s < %d)", int(small.sum()), count_col, min_count)
    return out


class OutputWriter:
    def __init__(self, out_dir, dpi=150, png_max_rows=30):
        self.out_dir = Path(out_dir)
        self.csv_dir = self.out_dir / "csv"
        self.png_dir = self.out_dir / "png"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.png_max_rows = png_max_rows
        self.manifest = []
        self.font = setup_korean_font()

    @staticmethod
    def assert_no_gps(df, name):
        def is_gps_column(c):
            s = str(c)
            if GPS_COLUMN.search(s):
                return True
            # 짧은 별칭(lat/lon/...)은 부분일치가 아니라 '_'·기호로 나뉜 토큰 전체 일치만 잡는다.
            # 그래야 merge suffix(lat_x)·파생 컬럼명(home_lat, site_lon)도 놓치지 않는다.
            tokens = _TOKEN_SPLIT.split(s.lower())
            return any(t in GPS_SHORT_TOKENS for t in tokens if t)

        bad = [c for c in df.columns if is_gps_column(c)]
        if bad:
            raise ValueError(f"[{name}] 좌표 컬럼 {bad} 이 포함된 표는 저장하지 않는다(반출 규칙). 법정동 단위로 집계할 것")

    def table(self, df, name, title, digits=4, note=""):
        """표 → CSV + PNG. PNG 는 상위 png_max_rows 행."""
        df = df.reset_index(drop=True)
        self.assert_no_gps(df, name)
        csv_path = self.csv_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self.manifest.append((name, "csv", str(csv_path)))

        shown = df.head(self.png_max_rows)
        nrows = len(shown)
        header = title if len(df) <= self.png_max_rows else f"{title}  (상위 {nrows}행 / 전체 {len(df)}행)"
        if note:
            header += f"\n{note}"
        labels = [png_text(c) for c in df.columns]
        cells = [[png_text(_fmt(v, digits)) for v in row] for row in shown.itertuples(index=False, name=None)]
        units = [max([_text_units(labels[j])] + [_text_units(r[j]) for r in cells]) + 2 for j in range(len(labels))] or [10]
        width = min(max(6.0, 0.075 * sum(units)), 30.0)
        fig = new_figure(width=width, height=0.25 * (max(nrows, 1) + 1) + (0.25 if note else 0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(png_text(header), fontsize=10, loc="left")
        if nrows == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=12)
        else:
            tbl = ax.table(cellText=cells, colLabels=labels, cellLoc="center", bbox=[0, 0, 1, 1],
                           colWidths=[u / sum(units) for u in units])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            for (r, _c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#E8EEF4")
        png_path = self.png_dir / f"{name}.png"
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        self.manifest.append((name, "png", str(png_path)))
        return csv_path, png_path

    def figure(self, fig, name):
        png_path = self.png_dir / f"{name}.png"
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        self.manifest.append((name, "png", str(png_path)))
        return png_path

    def manifest_table(self):
        return pd.DataFrame(self.manifest, columns=["산출물", "형식", "경로"])


# ---------------------------------------------------------------- 차트

def plot_activation_examples(daily, activation, max_regions=6):
    """처치 지역 일부의 월 일평균 충전량과 T_r. 법정동 집계값만 그린다."""
    treated = activation[activation["status"] == "treated"].head(max_regions)
    fig = new_figure(9, 5)
    ax = fig.add_subplot(111)
    for _, row in treated.iterrows():
        s = daily[daily["bjd_code"] == row["bjd_code"]].sort_values("mi")
        line = ax.plot(s["mi"], s["kwh_per_day"], lw=1.2, label=f"{row['bjd_code']} (T_r {row['T_r']})")[0]
        ax.axvline(row["T_r_mi"], color=line.get_color(), ls="--", lw=0.8)
    ticks = sorted(daily["mi"].unique())[::3]
    from common import mi_to_ym

    ax.set_xticks(ticks, [mi_to_ym(t) for t in ticks], rotation=45, fontsize=7)
    ax.set_ylabel("일평균 충전량 (kWh/일)")
    ax.set_title("2단계 활성화 시점 예시 (점선 = T_r)")
    ax.legend(fontsize=7)
    return fig


def plot_event_study(series, title):
    """series: {라벨: DataFrame(k, tau, ci_lo, ci_hi)}."""
    fig = new_figure(8, 4.5)
    ax = fig.add_subplot(111)
    offsets = np.linspace(-0.15, 0.15, max(len(series), 1))
    for off, (label, df) in zip(offsets, series.items()):
        if df is None or df.empty:
            continue
        ax.errorbar(df["k"] + off, df["tau"], yerr=[df["tau"] - df["ci_lo"], df["ci_hi"] - df["tau"]],
                    fmt="o-", ms=4, capsize=3, lw=1, label=png_text(label))
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(-0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("활성화 후 개월 k (k=-1 기준)")
    ax.set_ylabel("tau(k)  (Y_ddd 로그차)")
    ax.set_title(png_text(title))
    ax.legend(fontsize=8)
    return fig


def plot_quadrants(df, cate_cut, conc_cut):
    """4사분면 — x: CATE, y: 부하 집중도. 유보 지역은 빈 원으로 구분."""
    fig = new_figure(7.5, 6)
    ax = fig.add_subplot(111)
    colors = {"① 접근성·수요 확인": "#1F77B4", "② 부하 완화 검토": "#FF7F0E",
              "③ 공공 필요 확인": "#2CA02C", "④ 부하 완화·공공 필요 검토": "#7F7F7F"}
    for label, g in df.groupby("quadrant"):
        hold = g["reserved"].astype(bool)
        c = colors.get(label, "black")
        ax.scatter(g.loc[~hold, "cate"], g.loc[~hold, "concentration"], s=36, color=c, label=png_text(label))
        if hold.any():
            ax.scatter(g.loc[hold, "cate"], g.loc[hold, "concentration"], s=60, facecolors="none", edgecolors=c,
                       linewidths=1.5, label=png_text(f"{label} (유보)"))
    ax.axvline(cate_cut, color="gray", ls="--", lw=0.8)
    ax.axhline(conc_cut, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("CATE 추정치 (상권 효과)")
    ax.set_ylabel("부하 집중도 = max_h 평균kW / Σ_h 평균kW")
    ax.set_title("8단계 4사분면 처방 (빈 원 = 처치오염·사전추세 유보)")
    ax.legend(fontsize=7, loc="best")
    return fig


def plot_histogram(counts, title, xlabel):
    """counts: DataFrame(bin_label, n) — 이미 구간 집계된 값만 받는다."""
    fig = new_figure(8, 4)
    ax = fig.add_subplot(111)
    ax.bar(range(len(counts)), counts["n"], color="#4C72B0")
    ax.set_xticks(range(len(counts)), [png_text(b) for b in counts["bin"]], rotation=45, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("세션 수")
    ax.set_title(png_text(title))
    return fig


def plot_bar(df, x, y, title, ylabel):
    fig = new_figure(9, 4)
    ax = fig.add_subplot(111)
    ax.bar(range(len(df)), df[y], color="#55A868")
    ax.set_xticks(range(len(df)), [png_text(v) for v in df[x]], rotation=60, fontsize=7)
    ax.set_ylabel(png_text(ylabel))
    ax.set_title(png_text(title))
    return fig
