"""플랜 B(월별 KEPCO) 위험 축과 우선순위 지도 — 월별이면 8-A 없이 8-E·지도·동네 카드까지 가야 한다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mock_data  # noqa: E402
import priorityscore  # noqa: E402
from killcriteria import run_checks  # noqa: E402
from outputs import plot_priority_map  # noqa: E402
from pipeline import run  # noqa: E402


def _mh(level_by_month, code, n_days=30):
    """월 × 24시 표. 19시에만 level, 나머지는 절반."""
    rows = []
    for mi, level in enumerate(level_by_month):
        for h in range(24):
            rows.append({"bjd_code": code, "mi": 1000 + mi, "hour": h, "n_days": n_days,
                         "kwh": (level if h == 19 else level / 2) * n_days, "cust_min": 10})
    return pd.DataFrame(rows)


def test_monthly_risk_counts_months_over_calibration_max():
    grow = _mh([10.0] * 12 + [10.0, 12.0, 13.0, 11.5, 14.0, 10.5], "G")       # 교정 최대 10 → 1.2배(12) 넘은 달 3개
    flat = _mh([10.0] * 18, "F")
    low = _mh([0.5] * 18, "L")
    short = _mh([np.nan] * 8 + [10.0] * 10, "S")                               # 교정 12달 중 4달만
    r = priorityscore.monthly_risk(pd.concat([grow, flat, low, short]), [1.1, 1.2], min_calib_peak_kw=1.0)
    at = r[r["multiplier"] == 1.2].set_index("bjd_code")
    assert at.at["G", "surge_risk"] == pytest.approx(2 / 6) and at.at["G", "peak_ratio"] == pytest.approx(1.4)
    assert r[r["multiplier"] == 1.1].set_index("bjd_code").at["G", "surge_risk"] == pytest.approx(4 / 6)
    assert at.at["F", "surge_risk"] == 0 and at.at["F", "status"] == "assessed"
    assert at.at["L", "status"] == "not_assessed_low_load" and np.isnan(at.at["L", "surge_risk"])
    assert at.at["S", "status"] == "data_missing"
    with pytest.raises(ValueError, match="개월 이상"):
        priorityscore.monthly_risk(_mh([10.0] * 10, "X"), [1.2])


def test_priority_map_hides_coordinates_and_small_regions():
    pr = pd.DataFrame({"bjd_code": list("ABCD"), "rank": [1, 2, np.nan, 3], "rank_eligible": [True, True, False, True],
                       "PriorityScore": [0.9, 0.5, np.nan, 0.2]})
    cent = pd.DataFrame({"bjd_code": list("ABCD"), "lat": [37.5, 37.6, 37.55, 37.52], "lon": [127.0, 127.1, 126.9, 127.05]})
    fig = plot_priority_map(pr, cent, visible={"A", "C", "D"}, top_n=2)
    ax = fig.axes[0]
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []            # 좌표 눈금 없음
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1", "3"]                                                     # B(소표본)는 번호·색 없음


@pytest.fixture(scope="module")
def monthly_result(tmp_path_factory):
    base = tmp_path_factory.mktemp("planb")
    data = base / "data"
    mock_data.generate(data, seed=42)
    names = {"bjd_master": "bjd_master.txt", "emd_centroids": "bjd_centroids.csv", "kepco_001": "kepco_001.csv",
             "access_stations": "access_stations.csv", "ev_registration": "ev_registration.csv"}
    cfg = {"paths": {**{k: str(data / v) for k, v in names.items()}, "out_dir": str(base / "out")},
           "params": {"energy": {"enabled": False}, "priority": {"enabled": True},
                      "kill": {"sample_rows": 200000, "compare_rows": 200000}}}
    path = base / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path, run(path)


def test_monthly_kepco_still_gives_priority_map_and_cards(monthly_result):
    path, res = monthly_result
    st = res["stages"].set_index("이름")
    assert st.at["투자 검토 결합점수", "상태"] == "완료" and "플랜 B" in st.at["투자 검토 결합점수", "비고"]
    assert st.at["우선순위 지도", "상태"] == "완료" and st.at["발표 3숫자", "상태"] == "완료", res["stages"].to_string()
    out = Path(res["out_dir"])
    assert (out / "png" / "s9_priority_map.png").exists() and (out / "png" / "s9_region_cards.png").exists()
    story = pd.read_csv(out / "csv" / "s9_story.csv", encoding="utf-8-sig")
    assert "월별" in story.loc[0, "내용"]
    pr = pd.read_csv(out / "csv" / "s8e_priority.csv", encoding="utf-8-sig")
    assert "소표본억제" in pr and pr["rank"].notna().all()                           # 월별에서도 소표본 표시가 붙는다
    kill = run_checks(path, write=False)
    assert not (kill["판정"] == "오류").any(), kill.to_string()


def test_png_drops_columns_equal_to_raw_provided_values(tmp_path, monkeypatch):
    """반출 규칙: 제공데이터 값(기간 최대 kW·시간별 실측 등)이 그대로 든 열은 PNG 에서 빠지고 CSV(내부)에는 남는다."""
    import outputs
    seen, real = {}, outputs.new_figure
    monkeypatch.setattr(outputs, "new_figure", lambda *a, **k: seen.setdefault("fig", real(*a, **k)))
    df = pd.DataFrame({"bjd_code": ["1111010100"], "calib_peak_kw": [12.3], "target_kw": [14.8],
                       "surge_risk": [0.25], "actual_grid_kw": [9.9]})
    outputs.OutputWriter(tmp_path).table(df, "t", "제목")
    texts = " ".join(c.get_text().get_text() for c in seen["fig"].axes[0].tables[0].get_celld().values())
    assert "surge_risk" in texts and "12.3" not in texts and "14.8" not in texts and "9.9" not in texts
    assert "calib_peak_kw" in pd.read_csv(tmp_path / "csv" / "t.csv").columns       # 내부 CSV 는 그대로
