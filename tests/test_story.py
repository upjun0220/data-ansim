"""심사 대응 보강 — 8-G 충전기 추가 실험, 발견 문장·새 3숫자, 동네 카드, 계절 창 비교. 합성 자료."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import equityaccess
import headline
from fieldday1 import compare_windows, season_windows

# 동서로 1km 간격 법정동 5곳. 충전소는 서쪽 끝(A)에만 있다 → 동쪽 동네일수록 접근성 0.
POINTS = pd.DataFrame({"bjd_code": list("ABCDE"), "lat": 37.5, "lon": [127.0 + 0.0113 * i for i in range(5)],
                       "ev_count": [100, 80, 60, 40, 20]})
STATIONS = pd.DataFrame({"lat": [37.5], "lon": [127.0], "chargers": [10.0]})


def test_adding_chargers_where_access_is_zero_beats_adding_where_demand_is_high():
    sim = equityaccess.simulate_added_chargers(STATIONS, POINTS, {"접근성 낮은 곳": ["E", "D"], "수요 많은 곳": ["A", "B"]},
                                               chargers_each=2, share=0.4)
    t = sim.set_index("전략")
    assert t.at["접근성 낮은 곳", "접근성 0 동네(전)"] == 4
    assert t.at["접근성 낮은 곳", "접근성 0 동네(후)"] == 2          # D·E 가 0 에서 벗어남
    assert t.at["수요 많은 곳", "접근성 0 동네(후)"] == 3            # B 만 벗어남
    assert t.at["접근성 낮은 곳", "추가 충전기(기)"] == 4
    assert t.at["접근성 낮은 곳", "하위 40% 평균 접근성(후)"] > t.at["수요 많은 곳", "하위 40% 평균 접근성(후)"]


def _head(r_ai, r_b):
    return pd.DataFrame([
        {"번호": "숫자 1", "내용": "충전기 1기당 전기차 대수의 지역 간 배수", "값": 4.2, "비교": np.nan},
        {"번호": "숫자 2", "내용": "증설 0대 급증 경보 정밀도 — AI(값) 대 기준 모델(비교)", "값": 0.6, "비교": 0.4},
        {"번호": "숫자 2", "내용": "증설 0대 급증 경보 재현율 — AI(값) 대 기준 모델(비교)", "값": r_ai, "비교": r_b}])


def test_story_sentence_reports_ai_vs_baseline_honestly():
    sim = pd.DataFrame({"전략": ["우선순위 상위 20곳", "전기차 등록 상위 20곳(비교)"], "하위 20% 개선율": [0.30, 0.10]})
    story = headline.story_table(_head(0.72, 0.51), sim)
    sentence = story.loc[story["구분"] == "발견 문장(초안)", "내용"].iloc[0]
    assert "72%" in sentence and "51%" in sentence and "21%p" in sentence and "+30%" in sentence and "+10%" in sentence
    assert set(story["구분"]) >= {"숫자 A 격차", "숫자 B 예측", "숫자 C 개입"}

    zero_sim = equityaccess.simulate_added_chargers(STATIONS, POINTS, {"우선순위": ["E", "D"], "비교": ["A", "B"]}, 2, share=0.4)
    assert "접근성 0 동네가 4→2곳" in headline.story_table(_head(0.72, 0.51), zero_sim).loc[0, "내용"]   # 개선율 NaN 대체

    worse = headline.story_table(_head(0.40, 0.55)).loc[0, "내용"]
    assert "넘지 못했다" in worse                                     # 못 이긴 결과도 그대로
    missing = headline.story_table(pd.DataFrame(columns=["번호", "내용", "값", "비교"])).loc[0, "내용"]
    assert "미산출" in missing                                       # 값이 없으면 지어내지 않음


def test_region_cards_rank_recommend_and_hide_small_regions():
    priority = pd.DataFrame({"bjd_code": list("ABC"), "rank": [1, 2, 3], "rank_eligible": True,
                             "PriorityScore": [0.9, 0.7, 0.2], "risk_raw": [0.3, 0.1, 0.0],
                             "risk_norm": [0.9, 0.8, 0.1], "equity_norm": [0.7, 0.2, 0.1],
                             "access_2sfca": [0.01, 0.5, 0.9], "robust_top": [True, False, False]})
    risk = pd.DataFrame({"bjd_code": list("ABC"), "multiplier": 1.2, "tp_ai": [3, 1, 0], "actual_ai": [4, 2, 0]})
    master = pd.DataFrame({"bjd_code": list("ABC"), "region_name": ["서울 가구 가동", "서울 가구 나동", "서울 가구 다동"]})
    params = {"base_multiplier": 1.2, "card_top_n": 2}
    cards = headline.region_cards(priority, risk, None, master, params)
    assert cards["bjd_code"].tolist() == ["A", "B"]
    assert cards.loc[0, "권고(초안)"] == "부하 현장 점검 + 공용 충전기 추가 검토"
    assert cards.loc[1, "권고(초안)"] == "배전 부하 현장 점검(한전 협의)"
    assert cards.loc[0, "AI 경보 적중(급증일)"] == "3/4일" and cards.loc[0, "접근성 순위(낮은 순)"] == "1/3"
    hidden = headline.region_cards(priority, risk, None, master, params, codes={"B", "C"})
    assert "A" not in hidden["bjd_code"].tolist()


def test_season_windows_and_top_group_overlap():
    assert season_windows("2025-12-04") == ["2025-12-04", "2025-09-04", "2025-06-04", "2025-03-04"]
    frame = lambda codes: pd.DataFrame({"bjd_code": codes, "rank": range(1, len(codes) + 1)})
    table, common = compare_windows({"w1": frame(list("ABCD")), "w2": frame(list("ABDC")), "w3": frame(list("ACBE"))}, 2)
    assert common == ["A"]                                            # 상위 2: {A,B} {A,B} {A,C}
    summary = table[table["평가 시작"] == "요약"].iloc[0]
    assert summary["첫 창과 겹침"] == 1
    assert abs(summary["자카드(첫 창)"] - np.mean([1.0, 1 / 3, 1 / 3])) < 1e-9
