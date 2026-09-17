"""선택 패키지 실경로 검증. 설치 환경에서 실행하고 미설치 환경은 명시적으로 생략한다."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heterogeneity
import kepco_loader
from config import DEFAULT_PARAMS


def test_real_causal_forest_and_cross_validation():
    pytest.importorskip("econml.dml")
    rng = np.random.default_rng(42)
    n = 120
    params = copy.deepcopy(DEFAULT_PARAMS["heterogeneity"])
    features = pd.DataFrame({"bjd_code": [str(i) for i in range(n)],
                             params["split_features"][0]: rng.uniform(0, 1, n),
                             params["split_features"][1]: rng.uniform(0, 1, n)})
    w = np.tile([0, 1], n // 2)
    delta = w * (0.2 + features.iloc[:, 1].to_numpy()) + rng.normal(0, .05, n)
    outcomes = pd.DataFrame({"bjd_code": features.bjd_code, "W": w, "T_r": "2025-04", "delta": delta})
    # 채택 성능 기준 변경이 아니라 선택 패키지 코드 경로를 직접 검증하기 위한 명시적 호출이다.
    params["primary"] = "forest"
    result = heterogeneity.run_heterogeneity(features, outcomes, params)
    assert result["method"] == "Causal Forest(EconML)"
    assert len(result["cate"]) == n and np.isfinite(result["cate"].cate).all()
    assert len(result["importance"]) == 2
    v = result["validation"].set_index("방법")
    assert np.isfinite(v.loc["Causal Forest", "변환결과_MSE"])
    assert not v.loc["Causal Forest", "비고"].startswith("실패")


def test_ruptures_matches_builtin_on_known_changes():
    rpt = pytest.importorskip("ruptures")
    y = np.repeat([1.0, 5.0, 2.0], 12)
    external = rpt.Pelt(model="l2", min_size=2, jump=1).fit(y).predict(pen=5)[:-1]
    internal = kepco_loader.pelt_builtin(y, 5, min_size=2)
    assert external == internal == [12, 24]
