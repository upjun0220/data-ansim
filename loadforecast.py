"""8-A: 과거 이력으로 다음날 24시간 평균 부하 예측·백테스트. 실시간 제어가 아니다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_matrix(load_df, region):
    """날짜×0~23시. 없는 셀은 NaN으로 남기며 중복·음수는 거부한다."""
    g = load_df.loc[load_df["bjd_code"].astype(str) == str(region)].copy()
    if g.empty:
        raise ValueError(f"{region}: 시간별 부하 없음")
    g["date"] = pd.to_datetime(g["date"], errors="raise").dt.normalize()
    if g.duplicated(["date", "hour"]).any():
        raise ValueError("일자·시간 중복")
    if not g["hour"].isin(range(24)).all() or not np.isfinite(g["kw"]).all() or (g["kw"] < 0).any():
        raise ValueError("시간/부하 값 오류")
    return g.pivot(index="date", columns="hour", values="kw").reindex(columns=range(24)).sort_index()


def forecast_day(matrix, target_date, W=4, holidays=None):
    """날짜에서 7·w일을 뺀 뒤 동일 시간 열을 선택한다. 대상일 이후 값은 접근하지 않는다.

    holidays는 공식 달력을 확인한 {YYYY-MM-DD: 휴일유형} 선택 입력이다.
    동일 유형 휴일 표본이 W개 미만이면 예측을 실패 처리하며 임의 대체하지 않는다.
    """
    if not isinstance(W, int) or W < 1:
        raise ValueError("W는 양의 정수")
    day = pd.Timestamp(target_date).normalize()
    holiday_map = {pd.Timestamp(k).normalize(): v for k, v in (holidays or {}).items()}
    if day in holiday_map:
        dates = sorted((d for d, kind in holiday_map.items() if d < day and kind == holiday_map[day]),
                       reverse=True)[:W]
    else:
        dates = [day - pd.Timedelta(days=7 * w) for w in range(1, W + 1)]
    sample = matrix.reindex(dates)
    if len(dates) != W or sample.isna().any().any():
        raise ValueError(f"{day.date()}: 동일 요일·시간 {W}주 완전한 이력 부족")
    return sample.median(axis=0).rename("forecast_kw")


def forecast_load(load_df, target_date, region, W=4, holidays=None):
    return forecast_day(daily_matrix(load_df, region), target_date, W, holidays)


def backtest_forecast(load_df, region, n_weeks_holdout=8, W=4, holidays=None, end_date=None):
    """동일 관측 셀에서 계절 중앙값과 전날 값을 비교. end_date까지의 자료만 평가한다."""
    matrix = daily_matrix(load_df, region)
    end = pd.Timestamp(end_date).normalize() if end_date is not None else matrix.index.max()
    if n_weeks_holdout < 1:
        raise ValueError("홀드아웃은 1주 이상")
    rows, skipped = [], 0
    for day in pd.date_range(end - pd.Timedelta(days=7 * n_weeks_holdout - 1), end):
        actual = matrix.reindex([day]).iloc[0]
        previous = matrix.reindex([day - pd.Timedelta(days=1)]).iloc[0]
        try:
            pred = forecast_day(matrix, day, W, holidays)
        except ValueError:
            skipped += 1
            continue
        if actual.isna().any() or previous.isna().any():
            skipped += 1
            continue
        growing = actual.sum() > matrix.reindex([day - pd.Timedelta(days=7 * w)
                                                for w in range(1, W + 1)]).sum(axis=1).median()
        peak = int(actual.idxmax())
        for hour in range(24):
            rows.append((day, hour, actual[hour] - pred[hour], abs(actual[hour] - previous[hour]),
                         bool(growing), hour == peak))
    details = pd.DataFrame(rows, columns=["date", "hour", "error", "persistence_error", "growing", "peak"])
    if details.empty:
        raise ValueError(f"{region}: 검증 가능한 완전한 날짜 0개")
    growth = details.loc[details["growing"], "error"]
    return {"bjd_code": str(region), "mae_naive_seasonal": details["error"].abs().mean(),
            "mae_persistence": details["persistence_error"].mean(), "bias_actual_minus_forecast": details["error"].mean(),
            "growth_bias": growth.mean(), "growth_n_obs": len(growth),
            "peak_mae": details.loc[details["peak"], "error"].abs().mean(),
            "n_obs": len(details), "n_days": details["date"].nunique(), "skipped_days": skipped,
            "calendar": "설정 달력 적용" if holidays else "공휴일 미보정", "validation_end": str(end.date())}
