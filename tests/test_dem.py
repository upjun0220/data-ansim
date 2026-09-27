"""LX DEM 경사 보정 — TM 변환식, Tobler 배율, 가짜 래스터로 표고 추출, 경사가 접근성을 바꾸는 방향."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demloader
import equityaccess

WKT_5186 = ('PROJCS["Korea 2000 / Central Belt 2010",GEOGCS["Korea 2000",DATUM["Geocentric_datum_of_Korea",'
            'SPHEROID["GRS 1980",6378137,298.257222101]]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",38],PARAMETER["central_meridian",127],PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",200000],PARAMETER["false_northing",600000],UNIT["metre",1]]')


def _meridian_arc(lat_deg):
    """GRS80 자오선 호 길이 — 수치 적분(변환식과 독립인 대조값)."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    phi = np.linspace(0, np.radians(lat_deg), 200001)
    m = a * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
    return float(np.sum((m[1:] + m[:-1]) / 2 * np.diff(phi)))


def test_tm_forward_matches_origin_meridian_and_distance():
    x, y = demloader.to_raster_xy([38.0], [127.0], WKT_5186)
    assert x[0] == pytest.approx(200000, abs=1e-6) and y[0] == pytest.approx(600000, abs=1e-6)
    x, y = demloader.to_raster_xy([37.5], [127.0], WKT_5186)                 # 중앙경선: y 차 = 자오선 호 차
    assert y[0] == pytest.approx(600000 - (_meridian_arc(38.0) - _meridian_arc(37.5)), abs=0.01)
    lat, lon = np.array([37.55, 37.55]), np.array([126.95, 127.05])          # 서울, 동서 약 8.8km
    x, y = demloader.to_raster_xy(lat, lon, WKT_5186)
    geo = equityaccess._distance_km(lat[0], lon[0], lat[1], lon[1]) * 1000
    assert np.hypot(np.diff(x), np.diff(y))[0] == pytest.approx(geo, rel=3e-3)   # 구면 근사 대비 0.3% 이내
    with pytest.raises(ValueError):
        demloader.to_raster_xy([37.5], [127.0], WKT_5186.replace("GRS 1980\",6378137,298.257222101",
                                                                  "Bessel 1841\",6377397.155,299.1528128"))


def test_slope_factor_is_one_on_flat_and_grows_with_grade():
    f = equityaccess.slope_factor(np.array([0.0, 0.05, -0.05, 0.1, 0.2]))
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(f[2])        # 왕복이라 부호 대칭
    assert 1.0 < f[1] < f[3] < f[4]


class FakeRaster:
    """TM 좌표 평면: 동쪽으로 갈수록 1km 당 100m 높아지는 경사면(10%)."""
    wkt, bounds, nodata = WKT_5186, (150000.0, 500000.0, 260000.0, 620000.0), -9999.0

    def sample(self, x, y):
        return (np.asarray(x) - 190000.0) * 0.1


def test_sample_elevation_uses_available_backend_and_marks_outside(monkeypatch, tmp_path):
    img = tmp_path / "dem.img"
    img.write_bytes(b"")
    monkeypatch.setattr(demloader, "backend", lambda: None)
    with pytest.raises(ImportError):
        demloader.sample_elevation(img, [37.5], [127.0])
    monkeypatch.setattr(demloader, "backend", lambda: "rasterio")
    monkeypatch.setattr(demloader, "_open", lambda path, tool: FakeRaster())
    z = demloader.sample_elevation(tmp_path, [37.55, 37.55, 35.1], [126.95, 127.05, 129.0])   # 폴더도 받음
    assert z[1] - z[0] == pytest.approx(0.1 * 8800, rel=0.01) and np.isnan(z[2])            # 부산은 범위 밖


def test_slope_moves_hill_neighbourhood_down_and_summary_hides_codes():
    # 동네 A 는 충전소보다 45m 낮은 곳(450m 거리, 10% 경사 → 배율 1.21 → 545m), B 는 같은 거리 평지. A 만 500m 밖.
    pts = pd.DataFrame({"bjd_code": ["A", "B"], "lat": [37.50, 37.60], "lon": [127.0, 127.0], "ev_count": [10, 10]})
    st = pd.DataFrame({"lat": [37.50 + 0.45 / 111.0, 37.60 + 0.45 / 111.0], "lon": [127.0, 127.0], "chargers": [2, 2]})
    base = equityaccess.compute_2sfca(st, pts)
    slope, grades = equityaccess.slope_adjusted_2sfca(st, pts, [0.0, 50.0], [45.0, 50.0])
    b, s = base.set_index("bjd_code")["access_2sfca"], slope.set_index("bjd_code")["access_2sfca"]
    assert b["A"] == b["B"] > 0 and s["A"] == 0 and s["B"] == b["B"]
    assert grades.max() == pytest.approx(0.1, rel=0.02)
    table = equityaccess.compare_slope(base, slope, grades, (2, 2), (2, 2))
    assert table.set_index("항목").at["접근성 0 동네(기본 → 보정)", "값"] == "0 → 1"
    assert not table.astype(str).apply(lambda c: c.isin(["A", "B"])).any().any()   # 동네 코드는 싣지 않음


def test_missing_elevation_is_treated_as_flat():
    pts = pd.DataFrame({"bjd_code": ["A"], "lat": [37.5], "lon": [127.0], "ev_count": [5]})
    st = pd.DataFrame({"lat": [37.5 + 0.3 / 111.0], "lon": [127.0], "chargers": [1]})
    slope, _ = equityaccess.slope_adjusted_2sfca(st, pts, [np.nan], [30.0])
    assert slope["access_2sfca"].iloc[0] == equityaccess.compute_2sfca(st, pts)["access_2sfca"].iloc[0]


def test_relative_copy_path_becomes_absolute(tmp_path, monkeypatch):
    from common import locate
    (tmp_path / "d.csv").write_text("a\n1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert locate("d.csv").is_absolute()                                     # 설정 JSON 이 결과 폴더 기준으로 재해석해도 맞게


def test_gdal_branch_reads_pixel_by_geotransform(monkeypatch, tmp_path):
    """센터에 rasterio 없이 GDAL(osgeo)만 있을 때의 격자 인덱스 계산 — 가짜 gdal 모듈로 확인."""
    import types
    grid = np.arange(12, dtype=float).reshape(3, 4)                         # 3행 4열, 5m 격자
    grid[0, 0] = -9999.0

    class Band:
        def ReadAsArray(self, c, r, w, h):
            return grid[r:r + h, c:c + w]

        def GetNoDataValue(self):
            return -9999.0

    class Ds:
        RasterXSize, RasterYSize = 4, 3

        def GetGeoTransform(self):
            return (1000.0, 5.0, 0.0, 2000.0, 0.0, -5.0)                    # 왼쪽 위 (1000, 2000)

        def GetProjection(self):
            return WKT_5186

        def GetRasterBand(self, i):
            return Band()

    osgeo = types.ModuleType("osgeo")
    osgeo.gdal = types.SimpleNamespace(Open=lambda p: Ds())
    monkeypatch.setitem(sys.modules, "osgeo", osgeo)
    r = demloader._Raster(tmp_path / "x.img", "osgeo")
    assert r.bounds == (1000.0, 1985.0, 1020.0, 2000.0)
    z = r.sample(np.array([1002.0, 1017.0, 1012.0]), np.array([1999.0, 1986.0, 1993.0]))
    assert np.isnan(z[0]) and z[1] == 11.0 and z[2] == 6.0                  # nodata · (2행 3열) · (1행 2열)
