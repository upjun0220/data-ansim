"""LX DEM 5M(수도권 .img 래스터, GRS80) — 필요한 점의 표고만 뽑는다.

2.2GB 래스터를 통째로 올리지 않고 법정동 중심점·충전소 위치(수천 점)의 값만 읽는다.
읽기는 rasterio → GDAL(osgeo) 순으로 있는 것을 쓰고, 둘 다 없으면 ImportError(해당 단계만 생략).
좌표 변환은 pyproj 가 있으면 쓰고, 없으면 래스터 WKT 의 횡메르카토르(TM) 모수로 직접 계산한다(GRS80·WGS84 타원체만).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from common import locate, log

GRS80 = (6378137.0, 298.257222101)


def backend():
    """센터 파이썬에서 쓸 수 있는 래스터 읽기 도구 이름 또는 None."""
    import importlib.util as u

    return next((m for m in ("rasterio", "osgeo") if u.find_spec(m)), None)


def dem_files(path):
    """.img 파일 하나 또는 .img 가 든 폴더(시군구별로 나뉘어 있을 수 있음) → 파일 목록."""
    p = locate(path)
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.img")) if p.is_dir() else []
    if not files:
        raise FileNotFoundError(f"DEM .img 를 찾지 못함: {str(path)!r}")
    return files


def _wkt_params(wkt):
    params = {k.lower(): float(v) for k, v in re.findall(r'PARAMETER\["([^"]+)",\s*([-\d.eE+]+)', wkt)}
    sph = re.search(r'(?:SPHEROID|ELLIPSOID)\["[^"]*",\s*([\d.]+),\s*([\d.]+)', wkt)
    return params, (float(sph.group(1)), float(sph.group(2))) if sph else None


def tm_forward(lat, lon, lat0, lon0, k0, fe, fn, ellipsoid=GRS80):
    """위경도(도) → 횡메르카토르 x, y(m). Snyder(1987) 식 8-9~8-10, 중앙경선 ±3° 안에서 cm 수준."""
    a, invf = ellipsoid
    f = 1 / invf
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    phi0, lam0 = np.radians(lat0), np.radians(lon0)

    def arc(p):
        return a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * p)
                    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * p)
                    - (35 * e2 ** 3 / 3072) * np.sin(6 * p))

    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    t, c = np.tan(phi) ** 2, ep2 * np.cos(phi) ** 2
    aa = (lam - lam0) * np.cos(phi)
    x = fe + k0 * n * (aa + (1 - t + c) * aa ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * aa ** 5 / 120)
    y = fn + k0 * (arc(phi) - arc(phi0) + n * np.tan(phi) * (
        aa ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24
        + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * aa ** 6 / 720))
    return x, y


def to_raster_xy(lat, lon, wkt):
    """위경도 → 래스터 좌표. WKT 가 지리좌표면 그대로(lon, lat)."""
    if not wkt:
        raise ValueError("DEM 좌표계 정보 없음 — 변환 불가")
    try:
        from pyproj import Transformer

        return Transformer.from_crs("EPSG:4326", wkt, always_xy=True).transform(np.asarray(lon), np.asarray(lat))
    except ImportError:
        pass
    if not wkt.lstrip().upper().startswith(("PROJCS", "PROJCRS")):
        return np.asarray(lon, float), np.asarray(lat, float)
    params, ell = _wkt_params(wkt)
    if "transverse" not in wkt.lower() or ell is None or abs(ell[0] - GRS80[0]) > 1 or abs(ell[1] - GRS80[1]) > 0.01:
        raise ValueError("pyproj 없이 변환할 수 있는 좌표계는 GRS80·WGS84 횡메르카토르뿐 — DEM 좌표계 확인 필요")
    return tm_forward(lat, lon, params.get("latitude_of_origin", 0.0), params.get("central_meridian", 0.0),
                      params.get("scale_factor", 1.0), params.get("false_easting", 0.0),
                      params.get("false_northing", 0.0), ell)


class _Raster:
    """rasterio·GDAL 공통 최소 인터페이스: wkt, bounds(left, bottom, right, top), sample(x, y) → 표고(없으면 NaN)."""

    def __init__(self, path, tool):
        self.tool = tool
        if tool == "rasterio":
            import rasterio

            self.ds = rasterio.open(path)
            self.wkt = self.ds.crs.to_wkt() if self.ds.crs else ""
            self.bounds = tuple(self.ds.bounds)
            self.nodata = self.ds.nodata
        else:
            from osgeo import gdal

            self.ds = gdal.Open(str(path))
            self.wkt = self.ds.GetProjection()
            gt = self.gt = self.ds.GetGeoTransform()
            w, h = self.ds.RasterXSize, self.ds.RasterYSize
            xs, ys = (gt[0], gt[0] + gt[1] * w), (gt[3], gt[3] + gt[5] * h)
            self.bounds = (min(xs), min(ys), max(xs), max(ys))
            self.band = self.ds.GetRasterBand(1)
            self.nodata = self.band.GetNoDataValue()

    def sample(self, x, y):
        if self.tool == "rasterio":
            vals = np.array([v[0] for v in self.ds.sample(list(zip(x, y)))], float)
        else:
            gt = self.gt
            cols = ((np.asarray(x) - gt[0]) / gt[1]).astype(int)
            rows = ((np.asarray(y) - gt[3]) / gt[5]).astype(int)
            vals = np.array([self.band.ReadAsArray(int(c), int(r), 1, 1)[0, 0] for c, r in zip(cols, rows)], float)
        if self.nodata is not None:
            vals[np.isclose(vals, self.nodata)] = np.nan
        vals[(vals < -100) | (vals > 3000)] = np.nan   # 수도권 표고 범위 밖(결측 부호값 등)
        return vals


def _open(path, tool):
    return _Raster(path, tool)


def sample_elevation(path, lat, lon):
    """점마다 표고(m). DEM 범위 밖·결측은 NaN. 파일이 여러 개면 범위 안에 든 파일에서 채운다."""
    tool = backend()
    if tool is None:
        raise ImportError("rasterio·GDAL(osgeo) 둘 다 없음 — DEM(.img)을 읽을 수 없어 경사 보정 생략")
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out = np.full(len(lat), np.nan)
    files = dem_files(path)
    for f in files:
        r = _open(f, tool)
        x, y = (np.asarray(v, float) for v in to_raster_xy(lat, lon, r.wkt))
        left, bottom, right, top = r.bounds
        todo = np.isnan(out) & (x >= left) & (x < right) & (y > bottom) & (y <= top)
        if todo.any():
            out[todo] = r.sample(x[todo], y[todo])
    log.info("DEM %s: 파일 %d개, 표고 확보 %d/%d점", tool, len(files), int(np.isfinite(out).sum()), len(out))
    return out
