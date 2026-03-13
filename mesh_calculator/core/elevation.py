"""
Elevation data handling with caching for mesh calculator.
"""
import threading
from typing import Optional, Tuple
import h3
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from rasterio.features import geometry_mask
import numpy as np
from shapely.geometry import Polygon, mapping
import structlog

logger = structlog.get_logger(__name__)


class ElevationProvider:
    """
    Provides elevation data from GeoTIFF raster with caching.

    Attributes:
        dataset: Rasterio dataset handle
        transform: Affine transform for coordinate conversion
        _data: In-memory raster band array (lazy-loaded on first access)
    """

    def __init__(self, tif_path: str):
        """
        Initialize elevation provider from GeoTIFF file.

        Args:
            tif_path: Path to GeoTIFF elevation file
        """
        self.dataset = None
        self.dataset = rasterio.open(tif_path)
        self.transform = self.dataset.transform
        self._cell_max_cache = {}
        self._cell_anchor_cache = {}
        self._line_peak_cache = {}
        self._data = None  # Lazy-loaded full band array
        self._nodata = self.dataset.nodata
        self._lock = threading.Lock()

    def _ensure_data(self):
        """Load the full raster band into memory on first access."""
        if self._data is None:
            with self._lock:
                if self._data is None:  # Double-check under lock
                    data = self.dataset.read(1)
                    # Replace nodata with 0.0 in-place so lookups don't need to check
                    if self._nodata is not None:
                        data = data.astype(np.float32, copy=False)
                        data[data == self._nodata] = 0.0
                    self._data = data

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get elevation at a specific latitude/longitude.

        Uses in-memory raster array for fast lookups (no per-pixel GDAL I/O).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Elevation in meters
        """
        self._ensure_data()
        try:
            row, col = rowcol(self.transform, lon, lat)
            if 0 <= row < self.dataset.height and 0 <= col < self.dataset.width:
                return float(self._data[row, col])
        except Exception as e:
            logger.warning("Failed to get elevation", lat=lat, lon=lon, error=str(e))
        return 0.0

    def get_elevation_bilinear(self, lat: float, lon: float) -> float:
        """
        Get elevation via bilinear interpolation at a specific latitude/longitude.

        Uses in-memory raster array. Falls back to nearest-pixel sampling near
        raster edges or on read errors.
        """
        self._ensure_data()
        try:
            col_f, row_f = (~self.transform) * (lon, lat)
            row0 = int(np.floor(row_f))
            col0 = int(np.floor(col_f))
            row1 = row0 + 1
            col1 = col0 + 1

            if (
                row0 < 0 or col0 < 0
                or row1 >= self.dataset.height
                or col1 >= self.dataset.width
            ):
                return self.get_elevation(lat, lon)

            v00 = float(self._data[row0, col0])
            v01 = float(self._data[row0, col1])
            v10 = float(self._data[row1, col0])
            v11 = float(self._data[row1, col1])

            dx = float(col_f - col0)
            dy = float(row_f - row0)
            top = v00 * (1.0 - dx) + v01 * dx
            bottom = v10 * (1.0 - dx) + v11 * dx
            return float(top * (1.0 - dy) + bottom * dy)
        except Exception as e:
            logger.warning("Failed to get bilinear elevation", lat=lat, lon=lon, error=str(e))
            return self.get_elevation(lat, lon)

    def get_elevation_bilinear_bulk(self, coords: list[Tuple[float, float]]) -> np.ndarray:
        """
        Get bilinear-interpolated elevations for multiple coordinates via
        vectorized numpy operations.

        Falls back to nearest-pixel for points near raster edges.

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            NumPy array of elevations
        """
        if not coords:
            return np.array([], dtype=np.float64)

        self._ensure_data()

        lats, lons = zip(*coords)
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = np.asarray(lons, dtype=np.float64)

        # Inverse transform to get fractional row/col
        inv = ~self.transform
        col_f = inv.a * lons_arr + inv.b * lats_arr + inv.c
        row_f = inv.d * lons_arr + inv.e * lats_arr + inv.f

        row0 = np.floor(row_f).astype(np.intp)
        col0 = np.floor(col_f).astype(np.intp)
        row1 = row0 + 1
        col1 = col0 + 1

        h, w = self.dataset.height, self.dataset.width

        # Points where all 4 neighbors are in bounds
        interior = (row0 >= 0) & (col0 >= 0) & (row1 < h) & (col1 < w)

        elevations = np.zeros(len(coords), dtype=np.float64)

        if np.any(interior):
            r0 = row0[interior]
            c0 = col0[interior]
            r1 = row1[interior]
            c1 = col1[interior]
            dx = (col_f[interior] - c0).astype(np.float64)
            dy = (row_f[interior] - r0).astype(np.float64)

            v00 = self._data[r0, c0].astype(np.float64)
            v01 = self._data[r0, c1].astype(np.float64)
            v10 = self._data[r1, c0].astype(np.float64)
            v11 = self._data[r1, c1].astype(np.float64)

            top = v00 * (1.0 - dx) + v01 * dx
            bottom = v10 * (1.0 - dx) + v11 * dx
            elevations[interior] = top * (1.0 - dy) + bottom * dy

        # Edge points: fall back to nearest-pixel
        edge = ~interior
        if np.any(edge):
            rows_e, cols_e = rowcol(self.transform, lons_arr[edge], lats_arr[edge])
            rows_e = np.asarray(rows_e)
            cols_e = np.asarray(cols_e)
            valid = (rows_e >= 0) & (rows_e < h) & (cols_e >= 0) & (cols_e < w)
            edge_idx = np.where(edge)[0]
            if np.any(valid):
                elevations[edge_idx[valid]] = self._data[rows_e[valid], cols_e[valid]]

        return elevations

    def get_elevation_bulk(self, coords: list[Tuple[float, float]]) -> np.ndarray:
        """
        Get elevations for multiple coordinates efficiently via vectorized
        numpy array indexing (no per-pixel GDAL I/O).

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            NumPy array of elevations
        """
        if not coords:
            return np.array([], dtype=np.float64)

        self._ensure_data()

        lats, lons = zip(*coords)
        rows, cols = rowcol(self.transform, lons, lats)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        valid = (
            (rows >= 0) & (rows < self.dataset.height) &
            (cols >= 0) & (cols < self.dataset.width)
        )

        elevations = np.zeros(len(coords), dtype=np.float64)
        elevations[valid] = self._data[rows[valid], cols[valid]]
        return elevations

    def _line_cache_key(
        self,
        src_lat: float,
        src_lon: float,
        dst_lat: float,
        dst_lon: float,
    ) -> tuple[tuple[float, float, float, float], bool]:
        a = (round(src_lat, 6), round(src_lon, 6))
        b = (round(dst_lat, 6), round(dst_lon, 6))
        if a <= b:
            return (a[0], a[1], b[0], b[1]), False
        return (b[0], b[1], a[0], a[1]), True

    def _read_window(self, win: Window) -> np.ma.MaskedArray:
        """Read a window from the in-memory array (or dataset as fallback).

        Returns a masked array consistent with dataset.read(1, window=win, masked=True).
        """
        self._ensure_data()
        r0 = int(win.row_off)
        c0 = int(win.col_off)
        r1 = r0 + int(win.height)
        c1 = c0 + int(win.width)
        data = self._data[r0:r1, c0:c1].copy()
        # Since _ensure_data replaces nodata with 0.0, we have no masked values
        return np.ma.array(data, mask=False)

    def _clip_window(self, window: Window) -> Optional[Window]:
        full = Window(0, 0, self.dataset.width, self.dataset.height)
        try:
            clipped = window.intersection(full)
        except Exception:
            return None
        if clipped.width <= 0 or clipped.height <= 0:
            return None
        rounded = clipped.round_offsets().round_lengths()
        if rounded.width > 0 and rounded.height > 0:
            return rounded

        # For tiny intersections, round_lengths() can collapse to zero.
        # Build a conservative integer window via floor/ceil so raster reads
        # never see width/height == 0.
        col0 = max(0, int(np.floor(clipped.col_off)))
        row0 = max(0, int(np.floor(clipped.row_off)))
        col1 = min(self.dataset.width, int(np.ceil(clipped.col_off + clipped.width)))
        row1 = min(self.dataset.height, int(np.ceil(clipped.row_off + clipped.height)))
        if col1 <= col0 or row1 <= row0:
            return None
        return Window(col0, row0, col1 - col0, row1 - row0)

    def _safe_max(self, arr: np.ma.MaskedArray, valid_mask: np.ndarray) -> Optional[float]:
        if arr.size == 0 or not np.any(valid_mask):
            return None
        values = arr.data[valid_mask]
        if values.size == 0:
            return None
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.max(values))

    @staticmethod
    def _line_fraction(
        src_lat: float,
        src_lon: float,
        dst_lat: float,
        dst_lon: float,
        lat: float,
        lon: float,
    ) -> float:
        lat_mid = (src_lat + dst_lat) / 2.0
        cos_lat = np.cos(np.radians(lat_mid))
        dx = (dst_lon - src_lon) * cos_lat
        dy = dst_lat - src_lat
        denom = (dx * dx) + (dy * dy)
        if denom <= 1e-12:
            return 0.0
        frac = (((lon - src_lon) * cos_lat * dx) + ((lat - src_lat) * dy)) / denom
        return float(np.clip(frac, 0.0, 1.0))

    def get_h3_cell_max_elevation(self, h3_index: str) -> float:
        """Return maximum DEM elevation inside an H3 polygon."""
        if h3_index in self._cell_max_cache:
            return self._cell_max_cache[h3_index]

        try:
            boundary = h3.cell_to_boundary(h3_index)
            polygon = Polygon([(lon, lat) for lat, lon in boundary])
            if polygon.is_empty:
                self._cell_max_cache[h3_index] = 0.0
                return 0.0

            minx, miny, maxx, maxy = polygon.bounds
            raw_window = window_from_bounds(
                minx, miny, maxx, maxy, transform=self.transform
            )
            win = self._clip_window(raw_window)
            if win is None:
                self._cell_max_cache[h3_index] = 0.0
                return 0.0

            band = self._read_window(win)
            w_transform = window_transform(win, self.transform)
            inside = geometry_mask(
                [mapping(polygon)],
                out_shape=band.shape,
                transform=w_transform,
                invert=True,
                all_touched=True,
            )
            valid = inside & (~np.ma.getmaskarray(band))
            max_elev = self._safe_max(band, valid)
            if max_elev is None:
                max_elev = 0.0
            self._cell_max_cache[h3_index] = float(max_elev)
            return float(max_elev)
        except Exception as e:
            logger.warning("Failed to get H3 cell max elevation", h3_index=h3_index, error=str(e))
            self._cell_max_cache[h3_index] = 0.0
            return 0.0

    @staticmethod
    def _to_local_xy(polygon: Polygon) -> tuple[Polygon, float, float, float]:
        """Project lon/lat polygon to a local-meter plane for buffering."""
        lon0 = float(polygon.centroid.x)
        lat0 = float(polygon.centroid.y)
        cos_lat = float(np.cos(np.radians(lat0)))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6
        m_per_deg = 111_320.0

        def _ll_to_xy(lon: float, lat: float) -> tuple[float, float]:
            x = (lon - lon0) * cos_lat * m_per_deg
            y = (lat - lat0) * m_per_deg
            return (x, y)

        exterior = [_ll_to_xy(lon, lat) for lon, lat in polygon.exterior.coords]
        holes = [
            [_ll_to_xy(lon, lat) for lon, lat in ring.coords]
            for ring in polygon.interiors
        ]
        return Polygon(exterior, holes), lon0, lat0, cos_lat

    @staticmethod
    def _from_local_xy(
        polygon_xy: Polygon,
        lon0: float,
        lat0: float,
        cos_lat: float,
    ) -> Polygon:
        """Project local-meter polygon back to lon/lat."""
        m_per_deg = 111_320.0

        def _xy_to_ll(x: float, y: float) -> tuple[float, float]:
            lon = lon0 + (x / (cos_lat * m_per_deg))
            lat = lat0 + (y / m_per_deg)
            return (lon, lat)

        exterior = [_xy_to_ll(x, y) for x, y in polygon_xy.exterior.coords]
        holes = [
            [_xy_to_ll(x, y) for x, y in ring.coords]
            for ring in polygon_xy.interiors
        ]
        return Polygon(exterior, holes)

    def _shrink_polygon_m(self, polygon: Polygon, margin_m: float) -> Optional[Polygon]:
        """Shrink polygon inward by margin meters in local tangent plane."""
        if margin_m <= 0.0 or polygon.is_empty:
            return polygon
        poly_xy, lon0, lat0, cos_lat = self._to_local_xy(polygon)
        try:
            shrunk_xy = poly_xy.buffer(-float(margin_m))
        except Exception:
            return None
        if shrunk_xy.is_empty:
            return None
        if shrunk_xy.geom_type == "MultiPolygon":
            geoms = list(shrunk_xy.geoms)
            if not geoms:
                return None
            shrunk_xy = max(geoms, key=lambda g: g.area)
        if shrunk_xy.geom_type != "Polygon":
            return None
        return self._from_local_xy(shrunk_xy, lon0, lat0, cos_lat)

    def _max_pixel_in_polygon(self, polygon: Polygon) -> Optional[tuple[float, float, float]]:
        """Return (lat, lon, elevation) of the highest valid DEM pixel in polygon."""
        minx, miny, maxx, maxy = polygon.bounds
        raw_window = window_from_bounds(minx, miny, maxx, maxy, transform=self.transform)
        win = self._clip_window(raw_window)
        if win is None:
            return None
        band = self._read_window(win)
        w_transform = window_transform(win, self.transform)
        inside = geometry_mask(
            [mapping(polygon)],
            out_shape=band.shape,
            transform=w_transform,
            invert=True,
            all_touched=True,
        )
        valid = inside & (~np.ma.getmaskarray(band))
        if not np.any(valid):
            return None
        values = band.data[valid]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        max_elev = float(np.max(values))
        candidates = np.where(valid & (band.data == max_elev))
        if len(candidates[0]) == 0:
            return None

        centroid = polygon.centroid
        row_off = int(win.row_off)
        col_off = int(win.col_off)
        best_row = None
        best_col = None
        best_d2 = float("inf")
        for r, c in zip(candidates[0], candidates[1]):
            global_row = row_off + int(r)
            global_col = col_off + int(c)
            px_lon, px_lat = rasterio.transform.xy(
                self.transform, global_row, global_col, offset="center"
            )
            d2 = (float(px_lat) - centroid.y) ** 2 + (float(px_lon) - centroid.x) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_row = global_row
                best_col = global_col
        if best_row is None or best_col is None:
            return None
        px_lon, px_lat = rasterio.transform.xy(
            self.transform, best_row, best_col, offset="center"
        )
        return float(px_lat), float(px_lon), max_elev

    def get_h3_cell_anchor_point(
        self,
        h3_index: str,
        margin_m: float = 10.0,
    ) -> tuple[float, float, float]:
        """
        Return fixed subcell LOS anchor (lat, lon, elevation) for an H3 cell.

        The anchor is the highest DEM pixel inside the inward-buffered polygon.
        Falls back to zero-margin search, then to centroid elevation.
        """
        key = (h3_index, round(float(margin_m), 3))
        cached = self._cell_anchor_cache.get(key)
        if cached is not None:
            return cached

        lat_c, lon_c = h3.cell_to_latlng(h3_index)
        try:
            boundary = h3.cell_to_boundary(h3_index)
            polygon = Polygon([(lon, lat) for lat, lon in boundary])
            if polygon.is_empty:
                fallback = (float(lat_c), float(lon_c), float(self.get_elevation(lat_c, lon_c)))
                self._cell_anchor_cache[key] = fallback
                return fallback

            candidates: list[Polygon] = []
            shrunk = self._shrink_polygon_m(polygon, float(margin_m))
            if shrunk is not None and not shrunk.is_empty:
                candidates.append(shrunk)
            candidates.append(polygon)

            for poly in candidates:
                anchor = self._max_pixel_in_polygon(poly)
                if anchor is not None:
                    self._cell_anchor_cache[key] = anchor
                    return anchor
        except Exception as e:
            logger.warning("Failed to resolve H3 anchor point", h3_index=h3_index, error=str(e))

        fallback = (float(lat_c), float(lon_c), float(self.get_elevation(lat_c, lon_c)))
        self._cell_anchor_cache[key] = fallback
        return fallback

    def get_line_peak_elevation(
        self,
        src_lat: float,
        src_lon: float,
        dst_lat: float,
        dst_lon: float,
    ) -> tuple[float, float, float, float]:
        """
        Return maximum DEM elevation on a line and its position.

        Returns:
            (max_elev_m, peak_lat, peak_lon, frac_0_1)
            where frac is distance fraction from source to destination.
        """
        key, flipped = self._line_cache_key(src_lat, src_lon, dst_lat, dst_lon)
        cached = self._line_peak_cache.get(key)

        if cached is None:
            a_lat, a_lon, b_lat, b_lon = key
            try:
                if a_lat == b_lat and a_lon == b_lon:
                    elev = self.get_elevation(a_lat, a_lon)
                    cached = (float(elev), float(a_lat), float(a_lon), 0.0)
                else:
                    self._ensure_data()
                    h, w = self.dataset.height, self.dataset.width

                    # Get pixel coordinates for endpoints
                    r_a, c_a = rowcol(self.transform, a_lon, a_lat)
                    r_b, c_b = rowcol(self.transform, b_lon, b_lat)

                    # Bresenham-style line sampling: walk all pixels along the line
                    n_steps = max(abs(r_b - r_a), abs(c_b - c_a), 1)
                    t = np.linspace(0.0, 1.0, n_steps + 1)
                    rows = np.round(r_a + (r_b - r_a) * t).astype(np.intp)
                    cols = np.round(c_a + (c_b - c_a) * t).astype(np.intp)

                    # Filter to in-bounds pixels and deduplicate
                    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                    rows = rows[valid]
                    cols = cols[valid]
                    t_valid = t[valid]

                    if len(rows) == 0:
                        elev_a = self.get_elevation(a_lat, a_lon)
                        elev_b = self.get_elevation(b_lat, b_lon)
                        if elev_a >= elev_b:
                            cached = (float(elev_a), float(a_lat), float(a_lon), 0.0)
                        else:
                            cached = (float(elev_b), float(b_lat), float(b_lon), 1.0)
                    else:
                        # Deduplicate (keep unique row,col pairs preserving order)
                        rc = rows.astype(np.int64) * w + cols.astype(np.int64)
                        _, unique_idx = np.unique(rc, return_index=True)
                        unique_idx.sort()
                        rows = rows[unique_idx]
                        cols = cols[unique_idx]
                        t_valid = t_valid[unique_idx]

                        elevs = self._data[rows, cols]
                        finite_mask = np.isfinite(elevs)

                        if not np.any(finite_mask):
                            elev_a = self.get_elevation(a_lat, a_lon)
                            elev_b = self.get_elevation(b_lat, b_lon)
                            if elev_a >= elev_b:
                                cached = (float(elev_a), float(a_lat), float(a_lon), 0.0)
                            else:
                                cached = (float(elev_b), float(b_lat), float(b_lon), 1.0)
                        else:
                            max_elev = float(np.max(elevs[finite_mask]))
                            # Among pixels with max elevation, pick closest to midpoint
                            peak_mask = finite_mask & (elevs == max_elev)
                            peak_rows = rows[peak_mask]
                            peak_cols = cols[peak_mask]
                            peak_t = t_valid[peak_mask]

                            mid_lat = (a_lat + b_lat) / 2.0
                            mid_lon = (a_lon + b_lon) / 2.0

                            if len(peak_rows) == 1:
                                best_idx = 0
                            else:
                                # Compute distances to midpoint
                                peak_lons, peak_lats = rasterio.transform.xy(
                                    self.transform, peak_rows.tolist(), peak_cols.tolist(),
                                    offset="center"
                                )
                                peak_lats = np.asarray(peak_lats)
                                peak_lons = np.asarray(peak_lons)
                                d2 = (peak_lats - mid_lat) ** 2 + (peak_lons - mid_lon) ** 2
                                best_idx = int(np.argmin(d2))

                            best_lon, best_lat = rasterio.transform.xy(
                                self.transform, int(peak_rows[best_idx]),
                                int(peak_cols[best_idx]), offset="center"
                            )
                            frac = self._line_fraction(
                                a_lat, a_lon, b_lat, b_lon,
                                float(best_lat), float(best_lon)
                            )
                            cached = (max_elev, float(best_lat), float(best_lon), float(frac))
            except Exception as e:
                logger.warning(
                    "Failed to get line peak elevation",
                    src_lat=src_lat, src_lon=src_lon, dst_lat=dst_lat, dst_lon=dst_lon,
                    error=str(e),
                )
                elev_a = self.get_elevation(src_lat, src_lon)
                elev_b = self.get_elevation(dst_lat, dst_lon)
                if elev_a >= elev_b:
                    cached = (float(elev_a), float(src_lat), float(src_lon), 0.0)
                else:
                    cached = (float(elev_b), float(dst_lat), float(dst_lon), 1.0)
            self._line_peak_cache[key] = cached

        max_elev, peak_lat, peak_lon, frac = cached
        if flipped:
            return float(max_elev), float(peak_lat), float(peak_lon), float(1.0 - frac)
        return float(max_elev), float(peak_lat), float(peak_lon), float(frac)

    def cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and other stats
        """
        return {
            'raster_loaded': self._data is not None,
            'raster_memory_mb': (
                self._data.nbytes / (1024 * 1024) if self._data is not None else 0
            ),
            'cell_max_cache_size': len(self._cell_max_cache),
            'cell_anchor_cache_size': len(self._cell_anchor_cache),
            'line_peak_cache_size': len(self._line_peak_cache),
        }

    def clear_cache(self):
        """Clear the elevation cache."""
        self._cell_max_cache.clear()
        self._cell_anchor_cache.clear()
        self._line_peak_cache.clear()

    def close(self):
        """Close the rasterio dataset."""
        if getattr(self, 'dataset', None):
            self.dataset.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure dataset is closed."""
        self.close()
