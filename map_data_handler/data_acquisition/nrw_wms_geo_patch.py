"""
Extract DOP (digital orthophotos) data from open NRW DOP API.

Possibilities:
 A) Cut out at bounds of polygons, at given resolution (px/m)
 B) Cut out at bounds of polygons with additional border, at given resolution (px/m)
 C) Cut out polygon with given size (px) and given resolution (px/m). Centered at mid.
    Raises Exception if polygon is too big for size/resolution.
 D) Cut out centered at point with given size (px) and given resolution (px/m)
 E) Cutout from borders with given size (px) and given resolution (px/m).

# TODO: cleanup, docstrings, typehints
@author: Frank Ehebrecht
"""

import io
import math
from dataclasses import dataclass
from typing import Union, List

import geopy.distance as geo_dist
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from PIL import Image
from owslib.wms import WebMapService
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import rasterio.features

LAYERS = ['WMS_NW_DOP', 'nw_dop_utm_info', 'nw_dop_rgb', 'nw_dop_cir', 'nw_dop_nir']
NRW_WMS_URL = r'https://www.wms.nrw.de/geobasis/wms_nw_dop'


@dataclass
class GeoPatch:
    patch: np.ndarray
    bounds_deg: tuple
    bounds_m: tuple
    polygons_deg: Union[List, None]
    hull_deg: Union[Polygon, None]
    polygons_m: Union[List, None]
    hull_m: Union[Polygon, None]
    hull_mask: Union[Polygon, None]


class LinFit:
    def __init__(self, p1, p2):
        self.m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = (p1[1] * p2[0] - p2[1] * p1[0]) / (p2[0] - p1[0])

    def predict(self, x):
        return self.m * x + self.b


def polygon_union(polygons):

    if isinstance(polygons, list) or isinstance(polygons, tuple):
        return cascaded_union(polygons), polygons
    elif isinstance(polygons, shapely.geometry.Polygon):
        return polygons, [polygons]
    else:
        raise NotAPolygonError


class NotAPolygonError(ValueError):
    pass


def polygon_deg2m(polygon_deg, bounds_deg, bounds_m):

    lon_fit = LinFit((bounds_deg[0], bounds_m[0]), (bounds_deg[2], bounds_m[2]))
    lat_fit = LinFit((bounds_deg[1], bounds_m[1]), (bounds_deg[3], bounds_m[3]))
    polygon_m = []
    for x_, y_ in zip(*polygon_deg.exterior.xy):
        polygon_m.append((lon_fit.predict(x_), lat_fit.predict(y_)))
    return Polygon(polygon_m)


def calc_size_from_bounds(bounds, resolution, factor=0):
    """
    p_lat
    X------------
    |           |
    |           |
    |           |
    X-----------X
    p0         p_lon

    bounds: (lon0, lat0, lon1, lat1)

    """

    additional_lon = (bounds[2] - bounds[0]) / 2 * factor
    additional_lat = (bounds[3] - bounds[1]) / 2 * factor
    bounds = (bounds[0] - additional_lon, bounds[1] - additional_lat,
              bounds[2] + additional_lon, bounds[3] + additional_lat)

    p0 = (bounds[0], bounds[1])
    p_lon = (bounds[2], bounds[1])
    p_lat = (bounds[0], bounds[3])
    dist_lon_m = np.abs(calculate_distance(p0, p_lon))
    dist_lon_deg = bounds[2] - bounds[0]
    dist_lat_m = np.abs(calculate_distance(p0, p_lat))
    dist_lat_deg = bounds[3] - bounds[1]

    px_lon_float = resolution * dist_lon_m + 1
    px_lat_float = resolution * dist_lat_m + 1

    px_lon = math.ceil(px_lon_float)
    px_lat = math.ceil(px_lat_float)

    lon_fac = px_lon / px_lon_float
    lat_fac = px_lat / px_lat_float

    new_bounds_deg = (bounds[0],
                      bounds[1],
                      bounds[0] + dist_lon_deg * lon_fac,
                      bounds[1] + dist_lat_deg * lat_fac)

    # TODO: check this again for correctness
    bounds_meters = (0, 0, dist_lon_m * lon_fac, dist_lat_m * lat_fac)
    size = (px_lon, px_lat)

    return new_bounds_deg, bounds_meters, size


def calc_bounds_from_point_and_size(point, size, resolution):

    mid_lon, mid_lat = point
    deg_per_m_lon, deg_per_m_lat = calculate_reverse_distance((mid_lon, mid_lat))

    left_size = size[0] // 2
    right_size = size[0] - left_size
    lower_size = size[1] // 2
    upper_size = size[1] - lower_size

    meters_left = (left_size - 1) / resolution
    meters_lower = (lower_size - 1) / resolution
    meters_right = (right_size - 1) / resolution
    meters_upper = (upper_size - 1) / resolution

    bounds_meter = (0., 0., meters_right + meters_left, meters_upper + meters_lower)
    bounds_deg = (mid_lon - meters_left * deg_per_m_lon,
                  mid_lat - meters_lower * deg_per_m_lat,
                  mid_lon + meters_left * deg_per_m_lon,
                  mid_lat + meters_upper * deg_per_m_lat)

    return bounds_deg, bounds_meter


def calc_bounds_from_bounds_and_size(bounds_polygon, size, resolution):
    mid_lon = bounds_polygon[0] + (bounds_polygon[2] - bounds_polygon[0]) / 2
    mid_lat = bounds_polygon[1] + (bounds_polygon[3] - bounds_polygon[1]) / 2

    bounds_deg, bounds_meter = calc_bounds_from_point_and_size((mid_lon, mid_lat), size, resolution)

    if (bounds_polygon[0] < bounds_deg[0] or
            bounds_polygon[1] < bounds_deg[1] or
            bounds_polygon[2] > bounds_deg[2] or
            bounds_polygon[3] > bounds_deg[3]):
        raise OutOfBoundBetterNameError

    return bounds_deg, bounds_meter


class OutOfBoundBetterNameError(ValueError):
    pass


def calculate_distance(p1, p2):
    # p1/p2 -> lon, lat
    # geodesic needs other order (lat,lon)
    return geo_dist.geodesic(p1[::-1], p2[::-1]).m


def calculate_reverse_distance(p):
    eps = 0.0001
    p_lon_1 = p
    p_lon_2 = (p[0] + eps, p[1])
    p_lat_1 = p
    p_lat_2 = (p[0], p[1] + eps)

    deg_per_m_lon = eps / calculate_distance(p_lon_1, p_lon_2)
    deg_per_m_lat = eps / calculate_distance(p_lat_1, p_lat_2)

    return deg_per_m_lon, deg_per_m_lat


class NrwWmsPatchProvider:
    def __init__(self, resolution, layer='nw_dop_rgb'):
        self.wms = WebMapService(NRW_WMS_URL)
        if layer not in LAYERS:
            raise ValueError(f'Layer not found: {layer}.')
        self.layer = layer
        self.resolution = resolution

    def from_polygons(self, polygons, factor=0):
        hull_deg, polygons_deg = polygon_union(polygons)
        hull_bounds_deg = hull_deg.bounds

        bounds_deg, bounds_m, size = calc_size_from_bounds(hull_bounds_deg, self.resolution, factor=factor)
        image = self.get_data(bounds_deg, size)

        polygons_m = []
        for polygon in polygons_deg:
            polygons_m.append(polygon_deg2m(polygon, bounds_deg, bounds_m))
        hull_m = polygon_deg2m(hull_deg, bounds_deg, bounds_m)
        hull_mask = calculate_mask(image, bounds_m, hull_m)

        return GeoPatch(patch=image,
                        bounds_deg=bounds_deg,
                        bounds_m=bounds_m,
                        polygons_deg=polygons_deg,
                        hull_deg=hull_deg,
                        polygons_m=polygons_m,
                        hull_m=hull_m,
                        hull_mask=hull_mask,
                        )

    def from_point_and_size(self, point, size):

        bounds_deg, bounds_m = calc_bounds_from_point_and_size(point, size, self.resolution)
        image = self.get_data(bounds_deg, size)

        return GeoPatch(patch=image,
                        bounds_deg=bounds_deg,
                        bounds_m=bounds_m,
                        polygons_deg=None,
                        hull_deg=None,
                        polygons_m=None,
                        hull_m=None,
                        hull_mask=None,
                        )

    def from_polygons_and_size(self, polygons, size):
        hull_deg, polygons_deg = polygon_union(polygons)
        hull_bounds_deg = hull_deg.bounds

        bounds_deg, bounds_m = calc_bounds_from_bounds_and_size(hull_bounds_deg, size, self.resolution)
        image = self.get_data(bounds_deg, size)

        polygons_m = []
        for polygon in polygons_deg:
            polygons_m.append(polygon_deg2m(polygon, bounds_deg, bounds_m))
        hull_m = polygon_deg2m(hull_deg, bounds_deg, bounds_m)
        hull_mask = calculate_mask(image, bounds_m, hull_m)

        return GeoPatch(patch=image,
                        bounds_deg=bounds_deg,
                        bounds_m=bounds_m,
                        polygons_deg=polygons_deg,
                        hull_deg=hull_deg,
                        polygons_m=polygons_m,
                        hull_m=hull_m,
                        hull_mask=hull_mask,
                        )

    def from_bounds(self, bounds, size):

        bounds_deg, bounds_m = calc_bounds_from_bounds_and_size(bounds, size, self.resolution)
        image = self.get_data(bounds_deg, size)

        return GeoPatch(patch=image,
                        bounds_deg=bounds_deg,
                        bounds_m=bounds_m,
                        polygons_deg=None,
                        hull_deg=None,
                        polygons_m=None,
                        hull_m=None,
                        hull_mask=None,
                        )

    def get_data(self, bounds, size):

        img = self.wms.getmap(layers=[self.layer],
                              styles=['default'],
                              srs='EPSG:4326',
                              bbox=bounds,
                              size=size,
                              format='image/png',
                              transparent=True,
                              )
        # TODO: do I need Image?
        image = np.array(Image.open(io.BytesIO(img.read())))
        return image[::-1, ...]


def plot_geo_patch(ax, geo_patch, unit='m', plot_polygons=False, plot_hull=False):

    if unit == 'm':
        x_ = np.linspace(geo_patch.bounds_m[0], geo_patch.bounds_m[2], geo_patch.patch.shape[1])
        y_ = np.linspace(geo_patch.bounds_m[1], geo_patch.bounds_m[3], geo_patch.patch.shape[0])
        ax.set_aspect('equal')

    elif unit == 'deg':
        x_ = np.linspace(geo_patch.bounds_deg[0], geo_patch.bounds_deg[2], geo_patch.patch.shape[1])
        y_ = np.linspace(geo_patch.bounds_deg[1], geo_patch.bounds_deg[3], geo_patch.patch.shape[0])
        ratio_deg = ((geo_patch.bounds_deg[2] - geo_patch.bounds_deg[0])
                     / (geo_patch.bounds_deg[3] - geo_patch.bounds_deg[1]))
        ratio_m = geo_patch.patch.shape[1] / geo_patch.patch.shape[0]
        ax.set_aspect(ratio_deg/ratio_m)
    else:
        raise ValueError(f'Unit not known: {unit}!')

    ax.pcolormesh(x_, y_, geo_patch.patch[:, :, 0])

    if plot_polygons:
        if unit == 'm':
            for poly in geo_patch.polygons_m:
                ax.fill(*poly.exterior.xy, alpha=0.3, color='k')
        if unit == 'deg':
            for poly in geo_patch.polygons_deg:
                ax.fill(*poly.exterior.xy, alpha=0.3, color='k')

    if plot_hull:
        if unit == 'm':
            ax.fill(*geo_patch.hull_m.exterior.xy, alpha=0.3, color='w')
        if unit == 'deg':
            ax.fill(*geo_patch.hull_deg.exterior.xy, alpha=0.3, color='w')


def calculate_mask(patch, bounds, hull):
    if hull is None:
        raise ValueError('No hull available in GeoPatch!')

    lon_fit = LinFit((bounds[0], 0), (bounds[2], patch.shape[1]))
    lat_fit = LinFit((bounds[1], 0), (bounds[3], patch.shape[0]))

    polygon_idx = []
    for x_, y_ in zip(*hull.exterior.xy):
        polygon_idx.append((lon_fit.predict(x_), lat_fit.predict(y_)))
    idx_poly = Polygon(polygon_idx)
    mask = rasterio.features.rasterize([idx_poly], out_shape=patch.shape[:2])
    return mask


if __name__ == "__main__":
    # x_vals = [7.653916139785217, 7.653926422075301, 7.65392295083162, 7.653913398620634, 7.653841735377802,
    #           7.653851542746392, 7.6538530526378015, 7.653861727760526, 7.653916139785217]
    # y_vals = [51.95987939942744, 51.95987929275744, 51.95971228099227, 51.959704134490515, 51.95975312339673,
    #           51.95977216532745, 51.95984194001826, 51.959844957915536, 51.95987939942744]
    # x_vals = [7.653734934506774, 7.653756039350539, 7.653741682127827, 7.653738739656338, 7.653749229437768,
    #           7.653738294658533, 7.653683606128862, 7.653687251256463, 7.653734934506774]
    # y_vals = [51.959807495560106, 51.95977446800089, 51.95976561358222, 51.959751218706636, 51.959744764285354,
    #           51.959737118748706, 51.959770757861186, 51.95977659704969, 51.959807495560106]
    x_vals = [7.653510104335948, 7.653584438982496, 7.653596493216375, 7.653681749605959, 7.653672517473981,
              7.653510003602702, 7.653510104335948]
    y_vals = [51.95967307151488, 51.959713667704584, 51.959719535955635, 51.95966788068461, 51.95966796603301,
              51.95966946830835, 51.95967307151488]

    poly_ = shapely.geometry.Polygon(zip(x_vals, y_vals))
    coords_ = (7.653900080270315, 51.96065781446052, 7.654040915315746, 51.96087076663339)

    providr_ = NrwWmsPatchProvider(resolution=10)

    geo_patch_ = providr_.from_point_and_size((x_vals[0], y_vals[0]), (300, 500))
    fig1_, ax1_ = plt.subplots(1, 1)
    plot_geo_patch(ax1_, geo_patch_, unit='deg')
    plt.plot([x_vals[0]], [y_vals[0]], 'ko')
    plt.show()

    geo_patch_ = providr_.from_polygons(poly_, factor=3)
    fig2_, ax2_ = plt.subplots(1, 1)
    plot_geo_patch(ax2_, geo_patch_, unit='m', plot_polygons=True)
    plt.show()

    geo_patch_ = providr_.from_polygons_and_size(poly_, (900, 500))
    fig3_, ax3_ = plt.subplots(1, 1)
    plot_geo_patch(ax3_, geo_patch_, unit='m', plot_polygons=True)
    plt.show()

    geo_patch_ = providr_.from_bounds(poly_.bounds, (900, 500))
    fig4_, ax4_ = plt.subplots(1, 1)
    plot_geo_patch(ax4_, geo_patch_, unit='m')
    plt.show()

    geo_patch_ = providr_.from_polygons(poly_, factor=3)
    fig5_, ax5_ = plt.subplots(1, 1)
    ax5_.pcolormesh(geo_patch_.hull_mask)
    plt.show()
