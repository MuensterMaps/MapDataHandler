"""
Cut out satellite/drone images

@author: Frank Ehebrecht
"""


import argparse

import contextily as ctx
import matplotlib.pyplot as plt
import geopandas as gpd

# TODO: this should be done in a more elegant way!
EXPECTED_INCREMENT = 0.07464553543395595


class GeoPatch:
    src = ctx.providers.Esri.WorldImagery
    zoom = 21

    def __init__(self, patch, extent, bounds, polygon):
        self.patch = patch
        self.extent = extent
        self.bounds = bounds
        self.polygon = polygon
        self.hull = None
        self.x_incr = None
        self.y_incr = None
        self._get_axis_increments()
        if polygon is not None:
            self.hull = self.polygon.exterior.xy

    @classmethod
    def from_polygon(cls, polygon, bounds_factor=0.0, size_px=None):
        """
        mode 1: polygon provided, use this as bounds
        mode 2: polygon and bounds_factor provided: extent polygon bounds with factor
        mode 3: polygon and size_px provided: cut out data with dimensions size_px x size_px
        """
        if size_px is None:
            bounds = GeoPatch._calc_bounds(polygon, bounds_factor)
            img_data = GeoPatch._get_img_data(bounds)
            patch, extent = img_data
            return cls(patch, extent, bounds, polygon)
        else:
            bounds = GeoPatch._calc_bounds(polygon, 0.0)
            expected_extent = size_px * EXPECTED_INCREMENT * 1.1
            additional_extent_x = expected_extent - (bounds[1] - bounds[0])
            additional_extent_y = expected_extent - (bounds[3] - bounds[2])
            if additional_extent_x < 0 or additional_extent_y < 0:
                raise ValueError('PolyGon does not fit in required extent!')
            new_bounds = (bounds[0] - additional_extent_x / 2,
                          bounds[1] + additional_extent_x / 2,
                          bounds[2] - additional_extent_y / 2,
                          bounds[3] + additional_extent_y / 2,
                          )
            _ = GeoPatch._get_img_data(new_bounds)
            # TODO: cut out centric patch of given extent
            raise NotImplementedError()

    @classmethod
    def from_bounds(cls, bounds):
        patch, extent = self._set_img_data(bounds)
        return cls(patch, extent, bounds, None)

    @classmethod
    def from_geo_df(cls, df):
        raise NotImplementedError()

    @staticmethod
    def _calc_bounds(polygon, factor):

        if factor == 0.0:
            return polygon.bounds
        else:
            bounds = polygon.bounds
            length = bounds[2] - bounds[0]
            width = bounds[3] - bounds[1]
            extended_bounds = (bounds[0] - factor * length,
                               bounds[1] - factor * width,
                               bounds[2] + factor * length,
                               bounds[3] + factor * width,
                               )
            return extended_bounds

    @staticmethod
    def _get_img_data(bounds):
        try:
            return ctx.tile.bounds2img(*bounds,
                                       zoom=GeoPatch.zoom,
                                       source=GeoPatch.src,
                                       ll=False
                                       )
        except Exception as e:
            print(str(repr(e)))
            return None, None
    
    def _get_axis_increments(self):
        self.x_incr = (self.extent[3] - self.extent[2]) / self.patch.shape[0]
        self.y_incr = (self.extent[1] - self.extent[0]) / self.patch.shape[1]

    def plot(self, ax, show_polygon=False):
        if show_polygon and self.polygon is None:
            raise ValueError('No polygon given!')
        ax.imshow(self.patch, extent=self.extent)
        ax.fill(*self.hull, alpha=0.5)

    def __repr__(self):
        if self.polygon is not None:
            return f'Shape is {self.patch.shape} | x increment {self.x_incr} | y increment {self.y_incr}'
        # TODO: polygon if given
        # TODO: coordinates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    args = parser.parse_args()
    path_ = args.src

    # load data
    df_ = gpd.read_file(path_)
    df_ = df_.to_crs(epsg=3857)
    # select first polygon
    idx_ = 0
    poly_ = df_.iloc[idx_]['geometry']

    # create GeoPatch object
    gp = GeoPatch.from_polygon(poly_)
    # print it
    print(gp)
    # plot it
    fig, ax = plt.subplots(1, 1)
    gp.plot(ax, show_polygon=True)
    plt.show()
