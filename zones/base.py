import os
from collections import namedtuple

from .errors import logger, ValuesFileError, StatsError, ZonesFileError
from .stats import STAT_DICT
from .helpers import create_dictionary
from . import util

import numpy as np
from scipy.spatial import Voronoi
from osgeo import osr
import pandas as pd
import xarray as xr
import rasterio as rio
import geopandas as gpd
import six
import shapely
from shapely.geometry import Point

try:
    from earthpy import clip as cl
    EARTHPY_INSTALLED = True
except:
    EARTHPY_INSTALLED = False

shapely.speedups.enable()


class ZonesMixin(object):

    @property
    def stats_avail(self):
        return list(STAT_DICT.keys())

    def calculate(self, stats):

        """
        Args:
            stats (str list)

        Returns:
            DataFrame
        """

        self.stats = stats

        if isinstance(self.stats, str):
            self.stats = [self.stats]

        self.check_arguments(self.stats)

        if self.verbose > 0:
            logger.info('  Preparing files ...')

        self.prepare_files(self.zones, self.values)

        if self.verbose > 0:
            logger.info('  Preparing zones ...')

        self.zone_values = self.prepare_zones(self.unique_column)

        if self.verbose > 0:
            logger.info('  Calculating stats ...')

        self.zone_iter(self.stats)

        if self.verbose > 0:
            logger.info('  Finalizing data ...')

        self._close_files()

        return self.finalize_dataframe()

    @staticmethod
    def check_if_geodf(data_file):

        """
        Checks for file data type

        Args:
            data_file (GeoDataFrame or image file or Xarray Dataset)

                *If `data_file` is an `Xarray.Dataset` or `Xarray.DataArray`, `data_file` must have the
                following attributes:  projection (str) and res (tuple)

        Returns:
            data_file (GeoDataFrame)
        """

        if isinstance(data_file, gpd.GeoDataFrame):
            return data_file.reset_index(), None
        else:

            if isinstance(data_file, xr.Dataset) or isinstance(data_file, xr.DataArray):

                if isinstance(data_file, xr.Dataset):
                    array_shape = data_file['bands'].shape
                else:
                    array_shape = data_file.shape

                ImageInfo = namedtuple('ImageInfo', 'data bands crs res left right bottom top')

                if len(array_shape) > 2:

                    image_info = ImageInfo(data=data_file,
                                           bands=array_shape[0],
                                           crs=data_file.crs,
                                           res=data_file.res[0],
                                           left=data_file.x.values[0],
                                           right=data_file.x.values[-1],
                                           bottom=data_file.y.values[-1],
                                           top=data_file.y.values[0])

                else:

                    image_info = ImageInfo(data=data_file,
                                           bands=1,
                                           crs=data_file.crs,
                                           res=data_file.res[0],
                                           left=data_file.x.values[0],
                                           right=data_file.x.values[-1],
                                           bottom=data_file.y.values[-1],
                                           top=data_file.y.values[0])

                return None, image_info

            else:

                file_extension = os.path.splitext(os.path.split(data_file)[1])[1].lower().strip()

                if file_extension in ['.shp', '.gpkg']:
                    return gpd.read_file(data_file).reset_index(), None
                elif file_extension == '.csv':
                    return pd.read_csv(data_file), None
                else:
                    return None, rio.open(data_file, mode='r')

    def prepare_files(self, zones, values):

        """
        Prepares files

        Args:
            zones
            values
        """

        self.values_df = None
        self.values_src = None

        self.zones_df = self.check_if_geodf(zones)[0]
        self.values_df, self.values_src = self.check_if_geodf(values)

    def prepare_zones(self, unique_column):

        if self.values_src:
            self.n_bands = self.values_src.count
        else:
            self.n_bands = 0

        # TODO
        if isinstance(unique_column, str):
            return None
        else:

            return create_dictionary(self.zones_df.shape[0],
                                     len(self.stats),
                                     self.n_bands)

    def finalize_dataframe(self):

        if hasattr(self, 'band'):

            if isinstance(self.band, int):

                values_df = pd.DataFrame.from_dict(self.zone_values[1], orient='index')
                values_df.columns = ('_bd{:d},'.format(self.band).join(self.stats) + '_bd{:d}'.format(self.band)).split(',')

            else:

                for bidx in range(1, self.n_bands+1):

                    values_df_ = pd.DataFrame.from_dict(self.zone_values[bidx], orient='index')
                    values_df_.columns = ('_bd{:d},'.format(bidx).join(self.stats) + '_bd{:d}'.format(bidx)).split(',')

                    if bidx == 1:
                        values_df = values_df_.copy()
                    else:
                        values_df = pd.concat((values_df, values_df_), axis=1)

        else:

            values_df = pd.DataFrame.from_dict(self.zone_values, orient='index')
            values_df.columns = self.stats

        return pd.merge(self.zones_df, values_df, left_index=True, right_index=True)

    def _close_files(self):

        if self.values_src:

            if hasattr(self.values_src, 'close'):

                self.values_src.close()
                self.values_src = None

    def _prepare_proj4(self):

        proj4 = ''

        for k, v in six.iteritems(self.zones_df.crs):
            proj4 += '+{}={} '.format(k, v)

        if not proj4:

            sr = osr.SpatialReference()
            util.check_axis_order(sr)
            sr.ImportFromWkt(self.values_src.projection)
            proj4 = sr.ExportToProj4()

        return proj4

    def check_arguments(self, stats):

        """
        Args:
            stats (list)
        """

        if isinstance(self.values, str):

            if not os.path.isfile(self.values):

                logger.error('  The values file does not exist.')
                raise ValuesFileError

        if not isinstance(self.zones, gpd.GeoDataFrame):

            if not os.path.isfile(self.zones):

                logger.error('  The zones file does not exist.')
                raise ZonesFileError

        if list(set(stats).difference(STAT_DICT.keys())):

            logger.error('  The statistic, {}, is not available.'.format(list(set(stats).difference(STAT_DICT.keys()))))
            raise StatsError

    @staticmethod
    def melt_dist(df, id_field=None):

        """
        Melts records of distributions into columns

        Args:
            df (DataFrame): The DataFrame to melt.
            id_field (Optional[str]): An id field to include. Otherwise, only the band columns are melted.

        Example:
            >>> import zones
            >>>
            >>> zs = zones.RasterStats('raster.tif', 'vector.gpkg', n_jobs=1)
            >>> df = zs.calculate('dist')
            >>> df = zs.melt_dist(df, id_field='id')

        Returns:
            Melted DataFrame (DataFrame)
        """

        out_df = dict()

        for i, df_row in df.iterrows():

            if not isinstance(id_field, str):
                first_col = True
            else:
                first_col = False

            for col in df.columns.tolist():

                if col.startswith('dist'):

                    out_col = col.replace('dist_', '')

                    val_list = df_row[col].split(';')
                    val_list = list(map(float, val_list))

                    if not first_col:

                        if id_field in out_df:
                            out_df[id_field] = out_df[id_field] + [int(df_row.id)] * len(val_list)
                        else:
                            out_df[id_field] = [int(df_row.id)] * len(val_list)

                        first_col = True

                    if out_col in out_df:
                        out_df[out_col] = out_df[out_col] + val_list
                    else:
                        out_df[out_col] = val_list

        min_length = 1e9
        for key, value in six.iteritems(out_df):

            if len(value) < min_length:
                min_length = len(value)

        for key, value in six.iteritems(out_df):

            if len(value) > min_length:
                out_df[key] = value[:min_length]

        return pd.DataFrame(data=out_df)


def voronoi(dataframe, size=100):

    """
    Creates Voronoi polygons from random points

    Args:
        dataframe (GeoDataFrame): The dataframe.
        size (Optional[int]): The number of random points to generate. This number is not guaranteed because
            points are clipped to geometry.

    Returns:
        ``DataFrame``
    """

    if not EARTHPY_INSTALLED:
        logger.exception('  earthpy must be installed to create voronoi polygons')

    geom = dataframe.geometry.values[0]

    left, bottom, right, top = dataframe.bounds.values.flatten().tolist()

    randx = np.random.choice(np.linspace(left, right, (right - left) / 30.0), size=size, replace=False)
    randy = np.random.choice(np.linspace(top, bottom, (top - bottom) / 30.0), size=size, replace=False)

    points = [[x, y] for x, y in zip(randx, randy) if Point(x, y).within(geom)]

    vor = Voronoi(points)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    geom = [poly for poly in shapely.ops.polygonize(lines)]
    df_voronoi = gpd.GeoDataFrame(data=range(0, len(geom)), geometry=geom, crs=dataframe.crs)

    df_voronoi_clip = cl.clip_shp(df_voronoi, dataframe).reset_index()

    return df_voronoi_clip
