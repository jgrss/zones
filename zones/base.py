import os
from collections import namedtuple
import math
import itertools
from pathlib import Path

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
from shapely.geometry import Point, Polygon

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

        self.prepare_files()

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

    def prepare_files(self):

        """
        Prepares files
        """

        self.values_df = None
        self.values_src = None
        self.other_values_df = []
        self.other_values_src = []

        self.zones_df = self.check_if_geodf(self.zones)[0]
        self.values_df, self.values_src = self.check_if_geodf(self.values)

        if self.other_values:

            if not isinstance(self.other_values, list):
                logger.exception('  The other raster values must be a list.')
                raise TypeError

            for other_values in self.other_values:

                other_values_df_, other_values_src_ = self.check_if_geodf(other_values)
                self.other_values_df.append(other_values_df_)
                self.other_values_src.append(other_values_src_)

    def prepare_zones(self, unique_column):

        if self.values_src:
            self.n_bands = self.values_src.count
        else:
            self.n_bands = 0

        if self.other_values:
            self.n_other_images = len(self.other_values)
        else:
            self.n_other_images = 0

        if isinstance(unique_column, str):
            return None
        else:

            if self.other_values:

                dict_len = self.zones_df.shape[0]

                zones_dict_ = {}

                if self.n_bands > 1:

                    for bidx in range(1, self.n_bands+1):

                        zones_dict_[bidx] = {}

                        for a, b in itertools.combinations([self.values] + self.other_values, r=2):

                            a, b = Path(a).name, Path(b).name

                            zones_dict_[bidx][f'{a}-{b}'] = {}

                            for i in range(0, dict_len):
                                zones_dict_[bidx][f'{a}-{b}'][i] = [0.0]

                else:

                    zones_dict_[1] = {}

                    for a, b in itertools.combinations([self.values] + self.other_values, r=2):

                        a, b = Path(a).name, Path(b).name

                        zones_dict_[1][f'{a}-{b}'] = {}

                        for i in range(0, dict_len):
                            zones_dict_[1][f'{a}-{b}'][i] = [0.0]

                return zones_dict_

            else:

                return create_dictionary(self.zones_df.shape[0],
                                         len(self.stats),
                                         self.n_bands)

    def _finalize_frequencies(self):

        zones_list = sorted(list(self.zone_values[1][list(self.zone_values[1].keys())[0]].keys()))

        pairs_list = list(set(list(itertools.chain.from_iterable(
            [[[f'{pair} band-{bd}' for pair in self.zone_values[bd].keys()] for bd in range(1, self.n_bands+1)] for zone in zones_list][0]))))

        values_list = list(set(list(itertools.chain.from_iterable(
            [[[list(self.zone_values[bd][pair][zone][0].keys()) for pair in self.zone_values[bd].keys()] for bd in range(1, self.n_bands+1)] for zone in zones_list][0][0]))))

        arrays = [pairs_list * len(values_list), values_list]

        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['pair', 'zone'])
        df = pd.DataFrame(np.zeros((len(zones_list), len(values_list)), dtype='int64'), index=zones_list, columns=index)

        for z in zones_list:

            for pair in pairs_list:

                image_pair, band_str = pair.split(' ')
                band = int(band_str.split('-')[1])

                dft = df.loc[z, pair]
                dft.loc[:] = 0

                for value in values_list:

                    if value in dft:

                        if (image_pair in self.zone_values[band]) and (z in self.zone_values[band][image_pair]) and (value in self.zone_values[band][image_pair][z][0]):
                            dft.loc[value] = self.zone_values[band][image_pair][z][0][value]

                df.loc[z, pair] = dft.values

        return df

    def finalize_dataframe(self):

        if hasattr(self, 'band'):

            prefix = self.column_prefix if self.column_prefix else '_bd'

            if isinstance(self.band, int):

                values_df = pd.DataFrame.from_dict(self.zone_values[1], orient='index')
                values_df.columns = ('{}{:d},'.format(prefix, self.band).join(self.stats) + '{}{:d}'.format(prefix, self.band)).split(',')

            else:

                if self.other_values:
                    values_df = self._finalize_frequencies()
                else:

                    for bidx in range(1, self.n_bands+1):

                        values_df_ = pd.DataFrame.from_dict(self.zone_values[bidx], orient='index')

                        values_df_.columns = ('{}{:d},'.format(prefix, bidx).join(self.stats) + '{}{:d}'.format(prefix, bidx)).split(',')

                        if bidx == 1:
                            values_df = values_df_.copy()
                        else:
                            values_df = pd.concat((values_df, values_df_), axis=1)

        else:

            values_df = pd.DataFrame.from_dict(self.zone_values, orient='index')
            values_df.columns = self.stats

        if self.other_values:
            return values_df
        else:
            return pd.merge(self.zones_df, values_df, left_index=True, right_index=True)

    def _close_files(self):

        if self.values_src:

            if hasattr(self.values_src, 'close'):

                self.values_src.close()
                self.values_src = None

        if self.other_values_src:

            for other_src in self.other_values_src:

                if hasattr(other_src, 'close'):

                    other_src.close()
                    other_src = None

    def _prepare_crs(self):

        crs_wkt = util.check_crs(self.zones_df.crs)

        if not crs_wkt:

            sr = osr.SpatialReference()
            util.check_axis_order(sr)
            sr.ImportFromWkt(self.values_src.projection)
            crs_wkt = sr.ExportToWkt()

        return crs_wkt

    def check_arguments(self, stats):

        """
        Args:
            stats (list)
        """

        if isinstance(self.values, str):

            if not os.path.isfile(self.values):

                logger.exception('  The values file does not exist.')
                raise ValuesFileError

        if self.other_values:

            if isinstance(self.other_values, list):

                for other_value in self.other_values:

                    if not os.path.isfile(other_value):

                        logger.exception('  The other values {} file does not exist.'.format(other_value))
                        raise ValuesFileError

        if not isinstance(self.zones, gpd.GeoDataFrame):

            if not os.path.isfile(self.zones):

                logger.exception('  The zones file does not exist.')
                raise ZonesFileError

        if list(set(stats).difference(STAT_DICT.keys())):

            logger.exception('  The statistic, {}, is not available.'.format(list(set(stats).difference(STAT_DICT.keys()))))
            raise StatsError

    @staticmethod
    def melt_freq(df):

        """
        Melts records of frequencies

        Args:
            df (DataFrame): The DataFrame to melt.

        Example:
            >>> import zones
            >>>
            >>> # Cross-tabulation of two rasters
            >>> zs = zones.RasterStats('raster.tif', 'vector.gpkg', other_values='other_raster.tif', n_jobs=1)
            >>> df = zs.calculate('crosstab')
            >>>
            >>> df = zs.melt_freq(df)

        Returns:
            Melted DataFrame (DataFrame)
        """

        dfm = pd.melt(df.copy())
        dfm.columns = ['pair', 'values', 'freq']

        return dfm

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


def grid(bounds, gy, gx, celly, cellx, crs=None):

    """
    Creates polygon grids

    Args:
        bounds (tuple | BoundingBox): A tuple of (left, bottom, right, top) or a
            ``rasterio.coords.BoundingBox`` instance.
        gy (float): The target grid y size.
        gx (float): The target grid x size.
        celly (float): The y cell size.
        cellx (float): The x cell size.
        crs (Optional[str]): The CRS.

    Returns:
        ``geopandas.GeoDataFrame``

    Example:
        >>> import zones
        >>>
        >>> bounds = (left, bottom, right, top)
        >>>
        >>> # Create 1 ha grids
        >>> df = zones.grid(bounds, 100, 100, 30.0, 30.0)
    """

    left, bottom, right, top = bounds

    polys = list()

    nrows = int(abs(top - bottom) / abs(celly))
    ncols = int(abs(right - left) / abs(cellx))

    nrowsp = int(math.ceil((nrows * abs(celly)) / gy))
    ncolsp = int(math.ceil((ncols * abs(cellx)) / gx))

    for i in range(0, nrowsp):
        for j in range(0, ncolsp):

            polys.append(Polygon([(left + j * gx, top - i * gy),
                                  (left + (j * gx) + gx, top - i * gy),
                                  (left + (j * gx) + gx, top - (i * gy) - gy),
                                  (left + j * gx, top - (i * gy) - gy),
                                  (left + j * gx, top - i * gy)]))

    return gpd.GeoDataFrame(data=np.array(list(range(0, len(polys))), dtype='int64'),
                            columns=['grid'],
                            geometry=polys,
                            crs=crs)


def voronoi(dataframe, grid_size=100, sample_size=10):

    """
    Creates Voronoi polygons from random points

    Args:
        dataframe (GeoDataFrame): The dataframe.
        grid_size (Optional[int]): The number of x and y coordinates to sample from.
        sample_size (Optional[int]): The number of random points to generate. This number is not guaranteed because
            points are clipped to geometry.

    Returns:
        ``DataFrame``
    """

    if not EARTHPY_INSTALLED:
        logger.exception('  earthpy must be installed to create voronoi polygons')
        raise ImportError

    geom = dataframe.geometry.values[0]

    left, bottom, right, top = dataframe.total_bounds.flatten().tolist()

    randx = np.random.choice(np.linspace(left, right, grid_size), size=int(sample_size), replace=False)
    randy = np.random.choice(np.linspace(top, bottom, grid_size), size=int(sample_size), replace=False)

    points = [[x, y] for x, y in zip(randx, randy) if Point(x, y).within(geom)]

    vor = Voronoi(points)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    geom = [poly for poly in shapely.ops.polygonize(lines)]
    df_voronoi = gpd.GeoDataFrame(data=range(0, len(geom)), geometry=geom, crs=dataframe.crs)

    df_voronoi_clip = cl.clip_shp(df_voronoi, dataframe).reset_index()

    return df_voronoi_clip
