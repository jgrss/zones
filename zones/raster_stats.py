from __future__ import division

import itertools
import multiprocessing as multi
from pathlib import Path
import logging

from .base import ZonesMixin
from .stats import STAT_DICT
from .helpers import merge_dictionary_keys
from . import util
from .errors import add_handler

import numpy as np
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GA_ReadOnly
from affine import Affine
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
import bottleneck as bn
import rasterio as rio
from rasterio import features
from rasterio.windows import Window
from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def _merge_dicts(dict1, dict2):

    dict3 = dict1.copy()
    dict3.update(dict2)

    return dict3


def _warp(input_image,
          output_image,
          out_proj=None,
          in_proj=None,
          cell_size=0,
          **kwargs):

    """
    Warp transforms a dataset

    Args:
        input_image (str): The image to warp.
        output_image (str): The output image.
        out_proj (Optional[str]): The output proj4 projection code.
        in_proj (Optional[str]): An input projection string. Default is None.
        cell_size (Optional[float]): The output cell size. Default is 0.
        kwargs:
            format=None, outputBounds=None (minX, minY, maxX, maxY),
            outputBoundsSRS=None, targetAlignedPixels=False,
             width=0, height=0, srcAlpha=False, dstAlpha=False, warpOptions=None,
             errorThreshold=None, warpMemoryLimit=None,
             creationOptions=None, outputType=0, workingType=0,
             resampleAlg=resample_dict[resample], srcNodata=None, dstNodata=None,
             multithread=False, tps=False, rpc=False, geoloc=False,
             polynomialOrder=None, transformerOptions=None, cutlineDSName=None,
             cutlineLayer=None, cutlineWhere=None, cutlineSQL=None,
             cutlineBlend=None, cropToCutline=False, copyMetadata=True,
             metadataConflictValue=None, setColorInterpretation=False,
             callback=None, callback_data=None
             E.g.,
                creationOptions=['GDAL_CACHEMAX=256', 'TILED=YES']

    Returns:
        None, writes to `output_image'.
    """

    cell_size_ = (cell_size, -cell_size)

    awargs = _merge_dicts(dict(srcSRS=in_proj,
                               dstSRS=out_proj,
                               xRes=cell_size_[0],
                               yRes=cell_size_[1],
                               outputType=gdal.GDT_Float32,
                               resampleAlg=gdal.GRA_NearestNeighbour),
                          kwargs)

    warp_options = gdal.WarpOptions(**awargs)

    try:

        out_ds = gdal.Warp(output_image,
                           input_image,
                           options=warp_options)

    except:
        out_ds = None

    return out_ds


def _rasterize_zone(geom, src_crs, image_src, image_name, open_bands, return_poly=True):

    """
    Rasterizes a polygon geometry
    """

    left, bottom, right, top = geom.bounds

    # Check if any part of the Polygon is outside the raster bounds.
    if (left < image_src.bounds.left) or \
            (bottom < image_src.bounds.bottom) or \
            (right > image_src.bounds.right) or \
            (top > image_src.bounds.top):

        # Get the minimum overlapping extent bounds
        #   between the raster and the vector geometry
        max_left = max(left, image_src.bounds.left)
        min_top = min(top, image_src.bounds.top)
        min_right = min(right, image_src.bounds.right)
        max_bottom = max(bottom, image_src.bounds.bottom)

        # Create a Polygon geometry of the minimum bounds
        geom = Polygon([(max_left, max_bottom),
                        (max_left, min_top),
                        (min_right, min_top),
                        (min_right, max_bottom),
                        (max_left, max_bottom)])

        geom_df = gpd.GeoDataFrame(data=[0], geometry=[geom], crs=src_crs)

        if geom_df.empty:
            return None, None, None, None, None, None

        left, bottom, right, top = geom_df.total_bounds.tolist()

    # Get the y and x count of the vector geometry
    xcount = int(round((right - left) / abs(image_src.res[0])))
    ycount = int(round((top - bottom) / abs(image_src.res[1])))

    if (ycount < 1) or (xcount < 1):
        return None, None, None, None, None, None

    if return_poly:

        # Get the transform for the full vector extent
        transform = Affine(abs(image_src.res[0]), 0.0, left, 0.0, -abs(image_src.res[1]), top)

        poly_array = features.rasterize([geom],
                                        out_shape=(ycount, xcount),
                                        transform=transform,
                                        fill=0,
                                        out=None,
                                        all_touched=False,
                                        default_value=1,
                                        dtype='int32')

    if isinstance(image_name, str) or isinstance(image_name, Path):

        # Get the transform for the full image extent
        transform = Affine(abs(image_src.res[0]), 0.0, image_src.bounds.left, 0.0, -abs(image_src.res[1]), image_src.bounds.top)

        # Get the upper left offsets of the minimum bounds
        j, i = ~transform * (left + abs(image_src.res[1]) / 2.0, top - abs(image_src.res[0]) / 2.0)
        j, i = int(j), int(i)

        window = Window(row_off=i, col_off=j, width=xcount, height=ycount)

        if isinstance(open_bands, int):

            if open_bands == -1:
                open_bands_range = list(range(1, image_src.count+1))
            else:
                open_bands_range = list(range(open_bands, open_bands+1))

        else:
            open_bands_range = open_bands

        # Read the raster data as arrays
        image_array = np.array([image_src.read(band_idx, window=window)
                                for band_idx in open_bands_range], dtype='float64')

    else:

        if isinstance(image_src.data, xr.core.dataset.Dataset):

            image_array = image_src.data['bands'].sel(y=slice(top, bottom),
                                                      x=slice(left, right)).values

        else:

            image_array = image_src.data.sel(y=slice(top, bottom),
                                             x=slice(left, bottom)).values

    if len(image_array.shape) > 2:
        if image_array.shape[0] == 1:
            image_array = image_array[0]

    ycount_adjust = min(poly_array.shape[-2], image_array.shape[-2])
    xcount_adjust = min(poly_array.shape[-1], image_array.shape[-1])

    if len(image_array.shape) > 2:
        image_array = image_array[:, :ycount_adjust, :xcount_adjust]
    else:
        image_array = image_array[:ycount_adjust, :xcount_adjust]

    if return_poly:
        return poly_array[:ycount_adjust, :xcount_adjust], image_array, left, top, right, bottom
    else:
        return image_array


def _update_dict(didx, zones_dict, image_array, stats, band, no_data, image_bands):

    """
    Updates the stats dictionary
    """

    if len(image_array.shape) < 3:

        for sidx, stat in enumerate(stats):

            if stat == 'dist':

                zones_dict[1][didx] = ';'.join(list(map('{:.4f}'.format,
                                                        image_array[image_array != no_data].tolist())))

            else:

                stat_func = STAT_DICT[stat]

                # TODO: if zones are not unique
                if stat == 'mode':
                    zones_dict[1][didx][sidx] = float(stat_func(image_array).mode)
                elif stat == 'nanmode':
                    zones_dict[1][didx][sidx] = float(stat_func(image_array[np.where(~np.isnan(image_array))]).mode)
                else:
                    zones_dict[1][didx][sidx] = stat_func(image_array)

    else:

        if isinstance(band, int):

            for sidx, stat in enumerate(stats):

                if stat == 'dist':

                    zones_dict[band][didx] = ';'.join(list(map('{:.4f}'.format,
                                                                    image_array[bidx - 1][image_array[bidx - 1] != no_data].tolist())))

                else:

                    stat_func = STAT_DICT[stat]

                    # TODO: if zones are not unique
                    if stat == 'mode':
                        zones_dict[band][didx][sidx] = float(stat_func(image_array[band - 1]).mode)
                    elif stat == 'nanmode':
                        zones_dict[band][didx][sidx] = float(stat_func(image_array[band - 1][np.where(~np.isnan(image_array))]).mode)
                    else:
                        zones_dict[band][didx][sidx] = stat_func(image_array[band - 1])

        else:

            for bidx in range(1, image_bands + 1):

                for sidx, stat in enumerate(stats):

                    if stat == 'dist':

                        zones_dict[bidx][didx] = ';'.join(list(map('{:.4f}'.format,
                                                                   image_array[bidx - 1][image_array[bidx - 1] != no_data].tolist())))

                    else:

                        stat_func = STAT_DICT[stat]

                        # TODO: if zones are not unique
                        if stat == 'mode':
                            zones_dict[bidx][didx][sidx] = float(stat_func(image_array[bidx - 1]).mode)
                        elif stat == 'nanmode':
                            zones_dict[bidx][didx][sidx] = float(stat_func(image_array[bidx - 1][np.where(~np.isnan(image_array))]).mode)
                        else:
                            zones_dict[bidx][didx][sidx] = stat_func(image_array[bidx - 1])
    return zones_dict


def _update_crosstab_dict(didx,
                          zones_dict,
                          image_array,
                          other_image_array_list,
                          band,
                          image_bands,
                          values_name,
                          other_values_list):

    """
    Updates the stats dictionary
    """

    combined_arrays = [image_array] + other_image_array_list
    combined_list = [Path(values_name).name] + [Path(fn).name for fn in other_values_list]

    stat_func = STAT_DICT['crosstab']

    if len(image_array.shape) < 3:

        for a, b in itertools.combinations(combined_list, r=2):

            a, b = Path(a).name, Path(b).name

            idx1 = combined_list.index(a)
            idx2 = combined_list.index(b)

            zones_dict[1][f'{a}-{b}'][didx][0] = stat_func(combined_arrays[idx1], combined_arrays[idx2])

    else:

        if isinstance(band, int):
            zones_dict[band][didx][0] = stat_func(image_array[band-1], other_image_array_list[band-1])
        else:

            for bidx in range(1, image_bands+1):

                for a, b in itertools.combinations(combined_list, r=2):

                    a, b = Path(a).name, Path(b).name

                    idx1 = combined_list.index(a)
                    idx2 = combined_list.index(b)

                    zones_dict[bidx][f'{a}-{b}'][didx][0] = stat_func(combined_arrays[idx1][bidx-1], combined_arrays[idx2][bidx-1])

    return zones_dict


# def _worker(didx, df_row, stats, n_stats, src_wkt, raster_value, no_data, verbose, n, band, open_bands, values):

def _worker(*args):

    didx, df_row, stats, n_stats, src_wkt, raster_value, no_data, verbose, n, band, open_bands, values = list(itertools.chain(*args))

    if verbose > 1:

        if didx % 100 == 0:
            logger.info('    Zone {:,d} of {:,d} ...'.format(didx+1, n))

    if isinstance(values, str) or isinstance(values, Path):
        values_src = rio.open(values, mode='r')
    else:
        values_src = values

    if isinstance(band, int) or (values_src.count == 1):
        data_values = np.concatenate(([didx], np.zeros(n_stats, dtype='float64') + np.nan))
    else:
        data_values = np.concatenate(([didx], np.zeros(n_stats * values_src.count, dtype='float64')))

    return_early = False

    geom = df_row.geometry

    if not geom:

        if isinstance(values, str) or isinstance(values, Path):
            values_src.close()

        return data_values

    # Rasterize the data
    poly_array, image_array, left, top, right, bottom = _rasterize_zone(geom, src_wkt, values_src, values, open_bands)

    # Cases where multi-band images are flattened
    #   because of single-zone pixels
    if not isinstance(band, int) and (values_src.count > 1):

        if len(image_array.shape) == 1:

            if isinstance(band, int):
                image_array = image_array.reshape(1, 1, 1)
            else:
                image_array = image_array.reshape(values_src.count, 1, 1)

    # Binarize a specific value
    if isinstance(raster_value, int):

        image_array = np.where(image_array == raster_value, 1, 0)

        if image_array.max() == 0:
            return_early = True

    # Binarize a list of values
    elif isinstance(raster_value, list):

        image_array_ = np.zeros(image_array.shape, dtype='uint8')

        for raster_value_ in raster_value:
            image_array_ = np.where(image_array == raster_value_, 1, image_array_)

        image_array = image_array_

        if image_array.max() == 0:
            return_early = True

    if not return_early:

        if not isinstance(poly_array, np.ndarray):

            if isinstance(values, str) or isinstance(values, Path):
                values_src.close()

            return data_values

        else:

            null_idx = np.where(poly_array == 0)

            if null_idx[0].shape[0] > 0:

                if not isinstance(band, int) and (values_src.count > 1):

                    for ix in range(0, values_src.count):
                        image_array[ix][null_idx] = no_data

                else:
                    image_array[null_idx] = no_data

            if any(['nan' in x for x in stats]):

                no_data_idx = np.where(image_array == no_data)

                if no_data_idx[0].shape[0] > 0:
                    image_array[no_data_idx] = np.nan

                if np.isnan(bn.nanmax(image_array)):

                    if isinstance(values, str) or isinstance(values, Path):
                        values_src.close()

                    return data_values

        if isinstance(band, int) or (values_src.count == 1):

            for sidx, stat in enumerate(stats):

                stat_func = STAT_DICT[stat]

                if stat == 'mode':
                    data_values_stat = float(stat_func(image_array).mode)
                else:
                    data_values_stat = stat_func(image_array)

                data_values[1+sidx] = data_values_stat

        else:

            for bdidx in range(0, values_src.count):

                for sidx, stat in enumerate(stats):

                    stat_func = STAT_DICT[stat]

                    if stat == 'mode':
                        data_values_stat = float(stat_func(image_array[bdidx]).mode)
                    else:
                        data_values_stat = stat_func(image_array[bdidx])

                    data_values[1+(bdidx*n_stats)+sidx] = data_values_stat

    if isinstance(values, str) or isinstance(values, Path):
        values_src.close()

    return data_values


def _calc_parallel(stats, src_wkt, raster_value, no_data, verbose, n, zones_df, values, band, open_bands, n_jobs, chunksize):

    n_stats = len(stats)

    # results = Parallel(n_jobs=n_jobs,
    #                    max_nbytes=None)(delayed(_worker)(didx,
    #                                                      dfrow,
    #                                                      stats,
    #                                                      n_stats,
    #                                                      src_wkt,
    #                                                      raster_value,
    #                                                      no_data,
    #                                                      verbose,
    #                                                      n,
    #                                                      band,
    #                                                      open_bands,
    #                                                      values)
    #                                     for didx, dfrow in zones_df.iterrows())

    data_gen = ((didx,
                 dfrow,
                 stats,
                 n_stats,
                 src_wkt,
                 raster_value,
                 no_data,
                 verbose,
                 n,
                 band,
                 open_bands,
                 values) for didx, dfrow in zones_df.iterrows())

    results = []

    with multi.Pool(processes=n_jobs) as executor:

        for result in executor.imap(_worker, data_gen, chunksize=chunksize):
            results.append(result)

    if isinstance(band, int):
        n_bands = 1
    else:

        if isinstance(values, str) or isinstance(values, Path):

            with rio.open(values) as src_tmp:
                n_bands = src_tmp.count

        else:
            n_bands = values.bands

    return merge_dictionary_keys(np.array(results, dtype='float64'), stats, n_bands)


class RasterStats(ZonesMixin):

    """
    Args:
        values (str or xarray): The raster values file. Can be a float or categorical raster.
        zones (str or GeoDataFrame): The zones file.
            Accepted types are:
                str: vector file (e.g., shapefile, or geopackage)
                GeoDataFrame: the geometry type must be `Polygon`
        other_values (Optional[list | str]): Other raster values used in cross-tabulation.
        no_data (Optional[int or float]): A no data value to mask. Default is 0.
        raster_value (Optional[int | list]): A raster value to get statistics for. Default is None.
        unique_column (Optional[str]): A unique column identifier for `zones`.
            Default is None, which treats all zones as unique and uses the
            highest (resolution) level geometry.
        band (Optional[int]): The band to calculate (if multi-band). Default is None, or calculate all bands.
        column_prefix (Optional[str]): A name to prepend to each band (column) output name.
        verbose (Optional[int]): The verbosity level. Default is 0.
        n_jobs (Optional[int]): The number of parallel processes (zones). Default is 1.
            *Currently, this only works with one statistic.
        chunksize (Optional[int]): The ``chunksize`` argument passed to ``multiprocessing.Pool().imap``.

    Examples:
        >>> import zones
        >>>
        >>> zs = zones.RasterStats('values.tif', 'zones.shp')
        >>>
        >>> # Calculate the 'mean'.
        >>> df = zs.calculate('mean')
        >>>
        >>> # Calculate multiple statistics.
        >>> df = zs.calculate(['nanmean', 'nansum'])
        >>>
        >>> # Write data to file
        >>> df.to_file('stats.shp')
        >>> df.to_csv('stats.csv')
    """

    def __init__(self,
                 values,
                 zones,
                 other_values=None,
                 no_data=0,
                 raster_value=None,
                 unique_column=None,
                 band=None,
                 column_prefix=None,
                 verbose=0,
                 n_jobs=1,
                 chunksize=1):

        self.values = values
        self.zones = zones
        self.other_values = other_values
        self.no_data = no_data
        self.raster_value = raster_value
        self.unique_column = unique_column
        self.band = band
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.column_prefix = column_prefix

        self.stats = None
        self.zone_values = None
        self.open_bands = self.band if isinstance(self.band, int) else -1

        if self.other_values:

            if isinstance(self.other_values, str) or isinstance(self.other_values, Path):
                self.other_values = [self.other_values]

    def zone_iter(self, stats):

        src_crs = self._prepare_crs()

        n = self.zones_df.shape[0]

        if self.verbose > 0:
            logger.info('  Calculating statistics ...')

        if self.n_jobs != 1:

            self.zone_values = _calc_parallel(stats,
                                              src_crs,
                                              self.raster_value,
                                              self.no_data,
                                              self.verbose,
                                              n,
                                              self.zones_df,
                                              self.values,
                                              self.band,
                                              self.open_bands,
                                              self.n_jobs,
                                              self.chunksize)

        else:

            for didx, df_row in tqdm(self.zones_df.iterrows(),
                                     total=self.zones_df.shape[0]):

                geom = df_row.geometry

                if not geom:
                    continue

                # Rasterize the data
                poly_array, image_array, left, top, right, bottom = _rasterize_zone(geom,
                                                                                    src_crs,
                                                                                    self.values_src,
                                                                                    self.values,
                                                                                    self.open_bands)

                other_image_array_list = []

                if self.other_values and isinstance(self.other_values, list):

                    for other_values_src_, other_values_ in zip(self.other_values_src, self.other_values):

                        other_image_array = _rasterize_zone(geom,
                                                            src_crs,
                                                            other_values_src_,
                                                            other_values_,
                                                            self.open_bands,
                                                            return_poly=False)

                        other_image_array_list.append(other_image_array)

                if isinstance(self.raster_value, int):

                    image_array = np.where(image_array == self.raster_value, 1, 0)

                    if self.other_values and isinstance(self.other_values, list):

                        for oidx, other_array_ in enumerate(other_image_array_list):
                            other_image_array_list[oidx] = np.where(other_array_ == self.raster_value, 1, 0)

                    if image_array.max() == 0:
                        continue

                elif isinstance(self.raster_value, list):

                    image_array_ = np.zeros(image_array.shape, dtype='uint8')

                    if self.other_values and isinstance(self.other_values, list):

                        other_zeros_list = []

                        for oidx, other_array_ in enumerate(other_image_array_list):
                            other_zeros_list.append(np.zeros(other_array_.shape, dtype='uint8'))

                    for raster_value_ in self.raster_value:

                        image_array_ = np.where(image_array == raster_value_, 1, image_array_)

                        if self.other_values and isinstance(self.other_values, list):

                            for oidx, other_array_ in enumerate(other_image_array_list):
                                other_zeros_list[oidx] = np.where(other_array_ == raster_value_, 1, other_zeros_list[oidx])

                    image_array = image_array_

                    if self.other_values and isinstance(self.other_values, list):
                        other_image_array_list = other_zeros_list

                    if image_array.max() == 0:
                        continue

                if not isinstance(poly_array, np.ndarray):

                    image_array = np.array([0], dtype='float32')

                    if self.other_values and isinstance(self.other_values, list):
                        other_image_array_list = [np.array([0], dtype='float32') for i_ in range(0, len(other_image_array_list))]

                else:

                    null_idx = np.where(poly_array == 0)

                    if null_idx[0].shape[0] > 0:

                        if len(image_array.shape) > 2:

                            for ix in range(0, image_array.shape[0]):

                                image_array[ix][null_idx] = self.no_data

                                if self.other_values and isinstance(self.other_values, list):

                                    for oidx, other_array_ in enumerate(other_image_array_list):
                                        other_array_[ix][null_idx] = self.no_data
                                        other_image_array_list[oidx] = other_array_

                        else:

                            image_array[null_idx] = self.no_data

                            if self.other_values and isinstance(self.other_values, list):

                                for oidx, other_array_ in enumerate(other_image_array_list):
                                    other_array_[null_idx] = self.no_data
                                    other_image_array_list[oidx] = other_array_

                    if any(['nan' in x for x in stats]):

                        no_data_idx = np.where(image_array == self.no_data)

                        if no_data_idx[0].shape[0] > 0:
                            image_array[no_data_idx] = np.nan

                        if np.isnan(bn.nanmax(image_array)):
                            continue

                        if self.other_values and isinstance(self.other_values, list):

                            for oidx, other_array_ in enumerate(other_image_array_list):

                                other_no_data_idx = np.where(other_array_ == self.no_data)

                                if no_data_idx[0].shape[0] > 0:
                                    other_array_[other_no_data_idx] = np.nan

                if stats[0] == 'crosstab':

                    self.zone_values = _update_crosstab_dict(didx,
                                                             self.zone_values,
                                                             image_array,
                                                             other_image_array_list,
                                                             self.band,
                                                             self.values_src.count,
                                                             self.values,
                                                             self.other_values)

                else:

                    self.zone_values = _update_dict(didx,
                                                    self.zone_values,
                                                    image_array,
                                                    stats,
                                                    self.band,
                                                    self.no_data,
                                                    self.values_src.count)
