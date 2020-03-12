from __future__ import division

from .base import ZonesMixin
from .errors import logger
from .stats import STAT_DICT
from .helpers import merge_dictionary_keys
from . import util

import numpy as np
from osgeo import gdal, ogr, osr
import xarray as xr
import shapely
# from shapely.geometry import Polygon
import bottleneck as bn
import rasterio as rio

from joblib import Parallel, delayed
from tqdm import tqdm

shapely.speedups.enable()


def _merge_dicts(dict1, dict2):

    dict3 = dict1.copy()
    dict3.update(dict2)

    return dict3


def warp(input_image,
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


def rasterize(geom, proj4, image_src, image_name, open_bands):

    """
    Rasterizes a polygon geometry
    """

    # left, bottom, right, top = geom.bounds

    # Create a memory layer to rasterize from.
    datasource = ogr.GetDriverByName('Memory').CreateDataSource('wrk')
    sp_ref = osr.SpatialReference()
    sp_ref.ImportFromProj4(proj4)
    util.check_axis_order(sp_ref)

    # Transform the geometry
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(image_src.crs.to_wkt())
    util.check_axis_order(target_sr)
    transform = osr.CoordinateTransformation(sp_ref, target_sr)
    gdal_geom = ogr.CreateGeometryFromWkt(geom.to_wkt())
    gdal_geom.Transform(transform)

    # Get the transformation boundary
    left, right, bottom, top = gdal_geom.GetEnvelope()

    # Create the new layer
    lyr = datasource.CreateLayer('', geom_type=ogr.wkbPolygon, srs=target_sr)
    field_def = ogr.FieldDefn('Value', ogr.OFTInteger)
    lyr.CreateField(field_def)

    # Add a feature
    feature = ogr.Feature(lyr.GetLayerDefn())
    feature.SetGeometryDirectly(ogr.Geometry(wkt=str(gdal_geom)))
    feature.SetField('Value', 1)
    lyr.CreateFeature(feature)

    xcount = int(round((right - left) / image_src.res[0]))
    ycount = int(round((top - bottom) / image_src.res[0]))

    # Create a raster to rasterize into.
    try:
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    except:
        return None, None, None, None, None, None

    target_ds.SetGeoTransform([left, image_src.res[0], 0., top, 0., -image_src.res[0]])
    target_ds.SetProjection(target_sr.ExportToWkt())

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], lyr, options=['ATTRIBUTE=Value'])

    poly_array = np.uint8(target_ds.GetRasterBand(1).ReadAsArray())

    datasource = None
    target_ds = None

    bottom = top - (ycount * image_src.res[0])
    right = left + (xcount * image_src.res[0])

    if isinstance(image_name, str):

        # Reproject the feature
        # with rio.open(image_name, mode='r') as src:
        #
        #     # Get the transform and new shape of reprojecting the source image.
        #     transform, width, height = rio.warp.calculate_default_transform(src.crs,
        #                                                                     image_src.crs,  # destination crs
        #                                                                     src.width,
        #                                                                     src.height,
        #                                                                     left=src.bounds.left,
        #                                                                     bottom=src.bounds.bottom,
        #                                                                     right=src.bounds.right,
        #                                                                     top=src.bounds.top,
        #                                                                     resolution=image_src.res)
        #
        #     # Setup the output array
        #     if open_bands == -1:
        #
        #         bobj = rio.band(src, list(range(1, src.count+1)))
        #         dst_array = np.empty((src.count, height, width), dtype='float32')
        #
        #     else:
        #
        #         bobj = rio.band(src, open_bands)
        #
        #         if isinstance(open_bands, int):
        #             dst_array = np.empty((1, height, width), dtype='float32')
        #         else:
        #             dst_array = np.empty((len(open_bands), height, width), dtype='float32')
        #
        #     rio.warp.reproject(source=bobj,
        #                        destination=dst_array,
        #                        src_transform=src.transform,
        #                        src_crs=src.crs,
        #                        dst_transform=image_src.transform,
        #                        dst_crs=image_src.crs,
        #                        num_threads=1,
        #                        warp_mem_limit=256)

        out_ds = warp(image_name,
                      '',
                      format='MEM',
                      out_proj=image_src.crs.to_wkt(),
                      cell_size=image_src.res[0],
                      multithread=True,
                      outputBounds=[left, bottom, right, top],
                      warpMemoryLimit=256)

        if out_ds:

            if isinstance(open_bands, int):

                if open_bands == -1:
                    open_bands_range = list(range(1, out_ds.RasterCount+1))
                else:
                    open_bands_range = list(range(open_bands, open_bands+1))

            else:
                open_bands_range = open_bands

            image_array = np.array([out_ds.GetRasterBand(band_idx).ReadAsArray()
                                    for band_idx in open_bands_range], dtype='float32')

            out_ds = None

    else:

        if isinstance(image_src.data, xr.core.dataset.Dataset):

            image_array = image_src.data['bands'].sel(y=slice(top, bottom),
                                                      x=slice(left, right)).values

        else:

            image_array = image_src.data.sel(y=slice(top, bottom),
                                             x=slice(left, bottom)).values

    return poly_array, np.squeeze(image_array), left, top, right, bottom


def update_dict(didx, zones_dict, image_array, stats, band, no_data, image_bands):

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


def worker(didx, df_row, stats, n_stats, proj4, raster_value, no_data, verbose, n, band, open_bands, values):

    if verbose > 1:

        if didx % 100 == 0:
            logger.info('    Zone {:,d} of {:,d} ...'.format(didx+1, n))

    if isinstance(values, str):
        values_src = rio.open(values, mode='r')
    else:
        values_src = values

    if values_src.count == 1:
        data_values = np.concatenate(([didx], np.zeros(n_stats, dtype='float64') + np.nan))
    else:
        data_values = np.concatenate(([didx], np.zeros(n_stats * values_src.count, dtype='float64')))

    return_early = False

    geom = df_row.geometry

    if not geom:

        if isinstance(values, str):
            values_src.close()

        return data_values

    # Rasterize the data
    poly_array, image_array, left, top, right, bottom = rasterize(geom, proj4, values_src, values, open_bands)

    # Cases where multi-band images are flattened
    #   because of single-zone pixels
    if values_src.count > 1:
        if len(image_array.shape) == 1:
            image_array = image_array.reshape(values_src.count, 1, 1)

    if isinstance(raster_value, int):

        image_array = np.where(image_array == raster_value, 1, 0)

        if image_array.max() == 0:
            return_early = True

    elif isinstance(raster_value, list):

        image_array_ = np.zeros(image_array.shape, dtype='uint8')

        for raster_value_ in raster_value:
            image_array_ = np.where(image_array == raster_value_, 1, image_array_)

        image_array = image_array_

        if image_array.max() == 0:
            return_early = True

    if not return_early:

        if not isinstance(poly_array, np.ndarray):

            if isinstance(values, str):
                values_src.close()

            return data_values

        else:

            null_idx = np.where(poly_array == 0)

            if null_idx[0].shape[0] > 0:

                if values_src.count > 1:

                    for ix in range(0, values_src.count):
                        image_array[ix][null_idx] = no_data

                else:
                    image_array[null_idx] = no_data

            if any(['nan' in x for x in stats]):

                no_data_idx = np.where(image_array == no_data)

                if no_data_idx[0].shape[0] > 0:
                    image_array[no_data_idx] = np.nan

                if np.isnan(bn.nanmax(image_array)):

                    if isinstance(values, str):
                        values_src.close()

                    return data_values

        if values_src.count == 1:

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

    if isinstance(values, str):
        values_src.close()

    return data_values


def calc_parallel(stats, proj4, raster_value, no_data, verbose, n, zones_df, values, band, open_bands, n_jobs):

    n_stats = len(stats)

    results = Parallel(n_jobs=n_jobs,
                       max_nbytes=None)(delayed(worker)(didx,
                                                        dfrow,
                                                        stats,
                                                        n_stats,
                                                        proj4,
                                                        raster_value,
                                                        no_data,
                                                        verbose,
                                                        n,
                                                        band,
                                                        open_bands,
                                                        values)
                                        for didx, dfrow in zones_df.iterrows())

    if isinstance(band, int):
        n_bands = 1
    else:

        if isinstance(values, str):

            with rio.open(values) as src_tmp:
                n_bands = src_tmp.count

        else:
            n_bands = values.bands

    return merge_dictionary_keys(np.array(results, dtype='float64'), stats, n_bands)


class RasterStats(ZonesMixin):

    """
    Args:
        values (str or xarray): The raster values file. Can be float or categorical raster.
        zones (str or GeoDataFrame): The zones file.
            Accepted types are:
                str: vector file (e.g., shapefile, or geopackage)
                GeoDataFrame: the geometry type must be `Polygon`
        no_data (Optional[int or float]): A no data value to mask. Default is 0.
        raster_value (Optional[int or list]): A raster value to get statistics for. Default is None.
        unique_column (Optional[str]): A unique column identifier for `zones`.
            Default is None, which treats all zones as unique and uses the
            highest (resolution) level geometry.
        band (Optional[int]): The band to calculate (if multi-band). Default is None, or calculate all bands.
        column_prefix (Optional[str]): A name to prepend to each band (column) name.
        verbose (Optional[int]): The verbosity level. Default is 0.
        n_jobs (Optional[int]): The number of parallel processes (zones). Default is 1.
            *Currently, this only works with one statistic.

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
                 no_data=0,
                 raster_value=None,
                 unique_column=None,
                 band=None,
                 column_prefix=None,
                 verbose=0,
                 n_jobs=1):

        self.values = values
        self.zones = zones
        self.no_data = no_data
        self.raster_value = raster_value
        self.unique_column = unique_column
        self.band = band
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.column_prefix = column_prefix

        self.stats = None
        self.zone_values = None
        self.open_bands = self.band if isinstance(self.band, int) else -1

    def zone_iter(self, stats):

        proj4 = self._prepare_proj4()

        n = self.zones_df.shape[0]

        if self.verbose > 0:
            logger.info('  Calculating statistics ...')

        if self.n_jobs != 1:

            self.zone_values = calc_parallel(stats,
                                             proj4,
                                             self.raster_value,
                                             self.no_data,
                                             self.verbose,
                                             n,
                                             self.zones_df,
                                             self.values,
                                             self.band,
                                             self.open_bands,
                                             self.n_jobs)

        else:

            for didx, df_row in tqdm(self.zones_df.iterrows(), total=self.zones_df.shape[0]):

                geom = df_row.geometry

                if not geom:
                    continue

                # image_bounds = Polygon([(self.values_src.left, self.values_src.top),
                #                         (self.values_src.right, self.values_src.top),
                #                         (self.values_src.right, self.values_src.bottom),
                #                         (self.values_src.left, self.values_src.bottom)])
                #
                # # Check if the geometry is within the image bounds
                # if not geom.within(image_bounds):
                #     continue

                # Rasterize the data
                poly_array, image_array, left, top, right, bottom = rasterize(geom,
                                                                              proj4,
                                                                              self.values_src,
                                                                              self.values,
                                                                              self.open_bands)

                if isinstance(self.raster_value, int):

                    image_array = np.where(image_array == self.raster_value, 1, 0)

                    if image_array.max() == 0:
                        continue

                elif isinstance(self.raster_value, list):

                    image_array_ = np.zeros(image_array.shape, dtype='uint8')

                    for raster_value_ in self.raster_value:
                        image_array_ = np.where(image_array == raster_value_, 1, image_array_)

                    image_array = image_array_

                    if image_array.max() == 0:
                        continue

                if not isinstance(poly_array, np.ndarray):
                    image_array = np.array([0], dtype='float32')
                else:

                    null_idx = np.where(poly_array == 0)

                    if null_idx[0].shape[0] > 0:

                        if len(image_array.shape) > 2:

                            for ix in range(0, image_array.shape[0]):
                                image_array[ix][null_idx] = self.no_data

                        else:
                            image_array[null_idx] = self.no_data

                    if any(['nan' in x for x in stats]):

                        no_data_idx = np.where(image_array == self.no_data)

                        if no_data_idx[0].shape[0] > 0:
                            image_array[no_data_idx] = np.nan

                        if np.isnan(bn.nanmax(image_array)):
                            continue

                self.zone_values = update_dict(didx,
                                               self.zone_values,
                                               image_array,
                                               stats,
                                               self.band,
                                               self.no_data,
                                               self.values_src.count)
