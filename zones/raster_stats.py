from __future__ import division

from .base import ZonesMixin
from .errors import logger
from .stats import STAT_DICT

from mpglue import raster_tools

import numpy as np
from osgeo import gdal, ogr, osr

from joblib import Parallel, delayed
from tqdm import tqdm

shapely.speedups.enable()


def rasterize(geom, proj4, image_src, image_name):

    """
    Rasterizes a polygon geometry
    """

    # left, bottom, right, top = geom.bounds

    # Create a memory layer to rasterize from.
    datasource = ogr.GetDriverByName('Memory').CreateDataSource('wrk')
    sp_ref = osr.SpatialReference()
    sp_ref.ImportFromProj4(proj4)

    # Transform the geometry
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(image_src.projection)
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

    xcount = int(round((right - left) / image_src.cellY))
    ycount = int(round((top - bottom) / image_src.cellY))

    # Create a raster to rasterize into.
    try:
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    except:
        return None, None

    target_ds.SetGeoTransform([left, image_src.cellY, 0., top, 0., -image_src.cellY])
    target_ds.SetProjection(target_sr.ExportToWkt())

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], lyr, options=['ATTRIBUTE=Value'])

    poly_array = np.uint8(target_ds.GetRasterBand(1).ReadAsArray())

    datasource = None
    target_ds = None

    #image_array = image_src.read(bands2open=-1,
                                #x=left,
                                #y=top,
                                #rows=ycount,
                                #cols=xcount,
                                #d_type='float32')

    src = raster_tools.warp(image_name,
                            'temp.mem',
                            format='MEM',
                            out_proj=image_src.projection,
                            multithread=True,
                            outputBounds=[left, top-(ycount*image_src.cellY), left+(xcount*image_src.cellY), top],
                            cell_size=image_src.cellY,
                            d_type='float32',
                            return_datasource=True,
                            warpMemoryLimit=256)

    image_array = src.read(bands2open=-1)

    src.close_file()
    src = None

    return poly_array, image_array


def worker(didx, df_row, stat, proj4, raster_value, no_data):

    return_early = False

    geom = df_row.geometry

    # Rasterize the data
    poly_array, image_array = rasterize(geom, proj4, values_src_g, values_df_g)

    if isinstance(raster_value, int):

        image_array = np.where(image_array == raster_value, 1, 0)

        if image_array.max() == 0:
            
            values = 0.0
            return_early = True

    elif isinstance(raster_value, list):

        image_array_ = np.zeros(image_array.shape, dtype='uint8')

        for raster_value_ in raster_value:
            image_array_ = np.where(image_array == raster_value_, 1, image_array_)

        image_array = image_array_

        if image_array.max() == 0:

            values = 0.0
            return_early = True

    if not return_early:

        if not isinstance(poly_array, np.ndarray):
            image_array = np.array([0], dtype='float32')
        else:

            null_idx = np.where(poly_array == 0)

            if null_idx[0].size > 0:
                image_array[null_idx] = no_data

            if any(['nan' in x for x in stat]):

                no_data_idx = np.where(image_array == no_data)

                if no_data_idx[0].size > 0:
                    image_array[no_data_idx] = np.nan

        stat_func = STAT_DICT[stat]

        values = stat_func(image_array)

    return didx, values


class RasterStats(ZonesMixin):

    """
    Args:
        values (str): The raster values file. Can be float or categorical raster.
        zones (str or GeoDataFrame): The zones file.
            Accepted types are:
                str: vector file (e.g., shapefile, or geopackage)
                GeoDataFrame: the geometry type must be `Polygon`
        unique_column (Optional[str]): A unique column identifier. Default is None.
        no_data (Optional[int or float]): A no data value to mask. Default is 0.
        raster_value (Optional[int or list]): A raster value to get statistics for. Default is None.
        band (Optional[int]): The band to calculate (if multi-band). Default is None, or calculate all bands.
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
                 unique_column=None,
                 no_data=0,
                 raster_value=None,
                 band=None,
                 verbose=0,
                 n_jobs=1):

        self.values = values
        self.zones = zones
        self.unique_column = unique_column
        self.no_data = no_data
        self.raster_value = raster_value
        self.band = band
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.stats = None
        self.zone_values = None

    def zone_iter(self, stats):

        global values_df_g, values_src_g

        values_df_g = None
        values_src_g = None

        proj4 = self._prepare_proj4()

        n = self.zones_df.shape[0]

        if (self.n_jobs != 1) and (len(self.stats) == 1):

            values_df_g = self.values
            values_src_g = self.values_src

            results = Parallel(n_jobs=self.n_jobs)(delayed(worker)(didx,
                                                                   dfrow,
                                                                   stats[0],
                                                                   proj4,
                                                                   self.raster_value,
                                                                   self.no_data)
                                                   for didx, dfrow in self.zones_df.iterrows())

            self.zone_values = dict(results)

        else:

            for didx, df_row in tqdm(self.zones_df.iterrows(), leave=False):

                if self.verbose > 1:
                    logger.info('    Zone {:,d} of {:,d} ...'.format(didx+1, n))

                geom = df_row.geometry

                # Rasterize the data
                poly_array, image_array = rasterize(geom, proj4, self.values_src, self.values)

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

                    if null_idx[0].size > 0:
                        image_array[null_idx] = self.no_data

                    if any(['nan' in x for x in stats]):

                        no_data_idx = np.where(image_array == self.no_data)

                        if no_data_idx[0].size > 0:
                            image_array[no_data_idx] = np.nan

                if len(image_array.shape) == 2:

                    for sidx, stat in enumerate(stats):

                        stat_func = STAT_DICT[stat]

                        # TODO: if zones are not unique
                        self.zone_values[1][didx][sidx] = stat_func(image_array)

                else:

                    if isinstance(self.band, int):

                        for sidx, stat in enumerate(stats):

                            stat_func = STAT_DICT[stat]

                            # TODO: if zones are not unique
                            self.zone_values[self.band][didx][sidx] = stat_func(image_array[self.band-1])

                    else:

                        for bidx in range(1, self.values_src.bands+1):

                            for sidx, stat in enumerate(stats):

                                stat_func = STAT_DICT[stat]

                                # TODO: if zones are not unique
                                self.zone_values[bidx][didx][sidx] = stat_func(image_array[bidx-1])
