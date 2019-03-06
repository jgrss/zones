from __future__ import division

from .base import ZonesBase
from .errors import logger
from .stats import STAT_DICT

from mpglue import raster_tools

import numpy as np
from osgeo import gdal, ogr, osr

from tqdm import tqdm


class RasterStats(ZonesBase):

    """
    Args:
        values (str): The raster values file. Can be float or categorical raster.
        zones (str): The zones file. It should be a polygon vector file.
        unique_column (Optional[str]): A unique column identifier. Default is None.
        no_data (Optional[int or float]): A no data value to mask. Default is 0.
        band (Optional[int]): The band to calculate (if multi-band). Default is None, or calculate all bands.
        verbose (Optional[int]): The verbosity level. Default is 0.

    Examples:
        >>> import zones
        >>> zs = zones.RasterStats('values.tif', 'zones.shp')
        >>> df = zs.calculate(['mean'])
        >>> df = zs.calculate(['nanmean', 'nansum'])
        >>> df.to_file('stats.shp')
        >>> df.to_csv('stats.csv')
    """

    def __init__(self, values, zones, unique_column=None, no_data=0, band=None, verbose=0):

        self.values = values
        self.zones = zones
        self.unique_column = unique_column
        self.no_data = no_data
        self.band = band
        self.verbose = verbose

        self.stats = None
        self.zone_values = None

    def _iter(self, stats):

        proj4 = self._prepare_proj4()

        n = self.zones_df.shape[0]

        for didx, df_row in tqdm(self.zones_df.iterrows(), leave=False):

            if self.verbose > 1:
                logger.info('    Zone {:,d} of {:,d} ...'.format(didx+1, n))

            geom = df_row.geometry

            # Rasterize the data
            poly_array, image_array = self._rasterize(geom, proj4, self.values_src, self.values)

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

    @staticmethod
    def _rasterize(geom, proj4, image_src, image_name):

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
