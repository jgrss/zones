from __future__ import division

from .base import ZonesBase
from .errors import logger
from .stats import STAT_DICT

from mpglue import raster_tools, vector_tools

import numpy as np
from osgeo import gdal, ogr, osr


class RasterStats(ZonesBase):

    """
    Args:
        values (str): The values file. Can be float or categorical raster, or points.
        zones (str): The zones file. It should be a polygon vector file.
        unique_column (Optional[str]): A unique column identifier. Default is None.

    Examples:
        >>> import zones
        >>> zs = zones.RasterStats('values.tif', 'zones.shp')
        >>> df = zs.calculate(['mean'])
        >>> df = zs.calculate(['nanmean', 'nansum'])
        >>> df.to_file('stats.shp')
        >>> df.to_csv('stats.csv')
    """

    def __init__(self, values, zones, unique_column=None, verbose=0):

        self.values = values
        self.zones = zones
        self.unique_column = unique_column
        self.verbose = verbose

        self.stats = None
        self.stat_func = None
        self.zone_values = None

    def calculate(self, stats):

        """
        Args:
            stats (str list)

        Returns:
            DataFrame
        """

        self._check_arguments(stats)

        self.stats = stats

        if self.verbose > 0:
            logger.info('  Preparing files ...')

        self._prepare_files(self.zones, self.values)

        if self.verbose > 0:
            logger.info('  Preparing zones ...')

        self.zone_values = self._prepare_zones(self.unique_column)

        if self.verbose > 0:
            logger.info('  Calculating stats ...')

        self._iter(self.stats)

        if self.verbose > 0:
            logger.info('  Finalizing data ...')

        self._close_files()

        return self._finalize_dataframe()

    def _iter(self, stats):

        proj4 = self._prepare_proj4()

        n = self.zones_df.shape[0]

        for didx, df_row in self.zones_df.iterrows():

            if self.verbose > 1:
                logger.info('    Zone {:,d} of {:,d} ...'.format(didx+1, n))

            geom = df_row.geometry

            # Rasterize the data
            poly_array, image_array = self._rasterize(geom, proj4, self.values_src, self.values)

            # TODO: multi-band images
            # TODO: return empty stat
            #if poly_array.shape != image_array.shape:
                #continue

            null_idx = np.where(poly_array == 0)

            if null_idx[0].size > 0:
                image_array[null_idx] = 0

            if any(['nan' in x for x in stats]):

                no_data_idx = np.where(image_array == 0)

                if no_data_idx[0].size > 0:
                    image_array[no_data_idx] = np.nan

            for sidx, stat in enumerate(stats):

                self.stat_func = STAT_DICT[stat]

                # TODO: if zones are not unique
                self.zone_values[didx][sidx] = self.stat_func(image_array)

    @staticmethod
    def _rasterize(geom, proj4, image_src, image_name):

        """
        Rasterizes a polygon geometry
        """

        left, bottom, right, top = geom.bounds

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
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)

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
