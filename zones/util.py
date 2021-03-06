import logging

from .errors import add_handler

from osgeo import osr
import pyproj
from rasterio.crs import CRS


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def check_crs(crs):

    """
    Checks a CRS instance

    Args:
        crs (``CRS`` | int | dict | str): The CRS instance.

    Returns:
        ``str`` as WKT
    """

    if isinstance(crs, pyproj.crs.crs.CRS) or isinstance(crs, pyproj.crs.CRS) or isinstance(crs, pyproj.CRS):

        if isinstance(crs.to_epsg(), int):
            dst_crs = CRS.from_epsg(crs.to_epsg())
        else:
            dst_crs = CRS.from_string(crs.srs)

    elif isinstance(crs, CRS):
        dst_crs = crs
    elif isinstance(crs, int):
        dst_crs = CRS.from_epsg(crs)
    elif isinstance(crs, dict):
        dst_crs = CRS.from_dict(crs)
    elif isinstance(crs, str):

        if crs.startswith('+proj'):
            dst_crs = CRS.from_proj4(crs)
        else:
            dst_crs = CRS.from_string(crs)

    else:
        logger.exception('  The CRS was not understood.')
        raise TypeError

    return dst_crs.to_wkt()


def check_axis_order(spatial_reference):

    """
    Checks for GDAL 2/3 spatial reference axis order

    Args:
        spatial_reference (object): The spatial reference object to check.

    Returns:
        None
    """

    if hasattr(spatial_reference, 'SetAxisMappingStrategy'):
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
