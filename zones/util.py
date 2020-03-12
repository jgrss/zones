from osgeo import osr
import pyproj
from rasterio.crs import CRS


def check_crs(crs):

    """
    Checks a CRS instance

    Args:
        crs (``CRS`` | int | dict | str): The CRS instance.

    Returns:
        ``rasterio.crs.CRS``
    """

    if isinstance(crs, pyproj.crs.crs.CRS):
        dst_crs = CRS.from_proj4(crs.to_proj4())
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

    return dst_crs.to_proj4()


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
