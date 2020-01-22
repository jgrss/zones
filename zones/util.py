from osgeo import osr


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
