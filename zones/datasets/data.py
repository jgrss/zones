import os


def load_01_single():

    file_path = os.path.dirname(__file__)

    raster = os.path.join(file_path, 'raster', '01_single_band_utm.tif')
    vector = os.path.join(file_path, 'vector', '01_vector_wgs84.shp')

    return raster, vector


def load_01_multi():

    file_path = os.path.dirname(__file__)

    raster = os.path.join(file_path, 'raster', '01_multi_band_utm.tif')
    vector = os.path.join(file_path, 'vector', '01_vector_wgs84.shp')

    return raster, vector
