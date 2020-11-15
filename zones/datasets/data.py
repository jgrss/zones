import os
from pathlib import Path


def load_01_single():

    file_path = Path(os.path.dirname(__file__))

    raster = file_path / 'raster' / '01_single_band_utm.tif'
    vector = file_path / 'vector' / '01_vector_wgs84.gpkg'

    return raster, vector


def load_01_multi():

    file_path = Path(os.path.dirname(__file__))

    raster = file_path / 'raster' / '01_multi_band_utm.tif'
    vector = file_path / 'vector' / '01_vector_wgs84.gpkg'

    return raster, vector
