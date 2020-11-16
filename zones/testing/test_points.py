import logging

from ..datasets import load_01_single_points
from .. import RasterStats
from ..errors import add_handler

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def test_01_single_points():

    feat = np.array([318, 385, 429, 419, 434, 334, 398, 283, 349,
                     346, 462, 492, 528, 392, 387, 436, 346, 493,
                     484, 495, 517, 439, 392, 489, 398, 398, 377,
                     498, 533, 498, 474, 451, 404, 472, 461, 453], dtype='int64')

    logger.info('  Single-band tests:')

    raster, vector = load_01_single_points()

    with rio.open(raster) as src:

        df_points = gpd.read_file(vector).to_crs(src.crs)

        df_points = pd.merge(df_points.drop(columns='geometry'),
                             df_points.buffer(abs(src.res[0])/2.0)\
                             .to_frame()\
                             .rename(columns={0: 'geometry'}),
                             left_index=True,
                             right_index=True)

    df = RasterStats(raster, df_points, verbose=0)\
                .calculate('max')

    assert np.allclose(df.max_bd1.values, feat)
    logger.info('  Band 1 values with 1 CPU OK')

    df = RasterStats(raster, df_points, verbose=0, n_jobs=2)\
                .calculate('max')

    assert np.allclose(df.max_bd1.values, feat)
    logger.info('  Band 1 values with 2 CPUs OK')

    logger.info('  If there were no assertion errors, the point tests ran OK.')
