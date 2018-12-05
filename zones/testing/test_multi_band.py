# import unittest

from ..errors import logger
from ..datasets import load_01_multi
from .. import RasterStats

import numpy as np


def test_01_multi():

    feat01 = np.array([318, 385, 429, 346, 462, 492, 484, 495, 517, 498, 533, 498], dtype='int64')
    feat02 = np.array([419, 434, 334, 528, 392, 387, 439, 392, 489, 474, 451, 404], dtype='int64')
    feat03 = np.array([398, 283, 349, 436, 346, 493, 398, 398, 377, 472, 461, 453], dtype='int64')

    raster, vector = load_01_multi()
    zs = RasterStats(raster, vector, verbose=0)
    df = zs.calculate(['sum', 'mean'])

    assert int(df.loc[0, 'sum_bd1']) == feat01.sum()
    assert int(df.loc[1, 'sum_bd1']) == feat02.sum()
    assert int(df.loc[2, 'sum_bd1']) == feat03.sum()
    assert round(df.loc[0, 'mean_bd1'], 2) == round(feat01.mean(), 2)
    assert round(df.loc[1, 'mean_bd1'], 2) == round(feat02.mean(), 2)
    assert round(df.loc[2, 'mean_bd1'], 2) == round(feat03.mean(), 2)

    logger.info('  If there were no assertion errors, the tests ran OK.')


# class TestUM(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#     def test_isarray_gtiff(self):
#
#         feat01 = np.array([318, 385, 429, 346, 462, 492, 484, 495, 517, 498, 533, 498], dtype='int64')
#
#         raster, vector = load_01()
#         zs = RasterStats(raster, vector, verbose=2)
#         df = zs.calculate(['sum'])
#
#         self.assertEquals(int(df.iloc[0, 'sum']), feat01.sum())


# def test_01():
#     unittest.main()
