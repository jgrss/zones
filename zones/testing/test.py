from .test_single_band import test_01_single
from .test_single_band_parallel import test_01_single_parallel
from .test_multi_band import test_01_multi
from ..errors import logger


def test_raster():

    logger.info('  Testing single band image ...')

    test_01_single()

    logger.info('  Testing single band image in parallel ...')

    test_01_single_parallel()

    logger.info('  Testing multi-band image ...')

    test_01_multi()
