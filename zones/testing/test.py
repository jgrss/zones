import logging

from .test_single_band import test_01_single
from .test_single_band_parallel import test_01_parallel
from .test_multi_band import test_01_multi


logger = logging.getLogger(__name__)


def test_raster():

    logger.info('  Testing single band image ...')

    test_01_single()

    logger.info('  Testing single band image in parallel ...')

    test_01_parallel()

    logger.info('  Testing multi-band image ...')

    test_01_multi()
