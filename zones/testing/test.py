import logging

from .test_single_band import test_01_single
from .test_single_band_parallel import test_01_parallel
from .test_multi_band import test_01_multi
from .test_points import test_01_single_points
from ..errors import add_handler


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def test_raster():

    logger.info('  Testing single-band image ...')

    test_01_single()

    logger.info('  Testing single-band image in parallel ...')

    test_01_parallel()

    logger.info('  Testing multi-band image ...')

    test_01_multi()

    logger.info('  Testing single-band image with points ...')

    test_01_single_points()
