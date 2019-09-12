from .test_single_band import test_01_single
from .test_single_band_parallel import test_01_parallel
from .test_multi_band import test_01_multi
from ..errors import logger


def test_raster():

    """
import zones
from zones.datasets import load_01_single, load_01_multi
raster, vector = load_01_single()
zs = zones.RasterStats(raster, vector, n_jobs=2, verbose=2)
df = zs.calculate('sum')
    """

    logger.info('  Testing single band image ...')

    test_01_single()

    logger.info('  Testing single band image in parallel ...')

    test_01_parallel()

    logger.info('  Testing multi-band image ...')

    test_01_multi()
