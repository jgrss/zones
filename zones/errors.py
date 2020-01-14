import os
import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

logging.basicConfig(filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zones.log'),
                    filemode='w',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class ValuesFileError(NameError):
    """Raised when the values file does not exist"""


class ZonesFileError(NameError):
    """Raised when the zones file does not exist"""


class StatsError(NameError):
    """Raised when the requested statistic does not exist"""
