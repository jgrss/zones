import os
import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

_log_home = os.path.abspath(os.path.dirname(__file__))

if not os.access(_log_home, os.W_OK):
    _log_home = os.path.expanduser('~')

_log_file = os.path.join(_log_home, 'zones.log')

logging.basicConfig(filename=_log_file,
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
