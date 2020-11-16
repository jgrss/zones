import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler.setFormatter(_formatter)


def add_handler(logger):

    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

    return logger


class ValuesFileError(NameError):
    """Raised when the values file does not exist"""


class ZonesFileError(NameError):
    """Raised when the zones file does not exist"""


class StatsError(NameError):
    """Raised when the requested statistic does not exist"""
