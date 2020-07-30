class ValuesFileError(NameError):
    """Raised when the values file does not exist"""


class ZonesFileError(NameError):
    """Raised when the zones file does not exist"""


class StatsError(NameError):
    """Raised when the requested statistic does not exist"""
