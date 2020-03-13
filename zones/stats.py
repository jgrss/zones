import numpy as np
import bottleneck as bn
from scipy.stats import mode


def _n_unique(array):
    return np.unique(array).shape[0]


def _count(array, nodata):
    return (array != nodata).sum()


def _crosstab(array_a, array_b):

    frequencies = {}

    unique_values = np.unique(array_a).tolist() + np.unique(array_b).tolist()

    for a in unique_values:
        for b in unique_values:
            frequencies[f'{int(a)}-{int(b)}'] = ((array_a == a) & (array_b == b)).sum()

    return frequencies


STAT_DICT = dict(mean=np.mean,
                 nanmean=bn.nanmean,
                 median=np.median,
                 nanmedian=bn.nanmedian,
                 std=np.std,
                 nanstd=bn.nanstd,
                 var=np.var,
                 nanvar=bn.nanvar,
                 sum=np.sum,
                 mode=mode,
                 nanmode=mode,
                 nansum=bn.nansum,
                 min=np.min,
                 nanmin=bn.nanmin,
                 max=np.max,
                 nanmax=bn.nanmax,
                 dist=None,
                 unique=_n_unique,
                 count=_count,
                 crosstab=_crosstab)
