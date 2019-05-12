import numpy as np
import bottleneck as bn
from scipy.stats import mode


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
                 dist=None)
