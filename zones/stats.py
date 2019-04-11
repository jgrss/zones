import numpy as np
from scipy.stats import mode


STAT_DICT = dict(mean=np.mean,
                 nanmean=np.nanmean,
                 median=np.median,
                 nanmedian=np.nanmedian,
                 std=np.std,
                 nanstd=np.nanstd,
                 var=np.var,
                 nanvar=np.nanvar,
                 sum=np.sum,
                 mode=mode,
                 nanmode=mode,
                 nansum=np.nansum,
                 min=np.min,
                 nanmin=np.nanmin,
                 max=np.max,
                 nanmax=np.nanmax,
                 dist=None)
