from future.utils import viewitems

import os

from .errors import logger, ValuesFileError, StatsError, ZonesFileError
from .stats import STAT_DICT

import mpglue as gl

import pandas as pd
import geopandas as gpd


class ZonesBase(object):

    def calculate(self, stats):

        """
        Args:
            stats (str list)

        Returns:
            DataFrame
        """

        self._check_arguments(stats)

        self.stats = stats

        if self.verbose > 0:
            logger.info('  Preparing files ...')

        self._prepare_files(self.zones, self.values)

        if self.verbose > 0:
            logger.info('  Preparing zones ...')

        self.zone_values = self._prepare_zones(self.unique_column)

        if self.verbose > 0:
            logger.info('  Calculating stats ...')

        self._iter(self.stats)

        if self.verbose > 0:
            logger.info('  Finalizing data ...')

        self._close_files()

        return self._finalize_dataframe()

    def _prepare_files(self, zones, values):

        self.values_df = None
        self.values_src = None

        self.zones_df = gpd.read_file(zones)

        if self.values.lower().endswith('.shp'):
            self.values_df = gpd.read_file(values)
        elif self.values.lower().endswith('.csv'):
            self.values_df = pd.read_csv(values)
        else:
            self.values_src = gl.ropen(values)

    def _prepare_zones(self, unique_column):

        if self.values_src:
            self.n_bands = self.values_src.bands
        else:
            self.n_bands = 0

        # TODO
        if isinstance(unique_column, str):
            return None
        else:

            zone_values = dict()

            if self.n_bands > 0:

                for bidx in range(1, self.n_bands+1):

                    zone_values[bidx] = dict()

                    for idx, dfr in self.zones_df.iterrows():
                        zone_values[bidx][idx] = [0.0] * len(self.stats)

            else:

                for idx, dfr in self.zones_df.iterrows():
                    zone_values[idx] = [0.0] * len(self.stats)

            return zone_values

    def _finalize_dataframe(self):

        if hasattr(self, 'band'):

            if isinstance(self.band, int):

                values_df = pd.DataFrame.from_dict(self.zone_values[self.band], orient='index')
                values_df.columns = ('_bd{:d},'.format(self.band).join(self.stats) + '_bd{:d}'.format(self.band)).split(',')

            else:

                for bidx in range(1, self.n_bands+1):

                    values_df_ = pd.DataFrame.from_dict(self.zone_values[bidx], orient='index')
                    values_df_.columns = ('_bd{:d},'.format(bidx).join(self.stats) + '_bd{:d}'.format(bidx)).split(',')

                    if bidx == 1:
                        values_df = values_df_.copy()
                    else:
                        values_df = pd.concat((values_df, values_df_), axis=1)

        else:

            values_df = pd.DataFrame.from_dict(self.zone_values, orient='index')
            values_df.columns = self.stats

        return pd.merge(self.zones_df, values_df, left_index=True, right_index=True)

    def _close_files(self):

        if self.values_src:

            self.values_src.close()
            self.values_src = None

    def _prepare_proj4(self):

        proj4 = ''

        for k, v in viewitems(self.zones_df.crs):
            proj4 += '+{}={} '.format(k, v)

        return proj4[:-1]

    def _check_arguments(self, stats):

        """
        Args:
            stats (list)
        """

        if not os.path.isfile(self.values):

            logger.error('  The values file does not exist.')
            raise ValuesFileError

        if not os.path.isfile(self.zones):

            logger.error('  The zones file does not exist.')
            raise ZonesFileError

        if list(set(stats).difference(STAT_DICT.keys())):

            logger.error('  The statistic, {}, is not available.'.format(list(set(stats).difference(STAT_DICT.keys()))))
            raise StatsError
