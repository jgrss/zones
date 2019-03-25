from future.utils import viewitems

import os

from .errors import logger, ValuesFileError, StatsError, ZonesFileError
from .stats import STAT_DICT

import mpglue as gl

import pandas as pd
import geopandas as gpd


class ZonesMixin(object):

    @property
    def stats_avail(self):
        return list(STAT_DICT.keys())

    def calculate(self, stats):

        """
        Args:
            stats (str list)

        Returns:
            DataFrame
        """

        self.stats = stats

        if isinstance(self.stats, str):
            self.stats = [self.stats]

        self.check_arguments(self.stats)

        if self.verbose > 0:
            logger.info('  Preparing files ...')

        self.prepare_files(self.zones, self.values)

        if self.verbose > 0:
            logger.info('  Preparing zones ...')

        self.zone_values = self.prepare_zones(self.unique_column)

        if self.verbose > 0:
            logger.info('  Calculating stats ...')

        self.zone_iter(self.stats)

        if self.verbose > 0:
            logger.info('  Finalizing data ...')

        self._close_files()

        return self.finalize_dataframe()

    @staticmethod
    def check_if_geodf(data_file):

        """
        Checks for file data type

        Args:
            data_file

        Returns:
            data_file (GeoDataFrame)
        """

        if isinstance(data_file, gpd.GeoDataFrame):
            return data_file, None
        else:

            file_extension = os.path.splitext(os.path.split(data_file)[1])[1].lower().strip()

            if file_extension in ['.shp', '.gpkg']:
                return gpd.read_file(data_file), None
            elif file_extension == '.csv':
                return pd.read_csv(data_file), None
            else:
                return None, gl.ropen(data_file)

    def prepare_files(self, zones, values):

        """
        Prepares files

        Args:
            zones
            values
        """

        self.values_df = None
        self.values_src = None

        self.zones_df = self.check_if_geodf(zones)[0]
        self.values_df, self.values_src = self.check_if_geodf(values)

    def prepare_zones(self, unique_column):

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

    def finalize_dataframe(self):

        if hasattr(self, 'band'):

            if isinstance(self.band, int):

                values_df = pd.DataFrame.from_dict(self.zone_values, orient='index')
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

    def check_arguments(self, stats):

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
