from .base import ZonesBase
from .errors import logger
from .stats import STAT_DICT

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

shapely.speedups.enable()


class PointStats(ZonesBase):

    """
    Args:
        values (str): The points values file. It can be a .csv or point shapefile.
        zones (str): The zones file. It should be a polygon vector file.
        unique_column (Optional[str]): A unique column identifier. Default is None.

    Examples:
        >>> import zones
        >>> zs = zones.PointStats('values.csv', 'zones.shp')
        >>> df = zs.calculate(['mean'])
        >>> df = zs.calculate(['nanmean', 'nansum'])
        >>> df.to_file('stats.shp')
        >>> df.to_csv('stats.csv')
    """

    def __init__(self, values, zones, unique_column=None, x_column='X', y_column='Y', verbose=0):

        self.values = values
        self.zones = zones
        self.unique_column = unique_column
        self.x_column = x_column
        self.y_column = y_column
        self.verbose = verbose

        self.stats = None
        self.zone_values = None

    @staticmethod
    def pd2gpd(point_file,
               x_column,
               y_column,
               zone_df):

        """
        Converts a Pandas DataFrame to a GeoPandas DataFrame

        Args:
            point_file (str)
            x_column (int or str)
            y_column (int or str)
            zone_df (DataFrame)
        """

        point_df = pd.read_csv(point_file)

        # Create a Point column
        point_geometry = [shapely.geometry.Point(xy) for xy in zip(point_df[x_column], point_df[y_column])]

        point_df.drop(columns=[x_column,
                               y_column],
                      axis=1,
                      inplace=True)

        return gpd.GeoDataFrame(point_df,
                                crs=zone_df.crs,
                                geometry=point_geometry)

    def _load(self):

        """
        Loads the files into Geo DataFrames
        """

        zone_df = gpd.read_file(self.zone_file)

        # Merge zones
        g = zone_df.groupby(self.zone_column)

        self.zone_df = gpd.GeoDataFrame(g.mean().reset_index(),
                                        crs=zone_df.crs,
                                        geometry=zone_df.geometry)

        if self.point_file.lower().endswith('.csv'):

            self.point_df = self.pd2gpd(self.point_file,
                                        self.x_column,
                                        self.y_column,
                                        self.zone_df)

        else:
            self.point_df = gpd.read_file(self.point_file)

        self.point_index = self.point_df.sindex

    def _iter(self, stats):

        n = self.zones_df.shape[0]

        for didx, df_row in self.zones_df.iterrows():

            if self.verbose > 1:
                logger.info('    Zone {:,d} of {:,d} ...'.format(didx + 1, n))

            geom = df_row.geometry

            # Get points that intersect the
            #   (square) bounds of the zone.
            int_points = sorted(list(self.point_index.intersection(geom.bounds)))

            if int_points:

                # Get a subset of the DataFrame.
                point_df = self.values_df.iloc[int_points]

                # Take points within the zone
                point_list = [point_idx for point_idx, point_row in point_df.iterrows()
                              if geom.contains(point_row.geometry)]

                # Get the real subset of points.
                point_df = self.values_df.iloc[point_list]

                for sidx, stat in enumerate(stats):

                    stat_func = STAT_DICT[stat]

                    # TODO: if zones are not unique
                    # self.zone_values[didx][sidx] = stat_func(image_array)

                    # TODO
                    # Calculate statistics.
                    zone_values = point_df[value_columns].apply(stat_func).values

                if not isinstance(stat_array, np.ndarray):
                    stat_array = zone_values.copy()
                else:
                    stat_array = np.vstack((stat_array, zone_values))

                index_values.append(df_idx)

        self.stat_df.loc[index_values, value_columns] = stat_array
