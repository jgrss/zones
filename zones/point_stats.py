from .base import ZonesBase
from .errors import logger
from .stats import STAT_DICT

import pandas as pd
import geopandas as gpd
import shapely

from tqdm import tqdm

shapely.speedups.enable()


class PointStats(ZonesBase):

    """
    Args:
        values (str): The points values file. It can be a .csv or point shapefile.
        zones (str): The zones file. It should be a polygon vector file.
        value_column (str): The column in `values` to calculate statistics from.
        query (Optional[str]): A query expression to subset the values DataFrame by. Default is None.
        unique_column (Optional[str]): A unique column identifier. Default is None.
        x_column (str): TODO
        y_column (str): TODO
        point_proj (Optional[str]): The point projection string. Default is None.
        verbose (Optional[int]): The verbosity level. Default is 0.

    Examples:
        >>> import zones
        >>>
        >>> zs = zones.PointStats('values.csv', 'zones.shp', 'field_name')
        >>> df = zs.calculate(['mean'])
        >>> df = zs.calculate(['nanmean', 'nansum'])
        >>> df.to_file('stats.shp')
        >>> df.to_csv('stats.csv')
        >>>
        >>> # Calculate the point mean where DN is equal to 1.
        >>> zs = zones.PointStats('values.csv', 'zones.shp', 'field_name', query="DN == 1")
        >>> df = zs.calculate(['mean'])
    """

    def __init__(self,
                 values,
                 zones,
                 value_column,
                 query=None,
                 unique_column=None,
                 x_column='X',
                 y_column='Y',
                 point_proj=None,
                 verbose=0):

        self.values = values
        self.zones = zones
        self.value_column = value_column
        self.query = query
        self.unique_column = unique_column
        self.x_column = x_column
        self.y_column = y_column
        self.point_proj = point_proj
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

    def _prepare_values(self):

        if isinstance(self.query, str):
            self.values_df = self.values_df.query(self.query)

        if isinstance(self.point_proj, str):
            self.values_df.crs = self.point_proj

        if self.verbose > 0:
            logger.info('  Transforming value DataFrame CRS ...')

        self.values_df = self.values_df.to_crs(self.zones_df.crs)

        if self.verbose > 0:
            logger.info('  Setting up spatial index ...')

        self.point_index = self.values_df.sindex

    def _iter(self, stats):

        self._prepare_values()

        n = self.zones_df.shape[0]

        for didx, df_row in tqdm(self.zones_df.iterrows(), leave=False):

            # if self.verbose > 1:
            #     logger.info('    Zone {:,d} of {:,d} ...'.format(didx + 1, n))

            geom = df_row.geometry

            # Get points that intersect the
            #   (square) bounds of the zone.
            int_points = sorted(list(self.point_index.intersection(geom.bounds)))

            if int_points:

                # Get a subset of the DataFrame.
                point_df = self.values_df.iloc[int_points]

                if point_df.iloc[0].geometry.type.lower() == 'polygon':

                    point_df['geometry'] = point_df.geometry.centroid

                    # Take points within the zone
                    point_list = [point_idx for point_idx, point_row in point_df.iterrows()
                                  if geom.contains(point_row.geometry)]

                    # Get the real subset of points.
                    point_df = self.values_df.loc[point_list]

                for sidx, stat in enumerate(stats):

                    if stat == 'dist':

                        self.zone_values[didx] = ';'.join(list(map('{:.4f}'.format,
                                                                   point_df[self.value_column].values.tolist())))

                    else:

                        stat_func = STAT_DICT[stat]

                        self.zone_values[didx][sidx] = stat_func(point_df[self.value_column].values)

            else:

                for sidx, stat in enumerate(stats):

                    if stat == 'dist':
                        self.zone_values[didx] = ''
