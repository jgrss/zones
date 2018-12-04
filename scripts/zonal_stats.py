import argparse

import zones

#import numpy as np
#import pandas as pd
#import geopandas as gpd
#import shapely

#shapely.speedups.enable()


def main():

    parser = argparse.ArgumentParser(description='Zonal stats on raster data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--values-file', dest='values_file', help='The values file', default=None)
    parser.add_argument('--zones-file', dest='zones_file', help='The zones file', default=None)
    parser.add_argument('--stats-file', dest='stats_file', help='The stats file', default=None)
    parser.add_argument('--stats', dest='stats', help='The stats', default=None, nargs='+')

    args = parser.parse_args()

    zs = zones.RasterStats(args.values_file, args.zones_file, verbose=2)
    df = zs.calculate(args.stats)
    df.to_file(args.stats_file)


if __name__ == '__main__':
    main()


#class ZonalStats(object):

    #"""
    #Args:
        #point_file (str): A vector file or a .csv with x, y columns.
        #zone_file (str): The zone vector file.
        #zone_column (str or int): The unique zone id column name.
        #x_column (Optional[str])
        #y_column (Optional[str])
    #"""

    #def __init__(self,
                 #point_file,
                 #zone_file,
                 #zone_column,
                 #x_column='X',
                 #y_column='Y'):

        #self.point_file = point_file
        #self.zone_file = zone_file
        #self.zone_column = zone_column
        #self.x_column = x_column
        #self.y_column = y_column

        #self.zone_id = None
        #self.point_df = None
        #self.zone_df = None
        #self.point_index = None
        #self.stat_df = None

        #self._load()

    #@staticmethod
    #def pd2gpd(point_file,
               #x_column,
               #y_column,
               #zone_df):

        #"""
        #Converts a Pandas DataFrame to a GeoPandas DataFrame

        #Args:
            #point_file (str)
            #x_column (int or str)
            #y_column (int or str)
            #zone_df (DataFrame)
        #"""

        #point_df = pd.read_csv(point_file)

        ## Create a Point column
        #point_geometry = [shapely.geometry.Point(xy) for xy in zip(point_df[x_column], point_df[y_column])]

        #point_df.drop(columns=[x_column,
                               #y_column],
                      #axis=1,
                      #inplace=True)

        #return gpd.GeoDataFrame(point_df,
                                #crs=zone_df.crs,
                                #geometry=point_geometry)

    #def _load(self):

        #"""
        #Loads the files into Geo DataFrames
        #"""

        #zone_df = gpd.read_file(self.zone_file)

        ## Merge zones
        #g = zone_df.groupby(self.zone_column)

        #self.zone_df = gpd.GeoDataFrame(g.mean().reset_index(),
                                        #crs=zone_df.crs,
                                        #geometry=zone_df.geometry)

        #if self.point_file.lower().endswith('.csv'):

            #self.point_df = self.pd2gpd(self.point_file,
                                        #self.x_column,
                                        #self.y_column,
                                        #self.zone_df)

        #else:
            #self.point_df = gpd.read_file(self.point_file)

        #self.point_index = self.point_df.sindex

    #def calculate(self,
                  #value_columns,
                  #statistics=None):

        #"""
        #Calculates zonal statistics

        #Args:
            #value_columns (list): The list of point value column names.
            #statistics (Optional[str list]): The statistics to calculate. Default is ['mean'].

        #Attributes:
            #stat_df (DataFrame)
        #"""

        #if not statistics:
            #statistics = ['mean']

        #stat_funcs = [STAT_DICT[stat] for stat in statistics]

        #self.stat_df = pd.DataFrame(index=self.zone_df.index,
                                    #columns=[self.zone_column] + value_columns)

        #self.stat_df[self.zone_column] = self.zone_df[self.zone_column]

        #index_values = list()

        #stat_array = None

        ## Iterate over each zone.
        #for df_idx, df_row in self.zone_df.iterrows():

            ## Get the current zone geometry.
            #zone_geometry = df_row.geometry

            ## Get points that intersect the
            ##   (square) bounds of the zone.
            #int_points = sorted(list(self.point_index.intersection(zone_geometry.bounds)))

            #if int_points:

                ## Get a subset of the DataFrame.
                #point_df = self.point_df.iloc[int_points]

                #point_list = list()

                ## Check which points are within the zone.
                #for point_idx, point_row in point_df.iterrows():

                    #if zone_geometry.contains(point_row.geometry):
                        #point_list.append(point_idx)

                ## Get the real subset of points.
                #point_df = self.point_df.iloc[point_list]

                ## Calculate statistics.
                #zone_values = point_df[value_columns].apply(stat_funcs).values

                #if not isinstance(stat_array, np.ndarray):
                    #stat_array = zone_values.copy()
                #else:
                    #stat_array = np.vstack((stat_array, zone_values))

                #index_values.append(df_idx)

        #self.stat_df.loc[index_values, value_columns] = stat_array
