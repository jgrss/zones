import argparse

import zones


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
