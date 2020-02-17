[](#mit-license)[](#python-3.6)[](#package-version)

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.x-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-0.2.10-blue.svg?cacheSeconds=2592000)

Zonal statistics on raster data
---

The `zones` library calculates summary statistics with vector and raster data. The library manages projections on-the-fly,
so there is no need to ensure consistency prior to running. Statistics are processed per zone, so memory requirements
scale with the size of the vector polygons.

### Dependencies

- Cython
- NumPy
- GDAL

The following dependencies are installed automatically.

- Bottleneck
- EarthPy
- GeoPandas
- joblib
- Pandas
- Rasterio
- Shapely
- six
- tqdm
- Xarray

### Installation

Clone and build

```commandline
> git clone https://github.com/jgrss/zones.git
> cd zones/
> python setup.py build && python setup.py install 
>
> # Cleanup
> cd ../
> rm -rf zones/
```

or install directly from GitHub

```commandline
> pip install git+https://github.com/jgrss/zones
```

#### Test the installation

```commandline
> python -c "import zones;zones.test_raster()"
```

### Zonal stats with polygon and raster data

```python
>>> import zones
>>>
>>> zs = zones.RasterStats('values.tif', 'zones.shp', verbose=2)
>>>
>>> # One statistic
>>> df = zs.calculate('mean')
>>>
>>> # Multiple statistics
>>> df = zs.calculate(['nanmean', 'nansum'])
>>>
>>> # Save data to file
>>> df.to_file('stats.shp')
>>> df.to_csv('stats.csv')
```

For multi-band images, the default is to calculate all bands, but the raster band can be specified.

```python
# Calculate statistics for band 2
>>> zs = zones.RasterStats('values.tif', 'zones.shp', band=2)
>>> df = zs.calculate('var')
```

The default 'no data' value is 0, but it can be specified. Note that 'no data' values are only ignored if 'nanstats' are used.

```python
# Calculate statistics for band 3, ignoring values of 255
>>> zs = zones.RasterStats('values.tif', 'zones.shp', band=3, no_data=255)
>>> df = zs.calculate('nanmedian')
```

#### The zone data can also be a `GeoDataFrame` or any other vector format supported by `GeoPandas`.

```python
>>> import geopandas as gpd
>>>
>>> gdf = gpd.read_file('data.shp')
>>>
>>> zs = zones.RasterStats('values.tif', gdf)
>>> zs = zones.RasterStats('values.tif', 'zones.gpkg')
```

### Zonal stats with polygon and vector point data

```python
>>> import zones
>>>
>>> zs = zones.PointStats('points.shp', 'zones.shp', 'field_name')
>>>
>>> df = zs.calculate(['nanmean', 'nansum'])
>>>
>>> # Save data to file
>>> df.to_file('stats.shp')
>>> df.to_csv('stats.csv')
>>>
>>> # Calculate the point mean where DN is equal to 1.
>>> zs = zones.PointStats('points.shp', 'zones.shp', 'field_name', query="DN == 1")
>>> df = zs.calculate('mean')
```

### Parallel processing

#### Zones can be processed in parallel.

> Currently, only the `mean` and `sum` statistics are supported when `n_jobs` is not equal to 1.

```python
>>> # Process zones in parallel, using 8 CPUs.
>>> zs = zones.RasterStats('values.tif', 'zones.shp', n_jobs=8, no_data=255, band=1)
>>> zs.calculate('mean')
```

### Other methods

```python
>>> # Get available stats
>>> print(zs.stats_avail)
```

```python
>>> # To store the data as a distribution, use 'dist'.
>>> df = zs.calculate('dist')
>>>
>>> # Melt the data into columns
>>> df = zs.melt_dist(df)
```

### Testing

```python
>>> import zones
>>> zones.test_raster()
```

... should result in `If there were no assertion errors, the tests ran OK.`
