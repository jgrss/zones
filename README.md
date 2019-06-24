Zonal statistics on raster data
---

The `zones` library calculates summary statistics with vector and raster data. The library manages projections on-the-fly,
so there is no need to ensure consistency prior to running.  Statistics are processed per zone, so memory requirements
scale with the size of the vector polygons.

### Dependencies

Data I/O is handled with [MpGlue](https://github.com/jgrss/mpglue).

The following dependencies are installed, or upgraded, automatically.

- GDAL
- NumPy
- tqdm
- future
- GeoPandas
- Pandas
- Bottleneck

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

> Currently, only one statistic is supported when `n_jobs` is not equal to 1.

```python
>>> # Process zones in parallel, using all available CPUs.
>>> zs = zones.RasterStats('values.tif', 'zones.shp', n_jobs=-1)
>>> zs.calculate('var')
```

### Other methods

```python
>>> # Get available stats
>>> print(zs.stats_avail)
```

```python
>>> # To store the data as a distribution, use 'dist'.
>>> df = zs.calculate('dist')
```

### Testing

```python
>>> import zones
>>> zones.test_raster()
```

... should result in `If there were no assertion errors, the tests ran OK.`
