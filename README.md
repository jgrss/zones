Zonal statistics on raster data
---

The `zones` library calculates summary statistics with vector and raster data. The library manages projections on-the-fly, 
so there is no need to ensure consistency prior to running.  Statistics are processed per zone, so memory requirements 
scale with the size of the vector polygons. 

### Planned support

Currently, polygon zones are supported, but support for point data will 
also be added.


### Zonal stats with polygon and raster data

```python
>>> import zones
>>>
>>> zs = zones.RasterStats('values.tif', 'zones.shp', verbose=2)
>>>
>>> # One statistic
>>> df = zs.calculate(['mean'])
>>>
>>> # Multiple statistics
>>> df = zs.calculate(['nanmean', 'nansum'])
>>>
>>> # Save data to file
>>> df.to_file('stats.shp')
>>> df.to_csv('stats.csv')
```

### Zonal stats with polygon and point data (stored in a CSV file)

```python
>>> import zones
>>>
>>> zs = zones.PointStats('values.csv', 'zones.shp', x_column='X', y_column='Y', verbose=2)
>>>
>>> df = zs.calculate(['nanmean', 'nansum'])
>>>
>>> # Save data to file
>>> df.to_file('stats.shp')
>>> df.to_csv('stats.csv')
```

### Testing

```python
>>> import zones
>>> zones.test_01()
```

... should result in `If there were no assertion errors, the tests ran OK.`

