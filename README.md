Zonal statistics on raster data
---

The `zones` library calculates summary statistics with vector and raster data. The library manages projections on-the-fly, 
so there is no need to ensure consistency prior to running.  Statistics are processed per zone, so memory requirements 
scale with the size of the vector polygons. 

### Planned support

Currently, vector points are supported for `PointStats`, but support for tabular x,y data will 
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
>>> df = zs.calculate(['mean'])

```

### Testing

```python
>>> import zones
>>> zones.test_raster()
```

... should result in `If there were no assertion errors, the tests ran OK.`

