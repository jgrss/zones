Zonal statistics on raster data
---

The `zones` library calculates summary statistics with vector and raster data. The library manages projections on-the-fly,
so there is no need to ensure consistency prior to running. Statistics are processed per zone, so memory requirements
scale with the size of the vector polygons. 

### Planned support

Currently, polygon zones are supported, but support for point data will 
also be added.


### Zonal stats with polygon data

```python
>>> import zones
>>>
>>> zs = zones.RasterStats('values.tif', 'zones.shp')
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
