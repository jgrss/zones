Zonal statistics on raster data
---

## Zonal stats with polygon data

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
