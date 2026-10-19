var CONFIG = {
  startDate: '2003-01-01',
  endDateExclusive: '2024-10-01',
  baselineStart: '2003-01-01',
  baselineEndExclusive: '2015-01-01',
  exportFolder: 'gee_ncp_groundwater',
  statsScaleMeters: 10000,
  maxElevationMeters: 200,
  minCoverageRatio: 0.60,
  gridStepDegree: 1.0,
  bbox: [112.0, 34.5, 120.0, 40.6],
  provinceNames: ['Beijing Shi', 'Tianjin Shi', 'Hebei Sheng', 'Henan Sheng', 'Shandong Sheng']
};

var START = ee.Date(CONFIG.startDate);
var END = ee.Date(CONFIG.endDateExclusive);
var BASELINE_START = ee.Date(CONFIG.baselineStart);
var BASELINE_END = ee.Date(CONFIG.baselineEndExclusive);
var BBOX = ee.Geometry.Rectangle(CONFIG.bbox, 'EPSG:4326', false);

var provinces = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'China'))
    .filter(ee.Filter.inList('ADM1_NAME', CONFIG.provinceNames));
var provinceGeom = provinces.geometry().dissolve();
var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(BBOX);
var lowlandMask = elevation.lte(CONFIG.maxElevationMeters).selfMask().clip(provinceGeom);

function monthStarts(startDate, endDateExclusive) {
  var monthCount = endDateExclusive.difference(startDate, 'month');
  return ee.List.sequence(0, monthCount.subtract(1)).map(function(n) {
    return startDate.advance(n, 'month');
  });
}

function buildGrid() {
  var lonList = ee.List.sequence(
      CONFIG.bbox[0],
      CONFIG.bbox[2] - CONFIG.gridStepDegree,
      CONFIG.gridStepDegree);
  var latList = ee.List.sequence(
      CONFIG.bbox[1],
      CONFIG.bbox[3] - CONFIG.gridStepDegree,
      CONFIG.gridStepDegree);

  var cells = lonList.map(function(lon) {
    lon = ee.Number(lon);
    return latList.map(function(lat) {
      lat = ee.Number(lat);
      var rect = ee.Geometry.Rectangle(
          [lon, lat, lon.add(CONFIG.gridStepDegree), lat.add(CONFIG.gridStepDegree)],
          'EPSG:4326',
          false);
      var overlapDict = ee.Image.pixelArea().updateMask(lowlandMask).reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: rect,
        scale: 1000,
        bestEffort: true,
        maxPixels: 1e13
      });
      var overlap = ee.Number(ee.Algorithms.If(overlapDict.get('area'), overlapDict.get('area'), 0));
      var center = rect.centroid(1);
      var centerLat = ee.Number(center.coordinates().get(1));
      return ee.Feature(rect, {
        grid_id: ee.String('g_').cat(lon.format('%.1f')).cat('_').cat(lat.format('%.1f')),
        grid_lon: center.coordinates().get(0),
        grid_lat: centerLat,
        coverage_ratio: overlap.divide(rect.area(1)),
        region_tag: ee.String(ee.Algorithms.If(centerLat.gte(37.0), 'north', 'south'))
      });
    });
  });

  return ee.FeatureCollection(ee.List(cells).flatten())
      .filter(ee.Filter.gte('coverage_ratio', CONFIG.minCoverageRatio));
}

function mergeFeatureCollections(listOfCollections) {
  return ee.FeatureCollection(ee.List(listOfCollections).iterate(function(item, acc) {
    return ee.FeatureCollection(acc).merge(ee.FeatureCollection(item));
  }, ee.FeatureCollection([])));
}

var grid = buildGrid();
var grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI');
var gldas = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H');
var imerg = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07');
var mod16 = ee.ImageCollection('MODIS/061/MOD16A2GF');
var jrcMonthly = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');

function maskedEmpty(name) {
  return ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0));
}

function safeMean(collection, start, end, bandName, outName) {
  var filtered = collection.filterDate(start, end).select(bandName);
  return ee.Image(ee.Algorithms.If(
      filtered.size().gt(0),
      filtered.mean().rename(outName),
      maskedEmpty(outName)
  ));
}

function safeSum(collection, start, end, bandName, outName, scaleFactor) {
  var filtered = collection.filterDate(start, end).select(bandName);
  return ee.Image(ee.Algorithms.If(
      filtered.size().gt(0),
      filtered.sum().multiply(scaleFactor).rename(outName),
      maskedEmpty(outName)
  ));
}

function maskLandsatL2(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 0).eq(0)
      .and(qa.bitwiseAnd(1 << 1).eq(0))
      .and(qa.bitwiseAnd(1 << 2).eq(0))
      .and(qa.bitwiseAnd(1 << 3).eq(0))
      .and(qa.bitwiseAnd(1 << 4).eq(0))
      .and(qa.bitwiseAnd(1 << 5).eq(0));
  var optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
  return image.addBands(optical, null, true).updateMask(mask);
}

function monthlyWaterFromLandsat(monthStart) {
  var monthEnd = monthStart.advance(1, 'month');
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterDate(monthStart, monthEnd)
      .filterBounds(BBOX)
      .map(maskLandsatL2);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
      .filterDate(monthStart, monthEnd)
      .filterBounds(BBOX)
      .map(maskLandsatL2);
  var monthly = l8.merge(l9);

  return ee.Image(ee.Algorithms.If(
      monthly.size().gt(0),
      monthly.map(function(image) {
        var green = image.select('SR_B3');
        var red = image.select('SR_B4');
        var nir = image.select('SR_B5');
        var swir1 = image.select('SR_B6');
        var mndwi = green.subtract(swir1).divide(green.add(swir1));
        var ndvi = nir.subtract(red).divide(nir.add(red));
        return mndwi.gt(0).and(mndwi.gt(ndvi)).rename('water_frac');
      }).mean().rename('water_frac'),
      maskedEmpty('water_frac')
  ));
}

function monthlyWaterFraction(monthStart) {
  var year = ee.Number.parse(monthStart.format('Y'));
  var month = ee.Number.parse(monthStart.format('M'));
  var jrcFiltered = jrcMonthly
      .filter(ee.Filter.eq('year', year))
      .filter(ee.Filter.eq('month', month));

  return ee.Image(ee.Algorithms.If(
      monthStart.millis().lt(ee.Date('2022-01-01').millis()),
      ee.Image(ee.Algorithms.If(
          jrcFiltered.size().gt(0),
          ee.Image(jrcFiltered.first()).select('water').eq(2).rename('water_frac'),
          maskedEmpty('water_frac')
      )),
      monthlyWaterFromLandsat(monthStart)
  ));
}

function buildRawMonthlyImage(monthStart) {
  monthStart = ee.Date(monthStart);
  var monthEnd = monthStart.advance(1, 'month');
  var daysInMonth = monthEnd.difference(monthStart, 'day');
  var monthNumber = ee.Number.parse(monthStart.format('M'));
  var yearNumber = ee.Number.parse(monthStart.format('Y'));

  var tws = safeMean(grace, monthStart, monthEnd, 'lwe_thickness', 'tws_cm');
  var gldasMonthly = gldas.filterDate(monthStart, monthEnd).mean();
  var sms = gldasMonthly.select([
      'SoilMoi0_10cm_inst',
      'SoilMoi10_40cm_inst',
      'SoilMoi40_100cm_inst',
      'SoilMoi100_200cm_inst'
    ]).reduce(ee.Reducer.sum()).divide(10).rename('sms_cm');
  var swe = gldasMonthly.select('SWE_inst').divide(10).rename('swe_cm');
  var cwa = gldasMonthly.select('CanopInt_inst').divide(10).rename('cwa_cm');
  var precipRate = safeMean(imerg, monthStart, monthEnd, 'precipitation', 'precip_rate_mmhr');
  var precip = precipRate.multiply(24).multiply(daysInMonth).rename('precip_mm');
  var et = safeSum(mod16, monthStart, monthEnd, 'ET', 'et_mm', 0.1);
  var waterFrac = monthlyWaterFraction(monthStart);

  return ee.Image.cat([tws, sms, swe, cwa, precip, et, waterFrac])
      .updateMask(lowlandMask)
      .clip(BBOX)
      .set('system:time_start', monthStart.millis())
      .set('date', monthStart.format('YYYY-MM-dd'))
      .set('month', monthNumber)
      .set('year', yearNumber);
}

var months = monthStarts(START, END);
var monthlyRaw = ee.ImageCollection.fromImages(months.map(buildRawMonthlyImage));
var baseline = monthlyRaw.filterDate(BASELINE_START, BASELINE_END);
var baselineMeans = ee.Image.cat([
  baseline.select('tws_cm').mean().rename('tws_base'),
  baseline.select('sms_cm').mean().rename('sms_base'),
  baseline.select('swe_cm').mean().rename('swe_base'),
  baseline.select('cwa_cm').mean().rename('cwa_base')
]);

var monthlyFinal = monthlyRaw.map(function(image) {
  var twsa = image.select('tws_cm').subtract(baselineMeans.select('tws_base')).rename('twsa_cm');
  var smsa = image.select('sms_cm').subtract(baselineMeans.select('sms_base')).rename('smsa_cm');
  var swea = image.select('swe_cm').subtract(baselineMeans.select('swe_base')).rename('swea_cm');
  var cwa = image.select('cwa_cm').subtract(baselineMeans.select('cwa_base')).rename('cwa_cm');
  var gwa = twsa.subtract(smsa).subtract(swea).subtract(cwa).rename('gwa_cm');

  return ee.Image.cat([
      gwa,
      twsa,
      smsa,
      swea,
      cwa,
      image.select('precip_mm'),
      image.select('et_mm'),
      image.select('water_frac')
    ])
    .set('system:time_start', image.get('system:time_start'))
    .set('date', image.get('date'))
    .set('month', image.get('month'))
    .set('year', image.get('year'));
});

function monthlyGridFeatures(monthStart) {
  monthStart = ee.Date(monthStart);
  var image = ee.Image(monthlyFinal.filterDate(monthStart, monthStart.advance(1, 'month')).first());
  return image.reduceRegions({
    collection: grid,
    reducer: ee.Reducer.mean(),
    scale: CONFIG.statsScaleMeters,
    tileScale: 4
  }).map(function(feature) {
    return feature.set({
      date: monthStart.format('YYYY-MM-dd'),
      year: ee.Number.parse(monthStart.format('Y')),
      month: ee.Number.parse(monthStart.format('M')),
      system_time_start: monthStart.millis()
    });
  });
}

var monthlyPanel = mergeFeatureCollections(months.map(monthlyGridFeatures))
    .sort('system_time_start');

Map.centerObject(BBOX, 7);
Map.addLayer(provinces.style({color: '222222', fillColor: '00000000', width: 1}), {}, 'Province envelope');
Map.addLayer(lowlandMask, {palette: ['f4f1bb']}, 'Lowland mask');
Map.addLayer(grid.style({color: '08519c', fillColor: '00000000', width: 1}), {}, '1-degree grid');

print('Grid definition', grid);
print('Monthly panel sample', monthlyPanel.limit(10));

Export.table.toDrive({
  collection: monthlyPanel,
  description: 'grid_monthly_panel',
  folder: CONFIG.exportFolder,
  fileFormat: 'CSV',
  selectors: [
    'grid_id',
    'grid_lon',
    'grid_lat',
    'region_tag',
    'coverage_ratio',
    'date',
    'year',
    'month',
    'gwa_cm',
    'twsa_cm',
    'smsa_cm',
    'swea_cm',
    'cwa_cm',
    'precip_mm',
    'et_mm',
    'water_frac'
  ]
});

Export.table.toDrive({
  collection: grid,
  description: 'ncp_grid_definition',
  folder: CONFIG.exportFolder,
  fileFormat: 'CSV',
  selectors: ['grid_id', 'grid_lon', 'grid_lat', 'coverage_ratio', 'region_tag']
});
