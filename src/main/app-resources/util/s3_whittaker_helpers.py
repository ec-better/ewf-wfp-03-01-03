from __future__ import absolute_import, division, print_function

import os
import sys 
sys.path.append('/'.join([os.environ['_CIOP_APPLICATION_PATH'], 'util']))
sys.path.append('../util')
import numpy as np
import gdal
import osr
import urllib.parse as urlparse
#from urlparse import urlparse
import pandas as pd
import datetime
#from whittaker import ws2d, ws2doptv, ws2doptvp, lag1corr
from vam.whittaker import ws2d, ws2doptv, ws2doptvp, lag1corr
from itertools import chain
import cioppy
import array
import geopandas as gpd



def get_sub_tiles(data_pipeline_results, pipeline_parameters, tiling_factor):
    
    sub_tiles = pd.DataFrame()

    for index, entry in data_pipeline_results.iterrows():

        print(entry.jday)

        src_ds = gdal.Open(get_vsi_url(entry.enclosure, 
                                       pipeline_parameters['username'], 
                                       pipeline_parameters['api_key']))


        step_x = src_ds.RasterXSize // tiling_factor
        step_y = src_ds.RasterYSize // tiling_factor

        for x in range(0, src_ds.RasterXSize // step_x):

                cols = step_x
                start_x = x * step_x 


                for y in range(0, src_ds.RasterYSize // step_y):

                    temp_dict = dict()

                    rows = step_y
                    start_y = y * step_y
                    
                    #temp_dict['sub_tile'] = 'tile_{}_{}_{}'.format(x, y, entry.title[38:44])
                    temp_dict['sub_tile'] = 'tile_{}_{}_{}'.format(x, y, entry.title[-7:])
                    temp_dict['start_x'] = start_x
                    temp_dict['start_y'] = start_y
                    temp_dict['cols'] = cols
                    temp_dict['rows'] = rows
                    #temp_dict['self'] = entry.self
                    #temp_dict['title'] = entry.title
                    temp_dict['day'] = entry.day
                    temp_dict['jday'] = entry.jday
                    temp_dict['enclosure'] = entry.enclosure

                    pd.Series(temp_dict)

                    sub_tiles = sub_tiles.append(pd.Series(temp_dict), ignore_index=True)  

                    #ds_mem = None
    print('Done!')
    return sub_tiles

def get_vsi_url(enclosure, user, api_key):
    
    parsed_url = urlparse.urlparse(enclosure)

    url = '/vsicurl/%s://%s:%s@%s/api%s' % (list(parsed_url)[0],
                                            user, 
                                            api_key, 
                                            list(parsed_url)[1],
                                            list(parsed_url)[2])
    
    return url 

def analyse_row(row):
    
    series = dict()

    series['day'] = ''.join(row['startdate'].split('T')[0].split('-'))
    series['jday'] = '{}{}'.format(datetime.datetime.strptime(series['day'], '%Y%m%d').timetuple().tm_year,
                                   "%03d"%datetime.datetime.strptime(series['day'], '%Y%m%d').timetuple().tm_yday)
    
    series['col-row'] = '{}-{}'.format(row['title'].split('C:')[1][:2],row['title'].split('R:')[1][:2])
    
    return pd.Series(series)


def analyse_merge_row(row, band_to_process):
    
    series = dict()

    if 's_' in row.enclosure:
        output_type = 's'
    
    elif 'original_{}'.format(band_to_process) in row.enclosure:
        output_type = 'original_{}'.format(band_to_process)
    
    elif 'lag1_{}'.format(band_to_process) in row.enclosure:
        output_type = 'lag1'
    
    elif 'mask_{}'.format(band_to_process) in row.enclosure:
        output_type = 'mask'
        
    else: 
        output_type = band_to_process

    series['output_type'] = output_type
    series['tile'] = os.path.basename(row.enclosure).split('_')[-1].split('.')[0]
    return pd.Series(series)    

def analyse_subtile(row, parameters, band_to_analyse):
    
    series = dict()
    
    src_ds = gdal.Open(get_vsi_url(row.enclosure, 
                                   parameters['username'], 
                                   parameters['api_key']))
    
    bands = dict()

    for band in range(src_ds.RasterCount):

        band += 1
        bands[src_ds.GetRasterBand(band).GetDescription()] = band 
        
    vsi_mem = '/vsimem/t.tif'
   
    gdal.Translate(vsi_mem, 
                   src_ds,
                   srcWin=[row.start_x, row.start_y, row.cols, row.rows])
    
    ds_mem = gdal.Open(vsi_mem)
    
    if ds_mem is None:
        raise

    # get the geocoding for the sub-tile
    series['geo_transform'] = [ds_mem.GetGeoTransform()]
    series['projection'] = ds_mem.GetProjection()
    series['SCL']= np.array(ds_mem.GetRasterBand(bands['SCL']).ReadAsArray())
    series['SCL_mask'] = ((series['SCL'] == 2) | (series['SCL'] == 4) | (series['SCL'] == 5) | (series['SCL'] == 6) |(series['SCL'] == 7) | (series['SCL'] == 10) | (series['SCL'] == 11))
    if band_to_analyse == 'NDVI':
        
        for band in ['B04', 'B08']:
            # read the data
            series[band] = np.array(ds_mem.GetRasterBand(bands[band]).ReadAsArray(),np.float32)

        # NDVI calculation done by lazy evaluation structure lambda to avoid division-by-zero  
        # NDVI<-0.2 set to noData
        ndvi = lambda x,y,z: -3000 if(x+y)==0 or z==False  else (-3000 if ((x-y)/float(x+y))<-0.2 else (x-y)/float(x+y))
        vfunc = np.vectorize(ndvi, otypes=[np.float])
        series['NDVI']=vfunc(series['B08'] ,series['B04'],series['SCL_mask'] )

        



        # remove the no longer needed bands

        for band in ['B04', 'B08']:
            series.pop(band, None)

    else:

        # read the band as float as it is required in filter     
        band_data = np.array(ds_mem.GetRasterBand(bands[band_to_analyse]).ReadAsArray(),np.float32)

        #noData value for other bands set to zero
        masked_band = lambda x,y : x if y else 0
        vfunc_masked = np.vectorize(masked_band, otypes=[np.float])
        series[band_to_analyse]=vfunc_masked(band_data, series['SCL_mask'])
        #series[band_to_analyse] = np.where(series['SCL_mask'], series[band_to_analyse], 0)
    

    ds_mem.FlushCache()

    return pd.Series(series)

def fromjulian(x):
    """
    Parses julian date string to datetime object.

    Args:
        x: julian date as string YYYYJJJ

    Returns:
        datetime object parsed from julian date
    """

    return datetime.datetime.strptime(x, '%Y%j').date()
    
def generate_dates(startdate_string=None, enddate_string=None, delta=5):
    """
    Generates a list of dates from a start date to an end date.

    Args:
        startdate_string: julian date as string YYYYJJJ
        enddate_string: julian date as string YYYYJJJ
        delta: integer timedelta between each date

    Returns:
        list of string julian dates YYYYJJJ
    """

    
    startdate = datetime.datetime.strptime(startdate_string, '%Y%j').date()
    enddate = datetime.datetime.strptime(enddate_string, '%Y%j').date()
    
    date_generated = [startdate + datetime.timedelta(days=x) for x in range(0, (enddate-startdate).days+delta, delta)]
    
    datelist = ['{}{:03d}'.format(x.year, x.timetuple().tm_yday) for x in date_generated]

    return datelist



def whittaker(ts, date_mask, band_to_analyse):
    """
    Apply the whittaker smoothing to a 1d array of floating values.
    Args:
        ts: array of floating values
        date_mask: full list of julian dates as string YYYYJJJ
    Returns:
        list of floating values. The first value is the s smoothing parameter
    """
    if band_to_analyse == "NDVI":
        nan_value = 255
    else:
        nan_value = 0
        
    ts_double=np.array(ts,dtype='double')
    mask = np.ones(len(ts))
    mask[ts==nan_value]=0
    # the output is an  array full of np.nan by default
    data_smooth = np.array([nan_value]*len(date_mask))
    
    # check if all values are np.npn
    if (mask==0).all()==False:

        w=np.array((ts!=nan_value)*1,dtype='double')
        lrange = array.array('d', np.linspace(-2, 4, 61))
        
        try: 
            # apply whittaker filter with V-curve
            zv, loptv = ws2doptvp(ts_double, w, lrange, p=0.90)
            #parameters needed for the interpolation step
           
            dvec = np.zeros(len(date_mask))
            w_d=np.ones(len(date_mask), dtype='double')

            
            # adding new dates with no associated product to the weights
            for idx, el in enumerate(date_mask):
                if not el:
                    w_d[idx]= 0

            dvec[w_d==1]= zv
            
            # apply whittaker filter with very low smoothing to interpolate
            data_smooth = ws2d(dvec, 0.0001, w_d)
            
            # Calculates Lag-1 correlation
            
            lag1 = lag1corr(ts_double[:-1], ts_double[1:], nan_value)
            
            


        except Exception as e:
            loptv = 0
            lag1 = nan_value
            print(e)
            print(mask)

    else:
        loptv = 0
        lag1 = nan_value
        

    return tuple(np.append(np.append(loptv,lag1), data_smooth))

def cog(input_tif, output_tif,no_data=None):
    
    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine('-co TILED=YES ' \
                                                                    '-co COPY_SRC_OVERVIEWS=YES ' \
                                                                    '-co COMPRESS=LZW '))
    
    if no_data != None:
        translate_options = gdal.TranslateOptions(gdal.ParseCommandLine('-co TILED=YES ' \
                                                                        '-co COPY_SRC_OVERVIEWS=YES ' \
                                                                        '-co COMPRESS=LZW '\
                                                                        '-a_nodata {}'.format(no_data)))
    ds = gdal.Open(input_tif, gdal.OF_READONLY)

    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    ds.BuildOverviews('NEAREST', [2,4,8,16,32])
    
    ds = None

    ds = gdal.Open(input_tif)
    gdal.Translate(output_tif,
                   ds, 
                   options=translate_options)
    ds = None

    os.remove('{}.ovr'.format(input_tif))
    os.remove(input_tif)
