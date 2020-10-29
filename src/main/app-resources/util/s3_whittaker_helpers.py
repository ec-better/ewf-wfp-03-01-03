from __future__ import absolute_import, division, print_function

import os
import sys 
sys.path.append('/'.join([os.environ['_CIOP_APPLICATION_PATH'], 'util']))
sys.path.append('../util')
import numpy as np
import gdal
import osr
import urllib.parse as urlparse

import pandas as pd
import datetime

from vam.whittaker import ws2d, ws2doptv, ws2doptvp, lag1corr
from itertools import chain
import cioppy
import array
import geopandas as gpd




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



def analyse_gps(row, user, api_key ):
            
    enclosure_vsi_url = get_vsi_url(row.enclosure, 
                                    user, 
                                    api_key)
    data_gdal = gdal.Open(enclosure_vsi_url)

            
    ulx, xres, xskew, uly, yskew, yres  = data_gdal.GetGeoTransform()
    lrx = ulx + (data_gdal.RasterXSize * xres)
    lry = uly + (data_gdal.RasterYSize * yres)

    series = dict()

    series['ul_x'] = ulx
    series['ul_y'] = uly
    series['lr_x'] = lrx
    series['lr_y'] = lry

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



def whittaker(ts, date_mask):
    """
    Apply the whittaker smoothing to a 1d array of floating values.
    Args:
        ts: array of floating values
        date_mask: full list of julian dates as string YYYYJJJ
    Returns:
        list of floating values. The first value is the s smoothing parameter
    """
    nan_value = 255

        
    ts_double=np.array(ts,dtype='double')
    mask = np.ones(len(ts))
    mask[ts==nan_value]=0
    # the output is an  array full of np.nan by default
    data_smooth = np.array([nan_value]*len(date_mask))
    
    # check if all values are np.npn
    if np.sum(mask)>0:

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

