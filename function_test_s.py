import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import sys
import os

# function for deseason data from 30 year climatology ; climatology change for every 5 years
def detrend_data_(data): # 'time' should be the time coordinate name
    y_beg1=data.time.dt.year.min().values
    y_beg2=y_beg1+4 # 5 year band 
    y_end=data.time.dt.year.max().values
    dat_=[]  
    n=1
    
    while (y_beg1 <= y_end-15): # upto last - 15 year (-15 year, 5 year band , +10 year) 

        dat_.append(data.sel(time=slice(str(y_beg1),str(y_beg2))).groupby('time.month')-data.sel(time=slice(str(y_beg1-15),str(y_beg2+10))).groupby('time.month').mean(dim='time'))
        #print(n,y_beg1,y_beg2,y_beg1-15,y_beg2+10)
        y_beg1=y_beg1+5
        y_beg2=y_beg2+5
        n=n+1
    ## for last 15 year 30 year climatology from last 30 year
    dat_.append(data.sel(time=slice(str(y_beg1),str(y_end))).groupby('time.month')-data.sel(time=slice(str(y_end-29),str(y_end))).groupby('time.month').mean(dim='time'))
    #print('!!!!!!')
    #print(n,y_beg1,y_end,y_end-29,y_end)
    print('!!!!!! Deseasoned and estimated the anomalies for 30 year climatology with 5 year moving window')

    return(xr.concat(dat_,dim='time'))
######################

def normalize_data(data):
    norm_data=(data - np.min(data)) / (np.max(data) - np.min(data)) # 0 to 1
    print('!!! data normalized= min max scaler')
    return(norm_data)

def lag_data(data_nino,data2,lag_):
    given_time = pd.to_datetime(data_nino.time.values)
    lagged_time = given_time - pd.DateOffset(months=lag_)
    data2 = data2.isel(time=~data2.indexes['time'].duplicated())
    lag_data=data2.sel(time=lagged_time)
    print('!! lag data  estimated')
    
    return(lag_data)



# first four year nino values removes >> Otherwise we  will not get a 0-24 (complete lag months) lag SST and OHC for jan-febuary of first 3 years
#predict_mon=1 means February. # 0 = jan , 11 = Dec  # month of prediction 
def modify_soda_(sst,ohc,lag_,predict_mon):
    
    ohca=detrend_data_(ohc.sel(lat=slice(-55,60)))
    ssta=detrend_data_(sst.sel(lat=slice(-55,60)))
    print('!! SSTA, OHCA calculated !!')
    
    box_avg_ssta=ssta.sel(lat=slice(-5,5),lon=slice(190,240)).mean(dim=('lon','lat'))
    nino_=box_avg_ssta.rolling(time=3).mean()
    print('!! nino 3.4 index calculated !!')

    nino_m=[]
    for j in range(1,13):
        nino_m.append(nino_[nino_.time.dt.month.isin([j])])

    pre_month1=1
    pre_month2=2

    ssta2=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_)
    ssta2_pm1=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_+pre_month1)
    ssta2_pm2=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_+pre_month2)

    ohca2=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_)
    ohca2_pm1=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_+pre_month1)
    ohca2_pm2=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_+pre_month2)

    print('!! ssta, ohca lag data estimated !!')

    return (ssta2,ssta2_pm1,ssta2_pm2,
            ohca2,ohca2_pm1,ohca2_pm2,
            nino_m[predict_mon][4:])




model_name=['ACCESS-CM2','CNRM_CM5','CanESM2','CESM1-BGC','HadGEM3-GC31-LL','ACCESS1-0','GFDL-CM4','IPSL-CM6A-LR','CESM2',
            'CMCC-CMS','GFDL-CM3','MPI-ESM1-2-HR','MPI-ESM-LR','CESM2-FV2','MRI-ESM2-0','CNRM-CM6-1','GISS-E2R','MPI-ESM1-2-LR','MIROC-ES2L',
            'BCC-CSM2-MR','BCC-CSM1-1','HadGEM2_ES','MIROC6','CCSM4','GISS-E2-1-H']

path_merged_tos_2=['/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Omon.gn_tos_regrided_1.47.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_CNRM-CM5_historical_r1i1p1_185001-201212_regrided_1.67.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_CanESM2_historical_r1i1p1_185001-200512_regrided_1.82.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_CESM1-BGC_historical_r1i1p1_185001-200512_regrided_1.84.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MOHC.HadGEM3-GC31-LL.historical.Omon.gn_tos_regridded_1.91.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_ACCESS1-0_historical_r1i1p1_185001-200512_regrided_1.92.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.NOAA-GFDL.GFDL-CM4.historical.Omon.gr_hist_tos_regrided_1.92.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.IPSL.IPSL-CM6A-LR.historical.Omon.gn_tos_regrided_2.21.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.NCAR.CESM2.historical.Omon.gr_hist_tos_regrided_2.22.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_CMCC-CMS_historical_r1i1p1_185001-200512_regrided_2.22.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_GFDL-CM3_historical_r1i1p1_186001-200512_regrided_2.27.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MPI-M.MPI-ESM1-2-HR.historical.Omon.gn_tos_regrided_2.30.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_MPI-ESM-LR_historical_r1i1p1_185001-200512_regrided_2.35.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.NCAR.CESM2-FV2.historical.Omon.gr_hist_tos_regrided_2.42.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MRI.MRI-ESM2-0.historical.Omon.gr_hist_tos_regrided_2.42.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.Omon.gr1_tos_regrided_2.42.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_GISS-E2-R_historical_r1i1p1_185001-201212_regrided_2.45.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MPI-M.MPI-ESM1-2-LR.historical.Omon.gn_tos_regrided_2.45.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MIROC.MIROC-ES2L.historical.Omon.gr1_tos_regrided_2.47.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.BCC.BCC-CSM2-MR.historical.Omon.gn_tos_regrided_2.50.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_BCC-CSM1-1_historical_r1i1p1_185001-201212_regrided_1.94.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_HadGEM2-ES_historical_r1i1p1_185912-200512_regrided_2.09.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.MIROC.MIROC6.historical.Omon.gn_tos_regrided_2.10.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/tos/regrided/tos_Omon_CCSM4_historical_r1i1p1_185001-200512_regrided_2.11.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/tos/regrided/CMIP.NASA-GISS.GISS-E2-1-H.historical.Omon.gn_tos_regrided_2.19.nc']



path_merged_thetao_2=['/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_CNRM_CM5_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_CanESM2_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_CESM1_BGC_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MOHC.HadGEM3-GC31-LL.historical.Omon.gn_thetao_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_ACCESS1-0_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.NOAA-GFDL.GFDL-CM4.historical.Omon.gr_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.IPSL.IPSL-CM6A-LR.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.NCAR.CESM2.historical.Omon.gr_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_CMCC-CMS_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_GFDL_CM3_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MPI-M.MPI-ESM1-2-HR.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_MPI_ESM_LR_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.NCAR.CESM2-FV2.historical.Omon.gr_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MRI.MRI-ESM2-0.historical.Omon.gr_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.Omon.gn_thetao_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_GISS_E2R_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MPI-M.MPI-ESM1-2-LR.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MIROC.MIROC-ES2L.historical.Omon.gn_thetao_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.BCC.BCC-CSM2-MR.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_BCC-CSM-1-1_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_HadGEM2_ES_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.MIROC.MIROC6.historical.Omon.gn_thetao_0_300_mean_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip5/thetao/regrided/thetao_Omon_CCSM4_historical_r1i1p1_185001-200512_regrided_.nc',
 '/home/users/roms/balaji/athul/enso/data/cmip6/thetao/regrided/CMIP.NASA-GISS.GISS-E2-1-H.historical.Omon.gr_thetao_0_300_mean_regrided_.nc']
 
### loading data ##############
#len(path_merged_thetao_2)
# cmip5 data in kelvin and cmip6 in degree >> transfered to degree

tos_=[]
thetao_=[]
for ii in range(len(path_merged_tos_2)):
   ds_tos_= xr.open_dataset(path_merged_tos_2[ii])
   ds_thetao_= xr.open_dataset(path_merged_thetao_2[ii])
   if "cmip5" in path_merged_tos_2[ii]:
      tos_.append(ds_tos_.tos.sel(lat=slice(-55,60)).sel(time=slice('1860','2005'))-273.15)#.convert_calendar(calendar='standard',align_on='year'))
      thetao_.append(ds_thetao_.thetao.sel(lat=slice(-55,60)).sel(time=slice('1860','2005'))-273.15)#.convert_calendar(calendar='standard',align_on='year'))
   else:
      tos_.append(ds_tos_.tos.sel(lat=slice(-55,60))) # for cmip6 data
      thetao_.append(ds_thetao_.thetao.sel(lat=slice(-55,60)))


def select_cmip_model(model_no_,lag_,predict_mon):
    sst=tos_[model_no_].sel(lat=slice(-55,60))
    ohc=thetao_[model_no_].sel(lat=slice(-55,60))

    min_len=min(len(sst.time),len(ohc.time))
    sst=sst.isel(time=slice(0,min_len))
    ohc=ohc.isel(time=slice(0,min_len))

    yr_min=str(sst.time.dt.year.min().item())
    t_ax = pd.date_range(start=yr_min+'-01-01', periods=min_len, freq='M')+pd.DateOffset(day=15)+pd.DateOffset(hour=12)
    sst['time']=t_ax
    ohc['time']=t_ax

    ohca=detrend_data_(ohc)
    ssta=detrend_data_(sst)
    print('!! SSTA, OHCA calculated !!')

    box_avg_ssta=ssta.sel(lat=slice(-5,5),lon=slice(190,240)).mean(dim=('lon','lat'))
    nino_=box_avg_ssta.rolling(time=3).mean()
    print('!! nino 3.4 index calculated !!')

    nino_m=[]
    for j in range(1,13):
        nino_m.append(nino_[nino_.time.dt.month.isin([j])])

    pre_month1=1
    pre_month2=2

    ssta2=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_)
    ssta2_pm1=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_+pre_month1)
    ssta2_pm2=lag_data(nino_m[predict_mon][4:],ssta,lag_=lag_+pre_month2)


    ohca2=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_)
    ohca2_pm1=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_+pre_month1)
    ohca2_pm2=lag_data(nino_m[predict_mon][4:],ohca,lag_=lag_+pre_month2)


    print('!!ssta, ohca,  lag data estimated !!')

    return (ssta2,ssta2_pm1,ssta2_pm2,
            ohca2,ohca2_pm1,ohca2_pm2,
            nino_m[predict_mon][4:])


#####################

