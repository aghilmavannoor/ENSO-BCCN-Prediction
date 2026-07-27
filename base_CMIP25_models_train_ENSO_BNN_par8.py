######  pgrm to train BNN model for ENSO prediction using CMIP monthly data  #######
print("Pgrm started!!", flush=True)
import os

seed = 53

os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import warnings
warnings.simplefilter(action="ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import sys
import shutil
import random
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.config.experimental.enable_op_determinism()

# Check if three arguments (script name, prediction month, and lead months) are provided
'''if len(sys.argv) != 3:
    print('two user input required!!!')
    print("Usage: python script.py prediction_month  lead_month   eg: python script.py 11 23")
    print("Please provide month [0:11, 0=Jan, 11=Dec], and lead input [1,24]  as command-line argument")
    sys.exit(1)'''''


#### import my functions ######
from function_test_s import detrend_data_
from function_test_s import normalize_data
from function_test_s import lag_data
from function_test_s import modify_soda_
from function_test_s import select_cmip_model

####################(Taking input month ,lag, epoches, batch size etc)    ######################

os.makedirs("running", exist_ok=True)

# Get the input values for prediction month  and lead month from command line arguments
def run_model(mon_, lag_mon):
    Calender_month = ('January,February,March,April,May,June,July,August,September,October,November,December').split(',')
    month_ = Calender_month[mon_]


    print("!!!", flush=True)
    print("Epoches and Batch size are fixed, change in the code if required", flush=True)

    ####### !!!!!!!! #########  >>>>>>>>>>>>>  Check always !!!!!1
    batch_size = 100
    epochs = 1000
    tot_models = 25

    ###### >>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<
    print('month=', month_)
    print('lag= ' + str(lag_mon))
    print('epoches= ', epochs)
    print('batch size= ', batch_size)
    print('total models= ', str(tot_models))

    ## creating folders to save outputs ############
    out_dir_path = '/home/users/roms/balaji/aghil/nino_seed_version/nino/3_input/300m/out_final/' + month_ + '/lead' + str(lag_mon)

    def create_directory(directory_path):
        # Check if the directory already exists
        if os.path.exists(directory_path):
            # If it exists, remove it and its contents
            shutil.rmtree(directory_path)
        # Create the directory
        os.makedirs(directory_path)

    create_directory(out_dir_path)
    print("Directory " + out_dir_path + " created successfully", flush=True)

    ###################  Loading ML modules ##############

    from tensorflow import keras
    import tensorflow as tf
    import tensorflow_probability as tfp
    #from tensorflow.keras.mixed_precision import set_global_policy
    #set_global_policy("mixed_float16")

    dist = tfp.distributions
    tfd = tfp.distributions

    print("AI/ML Modules loaded!!", flush=True)

    # first four year nino values removes >> Otherwise we  will not get a 0-24 (complete lag months) lag SST and OHC for jan-febuary of first 3 years
    #predict_mon=1 means February. # 0 = jan , 11 = Dec  # month of prediction 
    #lag_=15 >> 15 lag months
    #####################

    # lists for 6-month lookback (t .. t-5)
    ssta_tau_0 = []
    ssta_tau_1 = []
    ssta_tau_2 = []

    ohca_tau_0 = []
    ohca_tau_1 = []
    ohca_tau_2 = []

    nino_lead = []

    # load SODA data and create training samples and append to CMIP samples ####
    ds=xr.open_dataset('/home/users/roms/balaji/athul/enso/data/soda/soda_1871_2010_2.5_reshaped.nc').drop_vars("lev")

    print(ds)
    sst=ds.t_5m.sel(lat=slice(-55,60)).sel(time=slice('1871','1980')) # in degree celsius
    thetao=ds.t_5m_300_avg.sel(lat=slice(-55,60)).sel(time=slice('1871','1980')) # in degree

    print('SODA data loaded')
    """
    ds = xr.open_dataset('/home/arya/hennath/DATA/SODA/soda1/sodadata_sst_and_average_5m_10001.nc')
    sst = ds.sst.sel(lat=slice(-55, 60)).sel(time=slice('1871', '1980'))  # in degree celsius
    sst = sst.drop_vars("LEV1_25")
    thetao = ds.temp_1000.sel(lat=slice(-55, 60)).sel(time=slice('1871', '1980'))  # in degree
    thetao = thetao.drop_vars("LEV1_25")
    print('SODA data loaded')
    """
    ##################################
    # modify_soda_ must now return:
    # ssta_t, ssta_t-1, ssta_t-2, ssta_t-3, ssta_t-4, ssta_t-5,
    # ohc_t, ohc_t-1, ohc_t-2, ohc_t-3, ohc_t-4, ohc_t-5,
    # nino_target_series
    (xx_soda, yy_soda, zz_soda, 
     aa_soda, bb_soda, cc_soda, 
     dd_soda) = modify_soda_(sst, thetao, predict_mon=mon_, lag_=lag_mon)


    print('SODA training sample is appended to CMIP samples', flush=True)

    # Append SODA arrays to lists (ensure types are xarray DataArray with dims time,lat,lon)
    ssta_tau_0.append(xx_soda)
    ssta_tau_1.append(yy_soda)
    ssta_tau_2.append(zz_soda)
    
    ohca_tau_0.append(aa_soda)
    ohca_tau_1.append(bb_soda)
    ohca_tau_2.append(cc_soda)

    nino_lead.append(dd_soda)

    ## load CMIP data ###############
    for jj in range(tot_models):
        # select_cmip_model must now return same 12 arrays + nino array
        (xx, yy, zz,
         aa, bb, cc,
         dd) = select_cmip_model(model_no_=jj, predict_mon=mon_, lag_=lag_mon)

        ssta_tau_0.append(xx)
        ssta_tau_1.append(yy)
        ssta_tau_2.append(zz)
        
        ohca_tau_0.append(aa)
        ohca_tau_1.append(bb)
        ohca_tau_2.append(cc)
        
        nino_lead.append(dd)

        print('model_no=', jj, ' is over!!', flush=True)

    ###############
    """
    def drop_lev(x):
        if 'lev' in x.coords:
            print("yes")
            print(x.drop('lev').coords,x.drop_vars('lev').coords)
            return x.drop_vars('lev')
        else:
            return x

    ssta_tau_0 = [drop_lev(da) for da in ssta_tau_0]
    ssta_tau_1 = [drop_lev(da) for da in ssta_tau_1]
    ssta_tau_2 = [drop_lev(da) for da in ssta_tau_2]
    ssta_tau_3 = [drop_lev(da) for da in ssta_tau_3]
    ssta_tau_4 = [drop_lev(da) for da in ssta_tau_4]
    ssta_tau_5 = [drop_lev(da) for da in ssta_tau_5]

    ohca_tau_0 = [drop_lev(da) for da in ohca_tau_0]
    ohca_tau_1 = [drop_lev(da) for da in ohca_tau_1]
    ohca_tau_2 = [drop_lev(da) for da in ohca_tau_2]
    ohca_tau_3 = [drop_lev(da) for da in ohca_tau_3]
    ohca_tau_4 = [drop_lev(da) for da in ohca_tau_4]
    ohca_tau_5 = [drop_lev(da) for da in ohca_tau_5]



    for i in ssta_tau_0:
        print(i)
    """    
    # Concatenate all returned arrays along time
    merged_sst0 = xr.concat(ssta_tau_0, dim='time')
    merged_sst1 = xr.concat(ssta_tau_1, dim='time')
    merged_sst2 = xr.concat(ssta_tau_2, dim='time')

    
    merged_ohc0 = xr.concat(ohca_tau_0, dim='time')
    merged_ohc1 = xr.concat(ohca_tau_1, dim='time')
    merged_ohc2 = xr.concat(ohca_tau_2, dim='time')
    
    merged_nino = xr.concat(nino_lead, dim='time')

    # Create synthetic time index (as before)
    new_time_sst = pd.date_range('1700-01-12', periods=len(merged_sst0.time), freq='M')

    # Assign time coordinate to all merged arrays
    for da in (merged_sst0, merged_sst1, merged_sst2,
           merged_ohc0, merged_ohc1, merged_ohc2,
           merged_nino):
        da['time'] = new_time_sst

    # Build input tensor with 12 channels (SST t..t-5, OHC t..t-5)
    im_ = np.empty([merged_sst0.shape[0], merged_sst0.shape[1], merged_sst0.shape[2], 6])
    nan_ = -999

    # SST channels: t, t-1, t-2, t-3, t-4, t-5
    im_[:, :, :, 0] = merged_sst0.fillna(nan_)
    im_[:, :, :, 1] = merged_sst1.fillna(nan_)
    im_[:, :, :, 2] = merged_sst2.fillna(nan_)


    # OHC channels: t, t-1, t-2, t-3, t-4, t-5
    im_[:, :, :, 3] = merged_ohc0.fillna(nan_)
    im_[:, :, :, 4] = merged_ohc1.fillna(nan_)
    im_[:, :, :, 5] = merged_ohc2.fillna(nan_)

    y_tr_ = merged_nino.values
    x_tr_ = im_

    ###############################################

    def create_model(len_sample, input_shape):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape, name="basket", dtype='float32'),

            tfp.layers.Convolution2DFlipout(32, kernel_size=5, strides=(1, 1), data_format="channels_last",
                                            padding="same", activation=tf.nn.tanh, name="conv_tfp_1a",
                                            kernel_divergence_fn=kl_divergence_function,seed=53),
            tf.keras.layers.MaxPool2D(strides=(4, 4), pool_size=(4, 4), padding="same"),

            tfp.layers.Convolution2DFlipout(64, kernel_size=3, strides=(1, 1), data_format="channels_last",
                                            padding="same", activation=tf.nn.tanh, name="conv_tfp_1b",
                                            kernel_divergence_fn=kl_divergence_function,seed=53),
            tf.keras.layers.MaxPool2D(strides=(4, 4), pool_size=(4, 4), padding="same"),

            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(1, kernel_divergence_fn=kl_divergence_function,seed=53)
        ])

        learning_rate = 1.0e-3
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate),
                      metrics=['mse'])

        return model

    fff = "running/"+f"{mon_}_{lag_mon}.txt"
    open(fff,"w").close()
    
    tf.keras.backend.clear_session()
    kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(len(x_tr_), dtype='float32')
    model_ = create_model(len_sample=len(x_tr_), input_shape=x_tr_[0].shape)

    print('Model training started', flush=True)
    ############################################################################################
    checkpoint_path = out_dir_path + '/bay_cnn.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=0)

    history_ = model_.fit(x=x_tr_, y=np.array(y_tr_),
                          epochs=epochs,
                          verbose=0,
                          batch_size=batch_size,
                          validation_split=0.1,
                          shuffle=False,  
                          validation_freq=1, callbacks=[cp_callback])

    ##################################
    history_dict = history_.history
    loss_tr = np.array(history_dict['loss'])
    loss_val = np.array(history_dict['val_loss'])
    mse_tr = np.array(history_dict['mse'])
    mse_val = np.array(history_dict['val_mse'])

    column_names = ['loss_tr', 'loss_val', 'mse_tr', 'mse_val']
    combined_array = np.column_stack([loss_tr, loss_val, mse_tr, mse_val])
    np.savetxt(out_dir_path + '/training_loss_mse.txt', combined_array, delimiter=',', header=','.join(column_names), comments='')

    print('training & validation loss, mse are saved!!', flush=True)
    print(' ')
    print('BNN predicted!!')
    print('Run completed !!', flush=True)
    tf.keras.backend.clear_session()
    import gc
    gc.collect()


### Serial 
if __name__ == '__main__':
    tasks = [(m, l) for m in range(4, 8) for l in range(1, 25)]

    for m, l in tasks:
        print(f"Running month={m}, lead={l}", flush=True)
        run_model(m, l)
        
"""        
### Parallel
if __name__ == '__main__':
    from multiprocessing import Pool
    
    # create tasks only for missing runs
    tasks = [(m, l) for m in range(0, 4) for l in range(1, 25)]# if (m, l) not in done]
    with Pool(processes=1) as p:
        p.starmap(run_model, tasks)
"""


"""
import ast
remaining = []
with open("test/remaining2.txt") as f:
    for line in f:
        remaining.append(ast.literal_eval(line.strip()))
"""
