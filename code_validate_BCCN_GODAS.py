###  code to train BNN model using SODA monthly data## 
print("Pgrm started!!")

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
import random
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.config.experimental.enable_op_determinism()

#################### Taking inputs (months, lags,epiches,batch size)  ####################
# Check if three arguments (script name, prediction month, and lead months) are provided


# Get the input values for prediction month  and lead month from command line arguments

def run_model(mon_,lag_mon):
    
    Calender_month=('January,February,March,April,May,June,July,August,September,October,November,December').split(',')
    month_=Calender_month[mon_]


    print('month=',month_)
    print('lag= '+str(lag_mon))

    out_dir_path='/home/users/roms/balaji/aghil/nino_seed_version/nino/3_input/300m/GODAS_direct_CMIP251/'+month_+'/lead'+str(lag_mon)
    checkpoint_path='/home/users/roms/balaji/aghil/nino_seed_version/nino/3_input/300m/out_final/'+month_+'/lead'+str(lag_mon)+'/bay_cnn.ckpt'

    import os
    import shutil

    def create_directory(directory_path):
        # Check if the directory already exists
        if os.path.exists(directory_path):
            # If it exists, remove it and its contents
            shutil.rmtree(directory_path)
            print("Directory "+out_dir_path+" already exist !!")

        # Create the directory
        os.makedirs(directory_path)
    create_directory(out_dir_path)
    #os.makedirs(out_dir_path)
    print("Directory "+out_dir_path+" created successfully")
    print(" ")
    print(" ")
    ###################  Loading ML modules ##############

    from tensorflow import keras
    import tensorflow as tf
    import tensorflow_probability as tfp
    dist = tfp.distributions
    tfd = tfp.distributions

    print("!!!!!!!!!")
    print("!!!")

    print("AI/ML Modules loaded!!")


    ####################  Loading GODAS data ############################
    ds =xr.open_dataset('/home/users/roms/balaji/athul/enso/data/godas/GODAS_1980_2023_global_monthly_2.5_res.nc')

    ## In this time slice  first 3 year and last 1 year will miss due to lead time checking form 1-24 
    # so if slice 1s ('1988','2021') >> model will predict from 1991-2020 ( ie 1998,1989,1990, 2021 will miss)
    t5m_sst = ds.t5m_sst.sel(time=slice('1980','2022')).sel(lat=slice(-55,60))
    t_5m_300_avg = ds.t_5m_300_avg.sel(time=slice('1980','2022')).sel(lat=slice(-55,60))

    print('NCEP GODAS data loaded')
    ##################################

    ## Load my functions
    print('loading our own functions!!!!!')
    from function_test_s import detrend_data_
    from function_test_s import normalize_data
    from function_test_s import lag_data
    from function_test_s import modify_soda_



    ssta_tau_0=[];ssta_tau_1=[];ssta_tau_2=[]
    ohca_tau_0=[];ohca_tau_1=[];ohca_tau_2=[]
    nino_lead=[]


    ssta_tau_0,ssta_tau_1,ssta_tau_2,ohca_tau_0,ohca_tau_1,ohca_tau_2,nino_lead=modify_soda_(t5m_sst,t_5m_300_avg,predict_mon=mon_,lag_=lag_mon)

    im_=np.empty([ssta_tau_0.shape[0],ssta_tau_0.shape[1],ssta_tau_0.shape[2],6])

    nan_=-999
    im_[:,:,:,0]=ssta_tau_0.fillna(nan_) # tau ## [1:] for removing one year ON should be last yr 
    im_[:,:,:,1]=ssta_tau_1.fillna(nan_) #tau-1
    im_[:,:,:,2]=ssta_tau_2.fillna(nan_) # tau-2

    im_[:,:,:,3]=ohca_tau_0.fillna(nan_) #tau
    im_[:,:,:,4]=ohca_tau_1.fillna(nan_) # tau-1
    im_[:,:,:,5]=ohca_tau_2.fillna(nan_)  #tau-2


    y_test_=nino_lead.values 
    x_test_=im_
    print("")
    print("input data for BCNN is ready!!")

    def create_model(len_sample,input_shape):
        
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape,name="basket", dtype='float32'),

            
            tfp.layers.Convolution2DFlipout(32, kernel_size=5, strides=(1,1), data_format="channels_last",  # i changed 16 >> 32
                                            padding="same", activation=tf.nn.tanh, name="conv_tfp_1a", 
                                            kernel_divergence_fn=kl_divergence_function,seed=53),
            tf.keras.layers.MaxPool2D(strides=(4,4), pool_size=(4,4), padding="same"), #strides=(4,4), pool_size=(4,4)
            
            tfp.layers.Convolution2DFlipout(64, kernel_size=3, strides=(1,1), data_format="channels_last",  ## 32 to 64
                                            padding="same", activation=tf.nn.tanh, name="conv_tfp_1b", 
                                            kernel_divergence_fn=kl_divergence_function,seed=53),
            tf.keras.layers.MaxPool2D(strides=(4,4), pool_size=(4,4), padding="same"),

            tf.keras.layers.Flatten(),
            #tfp.layers.DenseFlipout(128, activation=tf.nn.relu,kernel_divergence_fn=kl_divergence_function), # this layer makes wrong simulation
            #model.add(Dropout(0.3))
            tfp.layers.DenseFlipout(1,kernel_divergence_fn=kl_divergence_function,seed=53) # while adding acti function =softmax driving wrong simulation here
        ])
        

        learning_rate = 1.0e-3 ##
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate),
                      metrics=['mse'])

        return model

    ##############


    tf.keras.backend.clear_session() 
    kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(len(x_test_), dtype='float32')
    model_2=create_model(len_sample=len(x_test_),input_shape=x_test_[0].shape)
    #model_2.summary()
    model_2.load_weights(checkpoint_path)
    #model_2.summary()
    print('loaded BCNN  model & weights taken from CMIP trined BNN')
    ###############################
    print("")
    print("BCNN prediction started !!")
    enso_pred_=[]
    enso_pred_.append([model_2.predict(x_test_) for k in range(200)])
    print("")
    print("")
    print("BCNN predicted !!")
    vv=np.squeeze(enso_pred_)

    ###### saving outputs to ncfile 
    print('saving outputs.....')

    pred_nino = xr.DataArray(vv, coords=[np.arange(1,201),nino_lead.time], dims=['sample','time'])
    nino_obs = xr.DataArray(y_test_, coords=[nino_lead.time], dims=['time'])
    lead_time = xr.DataArray(np.arange(0,len(ssta_tau_0.time)), coords=[ssta_tau_0.time], dims=['time_lead'])



    dummy_ = xr.Dataset()
    dummy_['pred_nino'] = pred_nino
    dummy_['obs_nino'] = nino_obs
    dummy_['lead_time'] = lead_time


    dummy_.to_netcdf(out_dir_path+'/Bayesian_pred_for_all_'+month_+'_with_lead='+str(lag_mon)+'_1984_2022.nc', mode='w', format='NETCDF4',engine="netcdf4")
    print('output saved !!!')
    print("")
    print('prgm end !!!!!!!!!!!!!!!!################!!!!!!!!!!!!!!!!!')
"""  
if __name__ == '__main__':
    from multiprocessing import Pool
    # create tasks only for missing runs
    #tasks = [(0, 3)]
    tasks = [(m, l) for m in range(0, 12) for l in range(1, 25)]# if (m, l) not in done]
    #tasks = remaining[0:12]
    with Pool(processes=24) as p:
        p.starmap(run_model, tasks)
        
"""
if __name__ == '__main__':

    # Create all (month, lead) combinations
    tasks = [(m, l) for m in range(0, 12) for l in range(1, 25)]

    # Run sequentially (GPU safe)
    for m, l in tasks:
        print(f"\nRunning month={m}, lead={l}")
        run_model(m, l)

    print("\nAll runs completed successfully.")

