from custom_model.layers_keras import *
from custom_model.math_utils import *

import numpy as np
import tensorflow as tf
import math
from collections import defaultdict

from keras import Model
from keras.layers import Input, Dense, GRU, TimeDistributed, Lambda, Concatenate, RNN, dot
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras

def unstack(x):
    x = tf.unstack(x, axis=1)
    return x

def stack(x):
    y = K.stack(x, axis=1)
    return K.squeeze(y, axis=2)
    
#################################
def GCGRU_ScheduledSampling(obs_timesteps=29, pred_timesteps=4, nb_nodes=161, gru_units=64, k=4, gc='dgc'):
    input_obs = Input(shape=(obs_timesteps, nb_nodes, 1))

    encoder = GRU(gru_units, return_state=True)
    decoder = GRU(gru_units, return_sequences=True, return_state=True)
    readout = Dense(nb_nodes, activation='sigmoid')
    unstack_k = Lambda(unstack)
    choice = Scheduled()
    if gc == 'dgc':
        shared_1 = DynamicGraphConv(k)#, activation='sigmoid')
    elif gc == 'lcgc':
        shared_1 = LocallyConnectedGC(k)#, activation='sigmoid')
    elif gc == 'dgc_test':
        shared_1 = DGC_test(k)
    
    inner = TimeDistributed(shared_1)(input_obs)
    encoder_inputs = Lambda(lambda x: K.squeeze(x, axis = -1))(inner) # (None, 29, 208)
    encoder_outputs, state_h = encoder(encoder_inputs)

    input_gt = Input(shape=(pred_timesteps, nb_nodes, 1)) #(None, 4, 208, 1)
    unstacked = unstack_k(input_gt) # [(None, 208, 1) x 4] list
    
    initial = unstacked[0] #(None, 208, 1)
    inner = shared_1(initial) #(None, 208, 1)
    decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inner) #(None, 1, 208)
    decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
    
    state_h = state_h_new
    prediction = []
    decoded_results = readout(decoder_outputs_new)
    prediction.append(decoded_results)
    
    for i in range(1,pred_timesteps):
        decoder_inputs = choice([prediction[-1], unstacked[i]])#(None, 208, 1)
        decoder_inputs = shared_1(decoder_inputs)#(None, 208, 1)
        decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(decoder_inputs)#(None, 1, 208)
        decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
        state_h = state_h_new
        decoded_results = readout(decoder_outputs_new)
        prediction.append(decoded_results)
    
    outputs = Lambda(stack)(prediction)
    model = Model([input_obs, input_gt], outputs)
    return model

########################
def GRU_encoder_decoder(obs_timesteps=29, pred_timesteps=4, nb_nodes=161, gru_units=64):
    input_obs = Input(shape=(obs_timesteps, nb_nodes, 1))

    encoder = GRU(gru_units, return_state=True)
    decoder = GRU(gru_units, return_sequences=True, return_state=True)
    readout = TimeDistributed(Dense(nb_nodes, activation='hard_sigmoid'))
    unstack_k = Lambda(unstack)
    choice = Scheduled()
    
    encoder_inputs = Lambda(lambda x: K.squeeze(x, axis = -1))(input_obs) # (None, 29, 208)
    encoder_outputs, state_h = encoder(encoder_inputs)

    input_gt = Input(shape=(pred_timesteps, nb_nodes, 1)) #(None, 4, 208, 1)
    unstacked = unstack_k(input_gt) # [(None, 208, 1) x 4] list
    
    initial = unstacked[0] #(None, 208, 1)
    decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(initial) #(None, 1, 208)
    decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
    
    state_h = state_h_new
    prediction = []
    decoded_results = readout(decoder_outputs_new)
    prediction.append(decoded_results)
    
    for i in range(1,pred_timesteps):
        decoder_inputs = choice([prediction[-1], unstacked[i]])#(None, 208, 1)
        decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(decoder_inputs)#(None, 1, 208)
        decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
        state_h = state_h_new
        decoded_results = readout(decoder_outputs_new)
        prediction.append(decoded_results)
    
    outputs = Lambda(stack)(prediction)
    model = Model([input_obs, input_gt], outputs)
    return model

#######################
def create_embed_model(obs_timesteps=29, pred_timesteps=4, nb_nodes=161, k=1, dgc_mode='dgc', inner_act=None, model_name='dgcrnn'):
    if model_name == 'dgcgru':
        encoder = RNN(DGCRNNCell(k,dgc_mode=dgc_mode), return_state=True)
        decoder = RNN(DGCRNNCell(k,dgc_mode=dgc_mode), return_sequences=True, return_state=True)
    elif model_name == 'dfn':
        encoder = RNN(RecurrentDGC(k=k, units=64), return_sequences=True, return_state=True)
        decoder = RNN(RecurrentDGC(k=k, units=64), return_sequences=True, return_state=True)
           
    #readout = Dense(nb_nodes, activation='sigmoid')
    unstack_k = Lambda(unstack)
    choice = Scheduled()
    
    input_obs = Input(shape=(obs_timesteps, nb_nodes, 1)) 
    input_gt = Input(shape=(pred_timesteps, nb_nodes, 1)) #(None, T, N, 1)
    encoder_inputs = Lambda(lambda x: K.squeeze(x, axis = -1))(input_obs) # (None, T, N)
    
    encoder_outputs, state_h = encoder(encoder_inputs)
    #state_h = encoder(encoder_inputs)
    
    unstacked = unstack_k(input_gt) #[(None, N, 1) x T] list
    
    initial = unstacked[0] #(None, N, 1)
    decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(initial) #(None, 1, N)
    decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
    state_h = state_h_new
    prediction = []
    decoded_results = decoder_outputs_new
    #decoded_results = readout(decoder_outputs_new)
    prediction.append(decoded_results)
    
    if pred_timesteps > 1:       
        for i in range(1,pred_timesteps):
            decoder_inputs = choice([prediction[-1], unstacked[i]])#(None, 208, 1)
            decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(decoder_inputs)#(None, 1, 208)
            decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
            state_h = state_h_new
            decoded_results = decoder_outputs_new
            #decoded_results = readout(decoder_outputs_new)
            prediction.append(decoded_results)
    
    outputs = Lambda(stack)(prediction)
    model = Model([input_obs, input_gt], outputs)

    return model

def train_model(x_train, e_train, y_train, x_test, e_test, y_test, 
                OBS,PRED,k_max,lr=0.01,decay=1e-3,convergence=6, 
                EPOCH=256, BATCH=256, repeat=1, rf=0, model_name='dfn',dgc_mode='dgc'):
    MAE_avr=[]
    MAPE_avr=[]
    RMSE_avr=[]
    
    for j in range(repeat):
        print('repeat = ', j+1, end="\t")
        MAE_=[]
        MAPE_=[]
        RMSE_=[]
    
        for i in range(rf, k_max+1):
            print('k = ', i, end="\t")
            model = create_embed_model(obs_timesteps=OBS, pred_timesteps=PRED, nb_nodes=208,                                                                                      k=i,dgc_mode=dgc_mode,model_name=model_name)
            
            opt = keras.optimizers.Adam(lr=lr, decay=decay)
            model.compile(loss = 'mape',
                          optimizer=opt,
                          metrics=['mae',rmse])
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
                         ScheduledSampling(k=convergence)]
            history = model.fit([x_train,e_train], y_train,
                                 epochs=EPOCH,
                                 batch_size=BATCH,
                                 callbacks=callbacks,
                                 validation_split = 0.2)

            y = model.predict([x_test,e_test])

            Mae = MAP(y_test*120, y*120)
            Mape = MAPE(y_test*120, y*120)
            Rmse = RMSE(y_test*120, y*120)
            MAE_.append(Mae)
            MAPE_.append(Mape)
            RMSE_.append(Rmse)
            
        MAE_avr.append(MAE_)
        MAPE_avr.append(MAPE_)
        RMSE_avr.append(RMSE_)
        
    mae_avr = np.amin(np.array(MAE_avr), axis=0)
    mape_avr = np.amin(np.array(MAPE_avr), axis=0)
    rmse_avr = np.amin(np.array(RMSE_avr), axis=0)

    print('MAE:', mae_avr)
    print('MAPE:', mape_avr)
    print('RMSE:', rmse_avr)
    return MAE_avr, MAPE_avr, RMSE_avr
