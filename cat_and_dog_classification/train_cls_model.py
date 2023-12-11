import os
import json
import numpy as np

from marco import *
from network import *
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_cls_model(flag, hyp_params):
    '''
        args: 
            flag['IS_LOAD_MODEL']: define if load the model for further training
            flag['LOAD_MODEL_PATH']: define the path to load the best model
            flag['SAVE_MODEL_PATH']: define the path to save the best model
            flag['HIST_CALLBACK_PATH']: log the history at every epoch
            flag['HIST_PATH']: save the full history at the end of training
            flag['TRAIN_PATH']: define the training path
            flag['VALID_PATH']: define the validation path
            flag['CHK_PT_VERBOSE']: define checkpoint log verbose
            flag['TRAIN_VERBOSE']: define training verbose log verbose

            hyp_params['CLS_NAME']: define name of class
            hyp_params['IMG_SIZE']: define input image size
            hyp_params['EVAL_METRIC']: define evaluation metric
            hyp_params['EVAL_MODE']: define evalutation mode
            hyp_params['EPOCH']: define maximum training mode
            hyp_params['CLS_NUM']: define number of class
            hyp_params['BATCH']: define number of data in one batch
            hyp_params['LR']: define learning rate
            hyp_params['FC_DROPOUT']: define dropout ratio
            hyp_params['WORKER']: define number of cpu worker for handling image generator
            
        returns:
            None

    '''
    
    #====== define image generator ======#
    train_gen = ImageDataGenerator(
                            rescale = 1./255.,
                            horizontal_flip = True,
                            rotation_range = 20.,
                            zoom_range = 0.2
                            )
    valid_gen = ImageDataGenerator(rescale = 1./255.,)

    #====== define data iterator ======#
    train_iter = train_gen.flow_from_directory(
                                        flag['TRAIN_PATH'], 
                                        target_size=(hyp_params['IMG_SIZE'], 
                                                     hyp_params['IMG_SIZE']), 
                                        classes=hyp_params['CLS_NAME'],
                                        batch_size = hyp_params['BATCH'],
                                        shuffle= True
                                        )
    valid_iter = valid_gen.flow_from_directory(
                                        flag['VALID_PATH'], 
                                        target_size=(hyp_params['IMG_SIZE'],
                                                     hyp_params['IMG_SIZE']), 
                                        classes = hyp_params['CLS_NAME'],
                                        batch_size = hyp_params['BATCH'],
                                        shuffle= False
                                        )

    #====== define model ======#
    model = Network(cls_num = hyp_params["CLS_NUM"],
                    fc_dropout_ratio = hyp_params["FC_DROPOUT"],
                    is_train = True).obj_cls()

    if flag['IS_LOAD_MODEL']:
        try:
            model.load_weights(flag['LOAD_MODEL_PATH'])
        except:
            print("[WARNING] cannot load model")

    print(model.summary())

    #====== define optimizer ======#
    opt = tf.keras.optimizers.Adam(learning_rate=hyp_params['LR'], 
                                    beta_1=0.9, 
                                    beta_2=0.999, 
                                    epsilon=None, 
                                    amsgrad=False)

    #====== compile model ======#
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['acc'])

    #====== define model logger ======#
    model_log = ModelCheckpoint(
        filepath = flag['SAVE_MODEL_PATH'], 
        monitor  = hyp_params['EVAL_METRIC'], 
        verbose = flag['CHK_PT_VERBOSE'], 
        save_weights_only=True,
        save_best_only=True, 
        mode=hyp_params['EVAL_MODE'])
    
    #====== define history logger ======#
    history_logger = tf.keras.callbacks.CSVLogger(flag['HIST_CALLBACK_PATH'], 
                                                  separator=",", 
                                                  append=False)


    #====== start training ======#
    history = model.fit(
        train_iter, 
        validation_data = valid_iter,
        workers = hyp_params['WORKER'],
        epochs  = hyp_params['EPOCH'],
        callbacks = [model_log, history_logger],
        verbose=flag['TRAIN_VERBOSE'], 
    )

    #====== save history ======#
    np.save(flag['HIST_PATH'], history, allow_pickle=True)
