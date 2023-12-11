'''
LAST UPDATED ON 2023/04/21

1. aims at training classification model
'''


import os
import numpy as np
from marco import *
from train_cls_model import *

#====== define flag ======#ß
flag = {}
#define if load the model for further training
flag['IS_LOAD_MODEL'] = False 
#define the path to load the best model
flag['LOAD_MODEL_PATH']    = 'best_model/best_weights.h5'
#define the path to save the best model
flag['SAVE_MODEL_PATH']    = 'best_weights.h5'
#log the history at every epoch
flag['HIST_CALLBACK_PATH'] = 'hist_callback.csv'
#save the full history at the end of training
flag['HIST_PATH']     = 'model_history.npy'
#define the training path
flag['TRAIN_PATH']    = 'dog_cat_cls_data/train'
#define the validation path
flag['VALID_PATH']    = 'dog_cat_cls_data/valid'
#define checkpoint log verbose
flag['CHK_PT_VERBOSE']  = 1
#define training verbose log verbose
flag['TRAIN_VERBOSE']   = 1

for k in flag:
    print(f'[{k}]: ', flag[k])

#====== define hyper-parameters ======#
hyp_params  = {}
#define name of class
hyp_params['CLS_NAME'] = ('cat','dog')
#define input image size
hyp_params['IMG_SIZE'] = CLS_IMG_SIZE
#define evaluation metric
hyp_params['EVAL_METRIC'] = 'val_acc'
#define evalutation mode
hyp_params['EVAL_MODE'] = 'max'
#define maximum training mode
hyp_params['EPOCH']    = 150
#define number of class
hyp_params['CLS_NUM']  = 2
#define number of data in one batch
hyp_params['BATCH']    = BATCH
#define learning rate
hyp_params['LR']       = LR
#define dropout ratio
hyp_params['FC_DROPOUT'] = 0.35
#define number of cpu worker for handling image generator
hyp_params['WORKER']   = 10

for k in hyp_params:
    print(f'[{k}]: ', hyp_params[k])
    
#====== train model ======#
train_cls_model(flag, hyp_params)