from marco import *
from train_model import *

#======== define flag ========#
flag = {}
#define if testing the data (TODO: do not have testing function yet)
flag['IS_TEST']  = False
#define if the model is trained 
flag['IS_TRAIN'] = True
#define if loss history is plotted after training
flag['IS_PLOT']  = False
#define if model is saved during training
flag['IS_SAVE']  = False
#define if trained model is loaded for further training
flag['IS_LOAD_MODEL'] = False
#define if previous loss history is loaded for further record
flag['IS_LOAD_HIST']  = False
#define if pretrained - backbone model is loaded
flag['IS_TRANSFER']   = True
for k in flag:
    print(f'{k}: {flag[k]}')

#======== define hyper-parameters ========#
h_params = {}
#number of training epoch
h_params['EPOCH'] = 200
#number of learning rate
h_params['LR'] = LR
#number of class
h_params['CLS_NUM'] = CLS_NUM
#number of boxes in one grid cell
h_params['BOX_NUM'] = BOX_NUM
#number of grids in one row/col
h_params['GRID_NUM'] = GRID_NUM
#number of data in one batch
h_params['BATCH'] = BATCH
#number of data to be prefetched
h_params['PREFETCH'] = BATCH*2
#input image size of detector
h_params['DETECT_IMG_SIZE'] = DETECT_IMG_SIZE
#dropout ratio of fully connected layer
h_params['FC_DROPOUT_RATIO'] = 0.5
for k in h_params:
    print(f'{k}: {h_params[k]}')

DATA_FOLDERS = ['cat_and_dog_det_data']

#======== start training ========#
train_model(DATA_FOLDERS, flag, h_params)
