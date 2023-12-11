'''
    Updated on 2023/04/03
    
    1. aim at storing less - frequently changed parameters
'''

CLS2IND = {'cat': 0, 
           'dog': 1}

IND2CLS = {0: 'cat', 
           1: 'dog'}

#define number of grids in one row/col
GRID_NUM = 7
#define number of class
CLS_NUM = 2
#define number of boxes
BOX_NUM = 2

#input image size for detection
DETECT_IMG_SIZE = 448
#number of data in one batch
BATCH = 16
#learning rate
LR  = 1e-4
