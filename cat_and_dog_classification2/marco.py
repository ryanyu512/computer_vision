'''
    Updated on 2023/04/03
    
    1. aim at defining the architechure of yolo_v1
'''

'''
CLS2IND = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}

IND2CLS = {
    0: 'person',
    1: 'bird',
    2: 'cat',
    3: 'cow',
    4: 'dog',
    5: 'horse',
    6: 'sheep',
    7: 'aeroplane',
    8: 'bicycle',
    9: 'boat',
    10: 'bus',
    11: 'car',
    12: 'motorbike',
    13: 'train',
    14: 'bottle',
    15: 'chair',
    16: 'diningtable',
    17: 'pottedplant',
    18: 'sofa',
    19: 'tvmonitor'
}
'''


CLS2IND = {'cat': 0, 
           'dog': 1}

IND2CLS = {0: 'cat', 
           1: 'dog'}

#Augmentation parameters
CLS_NUM = 20
#number of augmented data per image
AUG_NUM = 5 #set to be 5 due to the limitation of hardware config

#training epoch
EPOCH = 2000
#input image size for detection
CLS_IMG_SIZE = 224
#number of data in one batch
BATCH = 32
#learning rate
LR  = 1e-4
