'''
UPDATED ON 2023/04/03

1. aims at providing custom library of data preparation of classification and detection
'''

import os
import cv2
import copy
import json
import uuid
import math
import random
import numpy as np

from marco import *
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

def load_data_path(roots, target = 'train'):
    '''
    args:
        roots:  root of data
        target: which type of data (train or valid or test)
        
    returns:
        img_path: path of images 
        lab_path: path of labels
    '''
    
    img_path = []
    lab_path = []
    if target != 'train' and target != 'valid' and target != 'test':
        return img_path, lab_path
    
    for root in roots:
        sub_name = (root.split('/')[0]).split('_')[-1]

        if target == 'train':
            if sub_name == 'trainval':
                file_load = os.path.join(root, 'ImageSets', 'Main', 'train.txt')
            else:
                file_load = os.path.join(root, 'ImageSets', 'Main', 'test.txt')
        elif target == 'valid':
            file_load = os.path.join(root, 'ImageSets', 'Main', 'val.txt')
        elif target == 'test':
            file_load = os.path.join(root, 'ImageSets', 'Main', 'test.txt')
            
        f = open(file_load, 'r')
        lines = f.readlines()
        img_path += [os.path.join(root,  'JPEGImages', line.replace('\n','') + '.jpg') for line in lines]
        lab_path += [os.path.join(root, 'Annotations', line.replace('\n','') + '.xml') for line in lines]
        
    return img_path, lab_path 

def split_ids(data, train=0.9, valid=0.1, test=0, seed = 0):
    """
    args:
       data : list of data paths
       train: train split size (between 0 - 1)
       valid: valid split size (between 0 - 1)
       test : test split size (between 0 - 1)
       seed : random seed
        
    returns:
        train_set: list of training data paths
        valid_set: list of validation data paths
         test_set: list of testing data paths
    """
    
    if train + valid + test != 1:
        return 
    
    list_copy = list(range(0, len(data)))
    random.Random(seed).shuffle(list_copy)
    
    #obtain the size of training, validation and testing data
    train_size = math.floor(len(list_copy) * train)
    valid_size = math.floor(len(list_copy) * valid)
    test_size  = len(list_copy) - train_size - valid_size
    
    train_set = [None]*train_size
    if train + valid == 1.0:
        valid_size += test_size 
        valid_set = [None]*valid_size
        test_set  = None
        test_size = 0
    else:
        valid_set = [None]*valid_size
        test_set  = [None]*test_size
    
    #split the data into training, validation and testing dataset
    idx = 0
    for i, rand_ind in enumerate(list_copy):
        
        if i == train_size or i == train_size + valid_size:
            idx = 0
            
        if i < train_size:
            train_set[idx]= data[rand_ind]
        elif i >= train_size and i < train_size + valid_size:
            valid_set[idx]= data[rand_ind]
        else:
            test_set[idx] = data[rand_ind]
        idx += 1
        
    return train_set, valid_set, test_set


def split_VOC_data(img_path, lab_path, cls2ind, target_root = None, is_save = True):
    
    '''
    aim:
        gather and save VOC image and label into specific folder
    args:
        img_path: path of images
        lab_path: path of labels
        cls2ind: class to one hot vector index
        target_root: the root of saved data
        is_save: flag to decide if data is saved 
    return:
        None
    '''
    
    seed = 0
    rd = random.Random()
    rd.seed(0)
    
    if target_root is None or len(img_path) != len(lab_path):
        return

    cls_num = len(list(cls2ind.keys()))
    for img_load, lab_load in zip(img_path, lab_path):

        img = cv2.imread(img_load)

        with open(lab_load, 'r') as f:
            lab = f.read()
        lab = BeautifulSoup(lab, "xml")
        img_name = lab.find_all('filename')[0].text

        lab_obj = lab.find_all('object')

        boxes   = [None]*len(lab_obj)
        objs    = [None]*len(lab_obj)
        d_one_hots = [None]*len(lab_obj)
        c_one_hot = [0]*cls_num
        for j, obj in enumerate(lab_obj):
            x1 = float(obj.find_all('xmin')[0].text)
            y1 = float(obj.find_all('ymin')[0].text)
            x2 = float(obj.find_all('xmax')[0].text)
            y2 = float(obj.find_all('ymax')[0].text)

            obj = obj.find_all('name')[0].text
            obj_ind = cls2ind[obj]
            d_one_hot = [0]*cls_num
            d_one_hot[obj_ind] = 1

            boxes[j] = [x1, y1, x2, y2]
            objs[j]  = obj
            d_one_hots[j] = d_one_hot
            c_one_hot[obj_ind] = 1

        annotation = {}
        annotation['img_name'] = img_name
        annotation['num'] = len(lab_obj)
        annotation['box'] = boxes
        annotation['obj'] = objs
        annotation['d_one_hot'] = d_one_hots
        annotation['c_one_hot'] = c_one_hot

        if is_save:
            uid = uuid.UUID(int=rd.getrandbits(128))
            img_save = os.path.join(target_root, f'{uid}' + '.jpg')
            lab_save = os.path.join(target_root, f'{uid}' + '.json')
            cv2.imwrite(img_save, img)
            with open(lab_save, 'w') as f:
                json.dump(annotation, f)
                


def get_aug_pretrain_data(img_path, lab_path, transform, aug_num = 1, target_root = 'data/aug_train'):
    
    '''
    aim:
        convert VOC data format into YOLO format and other additional data augmentation
    args:
        img_path: path of images
        lab_path: path of labels
        transform: augmentation transform
        target_root: the root of saved data
    return:
        None
    '''
    
    img_path = sorted(img_path)
    lab_path = sorted(lab_path)
    
    for img_load, lab_load in zip(img_path, lab_path):

        img = cv2.cvtColor(cv2.imread(img_load), cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        
        lab      = json.load(open(lab_load, 'r'))  

        boxes  = lab['box']     
        obj_dict = {}
        for i, k in enumerate(lab['obj']):
            if k not in obj_dict:
                obj_dict[k] = []
            obj_dict[k].append([i, boxes[i]])
        
        uid = (img_load.split('.')[0]).split('/')[-1]
        
        for i, k in enumerate(obj_dict):
            obj_boxes = obj_dict[k]
            d_one_hot = lab['d_one_hot'][obj_boxes[0][0]]

            msk = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
            for _, box in obj_boxes:
                x1, y1 = round(max(box[0], 0)), round(max(box[1], 0))
                x2, y2 = round(min(box[2], img_w)), round(min(box[3], img_h))
                msk[y1:y2, x1:x2,:] = img[y1:y2, x1:x2,:].copy()
                        
            if transform is None:
                aug_num = 1   
            
            for j in range(aug_num): 
                msk_cpy = copy.deepcopy(msk)
                
                tlab = {}
                tlab['img_name']  = lab["img_name"]
                tlab['d_one_hot'] = d_one_hot
                tlab['obj']       = k
                
                if j != 0:
                    transformed = transform(image = msk_cpy)
                    msk_cpy = copy.deepcopy(transformed['image'])
                
                img_save = os.path.join(target_root, f'{uid}-{i}{j}' + '.jpg')
                lab_save = os.path.join(target_root, f'{uid}-{i}{j}' + '.json')
                cv2.imwrite(img_save, cv2.cvtColor(msk_cpy, cv2.COLOR_RGB2BGR))
                with open(lab_save, 'w') as f:
                    json.dump(tlab, f)
            