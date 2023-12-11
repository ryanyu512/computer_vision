import os
import cv2
import json
import copy
import random
import numpy as np
import albumentations as A

from marco import *
from network import *
from tqdm import tqdm
from utilities import *
from matplotlib import pyplot as plt

def get_img_list(dir, file_type = 'jpg'):
    '''
        args:
            dir: directories that contains targeted images
            file_type: can be .jpg or .png...
        
        returns:
            f: image paths of face data
    '''
    
    
    f  = os.listdir(dir)
    f  = [os.path.join(dir, _) for _ in f if _.split('.')[-1] == file_type]
    
    return f

def load_img(path):
    '''
        args:
            path: image paths
            
        returns:
            img: image
    '''
    
    
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img

def get_imgs(img_list, new_img_h = None, new_img_w = None):
    '''
        args:
            img_list: list of image paths
           new_img_h: scaled image height
           new_img_w: scaled image width
           
        returns:
                imgs: scaled and normalised images
    '''
    #get the list of img paths
    imgs = tf.data.Dataset.from_tensor_slices(img_list)
        
    #load img
    imgs = imgs.map(load_img)
    #resize image
    if new_img_h is not None or new_img_w is not None:
        imgs = imgs.map(lambda x: tf.image.resize(x, (new_img_h, new_img_w)))
    #normalize image
    imgs = imgs.map(lambda x: x/255.)

    return imgs

def get_label_list(img_list):
    '''
        args:
            img_list: list of image path
            
        returns:
            lab_list: list of label path
    '''
    
    N_img = len(img_list)
    lab_list = [None]*N_img
    for i in range(N_img):
        path = img_list[i]
        uid  = (path.split('.')[0]).split('/')[-1]
        root = (path.split('.')[0]).split('/')
        root = '/'.join(root[0:len(root) - 1])
        lab_list[i] = os.path.join(root, uid + '.json')

    return lab_list

def load_labels(label_path):
    '''
        args:
            label_path: data path of label
        
        return:
            label['c_one_hot']: one hot vector target for classification
            label['d_one_hot']: one hot vector target for detection
            label['box']: coordinates of boxes in yolo format 
    '''
    
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    
    return label['class'], label['boxes']

def get_labels(lab_list):
    '''
        args:
            lab_list: list of label path
        returns
            labs: labels
    '''
    
    labs = tf.data.Dataset.from_tensor_slices(lab_list)
    labs = labs.map(lambda x: tf.py_function(load_labels, 
                                             [x], 
                                             [tf.float32, tf.float32]
                                            )
                   )

    return labs

def combine_imgs_and_labels(imgs, labels, batch, pre_fetch = None, is_shuffle = False):
    '''
        args:
            imgs: images
            labels: labels
            batch: number of data in one batch 
            per_fetch: number of data to feteh before processing
        
        returns:
            data: return one batch of images and cooresponding labels
    '''
    data = tf.data.Dataset.zip((imgs, labels))
    #valid and test data do not need shuffle
    if is_shuffle:
        data = data.shuffle(batch*5)
        
    #since some images contain > 1 bounding box, requires 
    #padding for batch extraction
    padded_shape = ([None, None, 3], 
                    ([None, 2], 
                     [None, 4]))
    data = data.padded_batch(batch, 
                             padded_shapes = padded_shape)
    
    #tune the prefetch value dynamically at runtime
    #reduce GPU and CPU idle time
    if pre_fetch is None:
        pre_fetch = tf.data.experimental.AUTOTUNE
    data = data.prefetch(buffer_size = pre_fetch)
    
    return data

def compute_aug_img_and_label_mat(imgs, labs, c_num = 2, g_num = 7, transform = None):
     
    '''
    get augmented image and corresponding label matrix
    
    args:
        imgs: batchs of images (N_batch, height, width, channels)
        labs: [class, bounding box]
        transform: transformer
    returns:
        t_imgs: transformed images
        t_boxes: transformed boxes
        label_mat: label matrix for computing yolo loss
    '''
     
    t_imgs  = [None]*imgs.shape[0]
    t_boxes = [None]*imgs.shape[0]
    label_mat = np.zeros((labs[0].shape[0], 
                        g_num, 
                        g_num, 
                        c_num + 5))
    for i in range(imgs.shape[0]):
        
        #remove padded labels
        remain_ind = np.where(np.sum(labs[0][i], axis = 1) == 1)[0]
        cls = np.argmax(labs[0][i][remain_ind], axis = 1)
        box = labs[1][i][remain_ind]
        
        t_box = np.empty((0, 4), dtype = np.float32)
        if transform is not None:
            while len(t_box) <= 0:
                transformed = transform(image = imgs[i], bboxes = box, category_ids = cls)
                
                t_img = transformed['image']
                t_box = transformed['bboxes']
                t_cid = transformed['category_ids']
            
            t_box = np.array(t_box)*imgs.shape[1]
            t_imgs[i] = copy.deepcopy(t_img)
        else:
            t_box = np.array(box)*imgs.shape[1]
            t_cid = cls
            t_imgs[i] = cv2.resize(imgs[i], (imgs.shape[1], imgs.shape[2]), cv2.INTER_AREA)

        t_boxes[i] = t_box
        y_box = minmax2yolo(t_box, t_imgs[i].shape[1], t_imgs[i].shape[0])

        for j in range(len(y_box)):
            col_ind, row_ind = int(y_box[j][0]*g_num), int(y_box[j][1]*g_num)
            xc_cell, yc_cell = y_box[j][0]*g_num - col_ind, y_box[j][1]*g_num - row_ind
            w_cell, h_cell   = y_box[j][2], y_box[j][3]
            
            if label_mat[i, row_ind, col_ind, c_num] == 0:
                if np.sum(label_mat[i, row_ind, col_ind, 0:c_num]) > 1:
                    print("[label bug] label more than 1 class to be 1")
                
                label_mat[i, row_ind, col_ind, t_cid[j]] = 1
                label_mat[i, row_ind, col_ind, c_num] = 1
                label_mat[i, row_ind, col_ind, c_num+1:] = [xc_cell, yc_cell, w_cell, h_cell]
    
    return np.array(t_imgs), label_mat, np.array(t_boxes)

def vectorized_loss(res, lab, loss_w = [5., 1., 0.25, 1.], cls_num = 2, g_num = 7, b_num = 2):
    
    '''
        args:
            res: predictions from model
            lab: labels
            loss_w: weights of loss 
            cls_num: number of classes
            g_num: number of grid in one row/col
            b_num: number of boxes in one grid
            
        returns:
            loss: training loss in one batch
    '''
    
    #0, 1: [cat class, dog class]
    #2, 3,  4,  5,  6  = [conf, x1, y1, w, h]
    #7, 8,  9, 10, 11  = [conf, x1, y1, w, h]

    N_batch = res.shape[0]
    lab = tf.Variable(lab, dtype = tf.float32)
    
    #extract which grid in the image has object
    is_obj = lab[:, :, :, cls_num]
    is_obj = tf.cast(tf.expand_dims(is_obj, axis = 3), tf.float32)
    #reshape the prediction into grid*grid*(cls_num + 5*b_num)
    res = tf.reshape(res, (res.shape[0], g_num, g_num, (cls_num + 5*b_num)))
    
    #check iou 
    x_cell_ind = tf.repeat(
                           tf.expand_dims(
                                          tf.cast(
                                                  tf.range(g_num), 
                                                  tf.float32
                                                 ), 
                                           axis = 0
                                         ), 
                           repeats = g_num, 
                           axis = 0)
    y_cell_ind = tf.transpose(x_cell_ind)
    x_cell_ind = tf.repeat(tf.expand_dims(x_cell_ind, axis = 0), 
                           repeats = N_batch, axis = 0)
    y_cell_ind = tf.repeat(tf.expand_dims(y_cell_ind, axis = 0), 
                           repeats = N_batch, axis = 0)
    
    box0_x = tf.expand_dims(1/g_num*(res[:, :, :, cls_num + 1] + x_cell_ind), 3)
    box0_y = tf.expand_dims(1/g_num*(res[:, :, :, cls_num + 2] + y_cell_ind), 3)
    box0_w = tf.expand_dims(res[:,:,:,cls_num + 3], axis = 3)
    box0_h = tf.expand_dims(res[:,:,:,cls_num + 4], axis = 3)
    box0_p = tf.concat([box0_x, box0_y, box0_w, box0_h], axis = -1)
    
    box1_x = tf.expand_dims(1/g_num*(res[:, :, :, cls_num + 6] + x_cell_ind), 3)
    box1_y = tf.expand_dims(1/g_num*(res[:, :, :, cls_num + 7] + y_cell_ind), 3)
    box1_w = tf.expand_dims(res[:,:,:,cls_num + 8], axis = 3)
    box1_h = tf.expand_dims(res[:,:,:,cls_num + 9], axis = 3)
    box1_p = tf.concat([box1_x, box1_y, box1_w, box1_h], axis = -1)
    
    box_x = tf.expand_dims(1/g_num*(lab[:, :, :, cls_num + 1] + x_cell_ind), 3)
    box_y = tf.expand_dims(1/g_num*(lab[:, :, :, cls_num + 2] + y_cell_ind), 3)
    box_w = tf.expand_dims(lab[:,:,:,cls_num + 3], axis = 3)
    box_h = tf.expand_dims(lab[:,:,:,cls_num + 4], axis = 3)
    box_t = tf.concat([box_x, box_y, box_w, box_h], axis = -1)
    
    iou0 = tf.expand_dims(compute_IOU_by_union(box0_p , 
                                               box_t), 3)
    iou1 = tf.expand_dims(compute_IOU_by_union(box1_p, 
                                               box_t), 3)
    ious = tf.concat([iou0, iou1], axis = 3)

    #get the best box index
    best_box_ind = tf.math.argmax(ious, axis = -1)
    best_box_ind = tf.expand_dims(best_box_ind, 3)
    
    #determine which box in each grid is the best
    is_gd_box0 = tf.cast((1 - best_box_ind), tf.float32)
    is_gd_box1 = tf.cast(best_box_ind, tf.float32)

    #compute coordinate loss

    
    box_p = is_obj*(is_gd_box0*res[:,:,:,cls_num + 1:cls_num + 5] + \
                    is_gd_box1*res[:,:,:,cls_num + 6:cls_num + 10])

    box_t = is_obj*lab[:,:,:,cls_num + 1:cls_num + 5]
    
    #2: width, 3: height
    box_p_wh_sq = tf.math.sign(box_p[:,:,:,2:4])*tf.math.sqrt(tf.math.abs(box_p[:,:,:,2:4]))
    box_t_wh_sq = tf.sqrt(box_t[:,:,:,2:4])
    
    reg_loss = (box_t_wh_sq - box_p_wh_sq)**2 + (box_t[:,:,:,0:2] - box_p[:,:,:,0:2])**2

    reg_loss = tf.reduce_sum(reg_loss, axis = (1, 2, 3))
    reg_loss = tf.reduce_mean(reg_loss, axis = 0)
    
    #compute obj conf loss
    #2: obj conf. of box 1
    #7: obj conf. of box 2
    obj_p = is_obj*(is_gd_box0*tf.expand_dims(res[:,:,:,cls_num], 3) + \
                    is_gd_box1*tf.expand_dims(res[:,:,:,cls_num + 5], 3))
    obj_t = is_obj*tf.expand_dims(lab[:,:,:,cls_num], 3) 
    
    obj_loss = (obj_t - obj_p)**2 
    obj_loss = tf.reduce_sum(obj_loss, axis = (1, 2, 3)) 
    obj_loss = tf.reduce_mean(obj_loss, axis = 0)
    
    #compute no obj conf loss 
    no_obj_p0 = (1 - is_obj)*tf.expand_dims(res[:,:,:,cls_num], 3) 
    no_obj_p1 = (1 - is_obj)*tf.expand_dims(res[:,:,:,cls_num + 5], 3) 
    no_obj_t  = (1 - is_obj)*tf.expand_dims(lab[:,:,:,cls_num], 3) 

    no_obj_loss = (no_obj_t - no_obj_p0)**2 + (no_obj_t - no_obj_p1)**2
    no_obj_loss = tf.reduce_sum(no_obj_loss, axis = (1, 2, 3))
    no_obj_loss = tf.reduce_mean(no_obj_loss, axis = 0)

    #compute class loss
    cls_p = is_obj*res[:,:,:,0:cls_num]
    cls_t = is_obj*lab[:,:,:,0:cls_num]

    cls_loss = (cls_t - cls_p)**2
    cls_loss = tf.reduce_sum(cls_loss, axis = (1, 2, 3))
    cls_loss = tf.reduce_mean(cls_loss, axis = 0)

    #compute total loss
    loss = reg_loss*loss_w[0] + obj_loss*loss_w[1] + no_obj_loss*loss_w[2] + cls_loss*loss_w[3]
       
    return loss

def verify_loss(res, lab, loss_w = [5., 1., 0.25, 1.], cls_num = 2, g_num = 7, b_num = 2):
    
    '''
        this function is the for - loop version of yolo loss

        args:
            res: predictions from model
            lab: labels
            loss_w: weights of loss 
            cls_num: number of classes
            g_num: number of grid in one row/col
            b_num: number of boxes in one grid
            
        returns:
            loss: training loss in one batch
    '''
    
    N_batch = res.shape[0]
    
    lab = tf.Variable(lab, dtype = tf.float32)
    
    #reshape the prediction into grid*grid*(cls_num + 5*b_num)
    res = tf.reshape(res, (res.shape[0], g_num, g_num, (cls_num + 5*b_num)))
    
    #0, 1: [cat class, dog class]
    #2, 3,  4,  5,  6  = [conf, x1, y1, w, h]
    #7, 8,  9, 10, 11  = [conf, x1, y1, w, h]
    err_w = 0
    err_h = 0
    err_x = 0
    err_y = 0
    cls_loss = 0
    obj_loss = 0
    no_obj_loss = 0
    
    nobj_cnt = 0
    obj_cnt = 0
    for i in range(N_batch):
        for j in range(g_num):
            for k in range(g_num):
                if lab[i, j, k, cls_num] == 1:
                    
                    box0_p = res[i, j, k, cls_num + 1: cls_num + 5].numpy()
                    box0_p[0], box0_p[1] = (box0_p[0] + k)/g_num, (box0_p[1] + j)/g_num
                    box1_p = res[i, j, k, cls_num + 6: cls_num + 10].numpy()
                    box1_p[0], box1_p[1] = (box1_p[0] + k)/g_num, (box1_p[1] + j)/g_num
                    
                    box_t  = lab[i, j, k, cls_num + 1: cls_num + 5].numpy()
                    box_t[0], box_t[1] = (box_t[0] + k)/g_num, (box_t[1] + j)/g_num
                    
                    iou0 = compute_IOU_by_union(np.expand_dims(box0_p,0), 
                                                np.expand_dims(box_t,0))
                    iou1 = compute_IOU_by_union(np.expand_dims(box1_p,0), 
                                                np.expand_dims(box_t,0))
                    
                    #print("======= pred info at ground true cell =======")
                    #print('t_obj_ind: ', j, k)
                    #print('cls: ' , res[i,j,k,:cls_num])
                    #print('iou0: ', iou0.numpy(), 'iou1:', iou1.numpy())
                    #print('box1: ', res[i,j,k,cls_num:cls_num + 5])
                    #print('box2: ', res[i,j,k,cls_num + 5:cls_num + 10])
                    #print("======= pred info at ground true cell =======")
                    
                    if iou0 >= iou1:
                        w_sq = tf.math.sign(res[i,j,k,cls_num + 3])*tf.math.sqrt(tf.math.abs(res[i,j,k,cls_num + 3]))
                        h_sq = tf.math.sign(res[i,j,k,cls_num + 4])*tf.math.sqrt(tf.math.abs(res[i,j,k,cls_num + 4]))
                        xc   = res[i,j,k,cls_num + 1]
                        yc   = res[i,j,k,cls_num + 2]
                        conf = res[i,j,k,cls_num]
                    else:
                        w_sq = tf.math.sign(res[i,j,k,cls_num + 8])*tf.math.sqrt(tf.math.abs(res[i,j,k,cls_num + 8]))
                        h_sq = tf.math.sign(res[i,j,k,cls_num + 9])*tf.math.sqrt(tf.math.abs(res[i,j,k,cls_num + 9]))
                        xc   = res[i,j,k,cls_num + 6]
                        yc   = res[i,j,k,cls_num + 7]
                        conf = res[i,j,k,cls_num + 5]
                    
                    wt_sq = tf.math.sqrt(lab[i,j,k,cls_num + 3])
                    ht_sq = tf.math.sqrt(lab[i,j,k,cls_num + 4])
                    xct   = lab[i,j,k,cls_num + 1]
                    yct   = lab[i,j,k,cls_num + 2]

                    err_w += (wt_sq - w_sq)**2
                    err_h += (ht_sq - h_sq)**2
                    err_x += (xct - xc)**2
                    err_y += (yct - yc)**2
                    
                    #print("======= true info at ground true cell =======")
                    #print('cls0: ', lab[i,j,k,0].numpy(), 'cls1: ', lab[i,j,k,1].numpy(), 'is_obj:', lab[i,j,k,2].numpy())
                    #print("======= true info at ground true cell =======")
                    cls_loss += ((lab[i,j,k,0] - res[i,j,k,0])**2 + (lab[i,j,k,1] - res[i,j,k,1])**2)
                    obj_loss += (lab[i,j,k,2] - conf)**2
                    obj_cnt += 1.
                else:
                    no_obj_loss += ((res[i,j,k,cls_num])**2 + (res[i,j,k,cls_num + 5])**2)
                    nobj_cnt += 1.
    reg_loss = err_w + err_h + err_x + err_y
    err = reg_loss*loss_w[0] + obj_loss*loss_w[1] + no_obj_loss*loss_w[2] + cls_loss*loss_w[3] 
    err = err/(N_batch + 1e-16)
    
    #print('reg_loss: ', reg_loss)
    #print('obj_loss: ', obj_loss)
    #print('no_obj_loss: ', no_obj_loss)
    #print('cls_loss: ', cls_loss)

    return err

class TrainModel(Model): 
    def __init__(self, model,  **kwargs): 
        super().__init__(**kwargs)
        self.model = model
        
    def compile(self, opt, train_loss, eval_loss, cls_num, g_num, b_num, transform, **kwargs):
        super().compile(**kwargs)
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.opt = opt
        self.cls_num = cls_num
        self.b_num = b_num
        self.g_num = g_num
        self.transform = transform 
        
    def train_step(self, batch, **kwargs): 
        
        imgs, labs = batch
        
        t_imgs, label_mat, _ = compute_aug_img_and_label_mat(imgs = imgs, 
                                                             labs = labs, 
                                                             c_num = self.cls_num, 
                                                             g_num = self.g_num, 
                                                             transform = self.transform)
        
        with tf.GradientTape() as tape: 
            
            res = self.model(t_imgs, training = True)
            
            batch_loss = self.train_loss(res, label_mat,
                                         cls_num = self.cls_num)
            
            grad = tape.gradient(batch_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"batch_loss": batch_loss}
    
    def eval_step(self, batch, **kwargs): 
        
        imgs, labs = batch
        
        t_imgs, label_mat, _ = compute_aug_img_and_label_mat(imgs = imgs, 
                                                             labs = labs, 
                                                             c_num = self.cls_num, 
                                                             g_num = self.g_num, 
                                                             transform = None)
        
        res = self.model(t_imgs, training = False)

        batch_loss = self.eval_loss(res, 
                                    label_mat,
                                    cls_num = self.cls_num)
        
        return {"batch_loss": batch_loss}

def train_model(data_folders, flag, h_params):
    '''
        args:
        #======== define flag ========#
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
 
        #======== define hyper-parameters ========#
        #number of training epoch
        h_params['EPOCH'] = 200
        #number of learning rate
        h_params['LR'] = 1e-4
        #number of class
        h_params['CLS_NUM'] = 2
        #number of boxes in one grid cell
        h_params['BOX_NUM'] = 2
        #number of grids in one row/col
        h_params['GRID_NUM'] = 7
        #number of data in one batch
        h_params['BATCH'] = BATCH
        #number of data to be prefetched
        h_params['PREFETCH'] = BATCH*2
        #input image size of detector
        h_params['DETECT_IMG_SIZE'] = DETECT_IMG_SIZE
        #dropout ratio of fully connected layer
        h_params['FC_DROPOUT_RATIO'] = 0.5

        returns:
            None
    '''
    
    curr_model_name = 'cur_det_model.h5'
    best_model_name = 'best_det_model.h5'
    traf_model_name = 'best_cls_model.h5'
    hist_file_name  = 'loss_history.npy'
    
    net_model = Det_Network(cls_num = h_params["CLS_NUM"], 
                            fc_dropout_ratio = h_params["FC_DROPOUT_RATIO"],
                            is_train = True, 
                            ).obj_det()
    if flag['IS_LOAD_MODEL']:
        net_model.load_weights('det_model/' + best_model_name)
    if flag['IS_TRANSFER']:
        net_model.load_weights('cls_model/' + traf_model_name, by_name = True)
        
    print("network summary:")
    print(net_model.summary())
    ####### Get img path of training, validation and testing data #######

    train_img_list = []
    valid_img_list = []
    
    if flag['IS_TEST']:
        test_img_list  = []
    for i, data_folder in enumerate(data_folders):
        train_img_path = get_img_list(os.path.join(data_folder, 'train'))
        valid_img_path = get_img_list(os.path.join(data_folder, 'valid'))
        if flag['IS_TEST']:
            test_img_path = get_img_list(os.path.join(data_folder, 'test'))

        train_img_list = train_img_list + train_img_path
        valid_img_list = valid_img_list + valid_img_path
        if flag['IS_TEST']:
            test_img_list  = test_img_list  + test_img_path
        
    ####### Get training, validation and testing image data #######
    
    train_img_list = random.sample(train_img_list, len(train_img_list))
    train_images = get_imgs(train_img_list, 
                            h_params['DETECT_IMG_SIZE'], 
                            h_params['DETECT_IMG_SIZE'])
    valid_images = get_imgs(valid_img_list, 
                            h_params['DETECT_IMG_SIZE'], 
                            h_params['DETECT_IMG_SIZE'])
    if flag['IS_TEST']:
        test_images  = get_imgs(test_img_list, 
                                h_params['DETECT_IMG_SIZE'], 
                                h_params['DETECT_IMG_SIZE'])

    ####### Get training, validation and testing labels #######
    train_lab_list = get_label_list(train_img_list)
    valid_lab_list = get_label_list(valid_img_list)
    if flag['IS_TEST']:
        test_lab_list  = get_label_list(test_img_list)

    train_labels = get_labels(train_lab_list)
    valid_labels = get_labels(valid_lab_list)
    if flag['IS_TEST']:
        test_labels  = get_labels(test_lab_list)

    ####### Combine labels and images #######
    train = combine_imgs_and_labels(train_images, 
                                    train_labels, 
                                    h_params['BATCH'], 
                                    h_params['PREFETCH'], 
                                    is_shuffle = True)
    valid = combine_imgs_and_labels(valid_images, 
                                    valid_labels, 
                                    h_params['BATCH'], 
                                    h_params['PREFETCH'], 
                                    is_shuffle = False)
    if flag['IS_TEST']:
        test  = combine_imgs_and_labels(test_images, 
                                        test_labels, 
                                        h_params['BATCH'], 
                                        h_params['PREFETCH'], 
                                        is_shuffle = False)

    batches_train = len(train)
    batches_valid = len(valid)
    if flag['IS_TEST']:
        batches_test  = len(test)

    print('train batchs:', batches_train)
    print('valid batchs:', batches_valid)
    if flag['IS_TEST']:
        print('test batchs:', batches_test)

    ####### Define optimizer and loss #######

    opt = tf.keras.optimizers.Adam(learning_rate = h_params['LR'])

    ####### Training #######
    if flag['IS_TRAIN']:
            
        #define transform for data augmentation
        transform = A.Compose([
                                A.HorizontalFlip(),
                                A.VerticalFlip(),
                                A.RandomBrightnessContrast(),
                                A.HueSaturationValue(hue_shift_limit = 0.0, 
                                                     sat_shift_limit = [0., 0.2],
                                                     val_shift_limit = 0.0),
                                A.Affine(scale  = [0.8,  1], 
                                         translate_percent = [-0.2, 0.2],
                                        p = 0.5),
                            ], bbox_params=A.BboxParams(format='albumentations', 
                                                        min_visibility=0.5, 
                                                        label_fields=['category_ids']))
        
        hist = {'avg_train_loss': [], 
                'avg_valid_loss': [],
                'best_valid_loss': None}
        
        if flag['IS_LOAD_HIST']:
            hist = np.load(hist_file_name, allow_pickle=True).flat[0]
            best_valid_loss = hist['best_valid_loss']
        else:
            best_valid_loss = None
            
        model = TrainModel(net_model)
        model.compile(opt = opt, 
                    train_loss = vectorized_loss,
                    eval_loss  = vectorized_loss,
                    cls_num = h_params["CLS_NUM"],
                    g_num   = h_params["GRID_NUM"],
                    b_num   = h_params["BOX_NUM"],
                    transform = transform)
        
        for epoch in range(h_params['EPOCH']):
            
            train_iter = train.as_numpy_iterator()
            valid_iter = valid.as_numpy_iterator()

            avg_train_loss = 0.0
            avg_valid_loss = 0.0
            
            for i in tqdm(range(batches_train), leave = True):
                train_loss = model.train_step(train_iter.next())
                avg_train_loss += train_loss['batch_loss']

            avg_train_loss /= batches_train  
            
            for i in tqdm(range(batches_valid), leave = True):
                val_loss = model.eval_step(valid_iter.next())
                avg_valid_loss += val_loss['batch_loss']

            avg_valid_loss /= batches_valid  
            
            if flag['IS_SAVE']:
                hist['avg_train_loss'].append(avg_train_loss)
                hist['avg_valid_loss'].append(avg_valid_loss)
            
            if best_valid_loss is None:
                best_valid_loss = avg_valid_loss
                if flag['IS_SAVE']:
                    net_model.save(best_model_name)
                    print("save the best model!")
            else:
                if best_valid_loss > avg_valid_loss:
                    best_valid_loss = avg_valid_loss
                    hist['best_valid_loss'] = best_valid_loss
                    if flag['IS_SAVE']:
                        net_model.save(best_model_name)
                        print("save the best model!")
                    
            if flag['IS_SAVE']:
                net_model.save(curr_model_name)
                np.save(hist_file_name, hist, allow_pickle=True)

            print(f'epoch {epoch + 1}')
            print(f'avg_train_loss: {avg_train_loss}')
            print(f'avg_valid_loss: {avg_valid_loss}')
            print(f'best_valid_loss:  {best_valid_loss}')
            
        if flag['IS_PLOT']:
            plt.plot(hist['avg_train_loss'], 
                    'r', 
                    label = 'training_loss')
            plt.plot(hist['avg_valid_loss'], 
                    'g', 
                    label = 'validation_loss')
            plt.ylabel('Loss')
            plt.xlabel('Number of Epoch')
            plt.legend()
            plt.show()

