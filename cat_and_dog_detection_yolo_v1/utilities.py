import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def compute_IOU_by_union(reg0, reg1):
    
    '''
        compute IOU among boxes
    
        args:
            reg0, reg1: coordinates of bounding box 1 and 2 
                        (in yolo format)
        returns:
            overlap: overlap percentage of two bounding box
    '''

    if len(reg0.shape) == 4:
        reg0_x1, reg0_y1 = reg0[:,:,:,0] - reg0[:,:,:,2]*0.5, reg0[:,:,:,1] - reg0[:,:,:,3]*0.5
        reg1_x1, reg1_y1 = reg1[:,:,:,0] - reg1[:,:,:,2]*0.5, reg1[:,:,:,1] - reg1[:,:,:,3]*0.5
        
        reg0_x2, reg0_y2 = reg0[:,:,:,0] + reg0[:,:,:,2]*0.5, reg0[:,:,:,1] + reg0[:,:,:,3]*0.5
        reg1_x2, reg1_y2 = reg1[:,:,:,0] + reg1[:,:,:,2]*0.5, reg1[:,:,:,1] + reg1[:,:,:,3]*0.5
        
        w0, h0 = reg0[:,:,:,2], reg0[:,:,:,3]
        w1, h1 = reg1[:,:,:,2], reg1[:,:,:,3]
    else:
        reg0_x1, reg0_y1 = reg0[:,0] - reg0[:,2]*0.5, reg0[:,1] - reg0[:,3]*0.5
        reg1_x1, reg1_y1 = reg1[:,0] - reg1[:,2]*0.5, reg1[:,1] - reg1[:,3]*0.5
        
        reg0_x2, reg0_y2 = reg0[:,0] + reg0[:,2]*0.5, reg0[:,1] + reg0[:,3]*0.5
        reg1_x2, reg1_y2 = reg1[:,0] + reg1[:,2]*0.5, reg1[:,1] + reg1[:,3]*0.5
        
        w0, h0 = reg0[:,2], reg0[:,3]
        w1, h1 = reg1[:,2], reg1[:,3]
            
    area_p = w0*h0
    area_t = w1*h1
    
    inter_x1 = tf.maximum(reg0_x1, reg1_x1)
    inter_y1 = tf.maximum(reg0_y1, reg1_y1)
    inter_x2 = tf.minimum(reg0_x2, reg1_x2)
    inter_y2 = tf.minimum(reg0_y2, reg1_y2)
    
    inter_a = tf.maximum(0, inter_x2 - inter_x1)*tf.maximum(0, inter_y2 - inter_y1)
    
    overlap = tf.math.divide(inter_a, area_p + area_t - inter_a)

    return overlap

def yolo2minmax(yboxes, img_w, img_h):
    '''
    Convert normalised yolo box format to min max corner format
    
    args:
        yboxes: [xc, yc, w, h] in normalized yolo format 
        img_w, img_h: shape of image
        
    returns:
        boxes: [x1, y1, x2, y2] - VOC format    
    '''
    
    boxes = [None]*len(yboxes)
    for i, ybox in enumerate(yboxes):
        xc, yc, w, h = ybox
        
        x1 = (xc - w/2)*img_w
        y1 = (yc - h/2)*img_h
        x2 = (xc + w/2)*img_w
        y2 = (yc + h/2)*img_h
        boxes[i] = [x1, y1, x2, y2]
    
    return boxes

def minmax2yolo(boxes, img_w, img_h):
    '''
    Convert min max corners to normalised yolo box format
    
    args:
        boxes: [x1, y1, x2, y2]
        img_w, img_h: shape of image
        
    returns:
        yboxes: [xc, yc, w, h] in normalized yolo format    
    '''
    
    yboxes = [None]*len(boxes)
    for i, box in enumerate(boxes):
        xc = (box[0] + box[2])/2./img_w
        yc = (box[1] + box[3])/2./img_h

        w = (box[2] - box[0])/img_w
        h = (box[3] - box[1])/img_h

        yboxes[i] = [xc, yc, w, h]
    
    return yboxes

def compute_nms(boxes, iou_t, conf_t):
    '''
    compute non-maximum suppression
    
    args:
        boxes: boxes in yolo format [img_ind, cls, obj_conf, xc, yc, w, h]
        iou_t:   threshold for overlapping 
        conf_t:  threshold for objectness confidence
    returns:
        nms_boxes: boxes after nms
    '''
    
    conf_ind  = 2
    coor_start_ind = 3
    
    boxes = np.array([box for box in boxes if box[conf_ind] > conf_t])
    if boxes.size == 0:
        return np.empty((0, 7))

    s_ind = np.argsort(boxes[:,conf_ind], axis = 0)[::-1]
    boxes = boxes[s_ind, :]

    nms_boxes = []
    while boxes.size > 0:
        #get the most confidence box
        c_box = boxes[0,  :]
        boxes = boxes[1:, :]
        #compute the iou between chosen box and other boxes
        ious = compute_IOU_by_union(np.expand_dims(c_box[coor_start_ind:], 0), 
                                    boxes[:, coor_start_ind:])
        #filter out boxes with high iou
        remain_box_ind = np.where(ious < iou_t, 
                                  np.ones( (ious.shape[0],), dtype=bool), 
                                  np.zeros((ious.shape[0],), dtype=bool))

        boxes = boxes[remain_box_ind, :]
        #store the chosen box
        nms_boxes.append(list(c_box))

    return np.array(nms_boxes)

def cvt_cell_ratio_to_img_ratio(res, g_num = 7, c_num = 2, b_num = 2):
    '''
    convert coordinates based on cell ratio to coordinates based on image ratio
    
    args:
        res: label or prediction relative to cell width and height
        g_num: grid number in the row/col of an image
        c_num: number of classes
        b_num: number of bounding boxes
        
    returns:
        cvt_boxes: boxes with coordinates relative to image width and height
    '''
    
    N_batch = res.shape[0]
    #check if it is ground truth or label
    if len(res.shape) != 4:
        res   = np.reshape(res, (N_batch, g_num, g_num, c_num + b_num * 5))
    cls_p = tf.cast(np.argmax(res[:,:,:,0:c_num], axis = -1), 
                    tf.float32)
    cls_p = tf.expand_dims(cls_p, 3)
    #get the best bounding box based on 
    #best objectness confidence
    box0 = res[:, :, :, c_num + 1:c_num + 5]

    box1 = None
    
    if res.shape[3] == c_num + 10:
        box1 = res[:, :, :, c_num + 6:c_num + 10]
        confs = tf.concat([
                           tf.expand_dims(res[:,:,:,c_num    ], 3), 
                           tf.expand_dims(res[:,:,:,c_num + 5], 3)
                           ], 3)
        best_box_ind = tf.expand_dims(tf.argmax(confs, 3), 3)
        best_box_ind = tf.cast(best_box_ind, tf.float32)
        best_boxes   = best_box_ind*box1 + (1-best_box_ind)*box0
        
        best_confs   = tf.math.maximum(res[:,:,:,c_num],
                                       res[:,:,:,c_num + 5])
        best_confs   = tf.expand_dims(best_confs, 3)
    else:
        best_boxes = box0
        best_confs = np.expand_dims(res[:,:,:,c_num], 
                                    axis = 3)
    best_confs = tf.cast(best_confs, tf.float32)
        
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
    
    x = tf.expand_dims(1/g_num*(best_boxes[:,:,:,0] + x_cell_ind), 3)
    y = tf.expand_dims(1/g_num*(best_boxes[:,:,:,1] + y_cell_ind), 3)
    
    w = tf.expand_dims(tf.cast(best_boxes[:,:,:,2], tf.float32), 3)
    h = tf.expand_dims(tf.cast(best_boxes[:,:,:,3], tf.float32), 3)
    
    cvt_boxes = tf.concat([cls_p,
                           best_confs,
                           x,
                           y,
                           w,
                           h], 
                           axis = -1)
    
    return cvt_boxes

def compute_mAP(p_boxes, t_boxes, cls_num = 3, iou_t = 0.5):
        
        '''
        To evaluate the performance of the detector
        
        args:
            p_boxes: boxes prediction 
            [train_ind, class_prediction, objectness_conf, xc, yc, h, w]
            t_boxes: ground truth 
            [train_ind, class_true, objectness_conf, xc, yc, h, w]
            cls_num: number of class
            iou_t : threshold to consider if the prediction is TP/FP
            
        returns:
            maP: mean average precision
        '''
        
        aP = []    
        
        for cls in range(cls_num):
            dets = []
            grds = []
            
            for p_box in p_boxes:
                if p_box[1] == cls:
                    dets.append(p_box)
            
            for t_box in t_boxes:
                if t_box[1] == cls:
                    grds.append(t_box)

            img_gt_check = {}
            for grd in grds:
                if grd[0] not in img_gt_check:
                    img_gt_check[grd[0]] = 0
                img_gt_check[grd[0]] += 1
            
            for k, v in img_gt_check.items():
                img_gt_check[k] = np.zeros((v, ))
            
            dets = sorted(dets, key=lambda x:x[2], reverse = True)
            
            tp = np.zeros((len(dets),))
            fp = np.zeros((len(dets),))
            t_box_num = len(grds)
            
            if t_box_num == 0:
                continue
            
            for det_ind, det in enumerate(dets):
                
                gt_boxes = np.array([box for box in grds if box[0] == det[0]])
                gt_boxes_num = len(gt_boxes)
                best_iou = 0
                
                for gt_ind, gt in enumerate(gt_boxes):
                    iou = compute_IOU_by_union(np.expand_dims(det[3:], 0), 
                                               np.expand_dims( gt[3:], 0))
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_ind = gt_ind

                if best_iou > iou_t:
                    if img_gt_check[det[0]][best_iou_ind] == 0:
                        tp[det_ind] = 1
                        img_gt_check[det[0]][best_iou_ind] = 1
                    else:
                        fp[det_ind] = 1
                else:
                    fp[det_ind] = 1
                
            tp_cumsum = tf.cast(tf.math.cumsum(tp, axis = 0), tf.float32)
            fp_cumsum = tf.cast(tf.math.cumsum(fp, axis = 0), tf.float32)
            
            recalls    = tp_cumsum/(t_box_num + 1e-10)
            precisions = tf.math.divide(tp_cumsum, tp_cumsum + fp_cumsum + 1e-10)
            precisions = tf.concat([np.ones([1,]), precisions], axis = 0)
            recalls    = tf.concat([np.zeros([1,]), recalls]   , axis = 0)
            
            aP.append(tfp.math.trapz(precisions, x = recalls))
        
        maP = sum(aP)/len(aP)
            
        return maP