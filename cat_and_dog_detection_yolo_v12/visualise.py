'''
UPDATED ON 2023/05/02

1. aims at providing custom library of label/prediction visualization
'''

import cv2
import json
from matplotlib import pyplot as plt

def visualise_boxes(img, boxes, color = [0,0,255], thickness = 2, fontscale = 2, is_show_scores = False, objs = None):
    
    '''
        args:
            img: image to be displayed
            boxes: bounding box [x_min, y_min, x_max, y_max, conf(optional)]
            thickness: line thickness of bounding box and text
            fontscale: fone size
            is_show_scores: define if show scores of bounding boxes
            objs: name of objects (i.e. cat, dog)
        
        returns:
            img: image with bounding boxes
    '''
    
    for i, box  in enumerate(boxes):
        #define bbox
        x_min, y_min, x_max, y_max = box[0:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        #draw rectangle
        cv2.rectangle(img, 
                      (x_min, y_min), 
                      (x_max, y_max),
                      color = color,
                      thickness = thickness)
        
        #draw scores
        if len(box) > 4 and is_show_scores:
            cv2.putText(img, 
                        str(round(box[4], 2)), 
                        ((x_min + 60), 
                          y_min + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        thickness = thickness, 
                        color = color) 
        
        #draw class name
        if objs is not None:
            cv2.putText(img, 
                        objs[i], 
                        (x_min, y_min + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        thickness = thickness, 
                        color = color) 
        
    return img