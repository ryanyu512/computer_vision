'''
    Updated on 2023/04/21
    
    1. aim at defining the architechure of object detector
'''

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import LeakyReLU, ReLU, PReLU, Softmax, Activation, Dropout

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 ch, 
                 pad_size = (1, 1), 
                 c_size   = (3, 3),
                 c_stride = (1, 1), 
                 m_size   = (2, 2),
                 m_stride = (2, 2),
                 m_pad = 'valid',
                 alpha = 0.1, 
                 is_max = True,
                 is_train = False):
        super(ConvBlock, self).__init__()
        self.ch = ch
        self.pad_size = pad_size
        self.c_size   = c_size
        self.c_stride = c_stride
        self.c_pad    = 'valid'
        self.m_size   = m_size
        self.m_stride = m_stride
        self.m_pad = m_pad
        self.alpha = alpha
        self.is_max = is_max
        self.is_use_bias = True
        self.is_train = is_train
        
        self.z2d = ZeroPadding2D(padding = self.pad_size)
        self.c2d = Conv2D(filters = self.ch,
                          kernel_size = self.c_size,
                          kernel_initializer='he_uniform',
                          strides = self.c_stride,
                          padding = self.c_pad,
                          use_bias = self.is_use_bias)
        self.pReLU = PReLU(shared_axes = [1,2])
        self.m2d  = MaxPooling2D(pool_size = self.m_size, 
                                 strides   = self.m_stride, 
                                 padding   = self.m_pad)
        
    def get_config(self):
    
        config = super().get_config().copy()
        config.update({
            'ch': self.ch,
            'pad_size': self.pad_size,
            'c_size': self.c_size,
            'c_stride': self.c_stride,
            'c_pad': self.c_pad,
            'm_size': self.m_size,
            'm_stride': self.m_stride,
            'm_pad':self.m_pad,
            'alpha':self.alpha,
            'is_max':self.is_max,
            'is_use_bias':self.is_use_bias,
        })
        return config
    
    def call(self, x):
        
        x = self.z2d(x)
        x = self.c2d(x)
        x = self.pReLU(x)
        if self.is_max:
            x = self.m2d(x)
        
        return x

class Det_Network:
    
    def __init__(self, 
                 g_num = 7,
                 b_num = 2, 
                 cls_num = 2, 
                 fc_dropout_ratio = 0.5,
                 is_train = False):
        
        self.backbone = Sequential(name = 'backbone')
        self.neck     = Sequential(name = 'neck')
        self.is_train = is_train
        self.fc_dropout_ratio = fc_dropout_ratio
        
        #define backbone
        self.backbone.add(ConvBlock(ch = 64, 
                                    pad_size = (3, 3),
                                    c_size   = (7, 7),
                                    c_stride = (2, 2), 
                                    is_max = True,
                                    is_train = is_train))
        self.backbone.add(ConvBlock(ch = 192, 
                                    pad_size = (1, 1),
                                    c_size   = (3, 3),
                                    c_stride = (1, 1),
                                    is_max = True,
                                    is_train = is_train))
        self.backbone.add(ConvBlock(ch = 256, 
                                    pad_size = (1, 1),
                                    c_size   = (3, 3),
                                    c_stride = (1, 1),
                                    is_max = False,
                                    is_train = is_train))
        self.backbone.add(ConvBlock(ch = 256, 
                                    pad_size = (1, 1),
                                    c_size   = (3, 3),
                                    c_stride = (1, 1),
                                    is_max = True,
                                    is_train = is_train))
        #define neck
        self.neck.add(ConvBlock(ch = 512, 
                                pad_size = (1, 1),
                                c_size   = (3, 3),
                                c_stride = (1, 1),
                                is_max = True,
                                is_train = is_train))
        self.neck.add(ConvBlock(ch = 1024, 
                                pad_size = (1, 1),
                                c_size   = (3, 3),
                                c_stride = (1, 1),
                                is_max = True,
                                is_train = is_train))

        #define fully connected network
        '''
        1. grid cell = 7*7
        2. class = 2
        3. # of bbox = 2
        4. 49*(cls_num + 2*5)
        '''
        self.flatten  = Flatten(name = 'det_flatten')
        self.dense1   = Dense(640, 
                              kernel_initializer= 'he_uniform',
                              name = 'det_dense1')
        self.LReLU    = LeakyReLU(alpha = 0.1, name = 'det_LReLU')
        self.dropout2 = Dropout(rate = self.fc_dropout_ratio, 
                                name = 'det_dropout2')
        self.dense2   = Dense(g_num*g_num*(cls_num + b_num*5), 
                              kernel_initializer= 'he_uniform',
                              name = 'det_dense2')
        
    def obj_det(self):
        
        feed = Input((448, 448, 3))
        x    = self.backbone(feed)
        x    = self.neck(x)
        x    = self.flatten(x)
        x    = self.dense1(x)
        x    = self.dropout2(x, training = self.is_train)
        x    = self.LReLU(x)
        out  = self.dense2(x)
                      
        model = Model(feed, [out])
        
        return model
