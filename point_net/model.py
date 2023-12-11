import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout

class conv_class(Layer):
    def __init__(self, 
                 out_ch, 
                 k_size, 
                 stride, 
                 pad = 'valid',
                 activation = None, 
                 is_bn = False, 
                 bn_momentum = 0.99, 
                 **kwargs):
        
        super(conv_class, self).__init__(**kwargs)
        self.out_ch = out_ch
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.activation = activation
        self.is_bn = is_bn
        self.bn_momentum = bn_momentum
        
        self.conv = Conv2D(filters = out_ch, 
                           kernel_size = k_size,
                           strides = stride,
                           padding = pad,
                           activation = activation, 
                           use_bias = not is_bn)
        
        if is_bn:
            self.bn = BatchNormalization(momentum = bn_momentum)
        
    def call(self, x, is_train = False):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x, training = is_train)
        if self.activation:
            x = self.activation(x)
            
        return x

    def get_config(self):
        config = super(conv_class, self).get_config()
        config.update(
            {
            'out_ch': self.out_ch,
            'k_size': self.k_size,
            'stride': self.stride,
            'pad': self.pad,
            'activation': self.activation,
            'is_bn': self.is_bn,
            'bn_momentum': self.bn_momentum
            }
        )
        
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
        
class dense_class(Layer):
    def __init__(self, 
                 unit_num, 
                 activation = None,
                 is_bn = False,
                 bn_momentum = 0.99,
                 **kwargs):
        super(dense_class, self).__init__(**kwargs)
        self.unit_num = unit_num
        self.activation = activation
        self.is_bn = is_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units = unit_num, 
                           activation=activation,
                           use_bias = not is_bn)
        
        if is_bn:
            self.bn = BatchNormalization(momentum=bn_momentum)
            
    def call(self, x, is_train = False):
        x = self.dense(x)
        if self.is_bn:
            x = self.bn(x, training = is_train)
        if self.activation:
            x = self.activation(x)
            
        return x
    
    def get_config(self):
        config = super(conv_class, self).get_config()
        config.update(
            {
            'unit_num': self.unit_num,
            'activation': self.self.activation,
            'is_bn': self.is_bn,
            'bn_momentum': self.bn_momentum,
            }
        )
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class t_net(Layer):
    def __init__(self, 
                 bn_momentum = 0.99, 
                 **kwargs):
        super(t_net, self).__init__(**kwargs)
        
        self.bn_momentum = bn_momentum
        self.conv0 = conv_class(out_ch = 64, 
                                k_size = (1, 1),
                                stride = (1, 1), 
                                bn_momentum = bn_momentum)
        self.conv1 = conv_class(out_ch = 128, 
                                k_size = (1, 1),
                                stride = (1, 1),
                                bn_momentum = bn_momentum)
        self.conv2 = conv_class(out_ch = 1024, 
                                k_size = (1, 1),
                                stride = (1, 1), 
                                bn_momentum = bn_momentum)
        self.fc0 = dense_class(unit_num = 512, 
                               activation = tf.nn.relu,
                               is_bn = True,
                               bn_momentum = bn_momentum)
        self.fc1 = dense_class(unit_num = 256, 
                               activation = tf.nn.relu, 
                               is_bn = True, 
                               bn_momentum = bn_momentum)        
    
    def build(self, input_shape):
        self.d = input_shape[-1]
        
        #initialise weight
        self.w = self.add_weight(shape = (256, self.d**2), 
                                 initializer=tf.zeros_initializer,
                                 trainable = True,
                                 name = 'w')
        #initialise bias
        self.b = self.add_weight(shape = (self.d, self.d), 
                                 initializer=tf.zeros_initializer,
                                 trainable = True,
                                 name = 'b')

        eye = tf.constant(np.eye(self.d), dtype = tf.float32)
        self.b = tf.math.add(self.b, eye)
        
    def call(self, x, is_train = False):
        #b*N*d (input: d => 3, feature: d => 64)
        in_x = x
        #b*N*d => b*N*1*d
        x = tf.expand_dims(in_x, axis = 2) 
        #b*N*1*d => b*N*1*64
        x = self.conv0(x, training = is_train)
        #b*N*1*64 => b*N*1*128
        x = self.conv1(x, training = is_train)
        #b*N*1*128 => b*N*1*1024
        x = self.conv2(x, training = is_train)
        #b*N*1*1024 => b*N*1024
        x = tf.squeeze(x, axis = 2)
        
        #global features
        #b*N*1024 => b*1024
        x = tf.reduce_max(x, axis = 1)
        
        #dense layers
        #b*1024 => b*512
        x = self.fc0(x, training = is_train)
        #b*512 => b*256
        x = self.fc1(x, training = is_train)
        
        #convert to 3*3 transformation matrix
        #b*256 => b*1*256
        x = tf.expand_dims(x, axis = 1)
        #matmul(b*1*256, 256*d^2) => b*1*d^2
        x = tf.matmul(x, self.w)
        #b*1*d^2 => b*d^2
        x = tf.squeeze(x, axis = 1)
        #b*d^2 => b*d*d
        x = tf.reshape(x, (-1, self.d, self.d))
        #add bias
        #b*d*d + 1*d*d => b*d*d
        x += self.b
        
        #
        return tf.matmul(in_x, x)
        
    def get_config(self):
        config = super(t_net, self).get_config()
        config.update({
            'is_reg': self.is_reg,
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def point_net_cls(bn_momentum):
    
    ''' input layer '''
    #define input
    '''
    note: we can treat N*3 input point cloud as a gray image with N rows and 3 columns
    '''
    #b*N*3 
    in_pt = Input(shape=(None, 3), dtype=tf.float32, name = "pt_cloud")
    
    #normalise input
    #b*N*3 => b*N*3
    t_in_pt = t_net(bn_momentum=bn_momentum)(in_pt)
    
    ''' 1st embedding layer '''
    #embed to  64-dim space
    #b*N*3 => b*N*1*3
    '''
    note: 
    1. The main purpose is to convert the point cloud as "2d image like format (N rows, 3 cols, 
       1 channels)" to fit into traditional covolutional neural network. 
    2. Similar operations will also be used for processing 64-dim feature vector
    '''
    t_in_pt = tf.expand_dims(t_in_pt, axis = 2)
    #b*N*1*3 => b*N*1*64
    e1_out = conv_class(out_ch = 64, 
                      k_size = (1, 1), 
                      stride = (1, 1), 
                      activation = tf.nn.relu,
                      is_bn = True)(t_in_pt)
    #b*N*1*64 => b*N*1*64
    e1_out = conv_class(out_ch = 64, 
                      k_size = (1, 1), 
                      stride = (1, 1), 
                      activation = tf.nn.relu,
                      is_bn = True)(e1_out)
    ##b*N*64 => b*N*64
    e1_out = tf.squeeze(e1_out, axis = 2)
    
    #normalise features
    #b*N*64 => b*N*64
    t_e1_out = t_net(bn_momentum=bn_momentum)(e1_out)
    
    # 2nd embedding layer 
    #embed to  64-dim space
    #b*N*64 => b*N*1*64
    t_e1_out = tf.expand_dims(t_e1_out, axis = 2)
    #b*N*1*64 => b*N*1*64
    e2_out = conv_class(out_ch = 64, 
                      k_size = (1, 1), 
                      stride = (1, 1), 
                      activation = tf.nn.relu,
                      is_bn = True)(t_e1_out)
    #b*N*1*128 => b*N*1*128
    e2_out = conv_class(out_ch = 128, 
                       k_size = (1, 1), 
                       stride = (1, 1), 
                       activation = tf.nn.relu,
                       is_bn = True)(e2_out)
    #b*N*1*1024 => b*N*1*1024
    e2_out = conv_class(out_ch = 1024, 
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(e2_out)
    #b*N*1*1024 => b*N*1024
    e2_out = tf.squeeze(e2_out, axis = 2)
    
    # global vector layer 
    #reduce to global descriptor
    #b*N*1024 => b*1024
    g_vec = tf.reduce_max(e2_out, axis = 1)
    
    # dense layer
    #b*1024 => b*512
    d_out = dense_class(unit_num = 512,
                        activation = tf.nn.relu,
                        is_bn = True,
                        bn_momentum = bn_momentum)(g_vec)
    d_out = Dropout(rate = 0.3)(d_out)
    #b*512 => b*256
    d_out = dense_class(unit_num = 256,
                        activation = tf.nn.relu,
                        is_bn = True,
                        bn_momentum = bn_momentum)(d_out)
    d_out = Dropout(rate = 0.3)(d_out)
    
    # output layer
    #b*256 => b*40
    out = dense_class(unit_num = 40, is_bn = False)(d_out)
    
    return Model(inputs = in_pt, outputs = out)
    
def point_net_seg(bn_momentum = 0.99, N_point = 1024):
    
    ''' input layer '''
    #define input
    #b*N*3 
    '''
    note: we can treat N*3 input point cloud as a gray image with N rows and 3 columns
    '''
    in_pt = Input(shape=(N_point, 3), dtype=tf.float32, name = "pt_cloud")
    
    #normalise input
    #b*N*3 => b*N*3
    t_in_pt = t_net(bn_momentum=bn_momentum)(in_pt)

    ''' 1st embedding layer => expand to 64 vector '''
    '''
    note: 
    1. The main purpose of tf.expand_dims(t_in_pt, axis = 2) is to convert the point cloud as 
       "2d image like format (N rows, 1 cols, 3 channels)" to fit into traditional covolutional neural network. 
    2. Similar operations will also be used for processing 64-dim feature vector
    '''
    #b*N*3 => b*N*1*3
    t_in_pt = tf.expand_dims(t_in_pt, axis = 2)
    #b*N*1*3 => b*N*1*64
    e1_out = conv_class(out_ch = 64, 
                        pad = 'valid',
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(t_in_pt)
    #b*N*1*64 => b*N*1*64
    e1_out = conv_class(out_ch = 64, 
                        pad = 'valid',
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(e1_out)
    #b*N*64 => b*N*64
    e1_out = tf.squeeze(e1_out, axis = 2)
    
    #normalise features
    #b*N*64 => b*N*64
    t_e1_out = t_net(bn_momentum=bn_momentum)(e1_out)
    
    ''' 2nd embedding layer => expand to 1024 vector '''
    #embed to  1024-dim space
    #b*N*64 => b*N*1*64
    t_e1_out = tf.expand_dims(t_e1_out, axis = 2)
    #b*N*1*64 => b*N*1*64
    e2_out = conv_class(out_ch = 64, 
                        pad = 'valid',
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(t_e1_out)
    #b*N*1*128 => b*N*1*128
    e2_out = conv_class(out_ch = 128, 
                        pad = 'valid',
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(e2_out)
    #b*N*1*1024 => b*N*1*1024
    e2_out = conv_class(out_ch = 1024, 
                        pad = 'valid',
                        k_size = (1, 1), 
                        stride = (1, 1), 
                        activation = tf.nn.relu,
                        is_bn = True)(e2_out)
    #b*N*1*1024 => b*N*1024
    e2_out = tf.squeeze(e2_out, axis = 2)
    
    ''' global vector layer '''
    #reduce to global descriptor
    #b*N*1024 => b*1024
    g_vec = tf.reduce_max(e2_out, axis = 1, keepdims = True)
    ''' concatenated feature layer '''
    #b*1024 => b*1*1*1024
    g_vec = g_vec[:,:,tf.newaxis,:]

    # b*1*1*1024 => b*N*1*1024
    g_vec = tf.tile(g_vec, [1, N_point, 1, 1])

    #b*N*1*64 + b*N*1*1024 => b*N*1*1088
    c_vec = tf.concat([t_e1_out, g_vec], axis = 3)
    
    '''segmentation layer'''
    s_out = conv_class(out_ch = 512, 
                       pad = 'valid',
                       k_size = (1, 1), 
                       stride = (1, 1), 
                       activation = tf.nn.relu,
                       is_bn = True)(c_vec)
    s_out = conv_class(out_ch = 256, 
                       pad = 'valid',
                       k_size = (1, 1), 
                       stride = (1, 1), 
                       activation = tf.nn.relu,
                       is_bn = True)(s_out)
    s_out = conv_class(out_ch = 128, 
                       pad = 'valid',
                       k_size = (1, 1), 
                       stride = (1, 1), 
                       activation = tf.nn.relu,
                       is_bn = True)(s_out)
    
    '''output layer'''
    out = conv_class(out_ch = 50, 
                     pad = 'valid',
                     k_size = (1, 1), 
                     stride = (1, 1), 
                     activation = None,
                     is_bn = True)(s_out)
    
    return Model(inputs = in_pt, outputs = out)