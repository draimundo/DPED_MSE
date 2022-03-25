#################################
# RAW-to-RGB Model architecture #
#################################

import tensorflow as tf
import time
import numpy as np


conv2d=tf.compat.v1.layers.conv2d
relu=tf.compat.v1.nn.relu
relu6=tf.compat.v1.nn.relu6
sigmoid=tf.nn.sigmoid
maxpool=tf.nn.max_pool
#avgpool2d=tf.compat.v1.nn.avg_pool2d
avgpool2d=tf.math.reduce_mean
conv2dTrans=tf.compat.v1.layers.conv2d_transpose
depthTospace=tf.nn.depth_to_space
depthwiseConv2d=tf.keras.layers.DepthwiseConv2D

def csanet(input, num_maps_base=16):
    with tf.compat.v1.variable_scope("generator"):
        # -----------------------------------------
        # Downsampling layers
        filters = num_maps_base                 
        norm       = input / 4095.0 
        conv_l1_d1 = conv2d(norm, filters=filters, kernel_size=3, strides=2, padding="SAME", name="conv2d")  # HWC=64x64x16
        conv_l1_d1 = relu(conv_l1_d1)                                                   
        conv_l1_d1 = conv2d(conv_l1_d1, filters=filters, kernel_size=3, padding="SAME", name="conv2d_1")     # HWC=64x64x16 
        conv_l1_d1 = relu(conv_l1_d1) 
        

        # -------------------------------------------
        # Processing: Level 2
        filters = num_maps_base
        conv_l2_d1 = DAB_RES(conv_l1_d1)                                                                     # HWC=64x64x32
        
        conv_l2_d2 = conv2d(conv_l2_d1, filters=filters, kernel_size=3, padding="SAME", name="conv2d_15")    # HWC=64x64x16
        conv_l2_d2 = relu(conv_l2_d2)
        
        conv_l2_d3 = tf.concat([conv_l2_d2,conv_l1_d1], axis=3)                                              # HWC=64x64x32                                            
                   
        conv_l2_d4 = conv2dTrans(conv_l2_d3, filters=filters, kernel_size=3, strides=2, padding='SAME', use_bias=False) 
                                                                                                             # HWC=128x128x16
        
        # -------------------------------------------
        # Processing: Level 1
        filters, upscale = num_maps_base//4, 2
        conv_l1_d2 = conv2d(conv_l2_d4, filters=filters*upscale**2, kernel_size=3, padding="SAME", name="conv2d_20")
        conv_l1_d2 = relu(conv_l1_d2)                                                                        # HWC=128x128x16
                                                                                                  
        conv_l1_d3 = depthTospace(conv_l1_d2, upscale)                                                       # HWC=256x256x4
        
        # ----------------------------------------------------------
        # Processing: Level 0 (x2 upscaling),  Input size: 128 x 128
        filters = 3
        conv_l0_d1 = conv2d(conv_l1_d3, filters=filters, kernel_size=3, padding='SAME', name="conv2d_21")    # HWC=256x256x3                     
        conv_l0_d2 = sigmoid(conv_l0_d1)                                               

    output = conv_l0_d2

    return output


def PAM_Module(x, name=None):
    res = x
    
    dep = depthwiseConv2d(kernel_size=5, strides=(1,1), padding='SAME', dilation_rate=2, name=name)
    out = sigmoid(dep(x))
    
    out = out * res

    return out


def CAM_Module(x, reduction, name=None):
    res = x
    _, H, W, C = x.get_shape().as_list()
    PREFIX = name[:7]
    ID = int(name.split('_')[-1])
    
    #out = avgpool2d(x, ksize=(H,W), strides=(1,1), padding='VALID')
    out = avgpool2d(x, axis=[1,2], keepdims=True)
    out = conv2d(out, filters=C//reduction, kernel_size=1, padding="SAME", name=name)
    out = relu(out)
    out = conv2d(out, filters=C, kernel_size=1, padding="SAME", name=PREFIX+str(ID+1))
    out = sigmoid(out)
    
    out = out * res
    
    return out


def DAB_Module(x, conv_name=None, dwconv_name=None):
    res = x
    _, _, _, C = x.get_shape().as_list()
    PREFIX = conv_name[:7]
    ID = int(conv_name.split('_')[-1])
    
    out = conv2d(x, filters=C*2, kernel_size=3, padding="SAME", name=conv_name)
    out = relu(out)
    out = conv2d(out, filters=C*2, kernel_size=1, padding="SAME", name=PREFIX+str(ID+1))
    
    reduction = 4
    pam_branch = PAM_Module(out, name=dwconv_name)
    cam_branch = CAM_Module(out, reduction, name=PREFIX+str(ID+2))
    out = tf.concat([pam_branch, cam_branch], axis=3)
    
    out = conv2d(out, filters=C, kernel_size=1, padding="SAME", name=PREFIX+str(ID+4)) 
    out = out + res

    return out


def DAB_RES(x):
    _, _, _, C = x.get_shape().as_list()
    
    dab1 = DAB_Module(x , conv_name="conv2d_4", dwconv_name="depthwise_conv2d")
    dab2 = DAB_Module(dab1, conv_name="conv2d_9", dwconv_name="depthwise_conv2d_1")

    return dab2
    