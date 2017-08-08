import sys, os, time
import math
import tensorflow as tf
import numpy as np
import cv2
import warnings
import test
warnings.filterwarnings('ignore')


input_height = 416
input_width = 416
bn_epsilon = 1e-3
n_input_imgs = 1
relu_alpha = 0.1

# Input placeholders for x and y
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[n_input_imgs, input_height, input_width, 3])
    labels = tf.placeholder(tf.float32, shape=[n_input_imgs, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def max_pool_layer(input_tensor,kernel_size,stride,padding):
    pooling_result = tf.nn.max_pool(input_tensor, ksize=[1,kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding)
    return pooling_result

def leaky_relu(x, alpha):
    return tf.maximum(alpha * x, x)

# IMPORTANT: keep track of the number of parameters needed in each layer to check the total with the binary file!
n_params = 0

#1 conv1     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16
w1 = weight_variable([3,3,3,16])
b1 = bias_variable([16])
h1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
o1 = leaky_relu(h1, relu_alpha)

n_params = 3*3*3*16 + 16*4
# IMPORTANT: This is where n_params comes from: 
# n_params = kernel_shape + n_biases + n_bn_means + n_bn_var + n_bn_gammas
# n_params = kernel_shape + n_biases + n_output_channels + n_output_channels + n_output_channels
# n_params = kernel_shape + n_output_channels + n_output_channels + n_output_channels + n_output_channels
# n_params = kernel_shape + n_output_channels*4
# IMPORTANT: YOLOv2 sets the biases in every convolution = 0 and keeps only the betas (offsets) of the Batch Normalization!
# So in the end there will be only mean,var,beta(offset),gamma(scale) for every single output channel!

#2 max1          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
max1 = max_pool_layer(o1,kernel_size=2,stride=2,padding='VALID')

#3 conv2     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32
w2 = weight_variable([3,3,16,32])
b2 = bias_variable([32])
h2 = tf.nn.conv2d(max1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
o2 = leaky_relu(h2, relu_alpha)
n_params = n_params + 3*3*16*32 + 32*4

#4 max2          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
max2 = max_pool_layer(o2,kernel_size=2,stride=2,padding='VALID')
# NOTE: the maxpool layer does not have parameters!

#5 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64
w3 = weight_variable([3,3,32,64])
b3 = bias_variable([64])
h3 = tf.nn.conv2d(max2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
o3 = leaky_relu(h3, relu_alpha)
n_params = n_params + 3*3*32*64 + 64*4

#6 max3          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
max3 = max_pool_layer(o3,kernel_size=2,stride=2,padding='VALID')

#7 conv4    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
w4 = weight_variable([3,3,64,128])
b4 = bias_variable([128])
h4 = tf.nn.conv2d(max3, w4, strides=[1, 1, 1, 1], padding='SAME') + b4
o4 = leaky_relu(h4, relu_alpha)
n_params = n_params + 3*3*64*128 + 128*4

#8 max4          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
max4 = max_pool_layer(o4,kernel_size=2,stride=2,padding='VALID')

#9 conv5    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
w5 = weight_variable([3,3,128,256])
b5 = bias_variable([256])
h5 = tf.nn.conv2d(max4, w5, strides=[1, 1, 1, 1], padding='SAME') + b5
o5 = leaky_relu(h5, relu_alpha)
n_params = n_params + 3*3*128*256 + 256*4

#10 max5          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
max5 = max_pool_layer(o5,kernel_size=2,stride=2,padding='VALID')

#11 conv6   512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 512
w6 = weight_variable([3,3,256,512])
b6 = bias_variable([512])
h6 = tf.nn.conv2d(max5, w6, strides=[1, 1, 1, 1], padding='SAME') + b6
o6 = leaky_relu(h6, relu_alpha)
n_params = n_params + 3*3*256*512 + 512*4

#12 max6          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
max6 = max_pool_layer(o6,kernel_size=2,stride=1,padding='SAME')

#13 conv7    1024  1 x 1 / 1    13 x  13 x512   ->    13 x  13 x 1024
w7 = weight_variable([3,3,512,1024])
b7 = bias_variable([1024])
h7 = tf.nn.conv2d(max6, w7, strides=[1, 1, 1, 1], padding='SAME') + b7
o7 = leaky_relu(h7, relu_alpha)
n_params = n_params + 3*3*512*1024 + 1024*4

#14 conv8   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
w8 = weight_variable([3,3,1024,1024])
b8 = bias_variable([1024])
h8 = tf.nn.conv2d(o7, w8, strides=[1, 1, 1, 1], padding='SAME') + b8
o8 = leaky_relu(h8, relu_alpha)
n_params = n_params + 3*3*1024*1024 + 1024*4

#15 conv9   125  1 x 1 / 1    13 x  13 x 1024   ->    13 x  13 x125
w9 = weight_variable([1,1,1024,125])
b9 = bias_variable([125])
h9 = tf.nn.conv2d(o8, w9, strides=[1, 1, 1, 1], padding='SAME') + b9
# Linear output!
o9 = h9
n_params = n_params + 1*1*1024*125 + 125 # There is not batch norm, so n_params is: kernel_size + n_biases

print('Total number of params = {}'.format(n_params))

#######################################################################################################################################

