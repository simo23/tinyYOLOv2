import tensorflow as tf
import os
import numpy as np
import os.path
import test
import net


# IMPORTANT: Weights order in the binary file is [ 'biases','gamma','moving_mean','moving_variance','kernel']
# IMPORTANT: biases ARE NOT the usual biases to add after the conv2d! They refer to the betas (offsets) in the Batch Normalization!
# IMPORTANT: the biases added after the conv2d are set to zero! 
# IMPORTANT: to use the weights they actually need to be de-normalized because of the Batch Normalization! ( see later )

def load_conv_layer_bn(name,loaded_weights,shape,offset):
    # Conv layer with Batch norm

    n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
    n_output_channels = shape[-1]
    n_bn_mean = n_output_channels
    n_bn_var = n_output_channels
    n_biases = n_output_channels
    n_bn_gamma = n_output_channels

    n_weights_conv_bn = (n_kernel_weights + n_output_channels * 4)

    # IMPORTANT: This is where (n_kernel_weights + n_output_channels * 4) comes from: 
    # n_params = kernel_shape + n_biases + n_bn_means + n_bn_var + n_bn_gammas
    # n_params = kernel_shape + n_biases + n_output_channels + n_output_channels + n_output_channels
    # n_params = kernel_shape + n_output_channels + n_output_channels + n_output_channels + n_output_channels
    # n_params = kernel_shape + n_output_channels*4
    # IMPORTANT: YOLOv2 sets the biases in every convolution = 0 and keeps only the betas (offsets) of the Batch Normalization!
    # So in the end there will be only mean,var,beta(offset),gamma(scale) for every single output channel!

    print('Loading '+str(n_weights_conv_bn)+' weights of '+name+' ...')

    biases = loaded_weights[offset:offset+n_biases]
    offset = offset + n_biases
    gammas = loaded_weights[offset:offset+n_bn_gamma]
    offset = offset + n_bn_gamma
    means = loaded_weights[offset:offset+n_bn_mean]
    offset = offset + n_bn_mean
    var = loaded_weights[offset:offset+n_bn_var]
    offset = offset + n_bn_var
    kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
    offset = offset + n_kernel_weights

    # IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
    kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')

    # IMPORTANT: Denormalize the weights with the Batch Normalization parameters
    for i in range(n_output_channels):

        scale = gammas[i] / np.sqrt(var[i] + net.bn_epsilon)
        kernel_weights[i,:,:,:] = kernel_weights[i,:,:,:] * scale
        biases[i] = biases[i] - means[i] * scale

    # IMPORTANT: Set weights to Tensorflow order: (height, width, in_dim, out_dim)
    kernel_weights = np.transpose(kernel_weights,[2,3,1,0])

    return biases,kernel_weights,offset

def load_conv_layer(name,loaded_weights,shape,offset):
    # Conv layer without Batch norm

    n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
    n_output_channels = shape[-1]
    n_biases = n_output_channels

    n_weights_conv = (n_kernel_weights + n_output_channels)
    # The number of weights is a conv layer without batchnorm is: (kernel_height*kernel_width + n_biases)
    print('Loading '+str(n_weights_conv)+' weights of '+name+' ...')

    biases = loaded_weights[offset:offset+n_biases]
    offset = offset + n_biases
    kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
    offset = offset + n_kernel_weights

    # IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
    # IMPORTANT: We would like to set these to Tensorflow order: (height, width, in_dim, out_dim)
    kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')
    kernel_weights = np.transpose(kernel_weights,[2,3,1,0])

    return biases,kernel_weights,offset


def load(sess,weights_path,ckpt_folder_path,saver):

    if(os.path.exists(ckpt_folder_path)):
        print('Found a checkpoint!') 
        checkpoint_files_path = os.path.join(ckpt_folder_path,"model.ckpt")
        saver.restore(sess,checkpoint_files_path)
        print('Loaded weights from checkpoint!') 
        return True

    print('No checkpoint found!') 
    print('Loading weights from file and creating new checkpoint...') 

    # Get the size in bytes of the binary
    size = os.path.getsize(weights_path)

    # Load the binary to an array of float32
    loaded_weights = []
    loaded_weights = np.fromfile(weights_path, dtype='f')

    # Delete the first 4 that are not real params...
    loaded_weights = loaded_weights[4:]

    print('Total number of params to load = {}'.format(len(loaded_weights)))

    # IMPORTANT: starting from offset=0, layer by layer, we will get the exact number of parameters required and assign them!

    # Conv1 , 3x3, 3->16
    offset = 0
    biases,kernel_weights,offset = load_conv_layer_bn('conv1',loaded_weights,[3,3,3,16],offset)
    sess.run(tf.assign(net.b1,biases))
    sess.run(tf.assign(net.w1,kernel_weights))

    # Conv2 , 3x3, 16->32
    biases,kernel_weights,offset = load_conv_layer_bn('conv2',loaded_weights,[3,3,16,32],offset)
    sess.run(tf.assign(net.b2,biases))
    sess.run(tf.assign(net.w2,kernel_weights))

    # Conv3 , 3x3, 32->64
    biases,kernel_weights,offset = load_conv_layer_bn('conv3',loaded_weights,[3,3,32,64],offset)
    sess.run(tf.assign(net.b3,biases))
    sess.run(tf.assign(net.w3,kernel_weights))

    # Conv4 , 3x3, 64->128
    biases,kernel_weights,offset = load_conv_layer_bn('conv4',loaded_weights,[3,3,64,128],offset)
    sess.run(tf.assign(net.b4,biases))
    sess.run(tf.assign(net.w4,kernel_weights))

    # Conv5 , 3x3, 128->256
    biases,kernel_weights,offset = load_conv_layer_bn('conv5',loaded_weights,[3,3,128,256],offset)
    sess.run(tf.assign(net.b5,biases))
    sess.run(tf.assign(net.w5,kernel_weights))

    # Conv6 , 3x3, 256->512
    biases,kernel_weights,offset = load_conv_layer_bn('conv6',loaded_weights,[3,3,256,512],offset)
    sess.run(tf.assign(net.b6,biases))
    sess.run(tf.assign(net.w6,kernel_weights))

    # Conv7 , 3x3, 512->1024
    biases,kernel_weights,offset = load_conv_layer_bn('conv7',loaded_weights,[3,3,512,1024],offset)
    sess.run(tf.assign(net.b7,biases))
    sess.run(tf.assign(net.w7,kernel_weights))

    # Conv8 , 3x3, 1024->1024
    biases,kernel_weights,offset = load_conv_layer_bn('conv8',loaded_weights,[3,3,1024,1024],offset)
    sess.run(tf.assign(net.b8,biases))
    sess.run(tf.assign(net.w8,kernel_weights))

    # Conv9 , 1x1, 1024->125
    biases,kernel_weights,offset = load_conv_layer('conv9',loaded_weights,[1,1,1024,125],offset)
    sess.run(tf.assign(net.b9,biases))
    sess.run(tf.assign(net.w9,kernel_weights))

    # These two numbers MUST be equal! 
    print('Final offset = {}'.format(offset))
    print('Total number of params in the weight file = {}'.format(len(loaded_weights)))


    # Saving checkpoint!
    if not os.path.exists(ckpt_folder_path):
        print('Saving new checkpoint to the new checkpoint directory ./ckpt/ !')
        os.makedirs(ckpt_folder_path)
        checkpoint_files_path = os.path.join(ckpt_folder_path, "model.ckpt")
        saver.save(sess,checkpoint_files_path)

#######################################################################################################################################