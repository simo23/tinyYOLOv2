## TinyYOLOv2 in Tensorflow: extract weights from binary file, assigns them to net, saves ckpt, performs detection on an input image or webcam. 

I've been searching for a Tensorflow implementation of YOLOv2 but the darknet version and derivatives are not really easy to understand. This one is an hopefully easier-to-understand version of Tiny YOLOv2. The weight extraction, weights structure, weight assignment, network, inference and postprocessing are made as simple as possible.

The output of this implementation on the test image "dog.jpg" is the following:

![alt text](https://github.com/simo23/tinyYOLOv2/blob/master/dog_output.jpg "YOLOv2 output")

Just to be clear, this implementation is called "tiny-yolo-voc" on pjreddie's site and can be found here:

![alt text](https://github.com/simo23/tinyYOLOv2/blob/master/pjsite.png "YOLOv2 site")

#### This is a specific implementation of "tiny-yolo-voc" but the code could be re-used to import other configurations! Of course you will need to change the network architecture and hyperparameters according to the "cfg" file you want to use.

### The code is organized in this way:

- weights_loader.py : loads the weights from pjreddie's binary weights file into the tensorflow network and saves the ckpt
- net.py : contains the definition of the Tiny YOLOv2 network as defined in pjreddie's cfg file https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg
- test.py : performs detection on an input_image that you can define in the main. Outputs the input_image with B-Boxes
- test_webcam.py: performs detection on the webcam

### To use this code:

- Clone the project and place it where you want
- Download the binary file (~60MB) from pjreddie's site: https://pjreddie.com/media/files/tiny-yolo-voc.weights and place it into the folder where the scripts are
- Launch test.py or test_webcam.py. Change the input_img_path and the weights_path in the main if you want. The code is now configured to run with weights and input image in the same folder as the script. 
- If you are launching them for the first time, the binary file will be extracted and a ckpt will be created. Next time only the ckpt will be used!

### Requirements:

I've implemented everything with Tensorflow 1.0, Ubuntu 16.04, Numpy 1.13.0, Python 3.4, OpenCV 3.0

#### How to use the binary weights file

I've been struggling on understanding how the binary weights file was written. I hope to save you some time by explaining how I imported the weights into a Tensorflow network:

- Download the binary file from pjreddie's site: https://pjreddie.com/media/files/tiny-yolo-voc.weights 
- Extract the weights from binary to a numpy float32 array with  weight_array = np.fromfile(weights_path, dtype='f')
- Delete the first 4 numbers because they are not relevant
- Define a function ( load_conv_layer ) to take a part of the array and assign it to the Tensorflow variables of the net
- IMPORTANT: the weights order is [ 'biases','gamma','moving_mean','moving_variance','kernel'] 
- IMPORTANT: the 'biases' here refer to the beta value of the Batch Normalization. It does not refer to the biases that must be added after the conv2d because they are set all to zero! ( According to the paper by Ioffe et al. https://arxiv.org/abs/1502.03167 ) 
- IMPORTANT: the kernel weights are written in Caffe style which means they have shape = (out_dim, in_dim, height, width). They must be converted into Tensorflow style which has shape = (height, width, in_dim, out_dim)
- IMPORTANT: in order to obtain the correct results from the weights they need to be DENORMALIZED according to Batch Normalization. It can be done in two ways: define the network with Batch Normalization and use the weights as they are OR define the net without BN ( this implementation ) and DENORMALIZE the weights. ( details are in weights_loader.py )
- In order to verify that the weights extraction is succesfull, I check the total number of params with the number of weights into the weight file. They are both 15867885 in my case. 

#### Notes

- The code runs at ~15fps on my laptop which has a 2GB Nvidia GeForce GTX 960M GPU
- This implementation does not have the training part, I'm working on it! 

If you have questions do not wait! I'm looking forward to help
