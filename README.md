# TinyYOLOv2 in Tensorflow made easier

## This code: extract weights from binary file, assigns them to TF network, saves ckpt, performs detection on an input image or webcam

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
- If you are launching them for the first time, the weights will be extracted from the binary file and a ckpt will be created. Next time only the ckpt will be used!

### Requirements:

I've implemented everything with Tensorflow 1.0, Ubuntu 16.04, Numpy 1.13.0, Python 3.4, OpenCV 3.0



#### How to use the binary weights file ( Only if you want to use it in another projects, here it is already done ) 

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

#### How to postprocess the predictions ( Only if you want to use it in another projects, here it is already done ) 

Another key point is how the predictions tensor is made. It is a 13x13x125 tensor. To process it better:

- Convert the tensor to have shape = 13x13x5x25 = grid_cells x n_boxes_in_each_cell x n_predictions_for_each_box
- The 25 predictions are: 2 coordinates and 2 shape values (x,y,h,w), 1 Objectness score, 20 Class scores
- Now access to the tensor in an easy way! E.g. predictions[row, col, b, :4] will return the 2 coords and shape of the "b" B-Box which is in the [row,col] grid cell
- They must be postprocessed according to the parametrization of YOLOv2. In my implementation it is made like this: 

```python

# Pre-defined anchors shapes!
# They are not coordinates of the boxes, they are height and width of the 5 anchors defined by YOLOv2
anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
image_height = image_width = 416
n_grid_cells = 13
n_b_boxes = 5

for row in range(n_grid_cells):
  for col in range(n_grid_cells):
    for b in range(n_b_boxes):

      tx, ty, tw, th, tc = predictions[row, col, b, :5]
      
      # IMPORTANT: (416) / (13) = 32! The coordinates and shape values are parametrized w.r.t center of the grid cell
      # They are parameterized to be in [0,1] so easier for the network to predict and learn
      # With the iterations on every grid cell at [row,col] they return to their original positions
      
      # The x,y coordinates are: (pre-defined coordinates of the grid cell [row,col] + parametrized offset)*32 
      center_x = (float(col) + sigmoid(tx)) * 32.0
      center_y = (float(row) + sigmoid(ty)) * 32.0

      # Also the width and height must return to the original value by looking at the shape of the anchors
      roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
      roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

      final_confidence = sigmoid(tc)

      class_predictions = predictions[row, col, b, 5:]
      class_predictions = softmax(class_predictions)
      
```

YOLOv2 predicts parametrized values that must be converted to full size by multiplying them by 32! You can see other EQUIVALENT ways to do this but this one works fine. I've seen someone who, instead of multiplying by 32, divides by 13 and then multiplies by 416 which at the end equals a single multiplication by 32.


#### Notes

- The code runs at ~15fps on my laptop which has a 2GB Nvidia GeForce GTX 960M GPU
- This implementation does not have the training part, I'm working on it! 

If you have questions or suggestions do not wait! I'm looking forward to help
