# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./images/train-stats.png "Metrics"
[image3]: ./images/resumed-train-stats.png "Metrics (resumed)"
[image4]: ./images/center.jpg "Center lane driving"
[image5]: ./images/recover_left.jpg "Recover from left"
[image6]: ./images/recover_right.jpg "Recover from right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* ```model.py``` containing the script to train the CNN model
* ```cnn.py``` containing a CNN architectured by NVidia
* ```drive.py``` for driving the car in autonomous mode
* ```model.h5``` containing a trained convolution neural network 
* ```video.mp3``` for video recording of the result

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model (model2.py) is based on the CNN architecture presented at https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

To reduce overfitting, my modifications to the model include:

* A ReLU activation layer after each Conv2D layer and fully connected layer
* A max-pool dropout applied after each Conv2D ReLU layer, and a dropout applied after each fully connected layer. (https://arxiv.org/ftp/arxiv/papers/1512/1512.00242.pdf)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of following scenarios:

* Center driving, and center driving in opposite direction
* Right hand side driving and left hand side driving, and in opposite direction each as well
* Zigzag driving for recovering training, and in opposite direction as well

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training (30%) and validation set (70%).

My first step was to use a simple neural network model with just 2 Conv2D layers and 2 fully connected layers, just get started. Later I swiched to use the architecture built by NVidia.

I found that the early version of the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include max pool dropout and regular dropout, learned from the paper mentioned earlier. It turns out to be working well. Training loss and validation loss eventually converge:

![image2]

However both numbers are still a bit high. The test drive still has problems when dealing with turning at curves. The situation does not improve with extra epochs. So I decided to drive the simulation few extra rounds to provide more training data. Afterall, the metrics showed some improvements:

![image3]


#### 2. Final Model Architecture


Here is a visualization of the reference architecture

![alt text][image1] (Courtesy: NVidia)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving, where 2 laps are default direction and the other 2 are opposite direction to help generalization. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle driving in zigzags for 2 laps, 1 in default direction and 1 in opposite direction. Here are two images showing 

![alt text][image5]
![alt text][image6]

However there is a problem. As part of the zigzag driving I had to continue heading to the other side with angle = 0, after recovering from the other side. The training would learn from the images while angle is 0 but heading to the side, and think that it is what it needs to learn. To solve the issue, I went to the ```driving_log.csv``` file, removed rows where angle equals to 0, as well as the corresponding images. After that the recover behavior is working.

The code would first read all rows from each ```driving_log.csv``` from each recording, aggregate them into one big list, the shuffle it.

I wrote a ```generator()``` function which takes the shuffled list, and pass the generator to ```fit_generator()``` of the model when training. This way it is able to train all the data without worrying about memory limit, but it is not loading everything into memory all at once. I also use a ```validation``` parameter in the generator function so when it is ```True``` it will return a validation set for validation purpose.

As part of the generator it would flip each image from the data set, as well as the measured angle, in order to augment for more data.

I had about 65k+ number of data points, including images from augmentation.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

A ```--resume``` parameter is implemented for ```model.py```. When it is specified, Keras will load weights from existing model and continue the training. This way it does not need to train from scratch every time new data is available. 

See ```video.mp4``` for the final result. 
