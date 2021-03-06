# **Behavioral Cloning** 

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890.png "Model Visualization"
[model1]: ./images/model.png "Model"
[image2]: ./images/train-stats.png "Metrics"
[image3]: ./images/train-stats-2.png "Metrics (resumed)"
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

The ```model.py``` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The ```cnn.py``` file contains the actual neural network model. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model ```cnn.py``` is based on the CNN architecture presented at https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

![alt text][image1] (Courtesy: NVidia)

It consists of a convolution neural network with 5x5 and 3x3 filters sizes and depths between 3 and 64 (```cnn.py``` lines 13-39) 

The model includes RELU layers to introduce nonlinearity, dropout layers for generalization and the data is normalized in the model using a Keras BatchNormalization layer. 

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, as mentioned previously my modifications to the model include:

* A ```ELU``` activation layer after each fully connected layer
* A dropout applied after each fully connected layer
* L2 regularization as part of each fully connected layer

About 1.5GB of data was also provided as the training set to reduce overfitting.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of following scenarios:

* Center driving, and center driving in opposite direction
* Right-hand side driving and left-hand side driving, and in opposite direction each as well
* Zigzag driving for recovering training, and in opposite direction as well

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training (30%) and validation set (70%).

My first step was to use a simple neural network model with just 2 Conv2D layers and 2 fully connected layers, just get started. Later I switched to use the architecture built by NVidia.

I found that the early version of the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include max pool dropout and regular dropout, learned from the paper mentioned earlier. It turns out to be working well. Training loss and validation loss eventually converge:

![image2]

However both numbers are still a bit high. The test drive still has problems when dealing with turning at curves. The situation does not improve with either extra epochs or data. Per feedback from Udacity mentor, it's probably due to difference between ```cv2.imread()``` and ```drive.py```. Along with fixing the color space, also made few updates such as removing unneccessary dropouts from ```Conv2D``` layers, switching to use ```ELU``` activation from ```ReLU``` activation, introducing ```EarlyStopping``` callback, etc. Afterall, the metrics showed some improvements:

![image3]


#### 2. Final Model Architecture

![model1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving, where 2 laps are default direction and the other 2 are opposite direction to help generalization. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle driving in zigzags for 2 laps, 1 in default direction and 1 in opposite direction. Here are two images showing 

![alt text][image5]
![alt text][image6]

However, there is a problem. As part of the zigzag driving, I had to continue heading to the other side with angle = 0, after recovering from the other side. The training would learn from the images while the angle is 0 but heading to the side, and think that it is what it needs to learn. To solve the issue, I went to the ```driving_log.csv``` file, removed rows where angle equals to 0, as well as the corresponding images. After that, the recovery behavior is working.

The code would first read all rows from each ```driving_log.csv``` from each recording, aggregate them into one big list, then shuffle it.

I wrote a ```generator()``` function which takes the shuffled list, and pass the generator to ```fit_generator()``` of the model when training. This way it is able to train all the data without worrying about memory limit, but it is not loading everything into memory all at once. I also use a ```validation``` parameter in the generator function so when it is ```True``` it will return a validation set for validation purpose.

As part of the generator, it would flip each image from the data set, as well as the measured angle, in order to augment for more data.

I had about 65k+ number of data points, including images from augmentation.

I used an Adam optimizer so that manually training the learning rate wasn't necessary.

A ```--resume``` parameter is implemented for ```model.py```. When it is specified, Keras will load weights from existing model and continue the training. This way it does not need to train from scratch every time new data is available. 

See ```video.mp4``` for the final result. 
