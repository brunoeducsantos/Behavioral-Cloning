[//]: # (Image References)

[image1]:https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/architecture.jpg  "Model Visualization"
[image2]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/gray_scale.png "Grayscaling"
[image3]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/crossing_1.jpg "Recovery Image"
[image4]:https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/crossing_2.jpg "Recovery Image"
[image6]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/crossing_3.jpg "Recovery Image"
[image5]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/initia_image.png "Normal Image"
[image7]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/flip.png "Flipped Image"
[image8]: https://github.com/BrunoEduardoCSantos/Behavioral-Cloning/blob/master/images/crop2.png "Crop image"


## Project files 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

## Step by step running code procedure
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 filter sizes and depths of 32,48 and 64 (model.py lines 125,128,131).

The model includes RELU layers to introduce nonlinearity (code line 125,128,131). 

The model contains a dropout layer in order to reduce overfitting (model.py lines 138). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 145).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well image processing to reduce the number of input features, which means improve training of the model.

For details about how I created the training data, see the next section. 

####  Solution Design Approach

The overall strategy for deriving a model architecture was to start with one convolution network containing a small depth and one fully connected layer and from here increase depth and number of layers to reduce mean square error.

My first step was to use a convolution neural network model similar to the Nvidia Architecture since I thought this model might be appropriate because it was sucessfull in real self-driving car to similar tracks as the first track in the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it reduce the number of features and for instance the number of model parameters. For this purpose, I applied max-pooling after each convolutional layer.

Then I applied dropout after the fully connection layers to improve their training and avoiding overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track since it hasn't learnt to recover from marginal to center. In order to improve the driving behavior in these cases, I perform data augmentation to generate these cases and help the model to learn these situations. As a result, the model generalized and performed better to all driving behaviours.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

The final model architecture (model.py lines 123-132) consisted of a convolution neural network with the following layers and layer sizes:
* Filter size : (3,3) ; Depth layer: 32
* Filter size : (3,3) ; Depth layer: 48
* Filter size : (3,3) ; Depth layer: 64

Here is a visualization of the architecture:

![alt text][image1]

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover its position to the center of track once it tries to go off road or simply cross the marginal left or right lines.
These images show what a recovery looks like starting from left margin line to center of track:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would balance the number negative and positive steering angles. This way the model will learn for all spectrum of steering angles. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 40000 number of data points.

In addition, to improving training performance I a crop and resize image to 64X64.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the fast decreasing of mean squared error of both training and validation dataset. I used an adam optimizer so that manually training the learning rate wasn't necessary.
