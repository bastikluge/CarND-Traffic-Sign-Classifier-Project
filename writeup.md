# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in [my implementation](Traffic_Sign_Classifier.html) of the [project code template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

[//]: # (Image References)

[img_label_count]: ./plots/class_label_occurance.png "Class label occurance"
[img_ex_double_curve]: ./examples/example_double-curve_32x32.png "Internet example: double curve"
[img_ex_limit_70]: ./examples/example_limit-70_32x32.png "Internet example: speed limit 70 km/h"
[img_ex_priority]: ./examples/example_priority-road_32x32.png "Internet example: priority-road"
[img_ex_road_works]: ./examples/example_road-works_32x32.png "Internet example: road works"
[img_ex_stop]: ./examples/example_stop_32x32.png "Internet example: stop"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is HxWxD = 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the test data set. It is a bar chart showing how often each output label occurs in the test data set.

![][img_label_count]

Here, the minimum number of occurances is 180 for label "Speed limit (20km/h)" and the maximum number of occurances is 2010 for label "Speed limit (50km/h)".

### Design and Test a Model Architecture

I started off my implementation with the LeNet architecture as proposed in the lecture. The only initial modifications I made were to normalize the pixel values as _pixel = (pixel - 128) / 128_, and to adapt the dimensions (input feature set size is 3 for RGB images, output label set size is 43 for the traffic signs). When running this model architecture for 50 epochs with a batch size of 128 and a learning rate of 0.001, I found that after approximately 20 epochs the model converged to an accuracy of 0.99 on the training set and an accuracy of 0.75 on the validation set, and oscillated around these accuracies during the remaining 30 epochs. This indicated to me that my initial implementation was overfitting the data of the training set. In the next subsections I will discuss the changes I made to the initial implementation in order to achieve a good accuracy also on the validation set.

#### Preprocessing of the data

After analyzing the error distributions on the validation sets of my first trained network, I observed that especially those classes were badly matched, for which the training set contained few (less than 500) examples. So as a first preprocessing step I implemented an augmentation of the training set, such that for each class there will be at least 1500 values. The techniques I used for augmentation were rotating, scaling, bulging, moving and shearing the images.

Although it didn't seem favorable to me at first glance (because in my human perception I rely on the coloring of traffic signs during their classification), I observed that a grayscale transformation of the input images lead to a significantly better matching rate than without (91% on grayscale images with initial net architecture as opposed to 75% on color images). Most likely, an overfitting of the color schemes of the training images made up a significant part of the overfitting observed in the first training run of the neural network. So my second preprocessing step was a grayscale transformation (Y value of YUV encoding of image) of the input images.

The third preprocessing step I applied was a normalization of the grayscale values according to _pixel = (pixel - 128) / 128_. I used this normalization to ensure that the model is fed with well-conditioned data. I tried also a different normalization technique (normalize on minimum and maximum value to ensure the value range is [-1.0, 1.0]), but the simple approach worked better in my experiments.

#### Final model architecture

My final model consisted of the 3 convolutional and 3 fully connected layers as shown in below table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 29x29x12 	|
| RELU					|												|
| Dropout		      	| 												|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18					|
| Convolution 4x4	    | 2x2 stride, valid padding, outputs 6x6x25		|
| RELU					|												|
| Dropout		      	| 												|
| Flattening			| outputs 900									|
| Fully connected		| outputs 300  									|
| RELU					|												|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|
 
The first convolutional layer applies a 4x4 kernel size using a stride size of 1x1 and outputs 12 feature dimensions, resulting in an output size of 29x29x12. Its architecture is completed by a RELU-activation and a dropout node. The second convolutional layer applies a 2x2 kernel size using a stride size of 1x1 and outputs 18 feature dimensions, resulting in an output size of 28x28x18. Its architecture is completed by a RELU-activation node and a max-pooling node with kernel size 2x2 and stride size of 2x2. The third convolutional layer applies a 4x4 kernel size using a stride size of 2x2 and outputs 25 feature dimensions, resulting in an output size of 6x6x25. Its architecture is completed by a RELU-activation and a dropout node. Note that a muliplication of the kernel sizes of the convolutional nodes results in the size of the traffic sign images.

The output of the convoluational layers is flattened to serve as input of size 900 to the first fully connected layer, which has an output size of 300. The architecture of the first fully connected layer is completed by a RELU activation. The second fully connected layer transforms the input to 150 output dimensions, activates it with RELU and applies a dropout node to avoid overfitting. The third fully connected layer, which is the last layer of the neural network, transforms the input to the 43 output dimension, where each output dimension is associated with one type of traffic sign.

For all layers, I initialized all weight matrices with a gaussian distribution with mean 0.0 and standard deviation 0.1. All offset vectors were initialized with 0.0.

#### Training method

To train the model, I minimized the reduced mean of the softmax cross entropy. As optimization method I chose the AdamOptimizer, a slightly refined gradient descent method, which was proposed in the lecture. One training epoch consisted of shuffling the training data, dividing it into batches and running the optimization method on each batch.

In my final solution, I used 30 epochs, a batch size of 128, a learning rate of 0.001 and a dropout probability of 0.5.

#### Iterative improvement of classification pipeline

After I had implemented a running prototype (with the LeNet architecture as described in the introduction of this section), I made a series of modifications to both preprocessing steps, training parameters and network configuration. For each modification I executed a test run and protocolled after how many epochs the network weights converged, and to which accuracies on training and validation set the model converged. I also wrote a textual output of the error distribution on the validation set, which provides the information to which other traffic signs each traffic sign was erroneously mapped. Based on these results and a visual inspection of the traffic signs, I defined measures to improve my implementation. In the next paragraphs, I will discuss the different results and measures in more detail.

In my first round of analysis, I worked with the original RGB version of the images, because I assumed the colors to be helpful for the classification process. Because the LeNet architecture had been defined for grayscale images (with one color channel), and the RGB images have 3 color channels, I suspected that the number of feature dimensions of the network architecture might be too low. Regardless of whether I increased it slightly (first layer from 6 to 10, second layer from 16 to 20) or a little more (first layer from 6 to 12, second layer from 16 to 24), I observed a model convergence after 15 epochs (as opposed to 15 with LeNet) to an accuracy of 77%-79% on the validation set (as opposed to 75% with LeNet).

In my second round of analysis, I added dropout nodes at different layers of the LeNet network in order to tackle the overfitting I had observed with the original LeNet architecture (accuracy on training set had converged to 99%, whereas accuracy on validation set got stuck around 75%). **@todo: Test again because there was a bug!**

As a next step, I tested the effect of transformation of the RGB image to YUV color space and to grayscale (motivated by the 2011 publication "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by P. Sermanet and Y. LeCun) using the LeNet architecture. In contrast to my initial expectation, I observed that a grayscale transformation of the input images lead to a significantly better matching rate than without (91%-93% on grayscale images as opposed to 75% on RGB images). Most likely, an overfitting of the color schemes of the training images made up a significant part of the overfitting observed in the first training run of the neural network. Interestingly also the YUV encoding of the images had a significantly positive effect on the accuracy on the validation set (85%-88% on YUV images as opposed to 75% on RGB images). The extent of the improvement is surprising as the RGB->YUV transformation is a linear operation and therefore has the same structure as a linear node of a neural network. So the transformation can be interpreted as an additional network layer. I also observed that, similar as for RGB images, the training procedure converged after 15 epochs.

In another round of analysis, I tested whether a different normalization scheme would have a positive effect on the prediction accuracy of the neural network. I compared the simple normalization _pixel = (pixel - 128) / 128_ with the slightly more sophisticated normalization _pixel = (pixel - min(pixels)) / (max(pixels) - min(pixels)) - 0.5', which scales each image channel to the interval [-0.5, 0.5]. I measured this to have a slightly positive effect on the YUV image (approximately 1%) and a slightly negative effect on the grayscale image (approximately -1%). The convergence rate remained unchanged in both cases.

As this configuration had resulted in the best prediction accuracy on the previous experiments, I continued my experiments on grayscale images with simple normalization using the LeNet network architecture. Because of the oscillation of the prediction accuracy, which I had observed in the training epochs after convergence, I next decided to modify both batch size and learning rate. I observed that increasing the batch size to 512 and 1024 resulted in later convergence (after 20 and 42 epochs, respectively) and lower prediction accuracy (90% and 89%, respectively). Also a variation of the learning rate didn't pay of in my experiments: When increasing the learning rate (to 0.002 and 0.005) I observed greater oscillations of the prediction accuracy after convergence, and when decreasing the learning rate (to 0.0005 and 0.0002) I observed a slight degradation in prediction accuracy (90% and 89%, respectively). As expected the required number of epochs until convergence slightly decreased when increasing the learning rate (approximately 10 epochs) and significantly increased when decreasing the learning rate (approximately 25 and 40 epochs, respectively).

As none of the above experiments had resulted in a classification pipeline near to the required accuracy, I next took a closer look at the images and at the error distributions observed in the grayscale image and LeNet setup.
**@todo**

**1st shot (as is)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 20 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 75%-79%

**2nd shot (increase feature set of convolution layers a bit)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x20	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20					|
| Flattening			| outputs 500									|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 77%-79%

**3rd shot (increase feature set of convolution layers some more)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24					|
| Flattening			| outputs 600									|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 77%-79%

**4th shot (add dropout to convolution layers)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Dropout		      	| 												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24					|
| Dropout		      	| 												|
| Flattening			| outputs 600									|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 35 epochs with batch size 128 and learning rate 0.001: training accuracy 85-87%, validation accuracy 66%-69%

**5th shot (add dropout to fully connected layers)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24					|
| Flattening			| outputs 600									|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|

After 30 epochs with batch size 128 and learning rate 0.001: training accuracy 93-95%, validation accuracy 75-77%

**6th shot (go back to initial and transform to YUV)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 85%-88%

**7th shot (YUV with min-max normalization)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 89%-90%

**8th shot (YUV + dropout in fully connected layers)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24					|
| Flattening			| outputs 600									|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|

After 30 epochs with batch size 128 and learning rate 0.001: training accuracy 94%, validation accuracy 88%.

**9th shot (Y grayscale)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 13 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 91%-93%.

**10th shot (Y grayscale with min-max normalization)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image) min-max-normalized	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 89%.

**11th shot (Y grayscale with dropout)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|

After 25 epochs with batch size 128 and learning rate 0.001: training accuracy 95%, validation accuracy 89%.

**12th shot (Y grayscale with very big batch size)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 20 epochs with batch size 512 and learning rate 0.001: training accuracy 99%, validation accuracy 90%.

**13th shot (Y grayscale with very big batch size)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 42 epochs with batch size 1024 and learning rate 0.001: training accuracy 99%, validation accuracy 89%.

**14th shot (Y grayscale with bigger learning rate)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 10 epochs with batch size 128 and learning rate 0.002: training accuracy 99%, validation accuracy 90%-93%.

**15th shot (Y grayscale with big learning rate)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.005: training accuracy 99%, validation accuracy 91%-95%.

**16th shot (Y grayscale with smaller learning rate)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 25 epochs with batch size 128 and learning rate 0.0005: training accuracy 99%, validation accuracy 89%-90%.

**17th shot (Y grayscale with small learning rate)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 40 epochs with batch size 128 and learning rate 0.0002: training accuracy 99%, validation accuracy 88%.

**18th shot (Y grayscale with augmented data)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 400									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 10 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 95%.

**19th shot (Y grayscale with increased depth of convolution layers)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flattening			| outputs 800									|
| Fully connected		| outputs 200  									|
| RELU					|												|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 5 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 92%-94%.

**20th shot (Y grayscale with additional convolution layer)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 29x29x12 	|
| RELU					|												|
| Dropout		      	| 												|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18					|
| Convolution 4x4	    | 2x2 stride, valid padding, outputs 6x6x25		|
| RELU					|												|
| Dropout		      	| 												|
| Flattening			| outputs 900									|
| Fully connected		| outputs 300  									|
| RELU					|												|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Fully connected		| outputs 43  									|

After 10 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 94%.

**21th shot (Y grayscale with additional convolution layer and additional dropout layer)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 29x29x12 	|
| RELU					|												|
| Dropout		      	| 												|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18					|
| Convolution 4x4	    | 2x2 stride, valid padding, outputs 6x6x25		|
| RELU					|												|
| Dropout		      	| 												|
| Flattening			| outputs 900									|
| Fully connected		| outputs 300  									|
| RELU					|												|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|

After 15 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 96%.

**22th shot (Y grayscale with additional convolution layer, additional dropout layer and augmented data)**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale (Y of YUV image)			| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 29x29x12 	|
| RELU					|												|
| Dropout		      	| 												|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18					|
| Convolution 4x4	    | 2x2 stride, valid padding, outputs 6x6x25		|
| RELU					|												|
| Dropout		      	| 												|
| Flattening			| outputs 900									|
| Fully connected		| outputs 300  									|
| RELU					|												|
| Fully connected		| outputs 150  									|
| RELU					|												|
| Dropout		      	| 												|
| Fully connected		| outputs 43  									|

After 8 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 96%-97%.

=> Analyzed error distribution: Classes with few occurances in test set couldn't be matched well on the validation set (over-fitting to test set). Should use data augmentation.
YUV: Speed limit 20km/h could only be matched in less than 30% of the cases (mostly matched to speed limit 30km/h) because of low occurance in training set. Other speed limits (30km/h, 60km/h) were also matched to wrong speed categories. "End of speed limit" was matched in 100% of the cases despite of the low frequency of this class in the training set, most likely due to its unique color scheme. Surprisingly "no passing" was also matched to speed limits. Other signs with bad matching quality were "General caution", "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Roundabout mandatory", which were all erroneously matched to several other signs with same shape and color. It could be that for this sign the color matching was given a bigger weight than the matching of the inner shape. Analysis of the error distribution for a grayscale classifier will shed more light on this. Other examples with bad matching rate were the ones with little frequency in the training set: "Turn left ahead, "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons".
2 main problems for YUV images: 
(1) Color is weighted more important than shape in some cases.
(2) Some classes have little occurance in training set.
Both problems relate to overfitting of the neural net to the training data.
Y-grayscale: Speed limit 20km/h could be matched in 60% of the cases (mostly matched to speed limit 70km/h) because of low occurance in training set. The same bad matching result was observed for "Turn left ahead, "Roundabout mandatory", "End of no passing". Other speed limits (30km/h, 60km/h, 120km/h) were also matched to wrong speed categories. "End of no passing by vehicles over 3.5 metric tons". "End of speed limit" was not matched as well as when using color, as the unique coloring of the sign is not available any more. Other signs with bad matching quality were "General caution", "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Roundabout mandatory", which were all erroneously matched to several other signs with same shape and color. So the grayscale transformation didn't really help to analyze this.

=> The inner part of signs is badly recognized despite of YUV or grayscale encoding. Might need smaller convolution kernel.

=> Oscillation in the end of the convergence: might need lower learning rate and/or bigger batch size

=> Observation: Dropout leads to more iterations until convergence.

**@todo**

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Example 1: Double curve][img_ex_double_curve]
![Example 2: Speed limit 70 km/h][img_ex_limit_70]
![Example 3: Priority road][img_ex_priority]
![Example 4: Road works][img_ex_road_works]
![Example 5: Stop sign][img_ex_stop]

I expected examples 2-4 to be simple to classify because of fairly good light conditions and fairly regular background. Example 1 has a irregular backround and the sign shows some reflections, example 5 was made with bad light conditions. Because of this, I expected examples 1 and 5 to be harder to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve     		| **@todo**   									| 
| Speed limit 70 km/h	| **@todo**										|
| Priority road			| **@todo**										|
| Road works      		| **@todo**						 				|
| Stop sign				| **@todo**		      							|


The model was able to correctly guess **@todo** of the 5 traffic signs, which gives an accuracy of **@todo**%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


