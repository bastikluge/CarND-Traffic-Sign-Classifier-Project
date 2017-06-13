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

I started off my implementation with the LeNet architecture as proposed in the lecture. The only initial modifications I made were to normalize the pixel values as _pixel = (pixel - 128) / 128_, and to adapt the dimensions (input feature set size is 3 for RGB images, output label set size is 43 for the traffic signs). When running this model architecture for 50 epochs with a batch size of 128 and a learning rate of 0.001, I found that after approximately 20 epochs the model converged to an accuracy of 99% on the training set and an accuracy of 75% on the validation set, and oscillated around these accuracies during the remaining 30 epochs. This indicated to me that my initial implementation was overfitting the data of the training set. In the next subsections I will discuss the changes I made to the initial implementation in order to achieve a good accuracy also on the validation set.

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

The output of the convolutional layers is flattened to serve as input of size 900 to the first fully connected layer, which has an output size of 300. The architecture of the first fully connected layer is completed by a RELU activation. The second fully connected layer transforms the input to 150 output dimensions, activates it with RELU and applies a dropout node to avoid overfitting. The third fully connected layer, which is the last layer of the neural network, transforms the input to the 43 output dimension, where each output dimension is associated with one type of traffic sign.

For all layers, I initialized all weight matrices with a gaussian distribution with mean 0.0 and standard deviation 0.1. All offset vectors were initialized with 0.0.

#### Training method

To train the model, I minimized the reduced mean of the softmax cross entropy. As optimization method I chose the AdamOptimizer, a slightly refined gradient descent method, which was proposed in the lecture. One training epoch consisted of shuffling the training data, dividing it into batches and running the optimization method on each batch.

In my final solution, I used 15 epochs, a batch size of 128, a learning rate of 0.001 and a dropout probability of 0.5.

#### Iterative improvement of classification pipeline

After I had implemented a running prototype (with the LeNet architecture as described in the introduction of this section), I made a series of modifications to both preprocessing steps, training parameters and network configuration. For each modification I executed a test run and protocolled after how many epochs the network weights converged, and to which accuracies on training and validation set the model converged. I also wrote a textual output of the error distribution on the validation set, which provides the information to which other traffic signs each traffic sign was erroneously mapped. Based on these results and a visual inspection of the traffic signs, I defined measures to improve my implementation. In the next paragraphs, I will discuss the different results and measures in more detail.

In my first round of analysis, I worked with the original RGB version of the images, because I assumed the colors to be helpful for the classification process. Because the LeNet architecture had been defined for grayscale images (with one color channel), and the RGB images have 3 color channels, I suspected that the number of feature dimensions of the network architecture might be too low. Regardless of whether I increased it slightly (first layer from 6 to 10, second layer from 16 to 20) or a little more (first layer from 6 to 12, second layer from 16 to 24), I observed a model convergence after 15 epochs (as opposed to 15 with LeNet) to an accuracy of 77%-79% on the validation set (as opposed to 75% with LeNet).

In my second round of analysis, I added dropout nodes at different layers of the LeNet network in order to tackle the overfitting I had observed with the original LeNet architecture (accuracy on training set had converged to 99%, whereas accuracy on validation set got stuck around 75%). When adding dropout nodes at the end of each convolutional layer I observed a degraded accuracy on both training data (only 85%-87%) and validation data (66%-69%), which indicated that using both max-pooling and dropout was too strong regularization for the convolutional layers. When adding dropout nodes at the end of the first 2 fully connected layers, I measured a training accuracy of 96% and an improved validation accuracy of 80%. This result suggested that dropout in the fully connected layers could actually improve the overfitting I had observed with the initial network architecture. In both experiments, I observed that the dropout nodes resulted in a later convergence of the training process (after 30 and 35 epochs, respectively).

As a next step, I tested the effect of transformation of the RGB image to YUV color space and to grayscale (motivated by the 2011 publication "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by P. Sermanet and Y. LeCun) using the LeNet architecture. In contrast to my initial expectation, I observed that a grayscale transformation of the input images lead to a significantly better matching rate than without (91%-93% on grayscale images as opposed to 75% on RGB images). Most likely, an overfitting of the color schemes of the training images made up a significant part of the overfitting observed in the first training run of the neural network. Interestingly also the YUV encoding of the images had a significantly positive effect on the accuracy on the validation set (85%-88% on YUV images as opposed to 75% on RGB images). The extent of the improvement is surprising as the RGB->YUV transformation is a linear operation and therefore has the same structure as a linear node of a neural network. So the transformation can be interpreted as an additional network layer. I also observed that, similar as for RGB images, the training procedure converged after 15 epochs.

In another round of analysis, I tested whether a different normalization scheme would have a positive effect on the prediction accuracy of the neural network. I compared the simple normalization _pixel = (pixel - 128) / 128_ with the slightly more sophisticated normalization _pixel = (pixel - min(pixels)) / (max(pixels) - min(pixels)) - 0.5_, which scales each image channel to the interval [-0.5, 0.5]. I measured this to have a slightly positive effect on the YUV image (approximately 1%) and a slightly negative effect on the grayscale image (approximately -1%). The convergence rate remained unchanged in both cases.

As this configuration had resulted in the best prediction accuracy on the previous experiments, I continued my experiments on grayscale images with simple normalization using the LeNet network architecture. Because of the oscillation of the prediction accuracy, which I had observed in the training epochs after convergence, I next decided to modify both batch size and learning rate. I observed that increasing the batch size to 512 and 1024 resulted in later convergence (after 20 and 42 epochs, respectively) and lower prediction accuracy (90% and 89%, respectively). Also a variation of the learning rate didn't pay of in my experiments: When increasing the learning rate (to 0.002 and 0.005) I observed greater oscillations of the prediction accuracy after convergence, and when decreasing the learning rate (to 0.0005 and 0.0002) I observed a slight degradation in prediction accuracy (90% and 89%, respectively). As expected the required number of epochs until convergence slightly decreased when increasing the learning rate (approximately 10 epochs) and significantly increased when decreasing the learning rate (approximately 25 and 40 epochs, respectively).

As none of the above experiments had resulted in a classification pipeline near to the required accuracy, I next took a closer look at the images and at the error distributions observed with the LeNet setup for YUV and grayscale images and discussed the results with fellow students of the Self-Driving-Car Nanodegree. Regardless of the image format, I found that those classes with few occurances (e.g., 'Speed limit 20 km/h', 'Turn left ahead', 'Roundabout mandatory', 'End of no passing') had a bad matching rate. To alleviate this issue, I decided to augment the training data for these classes of traffic signs. I also observed that, even if there was a large number of training samples available, some classes were erroneously matched as others: One type of traffic signs, for which I observed this were speed limits, which were often matched to wrong top speed. Another class of traffic signs, namely the triangular signs with red frame, showed bad matching results by confusing the classes 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Roundabout mandatory'. When visually inspecting the size of the inner part of these signs in the provided image data, I observed that they were mostly contained in a rectangular area of approximately 8 pixels edge length. In comparison to the MINST data, for which the LeNet architecture had worked out well, the digits of the speed limits (and also the other inner parts of the traffic signs) therefore had to be matched with a fraction of the pixel size. Because I suspected that for this image complexity within a small area the convolution layers might not be well dimensioned I decided to test with both a significantly greater number of feature dimensions and a different number and size of convolution kernels.

After I had implemented augmentation of the training data (by randomly rotating, scaling, bulging, moving and shearing the images from classes with less than 1500 occurances), I observed quicker convergence (only 10 epochs instead of 15) to an accuracy of approximately 95% on the validation set.

To test my other theory independently I next continued to measure the effect of a different dimensioning of the convolution layers on the original (not augmented) data. When increasing the feature dimensions of first and second convolution layer to 12 and 32, respectively, and using fully connected layers of sizes 800->200->120->43, I observed very quick convergency (only 5 epochs instead of 15) to an accuracy of approximately 92%-94% on the validation set. With redesigned convolutional layers (first layer: kernel size 4x4 with stride size 1 followed by dropout; second layer: kernel size 2x2 with stride size 1 followed by max pooling; third layer: kernel size 4x4 with stride size 2 followed by dropout) I measured an accuracy of 94% on the validation set, to which the training process converged after approximately 10 epochs. As the number of network parameters had significantly increased with this design, and I wanted to counteract possible overfitting, I executed another training run with the same network architecture but an additional dropout node before the last fully connected layer (i.e., the final network architecture). This resulted in convergence to an accuracy of 96% on the validation set after 15 epochs.

As final solution I combined the approaches described in the last two paragraphs, i.e., I trained the network on augmented data with the network architecture in which I had added an additional convolutional layer and a dropout node before the output layer. With this solution, the training process converged after 10-12 epochs to an accuracy of 96%-97% on the validation set. In my final training run consisting of 15 epochs, I measured a final accuracy on the training set of 99.4%, on the validation set of 96.2% and on the test set of 96.1%.

As a next step to further improve the matching, I could have started another round of analysis of the error distribution. Probably also further augmentation (possibly with different image transformations) could have improved the matching rate. According to the literature a greater number of feature dimensions in the convolution layers could also further improve the matching rate.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Example 1: Double curve][img_ex_double_curve]
![Example 2: Speed limit 70 km/h][img_ex_limit_70]
![Example 3: Priority road][img_ex_priority]
![Example 4: Road works][img_ex_road_works]
![Example 5: Stop sign][img_ex_stop]

I expected examples 2-4 to be simple to classify because of fairly good light conditions and fairly regular background. Example 1 is not contained in the training data. It is a double curve, consisting of a right and a left turn, whereas the training data only contains a double curve consisting of a left and a right turn. It should therefore be impossible to match this traffic sign. Example 5 was made with bad light conditions, because of which I expected it to be harder to classify.

Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Double curve (mirrored)	| Children crossing								| 
| Speed limit 70 km/h		| Speed limit (70km/h)							|
| Priority road				| Road work										|
| Road works      			| Priority road					 				|
| Stop sign					| Stop			      							|

The model was able to correctly predict 4 of the 5 traffic signs, which gives an accuracy of 80%. Since the traffic sign, which could not be correctly predicted was the right-left double curve, which is not contained in the training data, the network predicted 100% of the trained classes correctly. Given the small size of this internet test set, the prediction accuracy is similar to the prediction accuracy of the official test set of 96% (which rounds to 100% when considering a resolution of 20% steps). In the next paragraphs, I will discuss the top five soft max probabilities associated with these traffic signs in more detail.

For the first image (mirrored double curve), the model is relatively sure that this is a 'Children crossing' sign (probability of 83.7%), followed by a probability of 8.5% for the 'Dangerous curve to the right' sign. All probabilities are listed in the table below:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 83.7%        			| Children crossing								| 
|  8.5%   				| Dangerous curve to the right					|
|  5.3%					| Slippery road									|
|  1.6%	      			| Road work						 				|
|  0.8%				    | Pedestrians	      							|

The matching result is plausible, given the fact that the sign was not contained in the training data, as both 'Children crossing' and 'Dangerous curve to the right' are signs of same color and shape, and with comparably shaped black marking in the middle.

For the second image 'Speed limit (70km/h)' the top five soft max probabilities are spread over speed limits with different top speeds:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 59.2%        			| Speed limit (70km/h)							| 
| 20.3%   				| Speed limit (20km/h)							|
| 19.4%					| Speed limit (30km/h)							|
|  0.6%	      			| Speed limit (50km/h)			 				|
|  0.3%				    | Speed limit (120km/h)							|

In fact I had observed after another training run on the same network architecture (with less epochs) that the sign was erroneously matched to 'Speed limit (20km/h)'. Also during my analysis of the error distributions (as described in subsection _Iterative improvement of classification pipeline_) I had observed that the 70km/h and the 20km/h speed limit signs were sometimes interchanged.

The remaining example images showed almost perfect prediction probability for the correct sign. The top five softmax probabilities are provided below for completeness.

Third image 'Road work':

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| > 99.9%        		| Road work										| 
| <  0.1%   			| Slippery road									|
| <  0.1%				| Dangerous curve to the right					|
| <  0.1%     			| Bicycles crossing				 				|
| <  0.1%			    | Bumpy road									|

Fourth image 'Priority road':

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| > 99.9%        		| Priority road									| 
| <  0.1%   			| Roundabout mandatory							|
| <  0.1%				| Keep right									|
| <  0.1%     			| No vehicles					 				|
| <  0.1%			    | Speed limit (60km/h)							|

Fifth image 'Stop':

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|   99.7%        		| Stop											| 
|    0.2%   			| Speed limit (50km/h)							|
| <  0.1%				| Speed limit (30km/h)							|
| <  0.1%     			| Keep right					 				|
| <  0.1%			    | Speed limit (70km/h)							|