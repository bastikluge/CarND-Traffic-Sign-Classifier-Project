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

The only preprocessing I applied was a normalization of the RGB values according to _pixel = (pixel - 128) / 128_. I used this normalization to ensure that the model is fed with well-conditioned data. Conversion to grayscale didn't seem favorable to me as (at least in human perception) the colors of the traffic sign already give a good indication of both the appearance and type of a traffic sign (signal color red is often used for the shape of warning signs, whearas blue or yellow signs indicate other types of sign). In fact, when testing the effect of grayscaling the input, I found that the model accuracy decreased significantly both on training set and on validation set (validation set below 5% with the setup described in the beginning of this section).

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

**@todo**

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....
**@todo**

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

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

After 13 epochs with batch size 128 and learning rate 0.001: training accuracy 99%, validation accuracy 91%.

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

**12th shot (Y grayscale)**

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

**13th shot (Y grayscale)**

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

After ?? epochs with batch size 512 and learning rate 0.001: training accuracy ??%, validation accuracy ??%.

Next steps:
- test with smaller learning rates
- test with 3 convolution layers 4x4->2x2->5x5
- implement augmentation

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

