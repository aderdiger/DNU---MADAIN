# Skin-Cancer-Prediction-ML-Model_2

Skin Cancer Preditiction
Group 2
Project 4

Our group has decided to develop a machine learning model that can predict whether a mole is cancerous or benign. To accomplish this, we will be using a kaggle data set that contains 10,015 images of moles accross 7 classes. To accomplish our goal, we will start by benchmarking 3 convolutional neural networks (CNN) detailed in Aurelien Geron's book "Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow" (2022). These three CNNs are: InceptionV3, ResNet50, and VGG16. In addition to the three CNN architectures, we will also test 3 different optimizers for each: Adam, RMSprop, and SGD. In total, 9 models will be benchmarked, and from those 9, 1 will be chosen as our primary model. 

Once our primary model has been chosen, we will tune the model in an effor to increase our precision and recall scores. 

Once we have tuned our model, we hope to input new images into our trained model and recieve a predictied classification output. We intend to host our model on either github pages or andrewkoller.org, and plan to use streamlit to help us build a webapp for our model. Presentation style is subject to change, however. 

Steps Already Taken

1. Initial developement and POC (past_versions v1-3)
2. Model Benchmarking (run1)
    a. Bug: Did not run ResNet50 with correct preprocessing function
    b. Best two models from testing: InceptionV3.adam, ResNet50.adam
3. Retesting ResNet50 with correct preprocessing function (run2)
    a. Retest indicated no major changes between original preprocessing function and new preprocessing function
4. Running InceptionV3.adam at 150 epochs (run3)
    a. Removed image augmentation 
    b. Results from testing do not show improvement with higher epochs. 
    c. Resolution: test at lower epochs; weights need to be adjusted 
5. Weighting scheme testing (run4)
    a. All testing to this point utilized TensorFlow's 'balanced' weighting system to account for large imblanace in classes.
    b. Tested effectiveness of different weights 
6. Binary classification testing (run5)
    a. Testing conducted at same time as weighting scheme testing
    b. All testing to this point involved a multiclass classifier.
    c. Tested to effectiveness of a binary classifier as opposed to a multiclass classifier.
    d. Did not make any difference from previous models
7. Custom weighting scheme 1 (run7)
8. Custom weight scheme 1 (run8)
9. Class balanced loss approach weighting (run9)
    a. Attempted to implement balanced loss weighting, model performed poorly
10. Adding generated augmented images to training data (run10)
    a. Added a random imgage augementor and image generator 
    b. Added randomly generated images back into training data 
    c. Wanted to normalize percentage representation in data set of underrepresented classes
11. Increasing custom layer neuron density from 512 to 1024 and rerunning promissing models
    a. Top performers are as follows: InceptionV3.Adam, ResNet50.Adam, VGG16.SGD
    b. Will more than likely continue forward with InceptionV3.Adam
    c. May also run ResNet50 in tandum, as epoch count does not seem to make a drastic difference. (As determined in run3)
12. Augmented image generation with 1000 images for underrepresented classes (run12)
    a. Results not impressive


https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc

https://stackoverflow.com/questions/51798784/keras-transfer-learning-on-inception-resnetv2-training-stops-in-between-becaus