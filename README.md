# MADAIN
## Mole Analysis with Deep Adam-optimized Inception Network

### By: Amanda Derdiger, Andrew Koller, Mustafa Can Ayter, and Natalia Mitchell

### Introduction
We have built a convolutional neural network (CNN) to analyze images of skin lesions and categorize them into one of seven classes, three of which are cancerous and four of which are benign. We have embedded our model into a web app, which is displayed on GitHub pages. 

![image](https://github.com/aderdiger/MADAIN/assets/148494444/e6325a45-a732-4b5b-a062-8a81e86cea94)

### Data
Our dataset is from Kaggle and can be accessed by the link below. This dataset contains 10,015 images of skin lesions across the 7 classes detailed above.

![demographics](https://github.com/aderdiger/MADAIN/assets/148494444/b67fe672-58c2-47b1-9d63-bf7c1cfeb654)


https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset

### Process

#### Benchmarking
We started by benchmarking three CNN architectures detailed in Aurelien Geron's book "Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow" (2022). These three CNNs were: InceptionV3, ResNet50, and VGG16. In addition to the three architectures, we tested 3 different optimizers for each: Adam, RMSprop, and SGD. In total, nine models were benchmarked, and from those, we chose InceptionV3 with the Adam optimizer as our primary model.*

* For benchmarking metrics, see "run1/visualizations/"

Despite the InceptionV3.Adam slightly underperforming relative to ResNet50.Adam in the classification reports, InceptionV3.Adam was chosen for it's supperior performance on AUC metrics. (see \run1\visualizations\roc_curve\roc_curve_InceptionV3_Adam.png). We chose our primary model to maximize our precision and recall scores, with recall on our three cancerous classes more highly prioritized. This is because, in the precision/recall trade-off, favoring recall reduces false negatives. In a cancer identification model, such as this, false negatives in the cancerous classes would be our most detrimental outcome that should be minimized to the extent possible. 

#### Fine-Tuning the Model
Once we chose our primary model, we continued to fine-tune it in an effort to increase our precision and recall scores, again prioritizing recall on the three cancerous classes. Our fine-tuning steps, along with their corresponding run folders in our repo are detailed below.

1. Running InceptionV3.adam at 150 epochs (run3; v6)
    a. Removed image augmentation - original benchmarking involved preliminary image augmentation:

        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,

    b. Results from testing do not show improvement with higher epochs. 

    ![image](https://github.com/aderdiger/MADAIN/blob/main/run3/visualizations/roc_curve_InceptionV3_Adam.png)

    precision    recall  f1-score   support

       akiec       0.02      0.05      0.03        65
         bcc       0.05      0.09      0.07       103
         bkl       0.08      0.07      0.08       220
          df       0.00      0.00      0.00        23
         mel       0.11      0.17      0.13       223
          nv       0.68      0.50      0.57      1341
        vasc       0.01      0.04      0.02        28

    accuracy                           0.37      2003
   macro avg       0.14      0.13      0.13      2003
weighted avg       0.48      0.37      0.41      2003


    c. Resolution: test at lower epochs; weights need to be adjusted 

2. Weighting scheme testing (run4)
    a. All testing to this point utilized TensorFlow's 'balanced' weighting system to account for large imblanace in classes.
    b. Tested effectiveness of different weights 
    c. Increased weighting for underrepresented classes by a factor of 4x
    d. Results unremarkable 

    ![image](https://github.com/aderdiger/MADAIN/blob/a90a26d55caa387aa8614be2b8cad85adb77fffb/run4/run7/visualizations/roc_curve_InceptionV3_Adam.png)
    

    precision    recall  f1-score   support

       akiec       0.04      0.06      0.05        65
         bcc       0.06      0.07      0.06       103
         bkl       0.00      0.00      0.00       220
          df       0.00      0.00      0.00        23
         mel       0.11      0.70      0.19       223
          nv       0.63      0.12      0.20      1341
        vasc       0.02      0.04      0.03        28

    accuracy                           0.16      2003
   macro avg       0.12      0.14      0.08      2003
weighted avg       0.44      0.16      0.16      2003

3. Binary classification testing (run5; v8)
    a. Testing conducted at same time as weighting scheme testing
    b. All testing to this point involved a multiclass classifier.
    c. Tested to effectiveness of a binary classifier as opposed to a multiclass classifier.
    d. Did not make any difference from previous models


4. Custom weighting - 4x on 'bcc' and 'akeic', 20x on 'mel' (run7)
    a. 
6. Inverse proportional weighting (run8)
    a. Weighted classes based on the inverse of their frequency
   
7. Class balanced loss approach weighting (run9)
    a. Attempted to implement balanced loss weighting, model performed poorly
8. Adding generated augmented images to training data (run10)
    a. Added a random imgage augementor and image generator 
    b. Added randomly generated images back into training data 
    c. Wanted to normalize percentage representation in data set of underrepresented classes
9. Increasing custom layer neuron density from 512 to 1024 and rerunning promissing models
    a. Top performers are as follows: InceptionV3.Adam, ResNet50.Adam, VGG16.SGD
    b. Will more than likely continue forward with InceptionV3.Adam
    c. May also run ResNet50 in tandum, as epoch count does not seem to make a drastic difference. (As determined in run3)
10. Augmented image generation with 1000 images for underrepresented classes (run12)
    a. Results not impressive


#### Resources:

https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc

https://stackoverflow.com/questions/51798784/keras-transfer-learning-on-inception-resnetv2-training-stops-in-between-becaus
