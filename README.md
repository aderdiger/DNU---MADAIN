# MADAIN (Mole Analysis with Deep Adam-Optimized Inception Network): Skin Cancer Prediction Using Deep Learning

## 🔍 Introduction
Skin cancer is a prevalent and potentially life-threatening disease that requires early detection for effective treatment. In this project, we developed a deep learning model that can predict whether a mole is cancerous or benign based on image data. By leveraging convolutional neural networks (CNNs) and transfer learning, we aimed to create an accurate and reliable tool to assist in the early identification of skin cancer.

## 📊 Dataset
We utilized a Kaggle dataset containing 10,015 images of moles across 7 different classes. The dataset was preprocessed and organized into separate train and validation sets using stratified splitting to ensure balanced class representation. The images were resized to a consistent dimension of 299x299 pixels to match the input size of the chosen CNN architectures.

## 🏗️ Model Architecture
We benchmarked three popular CNN architectures: InceptionV3, ResNet50, and VGG16. These architectures were selected based on their proven performance in image classification tasks. Transfer learning was applied by leveraging pre-trained weights from the ImageNet dataset, allowing us to benefit from the knowledge gained on a large-scale dataset.

## 🏋️‍♀️ Model Training and Optimization
The models were trained using the preprocessed dataset, with data augmentation techniques applied to enhance model generalization. We experimented with different optimizers, including Adam, RMSprop, and SGD, to find the best configuration for each architecture. Class imbalance was addressed using TensorFlow's 'balanced' weighting system. Callbacks such as TensorBoard, ReduceLROnPlateau, and EarlyStopping were employed to monitor training progress and prevent overfitting.

## 📈 Model Evaluation and Selection
The trained models were evaluated using precision, recall, and AUC-ROC metrics to assess their performance in classifying skin lesions. The InceptionV3 architecture with the Adam optimizer emerged as the best-performing model based on these evaluation metrics. Further fine-tuning and hyperparameter optimization were conducted on the selected model to enhance its performance.

## 🌐 Deployment and Web App
To make the skin cancer prediction model accessible to users, we deployed it using Streamlit, a popular framework for building interactive web applications. The web app allows users to upload an image of a mole and receive a prediction on whether it is likely to be cancerous or benign. The app ensures that the uploaded image is resized to the required dimensions of 299x299 pixels before being fed into the model.

## 📊 Results and Visualizations
The final model achieved promising results on the validation set, with high values for precision, recall, and AUC-ROC. We generated visualizations, including a confusion matrix, classification report, and ROC curve, to provide insights into the model's performance. The visualizations showcase the model's ability to accurately classify skin lesions and highlight areas for further improvement.

## ⚠️ Limitations and Future Work
While the developed model demonstrates significant potential in assisting with skin cancer prediction, it is important to acknowledge its limitations. The model's performance may be affected by factors such as image quality, lighting conditions, and the diversity of skin tones and lesion types represented in the training data. Future work could involve expanding the dataset, exploring advanced CNN architectures, and incorporating additional clinical features to enhance the model's accuracy and generalization ability.

## 🚀 Getting Started
To run the skin cancer prediction model locally or contribute to the project, please follow the instructions in the [installation guide](link-to-installation-guide). The guide provides step-by-step instructions for setting up the necessary dependencies and running the Streamlit web app.

## 📜 License
This project is licensed under the [MIT License](link-to-license-file). Feel free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## 🙏 Acknowledgements
We would like to express our gratitude to the creators of the Kaggle dataset used in this project and the developers of the deep learning frameworks and libraries that made this work possible. We also extend our appreciation to the open-source community for their valuable contributions and inspiration.

## 📧 Contact Information
For any questions, suggestions, or collaborations, please feel free to reach out to the project maintainers:
- Amanda Derdiger - https://github.com/aderdiger
- Andrew Koller - https://github.com/AEKoller
- Natalia Mitchell - https://github.com/nmitchell1219
- Mustafa Can Ayter - https://github.com/mustafacanayter

We hope that this project contributes to the early detection and prevention of skin cancer and serves as a valuable resource for researchers and practitioners in the field. Let's work together to save lives! 💪
