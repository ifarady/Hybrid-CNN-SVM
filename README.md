This repository presents a workflow that combines deep learning and SVM to achieve improved image classification accuracy in computer vision tasks. By utilizing the capabilities of deep learning to extract high-level features from images and then utilizing a traditional machine learning algorithm like SVM for classification, this approach can provide a more robust and accurate image classification system. The original code used in this project was adapted from the work of [https://github.com/snatch59/cnn-svm-classifier](https://github.com/snatch59/cnn-svm-classifier) and has been modified to fit the specific needs of this project. All credit for the original code goes to the original author.

### Workflow:

1.  **Data preparation:** Collect and organize your image data into separate folders for each class. Store your dataset in the `/dataset/` folder.
    
2.  **Feature extraction:** Download a pre-trained deep learning model (we use Inceptionv3) as a feature extractor and extract high-dimensional features from the training images. In this code, we extract the features from the last layer before the fully connected layer to obtain a representation of the image that captures its semantic content.
    
3.  **Training:** Use the high-dimensional features as input to train an SVM classifier. The SVM will learn to separate the features of different classes and generate a classification boundary. Please experiment with the SVM parameter settings to achieve better results for your dataset.
    
4.  **Testing:** In inference time, load a testing image and extract its high-dimensional features using the pre-trained deep learning model. Then, pass the features through the SVM to classify the image.
    
5.  **Improvement:** Iterate on the workflow to improve the performance of the classification task. Experiment with different deep learning models, feature extraction techniques, and SVM parameters to optimize the performance of the classifier.
