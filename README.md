# MNIST Digit Classifier with PyTorch

A beginner machine learning project built with PyTorch that trains a neural network to recognize handwritten digits from the MNIST dataset.

This project was built to learn the fundamentals of PyTorch and understand the basic machine learning workflow, including data loading, neural networks, loss functions, backpropagation, optimization, model evaluation, and confusion matrices.

## Project Overview

The model takes a 28×28 grayscale image of a handwritten digit and predicts which number it represents (0–9).

The basic process is:

1. Load the MNIST dataset
2. Convert images into PyTorch tensors
3. Load the data in batches
4. Build a neural network
5. Make predictions
6. Calculate the loss
7. Use backpropagation to calculate gradients
8. Use the Adam optimizer to update the model's weights
9. Repeat the process for multiple epochs
10. Test the model on images it has never seen
11. Measure accuracy
12. Analyze predictions using a confusion matrix

## Dataset

The project uses the MNIST handwritten digit dataset.

- 60,000 training images
- 10,000 testing images
- Images are 28×28 pixels
- 10 possible classes: digits 0–9

The dataset is downloaded automatically using `torchvision.datasets.MNIST`.

## Neural Network

The final neural network architecture is:

784 → 128 → 64 → 10

input Layer

Each image is 28×28 pixels:

28 × 28 = 784

The image is flattened into 784 values before being passed into the network.

Hidden Layer 1
784 → 128

The first layer takes the 784 pixel values and learns 128 features from them.

A ReLU activation function is applied after this layer.

Hidden Layer 2
128 → 64

The second layer takes the 128 learned features and produces 64 higher-level features.

Another ReLU activation function is applied.

Output Layer
64 → 10

The final layer produces 10 scores, one for each possible digit.

The digit with the highest score becomes the model's prediction.

Training

The model uses:

Loss Function: Cross Entropy Loss
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 64
Epochs: 10

During training, the model follows this process:

Image
  ↓
Forward Pass
  ↓
Prediction
  ↓
Calculate Loss
  ↓
Backpropagation
  ↓
Adam Optimizer
  ↓
Update Weights

This process is repeated for every batch of training data.

Results
Baseline Model

The original model was:

784 → 128 → 10

Test accuracy:

97.29%

Deeper Model

The model was then changed to:

784 → 128 → 64 → 10

Results:

Metric	Result
Training Accuracy	99.31%
Testing Accuracy	97.56%
Final Training Loss	0.0212

Adding the second hidden layer improved test accuracy from 97.29% to 97.56%, an improvement of 0.27 percentage points.

The difference between training and testing accuracy also shows that the model performs slightly better on data it has already seen than on unseen data.

Confusion Matrix

A confusion matrix was used to analyze which digits the model has difficulty distinguishing.

The rows represent the actual digit, while the columns represent the predicted digit.

The diagonal represents correct predictions, while values outside the diagonal represent incorrect predictions.

For example, the model had some difficulty recognizing handwritten 8s, with some actual 8s being predicted as 0, 3, or 5.

This provides more information than accuracy alone because it shows which types of mistakes the model is making.

What I Learned

Through this project, I learned the fundamentals of building and training a neural network with PyTorch, including:

Loading datasets with torchvision
Using DataLoader to create batches
Converting images to tensors
Flattening image data
Creating neural network layers with nn.Linear
Using ReLU as an activation function
Understanding forward passes
Calculating loss with Cross Entropy Loss
Using backpropagation
Optimizing weights with Adam
Training models over multiple epochs
Evaluating models on unseen data
Calculating training and testing accuracy
Creating and interpreting confusion matrices
Experimenting with neural network architecture
Future Improvements

Some possible future experiments include:

Experimenting with different learning rates
Testing different numbers of neurons
Comparing different numbers of epochs
Visualizing the confusion matrix
Testing a convolutional neural network (CNN)
Comparing different activation functions
Experimenting with techniques to reduce overfitting
Technologies Used
Python
PyTorch
Torchvision
Scikit-learn
MNIST Dataset
Goal

The goal of this project was not just to create a digit classifier, but to understand the basic process behind training a neural network and learn how different model choices affect performance.
