MNIST Digit Classifier

A beginner PyTorch project that uses a neural network to recognize handwritten digits from the MNIST dataset.

Overview

This project was built to learn the fundamentals of PyTorch and neural networks by creating a simple image classification model.

The model takes a handwritten digit image as input and predicts which number it represents, from 0–9.

How It Works

The neural network follows this basic process:

28 × 28 Image
     ↓
784 Pixels
     ↓
Linear Layer (784 → 128)
     ↓
ReLU Activation
     ↓
Linear Layer (128 → 10)
     ↓
Digit Prediction (0–9)

The model is trained using:

Cross Entropy Loss — measures how wrong the model's predictions are
Adam Optimizer — updates the model's weights to improve its predictions
Backpropagation — calculates how the model's weights should change
Dataset

This project uses the MNIST handwritten digit dataset, which contains:

60,000 training images
10,000 testing images
Images are 28 × 28 pixels
10 possible classes: 0–9

The dataset is automatically downloaded using torchvision.

Results

After training the model for 5 epochs, it achieved:

97.29% test accuracy

This means the model correctly classified approximately 9,729 out of 10,000 previously unseen handwritten digits.

Technologies Used
Python
PyTorch
Torchvision
MNIST

What I Learned

Through this project, I learned the basic PyTorch machine learning workflow:

Loading and preparing a dataset
Using DataLoader to process data in batches
Building a neural network with torch.nn
Understanding forward passes
Using activation functions such as ReLU
Calculating loss with Cross Entropy Loss
Using backpropagation
Training a model with the Adam optimizer
Evaluating model accuracy on unseen data
Future Improvements

Some improvements I plan to make:

Calculate and track average loss for each epoch
Visualize model predictions
Create a confusion matrix
Experiment with different network architectures
Experiment with different numbers of epochs and learning rates
Improve the model's performance

Running the Project

Install the required packages:

pip install torch torchvision

Then run:

python model.py

The MNIST dataset will automatically download the first time the program is run.
