# Neural Network from Scratch (NumPy)

## Overview

This project is a simple neural network built entirely from scratch using NumPy. It is designed to help understand how neural networks actually work without using deep learning frameworks like TensorFlow or PyTorch.
The goal of this project was to understand the core concepts such as forward propagation, backpropagation, activation functions, and gradient descent.
The idea for this project was inspired by 3Blue1Brown’s neural networks course.

---

## What this model does

The neural network is trained to recognize handwritten digits using the MNIST dataset.

It takes 28x28 pixel images (784 input values) and predicts a digit from 0 to 9.

---

## Architecture

The model is a simple fully connected neural network:

```
Input layer:    784 neurons
Hidden layer:   64 neurons (ReLU activation)
Output layer:   10 neurons (Softmax activation)
```

## Key Concepts Implemented

* Forward propagation
* Backpropagation using gradient descent
* ReLU activation function
* Softmax for multi-class classification
* Cross-entropy loss function
* He initialization for weight scaling
* Numerical stability in softmax and log operations
* One-hot encoding for labels

---

## Training Results

After training for 500 iterations:

* Training loss decreases steadily from ~2.5 to ~0.29
* Training accuracy reaches ~91%
* Test accuracy: ~92.16%

training log:

```
Iteration:   0 | Loss: 2.5073 | Accuracy: 5.86%
Iteration:  50 | Loss: 0.6605 | Accuracy: 84.56%
Iteration: 100 | Loss: 0.4706 | Accuracy: 87.87%
Iteration: 150 | Loss: 0.4041 | Accuracy: 89.14%
Iteration: 200 | Loss: 0.3684 | Accuracy: 89.89%
Iteration: 250 | Loss: 0.3451 | Accuracy: 90.36%
Iteration: 300 | Loss: 0.3281 | Accuracy: 90.75%
Iteration: 350 | Loss: 0.3146 | Accuracy: 91.14%
Iteration: 400 | Loss: 0.3035 | Accuracy: 91.41%
Iteration: 450 | Loss: 0.2939 | Accuracy: 91.68%
```

---

## How to run

1. Download the MNIST dataset files (IDX format)
2. Update file paths in `main.py`
3. Run:

```
python main.py
```

Make sure NumPy is installed:

```
pip install numpy
```

---

## Project Structure

```
main.py              -> Runs training and evaluation
neural_network.py    -> All ML logic (forward/backward pass)
load_data.py         -> MNIST data loading 
```

---

## What I learned

This project helped me understand:

* How neural networks actually learn from data
* Why backpropagation works mathematically
* How gradients flow through a network
* Why activation functions like ReLU matter
* Why numerical stability is important in softmax and log operations

---

## Inspiration

This project was inspired by:

* [3Blue1Brown’s series on neural networks ]([url](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi))
* Introductory machine learning videos focused on intuition behind ReLU, softmax, and gradient descent
