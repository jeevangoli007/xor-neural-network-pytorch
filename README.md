# XOR Neural Network Classifier using PyTorch

## Project Overview

This project implements a simple 2-layer neural network in PyTorch to learn the XOR function.

The network is trained from scratch using:

- Forward Pass
- Binary Cross Entropy Loss
- Backpropagation
- Adam Optimizer

The model successfully predicts all four XOR outputs after training.

---

## Neural Network Architecture

Input Layer : 2 Neurons

↓

Hidden Layer : 4 Neurons (ReLU)

↓

Output Layer : 1 Neuron (Sigmoid)

---

## Dataset

| Input | Output |
|--------|--------|
|0 0|0|
|0 1|1|
|1 0|1|
|1 1|0|

---

## Requirements

- Python 3.10+
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch
```

---

## Run the Project

```bash
python xor_classifier.py
```

The program will:

- Train the neural network
- Print loss every 500 epochs
- Display final predictions

---

## Sample Output

```
Epoch 500 Loss: ...

Epoch 1000 Loss: ...

...

Final Predictions

Input [0,0] -> Prediction: 0

Input [0,1] -> Prediction: 1

Input [1,0] -> Prediction: 1

Input [1,1] -> Prediction: 0
```

---

## Concepts Covered

- Neural Networks
- Forward Pass
- Backpropagation
- Gradient Descent
- Binary Classification
- ReLU Activation
- Sigmoid Activation
- BCELoss
- Adam Optimizer

---

## Author

Jeevan R. Goli
