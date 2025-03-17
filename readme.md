#### DA6401 Assignment-01 DA24M0164

### Implementation of Backpropagation algorithm from scratch..

## Overview
This repository contains the implementation of a neural network from scratch for DA6401 Assignment 1. The project includes a neural network model, various optimization algorithms, and utilities for training and evaluation.

**GitHub Repository:** [DA6401 Assignment 1](https://github.com/saimanikumar-da24m016/da6401_assignment1)  

**WandB Report:** [DA6401 Assignment 1 - DA24M016](https://wandb.ai/da24m016-indian-institute-of-technology-madras/DL_Assignment_01/reports/DA6401-Assignment-1-DA24M016--VmlldzoxMTUwNTkxNg?accessToken=fuq9rzuml57y1oj7vf3t2yxh5p71yyjcy1mfdkwnh9lo5wy65f3y7p5kvz1hdzqg)  


## Directory Structure
```
DA6401_Assignment1/
│-- model.py             # Neural network implementation
│-- optimizer.py         # Optimizer implementations (SGD, Momentum, Adam, etc.)
│-- train.py             # Training pipeline
│-- utils.py             # Utility functions
│-- test_wandb.ipynb     # Experimentation and logging with Weights & Biases
│-- q1.py                # for sample images
│-- readme.md            # Project documentation
```


## Features
- Implementation of a fully connected neural network with configurable activation functions and loss functions.
- Multiple optimization algorithms, including SGD, Momentum, Adam, and RMSProp.
- Training and evaluation scripts with proper logging using Weights & Biases.
- Utility functions for data processing and model management.

## Installation & Setup

   ```
1. Install dependencies:
   ```bash
   pip install numpy matplotlib wandb
   ```
2. Set up Weights & Biases (optional):
   ```bash
   wandb login
   ```

## Usage
### Training the Model
To train the model, run:
```bash
python train.py
```

### Testing the Model
You can test the model using the test script (or Jupyter Notebook `test_wandb.ipynb`).

### Customizing the Model
- Modify `model.py` to adjust network architecture, activation functions, and loss functions.
- Modify `optimizer.py` to experiment with different optimization techniques.

## Model Implementation
The neural network supports various activation functions:
- **Sigmoid**
- **Tanh**
- **ReLU**
- **Identity**
- **Softmax** (for classification)

The supported loss functions are:
- **Cross-Entropy Loss** (for classification tasks)
- **Mean Squared Error (MSE)** (for regression tasks)

## Optimizers Implemented
- **SGD**
- **Momentum**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSProp**
- **Adam**
- **Nadam**



