import numpy as np

# -----------------------------
# Activation Functions and Derivatives
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

def identity(x):
    return x

def didentity(x):
    return np.ones_like(x)

def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

# -----------------------------
# Loss Functions
# -----------------------------
def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss

def mse_loss(y_pred, y_true):
    loss = np.mean(0.5 * np.square(y_true - y_pred))
    return loss

# -----------------------------
# Neural Network Class
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation, weight_init, loss_func):
        # Set activation function and its derivative
        self.activation_str = activation.lower()
        if self.activation_str == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = dsigmoid
            
        elif self.activation_str == 'tanh':
            self.activation = tanh
            self.activation_deriv = dtanh
            
        elif self.activation_str == 'relu':
            self.activation = relu
            self.activation_deriv = drelu
            
        elif self.activation_str == 'identity':
            self.activation = identity
            self.activation_deriv = didentity
            
        else:
            raise ValueError("Unknown activation function.")

        # Set loss function
        self.loss_func_name = loss_func.lower()
        if self.loss_func_name == 'cross_entropy':
            self.loss_func = cross_entropy_loss
            
        elif self.loss_func_name == 'mean_squared_error':
            self.loss_func = mse_loss
            
        else:
            raise ValueError("Unknown loss function.")

        # Define layer sizes: input -> hidden layers -> output
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1

        # Initialize weights and biases
        self.params = {}
        for i in range(1, len(self.layer_sizes)):
            if weight_init.lower() == 'xavier':
                limit = np.sqrt(6 / (self.layer_sizes[i-1] + self.layer_sizes[i]))
                self.params['W' + str(i)] = np.random.uniform(-limit, limit, 
                                                               (self.layer_sizes[i-1], self.layer_sizes[i]))
            else:  # 'random'
                self.params['W' + str(i)] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01
            self.params['b' + str(i)] = np.zeros((1, self.layer_sizes[i]))
    
    def forward(self, X):
        cache = {}
        A = X
        cache['A0'] = A
        # Forward through hidden layers
        for i in range(1, self.num_layers):
            Z = np.dot(A, self.params['W' + str(i)]) + self.params['b' + str(i)]
            cache['Z' + str(i)] = Z
            A = self.activation(Z)
            cache['A' + str(i)] = A
        # Output layer with softmax activation
        Z = np.dot(A, self.params['W' + str(self.num_layers)]) + self.params['b' + str(self.num_layers)]
        cache['Z' + str(self.num_layers)] = Z
        A = softmax(Z)
        cache['A' + str(self.num_layers)] = A
        return A, cache

    def compute_loss(self, y_pred, y_true):
        return self.loss_func(y_pred, y_true)

    def backward(self, X, y_true, cache):
        grads = {}
        m = X.shape[0]
        A_final = cache['A' + str(self.num_layers)]
        
        # For cross entropy loss with softmax activation
        if self.loss_func_name == 'cross_entropy':
            dZ = (A_final - y_true) / m
        else:
            dZ = (A_final - y_true) / m

        # Gradient for the output layer
        A_prev = cache['A' + str(self.num_layers - 1)]
        grads['dW' + str(self.num_layers)] = np.dot(A_prev.T, dZ)
        grads['db' + str(self.num_layers)] = np.sum(dZ, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 1, 0, -1):
            dA = np.dot(dZ, self.params['W' + str(i + 1)].T)
            Z = cache['Z' + str(i)]
            dZ = dA * self.activation_deriv(Z)
            A_prev = cache['A' + str(i - 1)]
            grads['dW' + str(i)] = np.dot(A_prev.T, dZ)
            grads['db' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
        
        return grads
