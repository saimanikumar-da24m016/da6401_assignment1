import numpy as np
import argparse
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
# For MNIST, you could import: from keras.datasets import mnist

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
        # Select activation function
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
        # Hidden layers
        for i in range(1, self.num_layers):
            Z = np.dot(A, self.params['W' + str(i)]) + self.params['b' + str(i)]
            cache['Z' + str(i)] = Z
            A = self.activation(Z)
            cache['A' + str(i)] = A
        # Output layer: apply softmax to get probabilities
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
        
        # For cross entropy with softmax, the gradient simplifies
        if self.loss_func_name == 'cross_entropy':
            dZ = (A_final - y_true) / m
        else:
            dZ = (A_final - y_true) / m

        # Output layer gradients
        A_prev = cache['A' + str(self.num_layers - 1)]
        grads['dW' + str(self.num_layers)] = np.dot(A_prev.T, dZ)
        grads['db' + str(self.num_layers)] = np.sum(dZ, axis=0, keepdims=True)
        
        # Backpropagation through hidden layers
        for i in range(self.num_layers - 1, 0, -1):
            dA = np.dot(dZ, self.params['W' + str(i + 1)].T)
            Z = cache['Z' + str(i)]
            dZ = dA * self.activation_deriv(Z)
            A_prev = cache['A' + str(i - 1)]
            grads['dW' + str(i)] = np.dot(A_prev.T, dZ)
            grads['db' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
        
        return grads

# -----------------------------
# Optimizer Class
# -----------------------------
class Optimizer:
    def __init__(self, params, optimizer_name, learning_rate, momentum=0.5,
                 beta=0.5, beta1=0.5, beta2=0.5, epsilon=1e-6, weight_decay=0.0):
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0  # timestep for Adam/Nadam

        # Initialize optimizer states
        self.v = {}  # for momentum or Adam's first moment
        self.s = {}  # for RMSProp or Adam's second moment
        for key, val in params.items():
            self.v[key] = np.zeros_like(val)
            self.s[key] = np.zeros_like(val)
        self.param_keys = list(params.keys())
    
    def update(self, params, grads):
        self.t += 1
        for key in self.param_keys:
            grad = grads['d' + key]
            # Apply weight decay to weights only
            if key.startswith('W'):
                grad += self.weight_decay * params[key]
            if self.optimizer_name == 'sgd':
                params[key] -= self.learning_rate * grad
            elif self.optimizer_name == 'momentum':
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad
                params[key] += self.v[key]
            elif self.optimizer_name in ['nag', 'nesterov']:
                v_prev = self.v[key].copy()
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad
                params[key] += -self.momentum * v_prev + (1 + self.momentum) * self.v[key]
            elif self.optimizer_name == 'rmsprop':
                self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (grad ** 2)
                params[key] -= self.learning_rate * grad / (np.sqrt(self.s[key]) + self.epsilon)
            elif self.optimizer_name == 'adam':
                self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grad
                self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grad ** 2)
                v_corr = self.v[key] / (1 - self.beta1 ** self.t)
                s_corr = self.s[key] / (1 - self.beta2 ** self.t)
                params[key] -= self.learning_rate * v_corr / (np.sqrt(s_corr) + self.epsilon)
            elif self.optimizer_name == 'nadam':
                self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grad
                self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grad ** 2)
                v_corr = self.v[key] / (1 - self.beta1 ** self.t)
                s_corr = self.s[key] / (1 - self.beta2 ** self.t)
                params[key] -= self.learning_rate * (self.beta1 * v_corr + (1 - self.beta1) * grad) / (np.sqrt(s_corr) + self.epsilon)
            else:
                raise ValueError("Unknown optimizer: " + self.optimizer_name)

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train(network, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    num_samples = X_train.shape[0]
    steps_per_epoch = num_samples // batch_size
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        epoch_loss = 0
        epoch_acc = 0
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            # Forward pass
            y_pred, cache = network.forward(X_batch)
            loss = network.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            # Accuracy (argmax along classes)
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            acc = np.mean(predictions == true_labels)
            epoch_acc += acc
            # Backward pass
            grads = network.backward(X_batch, y_batch, cache)
            # Update parameters using the chosen optimizer
            optimizer.update(network.params, grads)
        
        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        
        # Validation performance
        y_val_pred, _ = network.forward(X_val)
        val_loss = network.compute_loss(y_val_pred, y_val)
        predictions_val = np.argmax(y_val_pred, axis=1)
        true_labels_val = np.argmax(y_val, axis=1)
        val_acc = np.mean(predictions_val == true_labels_val)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    return network

def evaluate(network, X_test, y_test):
    y_pred, _ = network.forward(X_test)
    predictions = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    test_acc = np.mean(predictions == true_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    return predictions, true_labels, test_acc

def plot_confusion_matrix(true_labels, predictions, classes, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    plt.show()

# -----------------------------
# Data Loading Function
# -----------------------------
def load_data(dataset):
    if dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError("Unknown dataset.")
    # Normalize and reshape data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    # One-hot encode labels (assuming 10 classes)
    num_classes = 10
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]
    # Split training data into training and validation (90%/10%)
    split_index = int(0.9 * X_train.shape[0])
    X_val = X_train[split_index:]
    y_val = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    return X_train, y_train, X_val, y_val, X_test, y_test

# -----------------------------
# Command-line Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", type=str,
                        help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", default="myname", type=str,
                        help="WandB entity name")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], type=str)
    parser.add_argument("-e", "--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="Mini-batch size")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], type=str)
    parser.add_argument("-o", "--optimizer", default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], type=str)
    parser.add_argument("-lr", "--learning_rate", default=0.1, type=float, help="Learning rate")
    parser.add_argument("-m", "--momentum", default=0.5, type=float, help="Momentum value for momentum and nag")
    parser.add_argument("-beta", "--beta", default=0.5, type=float, help="Beta for RMSProp")
    parser.add_argument("-beta1", "--beta1", default=0.5, type=float, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", default=0.5, type=float, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", default=1e-6, type=float, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", default=0.0, type=float, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i", "--weight_init", default="random", choices=["random", "Xavier"], type=str, help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", default=1, type=int, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", default=4, type=int, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"], type=str, help="Activation function")
    args = parser.parse_args()
    return args

# -----------------------------
# Main Function
# -----------------------------
def main():
    args = parse_args()

    # Initialize WandB for experiment tracking
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    input_size = X_train.shape[1]
    output_size = 10  # for 10 classes

    # Build list of hidden layer sizes based on num_layers and hidden_size
    hidden_sizes = [args.hidden_size] * args.num_layers

    # Initialize neural network
    network = NeuralNetwork(input_size, hidden_sizes, output_size, args.activation, args.weight_init, args.loss)
    
    # Initialize optimizer with the chosen hyperparameters
    optimizer = Optimizer(network.params, args.optimizer, args.learning_rate,
                          momentum=args.momentum, beta=args.beta,
                          beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon,
                          weight_decay=args.weight_decay)
    
    # Train the network
    network = train(network, optimizer, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate on test set
    predictions, true_labels, test_acc = evaluate(network, X_test, y_test)
    
    # Plot and log confusion matrix
    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, predictions, classes)
    
    wandb.finish()

if __name__ == '__main__':
    main()
