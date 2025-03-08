import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist

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
    # One-hot encode labels (assumes 10 classes)
    num_classes = 10
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]
    # Split off 10% of training data for validation
    split_index = int(0.9 * X_train.shape[0])
    X_val = X_train[split_index:]
    y_val = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    return X_train, y_train, X_val, y_val, X_test, y_test

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
            # Compute accuracy
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            acc = np.mean(predictions == true_labels)
            epoch_acc += acc
            # Backward pass
            grads = network.backward(X_batch, y_batch, cache)
            # Update parameters
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
    plt.show()
