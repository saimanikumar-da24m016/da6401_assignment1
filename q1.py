### Printing Images
import wandb
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

wandb.init(project="DL_Assignment_01", name="fashion_mnist_sample_grid")

(X_train, y_train), (_, _) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap="gray")
    ax.set_title(f"{i}: {class_names[i]}")
    ax.axis("off")

plt.tight_layout()

wandb.log({"Fashion-MNIST Sample Images": wandb.Image(fig)})

plt.show()
wandb.finish()
