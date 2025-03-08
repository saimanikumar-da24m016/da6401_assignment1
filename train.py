import argparse
import wandb
from model import NeuralNetwork
from optimizer import Optimizer
from utils import load_data, train, evaluate, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="DL_Assignment_01", type=str,
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
    return parser.parse_args()

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

    # Initialize neural network from model.py
    network = NeuralNetwork(input_size, hidden_sizes, output_size, args.activation, args.weight_init, args.loss)
    
    # Initialize optimizer from optimizer.py
    optimizer = Optimizer(network.params, args.optimizer, args.learning_rate,
                          momentum=args.momentum, beta=args.beta,
                          beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon,
                          weight_decay=args.weight_decay)
    
    # Train the network (train function from utils.py)
    network = train(network, optimizer, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate the model on the test set (evaluate function from utils.py)
    predictions, true_labels, test_acc = evaluate(network, X_test, y_test)
    
    # Plot and log confusion matrix (plot_confusion_matrix function from utils.py)
    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, predictions, classes)
    
    wandb.finish()

if __name__ == '__main__':
    main()
