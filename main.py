

import neural_network as nn
import load_data as ld


if __name__ == "__main__":

    print("Loading MNIST dataset...")
    X_train = ld.load_mnist_images(r"C:\Users\kprk2\MNIST_NN\train-images.idx3-ubyte")
    Y_train = ld.load_mnist_labels(r"C:\Users\kprk2\MNIST_NN\train-labels.idx1-ubyte")
    X_test = ld.load_mnist_images(r"C:\Users\kprk2\MNIST_NN\t10k-images.idx3-ubyte")
    Y_test = ld.load_mnist_labels(r"C:\Users\kprk2\MNIST_NN\t10k-labels.idx1-ubyte")
    print("Loading complete.")

    print(f"X_train shape: {X_train.shape}")  #(784, 60000)
    print(f"Y_train shape: {Y_train.shape}")  #(60000,)

    print("Training MNIST dataset...")
    W1, b1, W2, b2 = nn.train(X_train, Y_train, alpha=0.1, iterations=500)
    print("Training Complete.")

    _, _, _, A2_test = nn.forward_propagation(W1, b1, W2, b2, X_test)
    test_predictions = nn.get_predictions(A2_test)
    test_accuracy = nn.get_accuracy(test_predictions, Y_test)

    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")