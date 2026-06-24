import numpy as np

def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)  # He Initialization

    b1 = np.zeros((64, 1))  # Set values to 0

    W2 = np.random.randn(10, 64) * np.sqrt(2.0 / 64)

    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)  # Return val if greater than 0. Else, return 0.


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)  # overflow prevention

    A = np.exp(Z_shifted)

    return A / np.sum(A, axis=0, keepdims=True)  # Between 0.00-1.00


def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1

    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + b2

    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def relu_deriv(Z):
    return Z > 0


def one_hot_encoding(Y):
    one_hot_Y = np.zeros((Y.size, 10))

    one_hot_Y[np.arange(Y.size), Y] = 1  # Converts label integers into training vectors

    return one_hot_Y.T


def backward_propagation(Z1, A1, Z2, A2, b1, b2, W1, W2, X, Y):
    m = X.shape[1]

    one_hot_Y = one_hot_encoding(Y)

    dZ2 = A2 - one_hot_Y

    dW2 = (1 / m) * np.dot(dZ2, A1.T)

    dZ1 = np.dot(W2.T, dZ2) * relu_deriv(Z1)

    dW1 = (1 / m) * np.dot(dZ1, X.T)

    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - (alpha * dW1)

    b1 = b1 - (alpha * db1)

    W2 = W2 - (alpha * dW2)

    b2 = b2 - (alpha * db2)

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def compute_loss(A2, Y):
    m = Y.size

    one_hot_Y = one_hot_encoding(Y)

    loss = -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / m  # avoiding Log(0)

    return loss


def train(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    print("--Starting Training--")

    for it in range(iterations):

        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)

        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, b1, b2, W1, W2, X, Y)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if it % 50 == 0:
            predictions = get_predictions(A2)

            accuracy = get_accuracy(predictions, Y)

            loss = compute_loss(A2, Y)

            print(f"Iteration: {it:3d} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    return W1, b1, W2, b2

