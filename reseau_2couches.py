import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres


def forward_propagation(X, parametres):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations


def back_propagation(X, y, activations, parametres):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

def update(gradients, parametres, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres


def predict(X, parametres):

    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    return A2 >= 0.5

def neural_network(X_train, y_train, n1, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X_train, parametres)
        A2 = activations['A2']

        # Train
        train_loss.append(log_loss(y_train.flatten(), A2.flatten()))
        y_pred = predict(X_train, parametres)
        current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
        train_acc.append(current_accuracy)

        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.plot(train_loss, label='train loss')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(train_acc, label='train acc')
    plt.legend()

    # plt.show()

    return parametres


X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

parametres = neural_network(X, y, n1=16, n_iter=1000, learning_rate=0.1)
W1, b1, W2, b2 = parametres['W1'], parametres['b1'], parametres['W2'], parametres['b2']


def plot_decision_boundary(X, y, parametres):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parametres)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.subplot(1,3,3)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    
plot_decision_boundary(X, y, parametres)
plt.title("Decision Boundary")
plt.show()

"""
plt.subplot(1,3,3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
x1 = np.linspace(-1, 4, 100)
x2 = (- W1[0][0] * x1 - b1[0]) / W1[0][1]
plt.plot(x1, x2, c='orange', lw=3)
plt.title("")

plt.tight_layout()
plt.show()
"""
