import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from sklearn.datasets import make_moons

def initialisation(dimensions):

    parametres = {}
    C = len(dimensions)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


def forward_propagation(X, parametres):

    activations = { 'A0': X }
    C = len(parametres) // 2

    for c in range (1, C+1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations


def back_propagation(X, y, parametres, activations):

    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradients


def update(gradients, parametres, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C+1):
        parametres['W' + str(c)] -= learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] -= learning_rate * gradients['db' + str(c)]

    return parametres


def predict(X, parametres):

    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    
    Af = activations['A' + str(C)]
    return Af >= 0.5


def deep_neural_network(X, y, hidden_layers = (2, 2, 2), learning_rate = 0.001, n_iter = 1000):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, parametres)
        gradients = back_propagation(X, y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        if i%100 == 0:
            training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
            y_pred = predict(X, parametres)
            training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Visualisation des résultats
    print(activations['A' + str(C)].shape)
    print(y.shape)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    ax[0].plot(training_history[:, 0], label='train loss')
    ax[0].legend()

    ax[1].plot(training_history[:, 1], label='train acc')
    ax[1].legend()
    #plt.show()

    return parametres




#X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
# X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)


X_inner, y_inner = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X_outer, y_outer = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=1)

y_inner = np.zeros_like(y_inner)
y_outer = np.ones_like(y_outer)

X_outer[:, 0] += 1  # Décalage horizontal
X_outer[:, 1] += 1  # Décalage vertical
X = np.vstack((X_inner, X_outer))
y = np.hstack((y_inner, y_outer))


X = X.T
y = y.reshape((1, y.shape[0]))


parametres = deep_neural_network(X, y, hidden_layers=(32, 32), learning_rate=0.01, n_iter=5000)


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
