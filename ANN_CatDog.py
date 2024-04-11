import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from sklearn.datasets import make_moons

from utilities import *

plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor":  (0.12 , 0.12, 0.12, 1),
    "axes.facecolor": (0.12 , 0.12, 0.12, 1),
})


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


def back_propagation(X, y, parametres, activations, lambd = 0.1):
    m = y.shape[1]
    C = len(parametres) // 2
    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        # Regularization term added to gradient calculation for weights            
        gradients['dW' + str(c)] = 1 / m * (np.dot(dZ, activations['A' + str(c - 1)].T)) + (lambd * parametres['W' + str(c)])/m
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


def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers = (32,32), learning_rate = 0.01, n_iter = 1000, n_donnes = 1000):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_donnes), 2))
    testing_history = np.zeros((int(n_donnes), 2))


    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        if i%(n_iter/n_donnes) == 0:

            index = i // (n_iter // n_donnes)

            #Train
            training_history[index, 0] = (log_loss(y_train.flatten(), Af.flatten()))
            y_pred = predict(X_train, parametres)
            training_history[index, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))

            #Test
            activations_test = forward_propagation(X_test, parametres)
            Af_test = activations_test['A' + str(C)]
            testing_history[index, 0] = (log_loss(y_test.flatten(), Af_test.flatten()))
            y_pred_test = predict(X_test, parametres)
            testing_history[index, 1] = (accuracy_score(y_test.flatten(), y_pred_test.flatten()))


    # Visualisation des résultats

    if(i==n_iter-1):
        print("Final train accuracy:", accuracy_score(y_train.flatten(), y_pred.flatten()))
        print("Final test accuracy:", accuracy_score(y_test.flatten(), y_pred_test.flatten()))
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

    # Plot loss
    ax1.plot(training_history[:, 0], label='train loss')
    ax1.plot(testing_history[:, 0], label='test loss')
    ax1.set_title('Training and Testing Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(training_history[:, 1], label='train acc')
    ax2.plot(testing_history[:, 1], label='test acc')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return parametres


X_train, y_train, X_test, y_test = load_data()

X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()  # / X_train.max() pour normaliser les données et éviter l'overflow
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()

X_train_reshape = X_train_reshape.T
y_train = y_train.reshape((1, y_train.shape[0]))

X_test_reshape = X_test_reshape.T
y_test = y_test.reshape((1, y_test.shape[0]))


# Pour réduire le temps de calcul, on peut réduire le nombre de données

m_train = 300
m_test = 80
X_train_reshape = X_train_reshape[:, :m_train]
X_test_reshape = X_test_reshape[:, :m_test]
y_train = y_train[:, :m_train]
y_test = y_test[:, :m_test]

#print(X_train_reshape.shape)
#print(y_train.shape)




parametres = deep_neural_network(X_train_reshape, y_train, X_test_reshape, y_test, hidden_layers = (16,16), learning_rate = 0.1, n_iter = 8000, n_donnes = 100)



