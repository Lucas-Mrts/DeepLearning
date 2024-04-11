import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

from sklearn.metrics import accuracy_score

from utilities import *

X_train, y_train, X_test, y_test = load_data()

"""
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

print(X_test.shape)
print(y_test.shape)
print(np.unique(y_test, return_counts=True))

plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()
"""

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    eps = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + eps) - (1 - y) * np.log(1 - A + eps))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5


from sklearn.metrics import accuracy_score
from tqdm import tqdm

def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)

        if i%10 == 0:
            # Train
            train_loss.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Test
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise à jour
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

    return (W, b)
    
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()  # / X_train.max() pour normaliser les données et éviter l'overflow
print(X_train_reshape[0].shape)

X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
print(X_test_reshape.shape)


W, b = artificial_neuron(X_train_reshape, y_train, X_test_reshape, y_test, learning_rate = 0.01, n_iter = 10000)