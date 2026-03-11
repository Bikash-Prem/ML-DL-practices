import numpy as np


def fit(x, y, w, b, lr):
    epoch = 1
    while True:
        error = 0
        for i in range(len(x)):
            z = np.dot(w, x[i]) + b
            y_pred = 1 if z >= 0 else 0

            if y_pred != y[i]:
                error += 1
                w = w + lr * (y[i] - y_pred) * x[i]
                b = b + lr * (y[i] - y_pred)
        print(f"Epoch {epoch}: {error} errors")
        epoch += 1
        if error == 0:
            break
    print("Weight: ", w)
    print("Bias: ", b)


def predict(x, w, b):
    z = np.dot(w, x) + b
    return (z >= 0).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred)