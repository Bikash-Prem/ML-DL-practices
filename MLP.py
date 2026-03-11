import numpy as np

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([[0],[1],[1],[0]])

np.random.seed(5)
w1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

w2 = np.random.randn(2, 2)
b2 = np.zeros((1,2))

w3 = np.random.randn(2,1)
b3 = np.zeros((1,1))

lr = 0.5


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


for epoch in range(50000):

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3) + b3
    y_pred = sigmoid(z3)

    loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    d_output = y_pred - y
    d2 = d_output.dot(w3.T) * sigmoid_derivative(a2)
    d1 = d2.dot(w2.T) * sigmoid_derivative(a1)

    w3 -= a2.T.dot(d_output) * lr
    b3 -= np.sum(d_output, axis=0, keepdims=True) * lr

    w2 -= a1.T.dot(d2) * lr
    b2 -= np.sum(d2, axis=0, keepdims=True) * lr

    w1 -= x.T.dot(d1) * lr
    b1 -= np.sum(d1, axis=0, keepdims=True) * lr

    if epoch%5000 == 0:
        print(loss)

print("Predictions: ")
print(y_pred)
