import numpy as np
import matplotlib.pyplot as plt
from numpy.random import PCG64 as pcg

# -----------------------
# DATASET
# -----------------------
rng = np.random.default_rng(pcg())  # random number generator object
x = np.linspace(1, 10, 100) # uniformly increasing values from 1 to 10
y = 6 * x + 9  # linear equation y = mx+c (y=wx+b for ML)
y += rng.uniform(-1, 1, size=x.shape) # adding noise


# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(w, x, b):
    return w * x + b


# -----------------------
# LOSS FUNCTION
# -----------------------
def compute_loss(w, x, y, b):
    y_pred = predict(w, x, b)
    return np.mean((y - y_pred) ** 2)


# -----------------------
# GRADIENT COMPUTATION
# -----------------------
def compute_gradients(w, b, x, y):
    n = len(x)
    y_pred = predict(w, x, b)
    error = y - y_pred

    dw = (-2 / n) * np.sum(x * error)
    db = (-2 / n) * np.sum(error)

    return dw, db


# -----------------------
# GRADIENT DESCENT TRAINING
# -----------------------
def train(x, y, lr=0.01, epochs=1000):
    w = 0.0
    b = 0.0

    for i in range(epochs):
        dw, db = compute_gradients(w, b, x, y)
        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            print(f"Epoch {i}: loss = {compute_loss(w, x, y, b):.4f}")

    return w, b


# Train Test Split
def train_test_split(x, y, test_ratio=0.2):
    n = len(x)
    split = int(n * (1 - test_ratio))
    return x[:split], x[split:], y[:split], y[split:]


x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train the model
w_final, b_final = train(x_train, y_train)

test_loss = compute_loss(w_final, x_test, y_test, b_final)
print("Test loss:", test_loss)

print("\nFinal w:", w_final)
print("Final b:", b_final)

# -----------------------
# PLOT RESULT
# -----------------------
plt.scatter(x_train, y_train, label="Train Data")
plt.scatter(x_test, y_test, label="Test Data")
plt.plot(x, predict(w_final, x, b_final), color="red", label="Model")
plt.legend()
plt.show()

