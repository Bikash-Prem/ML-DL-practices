import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


# ---------- Distance Function ----------
def euclidean_distance(qi, pi):
    return np.sqrt(np.sum((qi - pi) ** 2))


# ---------- KNN Implementation ----------
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, x):
        return np.array([self._predict(sample) for sample in x])


# ---------- Load Dataset ----------
iris = datasets.load_iris()
X = iris.data[:, :2]   # only 2 features for plotting
y = iris.target


# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------- Train Model ----------
knn = KNN(k=5)
knn.fit(X_train, y_train)


# ---------- Predict ----------
predictions = knn.predict(X_test)


# ---------- Accuracy ----------
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# ---------- Plot Dataset ----------
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(
        X[y == label, 0],
        X[y == label, 1],
        label=iris.target_names[label]
    )

plt.xlabel("Length")
plt.ylabel("Width")
plt.title("Iris Dataset (2D)")
plt.legend()
plt.show()


# ---------- Plot Predictions ----------
plt.figure(figsize=(8, 6))

# training points (faded)
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="viridis",
    alpha=0.3
)

# test predictions (bold)
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=predictions,
    cmap="viridis",
    edgecolor="k"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Predictions")
plt.show()
