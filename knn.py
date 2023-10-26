import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and split the dataset
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define a distance function (Euclidean distance)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Step 3: Implement the k-NN algorithm
def k_nearest_neighbors(X_train, y_train, x_test, k=3):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Step 4: Make predictions for the test set
y_pred = [k_nearest_neighbors(X_train, y_train, x, k=3) for x in X_test]

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Optionally, you can make predictions for new data points
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.3, 2.9, 5.6, 1.8]])
new_predictions = [k_nearest_neighbors(X_train, y_train, x, k=3) for x in new_data]
print("Predictions for new data:", new_predictions)
