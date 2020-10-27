
from knn import K_Nearest_Neighbor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans 
from sklearn.metrics import accuracy_score

# Load Data
iris = load_iris()
X = scale(iris.data)
y = iris.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# Choose K 
clf = KNeighborsClassifier(n_neighbors=10)

# Fit
clf.fit(X_train, y_train)

# Prediction
predict = clf.predict(X_test)
print("-------SKLEARN MODEL-------")
print("Prediction", predict)

# Accuracy Score
print(f"Scikit-learn accuracy: {accuracy_score(y_test, predict)}")

# y_pred
y_pred = clf.predict([X_test[0]])
print("y_pred", y_pred)

########################################################################################

# K_Nearest_Neighbor Model
np_clf = K_Nearest_Neighbor(k=10)

np_clf.fit(X_train, y_train)

predict = np_clf.predict(X_test)
print("-------PYTHON MODEL-------")
print("Prediction", predict)

# Accuracy Score
print(f"K_Nearest_Neighbor accuracy: {accuracy_score(y_test, predict)}")

##########################################################################

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}

# Prediction on new data

new_data = np.array([
    [6, 4, 3, 2],
    [10, 6, 8, 2]])

new_data2 = np.array([
    [5.1, 4.3, 3.2, 2.5],
    [5, 0.9, 3.4, 1.0]])

# Sklearn
y_predict1 = clf.predict(new_data)
print("-------NEW DATA1-------")
print(f"data 1:{classes[y_predict1[0]]}")
print(f"data 2:{classes[y_predict1[1]]}")


# K_Nearest_Neighbor
y_predict2 = np_clf.predict(new_data)

print(f"data 1:{classes[y_predict2[0]]}")
print(f"data 2:{classes[y_predict2[1]]}")


# Sklearn
y_predict3 = clf.predict(new_data2)
print("-------NEW DATA2-------")
print(f"data 1:{classes[y_predict3[0]]}")
print(f"data 2:{classes[y_predict3[1]]}")


# K_Nearest_Neighbor
y_predict4 = np_clf.predict(new_data2)

print(f"data 1:{classes[y_predict4[0]]}")
print(f"data 2:{classes[y_predict4[1]]}")