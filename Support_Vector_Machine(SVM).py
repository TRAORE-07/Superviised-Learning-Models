import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the dataset and store it in a variable data
data = pd.read_csv('D:\\Datasets\\Social_Network_Ads.csv')

# Selecting features and target variable
x = data.iloc[:, [2, 3]]
y = data.iloc[:, -1]

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Training the SVM classifier with a linear kernel
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Making predictions on the test set
y_pred = classifier.predict(x_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plotting the training and test set
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, label='Training data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', label='Test data')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Creation on hyperplane
w = classifier.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2.5, 2.5)
yy = a*xx-(classifier.intercept_[0])/w[1]
plt.plot(xx, yy)
plt.show()
