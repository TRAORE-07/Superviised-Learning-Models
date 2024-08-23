from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()

# Feature and Target Extraction
x = iris.data
y = iris.target

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Evaluate the model's accuracy
print(f'Accuracy = {accuracy_score(y_test,y_pred)*100} %')

# Make a prediction for a new sample with specified features
result = model.predict([[6.1, 2.9, 4.7, 1.4]])
# Print the predicted class name based on the target names
print(f'Result = {iris.target_names[result[0]]}')

# Load the Iris dataset from seaborn for plotting
iris = sns.load_dataset("iris")

# Plot a scatterplot with a linear fit
g = sns.lmplot(x="sepal_length", y="sepal_width", hue="species", truncate=True, height=6, data=iris)
plt.show()