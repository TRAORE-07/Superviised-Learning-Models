from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine dataset
data = load_wine()

# Feature and Target Extraction
x = data.data
y = data.target

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize a neural network with three hidden layers, each containing 10 neurons.
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=2000)
mlp.fit(x_train, y_train)

# Make predictions on the test set
pred = mlp.predict(x_test)

# Evaluate the model
DT = confusion_matrix(y_test, pred)
print(f'Accuracy = {accuracy_score(y_test,pred)*100} %')

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(data=DT, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()