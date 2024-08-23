from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast_cancer dataset
data = load_breast_cancer()

# Feature and Target Extraction
x = data.data
y = data.target

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=1000, max_features=2, random_state=42)
rf.fit(x_train, y_train)

# Predict on the test set
y_pred = rf.predict(x_test)

# Function to plot a specific tree from the Random Forest
def plot_rf_tree(rf_model, tree_index=0, feature_names=None, class_names=None):
    if tree_index < 0 or tree_index >= len(rf_model.estimators_):
        raise ValueError(
            f"Tree index {tree_index} is out of range. Must be between 0 and {len(rf_model.estimators_) - 1}.")

    tree_to_plot = rf_model.estimators_[tree_index]

    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, filled=True, feature_names=feature_names, class_names=class_names, proportion=False)
    plt.title(f"Random Forest Tree {tree_index}")
    plt.show()


# Plot the first tree in the Random Forest
plot_rf_tree(rf, tree_index=0, feature_names=data.feature_names, class_names=data.target_names)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(data=cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Compute and print accuracy score
accuracy = rf.score(x_test, y_test)
print(f'Accuracy = {accuracy * 100:.2f} %')
