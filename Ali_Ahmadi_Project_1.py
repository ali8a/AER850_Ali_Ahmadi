# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

# Step 1: Data Processing
# Load the data
df = pd.read_csv('Project_1_Data.csv')  # Replace with your actual file path

# Check the DataFrame structure
print(df.head())

# Step 2: Data Visualization (3D Plot with Color Coding by Step)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a colormap based on the unique steps
steps = df['Step']
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=steps, cmap='viridis', marker='o', alpha=0.8)

# Labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of X, Y, Z Coordinates Color-Coded by Step')

# Colorbar to show the step values
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Step')

plt.show()

# Step 3: Scaling the Data and Correlation Analysis

# Define the features to be scaled
features = ['X', 'Y', 'Z']

# Scale the coordinate features (X, Y, Z)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Create a DataFrame with the scaled data
df_scaled = pd.DataFrame(scaled_data, columns=features)

# Add the 'Step' column back to the DataFrame
df_scaled['Step'] = df['Step']

# Calculate the correlation matrix (absolute values) and drop original X, Y, Z
corr_matrix_scaled = df_scaled.corr().abs()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_scaled, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
plt.title('Correlation Matrix with Scaled Features (0 to 1)')
plt.show()

# Step 4.1: Splitting the Data
# Separating the features (X, Y, Z) and the target (Step)
X = df_scaled[['X', 'Y', 'Z']]
y = df_scaled['Step']

# Splitting the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 4.2: Creating Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Define the models
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
svm = SVC(random_state=42)

# Define the hyperparameters grid for each model
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Perform grid search for each model
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1)
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', verbose=1)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=1)

# Fitting the models to the training data
grid_search_rf.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)

# Step 4.3: RandomizedSearchCV for Hyperparameter Tuning
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42, verbose=1)
random_search_rf.fit(X_train, y_train)

# Step 5.1: Performance Metrics and Model Evaluation
model_performance = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'F1 Score': []
}

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    model_performance['Model'].append(model_name)
    model_performance['Accuracy'].append(accuracy)
    model_performance['Precision'].append(precision)
    model_performance['F1 Score'].append(f1)

    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate each model
evaluate_model(grid_search_rf.best_estimator_, X_test, y_test, "Random Forest")
evaluate_model(grid_search_knn.best_estimator_, X_test, y_test, "K-Nearest Neighbors")
evaluate_model(grid_search_svm.best_estimator_, X_test, y_test, "SVM")
evaluate_model(random_search_rf.best_estimator_, X_test, y_test, "Randomized Search RF")

# Step 6.1: Stacking Classifier
estimators = [
    ('rf', grid_search_rf.best_estimator_),
    ('knn', grid_search_knn.best_estimator_)
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=grid_search_svm.best_estimator_)
stacking_clf.fit(X_train, y_train)

# Evaluate the stacked model
evaluate_model(stacking_clf, X_test, y_test, "Stacked Model")

# Step 7: Saving the Model and Making Predictions
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
joblib.dump(stacking_clf, f'Stacked_model_{timestamp}.joblib')

# Step 7: Making Predictions
# Define the coordinates for predictions
coordinates = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Convert to DataFrame with the correct feature names for scaling
coordinates_df = pd.DataFrame(coordinates, columns=['X', 'Y', 'Z'])

# Scale the input coordinates
coordinates_scaled = scaler.transform(coordinates_df)

# Make predictions using the stacked model
predictions = stacking_clf.predict(coordinates_scaled)

# Display the predictions
for coord, pred in zip(coordinates, predictions):
    print(f"Coordinates: {coord}, Predicted Maintenance Step: {pred}")
