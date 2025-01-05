import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from datetime import datetime

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'data/500Hits.csv')
print("Data file path::", file_path)
# Load the data
df = pd.read_csv(file_path, encoding='Latin 1')

# Observing the data
print('\nObserving the data:')
print(df.head())

# Dropping the columns that are not required
df = df.drop(columns=['PLAYER', 'CS'])

# Prepare the data for training
X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

# Split the data into training and test sets
print('\nSplitting data set into training and test sets')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

# Scaling the data
print('\nScaling the data set')
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the test data
X_test = scaler.fit_transform(X_test)

# Define different values for n_neighbors
n_neighbors_list = [3, 5, 8]

# Create a variable to store date and time in ddmmyyhhmmss format
current_time = datetime.now().strftime("%d%m%y%H%M%S")

for n_neighbors in n_neighbors_list:
    with mlflow.start_run():
        # Set a custom run name with an identifier
        run_name = f"KNN_model_n_neighbors_{n_neighbors}_{current_time}"
        mlflow.set_tag("mlflow.runName", run_name)

        # Train the KNN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        print(f'\nTraining the KNN model with n_neighbors={n_neighbors}')
        knn.fit(X_train, y_train)

        print('\nPredicting the test set results')
        # Predict the test set results
        y_pred = knn.predict(X_test)
        print(y_pred)

        # Evaluate the model
        print('\nConfusion Matrix and Classification Report')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        cr = classification_report(y_test, y_pred)
        print(cr)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters, metrics, and model
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create an input example
        input_example = X_test[:5]
        
        # Log the model with input example
        mlflow.sklearn.log_model(knn, "model", input_example=input_example)
        mlflow.log_artifact(file_path)