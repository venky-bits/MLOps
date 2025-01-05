import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

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

# Train the KNN model with n_neighbors=8
knn = KNeighborsClassifier(n_neighbors=8)

print('\nTraining the KNN model')
knn.fit(X_train, y_train)

print('\nPredicting the test set results')

# Predict the test set results
y_pred = knn.predict(X_test)

print(y_pred)

# Evaluate the model
print('\nConfusion Matrix and Classification Report')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print(cr)