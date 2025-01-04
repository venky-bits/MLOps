import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pytest


def test_data_loading():
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    assert not df.empty, "Dataframe is empty"
    assert 'PLAYER' in df.columns, "Expected column 'PLAYER' not found"
    assert 'CS' in df.columns, "Expected column 'CS' not found"


def test_data_splitting():
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"
    assert len(y_train) > 0, "Training labels are empty"
    assert len(y_test) > 0, "Test labels are empty"


def test_model_training():
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    assert knn, "Model training failed"


def test_model_prediction():
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    assert len(y_pred) == len(y_test), "Prediction length mismatch"


def test_evaluation_metrics():
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    assert cm is not None, "Confusion matrix is None"
    assert cr is not None, "Classification report is None"