import os
import pandas as pd
from unittest.mock import patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pytest


@pytest.fixture
def mock_data():
    data = {
        'PLAYER': ['Ty Cobb', 'Stan Musial'],
        'YRS': [24, 22],
        'G': [3035, 3026],
        'AB': [11434, 10972],
        'R': [2246, 1949],
        'H': [4189, 3630],
        '2B': [724, 725],
        '3B': [295, 177],
        'HR': [117, 475],
        'RBI': [726, 1951],
        'BB': [1249, 1599],
        'SO': [357, 696],
        'SB': [892, 78],
        'CS': [178, 31],
        'BA': [0.366, 0.331],
        'HOF': [1, 1]
    }
    return pd.DataFrame(data)


@patch('pandas.read_csv')
def test_data_loading(mock_read_csv, mock_data):
    mock_read_csv.return_value = mock_data
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    assert not df.empty, "Dataframe is empty"
    assert 'PLAYER' in df.columns, "Expected column 'PLAYER' not found"
    assert 'CS' in df.columns, "Expected column 'CS' not found"


@patch('pandas.read_csv')
def test_data_splitting(mock_read_csv, mock_data):
    mock_read_csv.return_value = mock_data
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


@patch('pandas.read_csv')
def test_model_training(mock_read_csv, mock_data):
    mock_read_csv.return_value = mock_data
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


@patch('pandas.read_csv')
def test_model_prediction(mock_read_csv, mock_data):
    mock_read_csv.return_value = mock_data
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    assert len(y_pred) == len(y_test), "Prediction length mismatch"


@patch('pandas.read_csv')
def test_evaluation_metrics(mock_read_csv, mock_data):
    mock_read_csv.return_value = mock_data
    file_path = os.path.join('data', '500Hits.csv')
    df = pd.read_csv(file_path, encoding='Latin 1')
    df = df.drop(columns=['PLAYER', 'CS'])
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    assert cm is not None, "Confusion matrix is None"
    assert cr is not None, "Classification report is None"