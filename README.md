# MLOps Project

This project demonstrates the implementation of MLOps practices using various machine learning models and tools.

Python files have been appended with the assignment module # for which the program is written.

## Setup

1. Clone the repository:
    ```sh
    git clone git@github.com:venky-bits/MLOps.git
    cd MLOps
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the application:
    ```sh
    python app.py
    ```

## Training Models

- To train the models, you can run the following scripts:
    - [M1-train.py](http://_vscodecontentref_/9)
    - [M2-train_with_mlflow.py](http://_vscodecontentref_/10)
    - [M3-train_tune.py](http://_vscodecontentref_/11)

- To test the trained models, you can run the following scripts:
    - [M1-test_train.py](http://_vscodecontentref_/12)
    - [M2-test_train_with_mlflow.py](http://_vscodecontentref_/13)

## Docker

- Build the Docker image:
    ```sh
    docker build -t mlops-project -f docker/flask-model-api .
    ```

- Run the Docker container:
    ```sh
    docker run -p 5001:5001 docker/flask-model-api
    ```

## DVC

- Data version control is managed using DVC. To pull the data:
    ```sh
    dvc pull
    ```

## MLflow

- MLflow is used for experiment tracking. The experiment runs are stored in the [mlruns](http://_vscodecontentref_/14) directory.