from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
import warnings

from mlflow import MlflowClient
from mlflow.models import ModelSignature, infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from collections import Counter
# import joblib
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path
import mlflow
import mlflow.sklearn
import logging

import json
import os

# os.system()
# os.environ['HTTP_PROXY']="http://genproxy.amdocs.com:8080"
# os.environ['HTTPS_PROXY']="http://genproxy.amdocs.com:8080"
# os.environ['no_proxy']="localhost,127.0.0.1,.svc,.local,.amdocs.com,.sock,docker.sock,localaddress,.localdomain.com"

warnings.filterwarnings('ignore')

# Read properties file
with open('properties.json') as f:
    data = json.load(f)

mlflow.set_tracking_uri(data['tracking_uri'])


def evaluation_metrics(X_test, X_train, y_train, y_test, model):  # Function to get eval metrics for models
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred, average='weighted')
    Recall = recall_score(y_test, y_pred, average='weighted')
    F1_score = f1_score(y_test, y_pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Accuracy Score: {Accuracy}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")
    print(f"F1_score: {F1_score}")
    return disp, [Accuracy, F1_score, mae, r2, Precision, Recall, rmse]


def search_experiments():  # Function to search MLFLOW experiments
    experiments = mlflow.search_experiments(order_by=["name"])
    return experiments


def delete_models():  # Function to run delete Models
    client = MlflowClient()
    versions = [981219]
    models = ["LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier",
              "AdaBoostClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier"]
    for model_name in models:
        for version in versions:
            try:
                client.delete_model_version(name=model_name, version=version)
            except mlflow.exceptions.RestException:
                pass
        try:
            client.delete_registered_model(name=model_name)
        except mlflow.exceptions.RestException:
            pass


# def fetch_logged_data(run_id):
#     client = MlflowClient()
#     data = client.get_run(run_id).data
#     tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
#     return data.params, data.metrics, tags, artifacts


def run_experiments():  # Function to run MLFLOW experiments
    global data
    # set experiment name
    # experiment_name = data['experiment']
    # experiment = mlflow.set_experiment(experiment_name=experiment_name)

    # Data ingestion
    train_data_path = Path("Dataset/train.csv")  # test_data_path = Path("Dataset/test.csv")
    df = pd.read_csv(train_data_path)

    # separate X and y
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=True)

    # create models
    lr = LogisticRegression(random_state=0)
    rf = RandomForestClassifier(n_estimators=200)
    ab = AdaBoostClassifier(n_estimators=200)
    gtb = GradientBoostingClassifier(n_estimators=200)
    dtc = DecisionTreeClassifier(random_state=0)
    # etc = ExtraTreesClassifier(random_state
    # =0)

    models = {"LogisticRegression": lr, "RandomForestClassifier": rf, "DecisionTreeClassifier": dtc,
              "AdaBoostClassifier": ab, "GradientBoostingClassifier": gtb}  # , "ExtraTreesClassifier": etc}
    # models = {"ExtraTreesClassifier": etc}
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # print(tracking_url_type_store)
    for model_name, model_obj in models.items():
        print(50 * '*')
        print(f"{model_name}")
        print(50 * '*')
        # print(f"experiment_id is {experiment.experiment_id}")
        # start MLFLOW Run
        mlflow.autolog()
        with mlflow.start_run(run_name="Alpha") as run:
            print(run.info.run_id)
            disp, params = evaluation_metrics(X_test, X_train, y_train, y_test, model_obj)
            disp.plot()
            # plt.grid(False)
            # plt.show()
            # Save Confusion Matrix Plot
            plt.savefig(model_name + "_cm.png")

            # Log Metrics
            mlflow.log_metric("Accuracy", params[0])
            mlflow.log_metric("F1_score", params[1])
            mlflow.log_metric("mae", params[2])
            mlflow.log_metric("r2", params[3])
            mlflow.log_metric("Precision", params[4])
            mlflow.log_metric("Recall", params[5])
            mlflow.log_metric("rmse", params[6])
            # artifact_uri = run.info.artifact_uri
            # print(artifact_uri)

            # Log artifacts
            mlflow.log_artifact(model_name + "_cm.png", "confusion_matrix")
            mlflow.log_artifact("Dataset/train.csv")
            mlflow.set_tag("model_name", model_name)

            # create model signature
            signature = infer_signature(X_train, model_obj.predict(X_train))

            # log and register model
            mlflow.sklearn.log_model(model_obj, "model", registered_model_name=model_name, signature=signature)
            # params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)


if __name__ == "__main__":
    # read data
    run_experiments()
# print(search_experiments())
# delete_models()
