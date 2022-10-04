from urllib.parse import urlparse

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
import warnings
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

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri('http://ilcepoc2353:5000/')
mlflow.set_experiment("alpha experiment")


def evaluation_metrics(X_train, y_train, y_test, model):
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


if __name__ == "__main__":
    # read data
    train_data_path = Path("Dataset/train.csv")
    test_data_path = Path("Dataset/test.csv")
    df = pd.read_csv(train_data_path)

    # separate X and y
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=True)

    lr = LogisticRegression(random_state=0)
    rf = RandomForestClassifier(n_estimators=200)
    ab = AdaBoostClassifier(n_estimators=200)
    gtb = GradientBoostingClassifier(n_estimators=200)
    dtc = DecisionTreeClassifier(random_state=0)
    etc = ExtraTreesClassifier(random_state=0)

    models = {"LogisticRegression": lr, "RandomForestClassifier": rf, "DecisionTreeClassifier": dtc,
              "AdaBoostClassifier": ab, "GradientBoostingClassifier": gtb, "ExtraTreesClassifier": etc}
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(tracking_url_type_store)
    for model_name, model_obj in models.items():
        print(50 * '*')
        print(f"{model_name}")
        print(50 * '*')
        with mlflow.start_run():
            disp, params = evaluation_metrics(X_train, y_train, y_test, model_obj)
            # disp.plot()
            # plt.grid(False)
            # plt.show()
            mlflow.log_metric("Accuracy", params[0])
            mlflow.log_metric("F1_score", params[1])
            mlflow.log_metric("mae", params[2])
            mlflow.log_metric("r2", params[3])
            mlflow.log_metric("Precision", params[4])
            mlflow.log_metric("Recall", params[5])
            mlflow.log_metric("rmse", params[6])
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(lr, "model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(lr, "model")
