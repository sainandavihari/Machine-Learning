import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()
x= iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

mlflow.set_experiment("iris-experiment")

with mlflow.start_run():
    lr_model = LogisticRegression(max_iter=50,solver='liblinear')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_conf_matrix = confusion_matrix(y_test, lr_pred)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", lr_acc)
    plt.figure(figsize=(6,6))
    sns.heatmap(lr_conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("lr_confusion_matrix.png")
    mlflow.log_artifact("lr_confusion_matrix.png")
    mlflow.sklearn.log_model(lr_model, "Logistic Regression Model")
    log_model_uri="runs:/{}/Logistic Regression Model".format(mlflow.active_run().info.run_id)
    registered_model = mlflow.register_model(log_model_uri, "IrisLogisticRegressionModel")
    mlflow.end_run()
    print("model accuracy: ", lr_acc)
    print("confusion matrix: ", lr_conf_matrix)

# random_forest
from sklearn.ensemble import RandomForestClassifier 
with mlflow.start_run():
    rf_model = RandomForestClassifier(n_estimators=10, random_state=0,max_depth=5,criterion='entropy')  
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f'random forest accuracy: {rf_acc}') 
    rf_conf_matrix = confusion_matrix(y_test, rf_pred)
    print(f'random forest confusion matrix: \n{rf_conf_matrix}')
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", rf_acc)
    plt.figure(figsize=(6,6))
    sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens',xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("rf_confusion_matrix.png")
    mlflow.log_artifact("rf_confusion_matrix.png")
    mlflow.sklearn.log_model(rf_model, "Random Forest Model")
    log_model_uri="runs:/{}/Random Forest Model".format(mlflow.active_run().info.run_id)
    registered_model = mlflow.register_model(log_model_uri, "IrisRandomForestModel")
    mlflow.end_run()
    print("model accuracy: ", rf_acc)
    print("confusion matrix: ", rf_conf_matrix)

logistic_model_uri="models:/IrisLogisticRegressionModel/latest"
loaded_lr_model = mlflow.sklearn.load_model(logistic_model_uri)

rf_model_uri="models:/IrisRandomForestModel/latest"
loaded_rf_model = mlflow.sklearn.load_model(rf_model_uri)






 

