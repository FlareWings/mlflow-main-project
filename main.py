import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the Telecom Churn Dataset csv file
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Churn_Data_Final.csv")
    Churn_Data_01 = pd.read_csv(dataset_path)

    # Independent variable
    X = Churn_Data_01.drop(['Churn'], axis=1)
    # Dependent variables
    y = Churn_Data_01.Churn
    
    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.20, random_state=42)
    
    # Scalining training data sets
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    max_leaf_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    n_estimators = int(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    max_features = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    random_state = int(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    

    with mlflow.start_run():
        GBCModel = GradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=random_state)
        GBCModel.fit(X_train_scaled, y_train)

        predicted_qualities = GBCModel.predict(X_test_scaled)

        Accuracy_Score = round(accuracy_score(y_test, predicted_qualities), 2)*100

        print("Gradient Boosting Classifier model (learning_rate={:.2f}, max_leaf_nodes={:.0f}, n_estimators={:.0f}, max_features={:.2f}, max_depth={:.0f}, random_state={:.0f}):".format(learning_rate, max_leaf_nodes, n_estimators, max_features, max_depth, random_state))
        print("  Accuracy of the Experiment: {:.2f}%".format(Accuracy_Score))

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("Accuracy Score in Percentage", Accuracy_Score)

        mlflow.sklearn.log_model(GBCModel, "model", registered_model_name="Gradient Boosting Classifier")