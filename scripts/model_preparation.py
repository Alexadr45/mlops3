#создайте python-скрипт , который создает и обучает модель машинного обучения на построенных данных из папки “train”.
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("model_preparation")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    X_train = pd.read_csv('/home/antosha/project/scripts/datasets/X_train.csv', delimiter = ',')
    y_train = pd.read_csv('/home/antosha/project/scripts/datasets/Y_train.csv', delimiter = ',')
    penalty = 'l1'
    solver='liblinear'
    C = 8.75
    mlflow.log_param("penalty", penalty)
    mlflow.log_param("solver", solver)
    mlflow.log_param("C", C)
    model = LogisticRegression(fit_intercept=True,
                                penalty=penalty,solver=solver,
                                C=C,
                                max_iter=10000)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "")
    pickle.dump(model, open('/home/antosha/project/scripts/model/model.pkl', 'wb'))
