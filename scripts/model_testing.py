import pandas as pd
import pickle
from sklearn.metrics import f1_score
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("model_testing")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    LogReg = pickle.load(open('/home/antosha/project/scripts/model/model.pkl', 'rb'))
    X_test = pd.read_csv('/home/antosha/project/scripts/datasets/X_test.csv', delimiter = ',')
    Y_test = pd.read_csv('/home/antosha/project/scripts/datasets/Y_test.csv', delimiter = ',')
    y_preds = LogReg.predict(X_test)
    f1 = f1_score(Y_test, y_preds, average="micro")
    mlflow.log_metric("F1_score", float(f1))
    print(f'f1_score: {f1}')
