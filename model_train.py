from sklearn.model_selection import train_test_split
from flaml.ml import sklearn_metric_loss_score
import numpy as np
from flaml import AutoML
import pickle
import pandas as pd

class Model_train:
    def __init__(selfX, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        automl = AutoML()
        settings = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'accuracy',
            # check the documentation for options of metrics (https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#optimization-metric)
            "task": 'classification',  # task type
            "log_file_name": 'pm.log',  # flaml log file
            "seed": 7654321,  # random seed
        }
        print("Next Automl train")

        automl.fit(X_train=X_train, y_train=y_train, **settings)
        # print('Best ML leaner:', automl.best_estimator)
        # print('Best hyperparmeter config:', automl.best_config)
        # print('Best accuracy on validation data: {0:.4g}'.format(1 - automl.best_loss))
        # print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

        with open('automl.pkl', 'wb') as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        '''load pickled automl object'''
        with open('automl.pkl', 'rb') as f:
            automl = pickle.load(f)

        y_pred = automl.predict(X_test)
        y_pred = np.array(y_pred, dtype=str)  # Convert to string if categorical
        y_test = np.array(y_test, dtype=str)  # Ensure both match

        print('accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))

    def predict(self, data, columns):
        with open('automl.pkl', 'rb') as f:
            automl = pickle.load(f)

        testing = pd.DataFrame([data], columns=columns)
        t_pred = automl.predict(testing)
        print('Predicted labels', t_pred)

