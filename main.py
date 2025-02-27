import threshold
import pandas as pd

from model_train import Model_train
from threshold import Threshold

print("Start")

df = pd.read_csv('predictive_maintenance_large.csv')

threshold = Threshold(df)

# features = list(df.columns)
# X = df.drop([features[-1]], axis=1)
# y = df[features[-1]]
# columns = list(X.columns)
# data = [987231, 1817, 1000, 1.73, 3.84, 42.25, 16, 4954]
# model = Model_train(X,y)
# print("Predicting the output:")
# model.predict(data, columns)

