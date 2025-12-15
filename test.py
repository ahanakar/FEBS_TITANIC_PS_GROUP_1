import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import preprocess
from old import feature_engineering
import pickle

df_test = pd.read_csv("test.csv")

passenger_ids = df_test['PassengerId']

df_processed = feature_engineering(df_test)
df_final = preprocess(df_processed)
X_test = df_final.drop(columns=['PassengerId', 'Group', 'AgeGroup', 'SpendGroup'], errors='ignore')

with open("artifacts/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': pred})
submission.to_csv('submission.csv', index = False)