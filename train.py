import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import preprocess_for_plots, plots_by_Ahana, preprocess
from titanic import plots_by_Aditya
from old import feature_engineering
from lor import logistic_regression

df = pd.read_csv("train.csv")

df_processed = feature_engineering(df)
df_processed = preprocess_for_plots(df)

'''#plots and analysis 
plots_by_Aditya(df_processed)
plots_by_Ahana(df_processed)'''

df_final = preprocess(df_processed)

#training logistic regression model
y = df_final['Transported']
logreg = logistic_regression(df_final, y)

#saving the model
import pickle
import os

def save_model(model, path="artifacts/logistic_model.pkl"):
    os.makedirs("artifacts", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

save_model(logreg)