import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

print(train_df.head())