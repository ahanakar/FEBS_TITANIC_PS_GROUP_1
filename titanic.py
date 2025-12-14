import numpy as np
import pandas as pd
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

def feature_engineering(train):
    train['Group'] = train['PassengerId'].str.split('_').str[0]
 
    train['GroupSize'] = train.groupby('Group')['Group'].transform('count')
    
    train[['Deck', 'CabinNumber', 'Side']] = train['Cabin'].str.split('/', expand=True)
    
    train['CabinNumber'] = pd.to_numeric(train['CabinNumber'], errors='coerce')
    
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    train[spend_cols] = train[spend_cols].fillna(0)

    train['TotalSpend'] = train[spend_cols].sum(axis=1)
    return train
train_fe = feature_engineering(train)
print(train_fe)
