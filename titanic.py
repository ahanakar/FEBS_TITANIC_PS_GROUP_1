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
test_fe = feature_engineering(test)
print(train_fe)
print(test_fe)

import matplotlib.pyplot as plt
import seaborn as sns
# age vs transported
plt.figure(figsize=(10,6))
sns.boxplot(x='Transported', y='Age', data=train)
plt.title('Age vs Transported')
plt.show()

# spending vs transported
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_melt = train.melt(id_vars='Transported', value_vars=spending_cols, 
                  var_name='SpendingType', value_name='Amount')

# plt.figure(figsize=(10,6))
sns.boxplot(x='SpendingType', y='Amount', hue='Transported', data=train_melt)
plt.title('Spending Features vs Transported')
plt.xticks(rotation=0)
plt.show()

# groupsize vs transported
train['Group'] = train['PassengerId'].str.split('_').str[0]
 
train['GroupSize'] = train.groupby('Group')['Group'].transform('count')

# plt.figure(figsize=(10,5))
sns.countplot(x='GroupSize', hue='Transported', data=train)
plt.title('Group Size Distribution by Transported')
plt.show()
