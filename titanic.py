import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train.csv')

# Define age bins
bins = list(range(0, 81, 10))  # 0-9, 10-19, ..., 70-79, 80
labels = [f'{i}-{i+9}' for i in range(0, 80, 10)]
train['AgeGroup'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)

# Age Group vs Number of Passengers
plt.figure(figsize=(10,6))
sns.countplot(x='AgeGroup', data=train,hue='Transported', palette='viridis')
plt.title('Count of Passengers in Different Age Groups (0-80)')
plt.xlabel('Age Group')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=45)
# plt.show()

# INTERPRETATION
'''
The 20–29 age group has the highest number of passengers overall so it the most dominant age group in the dataset.

(0–9 years) age group shows a higher number of transported passengers compared to non-transported, indicating a higher chances of being transported.

(10–19 years) age group also have more transported passengers than non-transported,i ndicating younger passengers had more transport outcomes.

In the 20–29 and 30–39 age groups, the number of non-transported passengers is higher than transported it shows a lower transport probability for young and middle-aged adults.

Passenger counts reduces repidly after age 40, it shows that there are fewer older passengers in the dataset.

This trend suggests that Age is an important feature and should be retained for further modeling and analysis.
'''


# age vs transported
train['Age'] = train['Age'].fillna(train['Age'].median())

plt.figure(figsize=(8,5))
sns.boxplot(x='Transported',y='Age',data=train,)
plt.xlabel('Transported')
plt.ylabel('Age')
plt.title('Age Distribution by Transported Status')
# plt.show()

# INTERPRETATION
'''
The median age of transported and non-transported passengers is almost the same (around mid-20s), indicating age alone does not strongly separate the two groups.

IQR is similar for both groups, showing that the age spread of transported and non-transported passengers is comparable.

Both groups contain older age outliers (60–80 years), suggesting that transportation occurred across all age ranges, not limited to younger passengers.
'''

spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train[spend_cols] = train[spend_cols].fillna(0)
train['TotalSpend'] = train[spend_cols].sum(axis=1)
bins = [0, 100, 500, 1000, 2000, 5000, 10000, train['TotalSpend'].max()+1]
labels = ['0-100', '101-500', '501-1000', '1001-2000', '2001-5000', '5001-10000', '10000+']

train['SpendGroup'] = pd.cut(train['TotalSpend'], bins=bins, labels=labels, right=False)
# Plot countplot
plt.figure(figsize=(12,6))
sns.countplot(x='SpendGroup', hue='Transported', data=train, palette='Set2')
plt.title('Count of Passengers by Total Spending and Transported Status')
plt.xlabel('Total Spending Group')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0)
plt.legend(title='Transported')
# plt.show()

# INTERPRETATION
'''
Passengers with very low total spending (0–100) have a much higher count of transported passengers, indicating that low spenders were more chances of being transported.

As total spending increases, the number of non-transported passengers becomes higher than transported, especially in spending ranges above 500.

High-spending groups (2000+) contain fewer passengers overall, but they show a lower likelihood of being transported, suggesting an inverse relationship between spending and transport outcome.
'''

# groupsize vs transported count plot
train['Group'] = train['PassengerId'].str.split('_').str[0]
train['GroupSize'] = train.groupby('Group')['Group'].transform('count')
sns.countplot(x='GroupSize', hue='Transported', data=train)
plt.title('Group Size Distribution by Transported')
# plt.show()

# INTERPRETATION
'''
Solo travelers form the largest category, and a higher number of them are not transported compared to transported.

As group size increases (2–6 members), the transported count becomes higher than non-transported, suggesting passengers traveling in groups (family or non-family) had a better chance of being transported.

Larger groups (7–8 members) are fewer in number overall, but they still show a slightly higher transported count, indicating group association positively influenced transportation outcomes regardless of whether the group was a family or not.
'''


plt.figure(figsize=(8,5))
sns.countplot(x='HomePlanet', hue='Transported', data=train, palette='Set2')
plt.title('HomePlanet vs Transported Count')
plt.xlabel('HomePlanet')
plt.ylabel('Number of Passengers')
# plt.show()

# INTERPRETATION
'''
Passengers from Europa show a much higher transported count than non-transported, indicating that Europa had better transportation.

Earth has the largest number of passengers overall, but the non-transported count is higher than transported,  Earth passengers had less chance to be transported.

Passengers from Mars have nearly balanced counts of transported and non-transported, indicating a relatively neutral impact of HomePlanet on transportation for this group.
'''

plt.figure(figsize=(6,5))
sns.countplot(x='CryoSleep', hue='Transported', data=train, palette='Set2')
plt.ylabel('Number of Passengers')
plt.title('CryoSleep vs Transported Count')
# plt.show()

# INTERPREATATION
'''
CryoSleep is a very strong predictive feature, as passengers in cryosleep had a significantly higher probability of being transported compared to those who were not there.
'''

#deck vs transported 
train[['Deck', 'CabinNumber', 'Side']] = train['Cabin'].str.split('/', expand=True)  
train['CabinNumber'] = pd.to_numeric(train['CabinNumber'], errors='coerce')

plt.figure(figsize=(8,5))
sns.countplot(x='Deck', hue='Transported', data=train, palette='Set2')
plt.ylabel('Number of Passengers')
plt.title('Deck vs People Transported')
# plt.show()
# INTERPRETATION
'''
Deck B, C, and G have a higher number of transported passengers, indicating a greater chance of transportation for passengers on these decks.

Deck F and E show a higher count of non-transported passengers, suggesting passengers from these decks were less likely to be transported.

Deck A and D show relatively balanced counts, while Deck T has very few passengers and does not provide meaningful insight.

Cabin Deck is an important categorical feature, as transportation chances varies noticeably across different decks
'''

# FROM AHANA'S BRANCH

def preprocess_basic(train): 
    train = train.copy()

    #categorizing columns 
    train = train.drop(columns = ['Name'], errors = 'ignore')
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    num_gaussian = ['Age']
    num_spent = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    train[['Deck', 'CabinNumber', 'Side']] = train['Cabin'].str.split('/', expand=True)
    
    for col in ['Deck', 'Side']:
        if col in train.columns:
            cat_cols.append(col)
            
    #filling in null values
    from sklearn.impute import SimpleImputer
    train[num_gaussian] = SimpleImputer(strategy = 'median').fit_transform(train[num_gaussian])
    #simple imputer used for age as it has normal distribution
            
    from sklearn.impute import KNNImputer
    train[num_spent] = KNNImputer(n_neighbors = 4, weights = 'distance').fit_transform(train[num_spent])
    #add_indicator not used because extra columns will be added and it will be more complex 

    train[cat_cols] = SimpleImputer(strategy = 'most_frequent').fit_transform(train[cat_cols])
    for col in ['CryoSleep', 'VIP']:
        if col in train.columns:
            train[col] = train[col].astype(bool)

    #encoding categorical data
    from sklearn.preprocessing import OneHotEncoder
    drop_list = ['Earth', False, '55 Cancri e', False]
    if 'Deck' in cat_cols:
        drop_list.append(None)
    if 'Side' in cat_cols:
        drop_list.append(None)
    ohe = OneHotEncoder(handle_unknown = 'ignore', drop = drop_list, sparse_output = False)
    cat_cols_encoded = ohe.fit_transform(train[cat_cols])
    cat_cols_encoded = pd.DataFrame(cat_cols_encoded, columns = ohe.get_feature_names_out(cat_cols), index = train.index)
    train = train.drop(columns = cat_cols)
    train = pd.concat([train, cat_cols_encoded], axis = 1)
            
    #scaling numerical data
    from sklearn.preprocessing import StandardScaler
    train[num_gaussian] = StandardScaler().fit_transform(train[num_gaussian])
            
    from sklearn.preprocessing import MinMaxScaler
    train[num_spent] = MinMaxScaler().fit_transform(train[num_spent])
    return(train)

