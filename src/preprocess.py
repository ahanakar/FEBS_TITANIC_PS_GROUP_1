import seaborn as sns 
import matplotlib.pyplot as plt

df = df.copy()

#some preprocessing steps for plotting 
import pandas as pd
#categorizing columns 
df = df.drop(columns = ['Name'], errors = 'ignore')
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
num_gaussian = ['Age']
num_spent = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
for col in ['Deck', 'Side']:
    if col in df.columns:
        cat_cols.append(col)
            
#filling in null values
from sklearn.impute import SimpleImputer
df[num_gaussian] = SimpleImputer(strategy = 'median').fit_transform(df[num_gaussian])
#simple imputer used for age as it has normal distribution
            
from sklearn.impute import KNNImputer
df[num_spent] = KNNImputer(n_neighbors = 4, weights = 'distance').fit_transform(df[num_spent])
#add_indicator not used because extra columns will be added and it will be more complex 

df[cat_cols] = SimpleImputer(strategy = 'most_frequent').fit_transform(df[cat_cols])
for col in ['CryoSleep', 'VIP']:
    if col in df.columns:
        df[col] = df[col].astype(bool)


#home planet vs transported 
plot_hp = df.groupby('HomePlanet')['Transported'].mean().reset_index()
sns.barplot(x = 'HomePlanet', y = 'Transported', data = plot_hp, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('HomePlanet vs Transported')
plt.show()
#it is clear that passengers from Europa were transported the most followed by Mars and least from Earth

#cryosleep vs transported 
plot_cs = df.groupby('CryoSleep')['Transported'].mean().reset_index()
sns.barplot(x = 'CryoSleep', y = 'Transported', data = plot_cs, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('CryoSleep vs Transported')
plt.show()
#cryosleep has a much higher transportation rate when true

#destination vs transported 
plot_dst = df.groupby('Destination')['Transported'].mean().reset_index()
sns.barplot(x = 'Destination', y = 'Transported', data = plot_dst, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Destination vs Transported')
plt.show()
#passengers going to 55 Cancri e had a higher chance of being transported follwed by PSO J318.5-22
#that is followed closely by TRAPPIST-1e

#vip vs transported 
plot_vip = df.groupby('VIP')['Transported'].mean().reset_index()
sns.barplot(x = 'VIP', y = 'Transported', data = plot_vip, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('VIP vs Transported')
plt.show()
#passengers who weren't given VIP status had a higher chance of being transported 

#deck vs transported 
plot_deck = df.groupby('Deck')['Transported'].mean().reset_index()
sns.barplot(x = 'Deck', y = 'Transported', data = plot_deck, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Deck vs Transported')
plt.show()
#passengers from deck B had the highest chances of being transported followed closely by ones from deck C 
#passengers from decks A and G had alomst the same medium chance of transportation 
#passengers from decks D and F also had almost the same chance of transportation, but less than that of A and G
#passengers from deck E had a low chance of transportation and the lowest was from deck T

#side vs transported 
plot_side = df.groupby('Side')['Transported'].mean().reset_index()
sns.barplot(x = 'Side', y = 'Transported', data = plot_side, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Side vs Transported')
plt.show()
#passengers from starboard had a slightly higher chance of being transported than the ones from port but it was still low

#combo plot of homeplanet and cryosleep
plot_cmb = df.groupby(['HomePlanet', 'CryoSleep'])['Transported'].mean().reset_index()
sns.barplot(x = 'HomePlanet', y = 'Transported', data = plot_cmb, hue = 'CryoSleep', errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Variation of transportation rates with both CryoSleep and HomePlanet')
plt.show()
#the variation seen across the previous plots is maintained
#passengers are transported the most when they ahve opted for CryoSleep and are from Europa





def preprocess_basic(df): 

    import pandas as pd

    #categorizing columns 
    df = df.drop(columns = ['Name'], errors = 'ignore')
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    num_gaussian = ['Age']
    num_spent = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    for col in ['Deck', 'Side']:
        if col in df.columns:
            cat_cols.append(col)
            
    #filling in null values
    from sklearn.impute import SimpleImputer
    df[num_gaussian] = SimpleImputer(strategy = 'median').fit_transform(df[num_gaussian])
    #simple imputer used for age as it has normal distribution
            
    from sklearn.impute import KNNImputer
    df[num_spent] = KNNImputer(n_neighbors = 4, weights = 'distance').fit_transform(df[num_spent])
    #add_indicator not used because extra columns will be added and it will be more complex 

    df[cat_cols] = SimpleImputer(strategy = 'most_frequent').fit_transform(df[cat_cols])
    for col in ['CryoSleep', 'VIP']:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    #encoding categorical data
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown = 'ignore', drop = 'first', sparse_output = False)
    cat_cols_encoded = ohe.fit_transform(df[cat_cols])
    cat_cols_encoded = pd.DataFrame(cat_cols_encoded, columns = ohe.get_feature_names_out(cat_cols), index = df.index)
    df = df.drop(columns = cat_cols)
    df = df.drop(columns = ['Cabin', 'CabinNumber'])
    df = pd.concat([df, cat_cols_encoded], axis = 1)
            
    #scaling numerical data
    from sklearn.preprocessing import StandardScaler
    df[num_gaussian] = StandardScaler().fit_transform(df[num_gaussian])
            
    from sklearn.preprocessing import MinMaxScaler
    df[num_spent] = MinMaxScaler().fit_transform(df[num_spent])

    return df
