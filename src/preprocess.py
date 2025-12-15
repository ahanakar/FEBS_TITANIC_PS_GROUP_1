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
#it isn't clear whether passengers from Mars will be transported or not as the proprotion is nearly 0.5
#however it is likely that passengers from Europa will be transported as their proprotion is nearly 0.65
#passengers from Earth may not be transported that much as the proprotion is nearly 0.4

#cryosleep vs transported 
plot_cs = df.groupby('CryoSleep')['Transported'].mean().reset_index()
sns.barplot(x = 'CryoSleep', y = 'Transported', data = plot_cs, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('CryoSleep vs Transported')
plt.show()
#cryosleep has a much higher transportation rate when true
#so it very likely that apssengers opting for cryosleep will be transported as the proportion is nearly 0.8 (very high)
#however those not in cryosleep are highly likely to not be transported (proportion around 0.3)

#destination vs transported 
plot_dst = df.groupby('Destination')['Transported'].mean().reset_index()
sns.barplot(x = 'Destination', y = 'Transported', data = plot_dst, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Destination vs Transported')
plt.show()
#passengers going to 55 Cancri e had a higher chance of being transported follwed by PSO J318.5-22
#that is followed closely by TRAPPIST-1e
#it is not clear whether passengers from PSO J318.5-22 and TRAPPIST-1e will be transported as their proportion is nearly 0.5
#however there is a slightly higher chance that passengers from 55 Cancri e will be transported as the proportion is nearly 0.6

#vip vs transported 
plot_vip = df.groupby('VIP')['Transported'].mean().reset_index()
sns.barplot(x = 'VIP', y = 'Transported', data = plot_vip, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('VIP vs Transported')
plt.show()
#passengers who weren't given VIP status had a higher chance of being transported 
#however, for non-VIP passengers it is unclear whether they will be transported 
#for VIP passengers there's a chance that they won't be transported as the proportion is around 0.37

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
#out of these, passengers from decks B and C have a high chance of being transported (proportion 0.7 and 0.65)
#it is unclear whether passengers from A and G will be transported or not as the proportion is nearly 0.5
#however, it is very likely that passengers from deck T will not be transported (proportion 0.2)

#side vs transported 
plot_side = df.groupby('Side')['Transported'].mean().reset_index()
sns.barplot(x = 'Side', y = 'Transported', data = plot_side, errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Side vs Transported')
plt.show()
#passengers from starboard had a slightly higher chance of being transported than the ones from port but it was still low
#however, both port and starboard have proportions of nearly 0.45 and 0.5 so it is unclear if they will be transported or not

#combo plot of homeplanet and cryosleep
plot_cmb = df.groupby(['HomePlanet', 'CryoSleep'])['Transported'].mean().reset_index()
sns.barplot(x = 'HomePlanet', y = 'Transported', data = plot_cmb, hue = 'CryoSleep', errorbar = None)
plt.ylabel('Proportion transported')
plt.title('Variation of transportation rates with both CryoSleep and HomePlanet')
plt.show()
#the variation seen across the previous plots is maintained
#passengers are transported the most when they ahve opted for CryoSleep and are from Europa
#passengers are transported the most when they opt for CryoSleep and are from Europa (propotion is 1, so it's basically sure)
#CryoSleep remains one of the most indicative features

#cabin number vs transported 
sns.boxplot(y = 'CabinNumber', x = 'Transported', data = df)
plt.title('Cabin number vs Transported')
plt.show()
#cabin number is not a very important feature as the difference is not a lot
#however the transporeted passengers have a slightly lower avergae cabin number

#age and totalspend vs transported 
sns.scatterplot(x = 'Age', y = 'TotalSpend', hue = 'Transported', data = df)
plt.title('Age and total amount spent as compared to transportation')
plt.show()
#substantial overlap between transported and non trasnported so neither age nor total spending alone are good indicators
#their interaction also isn't a good enough feature





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
