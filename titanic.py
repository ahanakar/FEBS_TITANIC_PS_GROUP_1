import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plots_by_Aditya(train):
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

    plt.figure(figsize=(8,5))
    sns.countplot(x='VIP',hue='Transported',data=train, palette='Set2')
    plt.ylabel('No. of people transported')
    plt.title('VIP/Non VIP vs Transpoted')
    # plt.show()
    # INTERPRETAION
    '''
    Most passengers are Non-VIP, and their transported vs not-transported counts are almost same.

    VIP passengers are very few compared to Non-VIPs.

    Among VIPs, more passengers were not transported than transported, showing no strong positive impact of VIP status.

    VIP shows weak influence on the target due to very small VIP sample size
    '''
