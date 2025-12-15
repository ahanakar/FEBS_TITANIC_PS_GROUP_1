# from Ahana's branch
import numpy as np 
import pandas as pd
df=pd.read_csv('train.csv')
df = df.copy()

def preprocess_basic(df): 

    import pandas as pd

    #categorizing columns 
    df = df.drop(columns = ['Name'], errors = 'ignore')
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    num_gaussian = ['Age']
    num_spent = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[['Deck', 'CabinNumber', 'Side']] = df['Cabin'].str.split('/', expand=True) 
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


def logistic_regression(df_processed, y):
    #Now doing train test split and applying logistic regression 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

    y = df_processed['Transported']
    X = df_processed.drop(columns=['Transported', 'PassengerId', 'Group', 'AgeGroup', 'SpendGroup'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42,)

    #Train Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1] 

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))
    return logreg