def preprocess_basic(df): 

    import pandas as pd
    
    df = df.copy()

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
    drop_list = ['Earth', False, '55 Cancri e', False]
    if 'Deck' in cat_cols:
        drop_list.append(None)
    if 'Side' in cat_cols:
        drop_list.append(None)
    ohe = OneHotEncoder(handle_unknown = 'ignore', drop = drop_list, sparse_output = False)
    cat_cols_encoded = ohe.fit_transform(df[cat_cols])
    cat_cols_encoded = pd.DataFrame(cat_cols_encoded, columns = ohe.get_feature_names_out(cat_cols), index = df.index)
    df = df.drop(columns = cat_cols)
    df = pd.concat([df, cat_cols_encoded], axis = 1)
            
    #scaling numerical data
    from sklearn.preprocessing import StandardScaler
    df[num_gaussian] = StandardScaler().fit_transform(df[num_gaussian])
            
    from sklearn.preprocessing import MinMaxScaler
    df[num_spent] = MinMaxScaler().fit_transform(df[num_spent])

    return df