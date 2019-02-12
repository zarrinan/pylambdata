""
" lambdata - a Data Science Helper
""
"
import numpy as np
import pandas as pd

import . zee

VERSION = 0
ONES = np.ones(100)
ONES_DF = pd.DataFrame(ONES)


#LabelEncoder Auto encodes any dataframe column of type category or object
def dummy_encode(df):
    from sklearn.preprocessing import LabelEncoder
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df


# Encode categorical features in a dataframe
def encode_cat(df):
    for col in df.columns:
        if (df[col].dtype == 'object'):
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df


# checks for nulls
def no_nulls(df):
    return not any(df.isnull().sum())


#training, test, validation split
def train_validation_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1,
        random_state=None, shuffle=True):

    from sklearn.model_selection import train_test_split
    assert train_size + val_size + test_size == 1

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, shuffle = shuffle)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
    random_state = random_state, shuffle = shuffle)

return X_Train, X_val, X_test, y_train, y_val, y_test


# print unique values
def unique(df):
    for col in df.columns:
    print(col, df[col].unique())

# cut outliers
def cut_outliers(df):
    from scipy import stats
    print(df.shape)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis = 1)]
    print(df.shape)
    return df

# check for non - numeric columns
def all_numeric(df):
    from pandas.api.types import is_numeric_dtype
    return all(is_numeric_dtype(df[col]) for col in df)