import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def split_data(path:str,stage='lgb',cols_drop=[]) -> tuple:
    '''

    Prepare data for train and validation sets.

    '''

    columns_to_drop = ['SalePrice', 'Id'] + cols_drop

    df = pd.read_csv(path, encoding='utf8')
    if stage == 'df':
        return df
    df['SalePrice'] = np.log1p(df['SalePrice'])
    X = df.drop(columns_to_drop, axis=1)
    y = df['SalePrice']
    
    cat_cols = X.select_dtypes(include=['string']).columns
    for col in cat_cols:
        X[col] = X[col].astype('category')

    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42) 

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=list(cat_cols))
    test_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    if stage == 'lgb':

        return train_data, test_data


def test_data_func(path:str, cols_drop=[]):
    '''
    
    Returns testing dataset to make a prediction
    
    '''


    columns_to_drop = cols_drop

    df = pd.read_csv(path, encoding='utf8')
    df = df.drop(columns_to_drop, axis=1)

    cat_cols = df.select_dtypes(include=['string']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df, cat_cols


