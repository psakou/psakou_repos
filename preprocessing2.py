# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def preprocess(train, test):
    # Combining test and train for easy feature engineering.
    target = train.target_pct_vunerable

    train['separator'] = 0
    test['separator'] = 1

    train, test = train.align(test, join = 'inner', axis = 1)

    comb = pd.concat([train, test])

    # Examining feature interactions from the most important features from model's feature importances graph and creating new magic features.
    # While there is no science into it and it's mostly trial and error, the new features improved the score greatly and if we had computational power,
    # we could have explored more interactions.

    comb['household_size'] = comb.total_individuals/comb.total_households
    comb['gf_1'] = comb['dw_01'] * comb['psa_01']
    comb['gf_2'] = comb['gf_1'] * comb['psa_00']
    comb['gf_3'] = comb['gf_1'] * comb['psa_02']
    comb['gf_4'] = comb['gf_1'] * comb['psa_03']
    comb['gf_5'] = comb['gf_1'] * comb['gf_2']
    comb['gf_6'] = comb['gf_5'] * comb['gf_2']
    comb['dw_01_2'] = comb['dw_01'] ** 2
    comb['psa_00_2'] = comb['psa_00'] ** 2
    luxury_stuff = ['psa_01','car_01','stv_00']
    not_luxury_stuff = ['psa_00','car_00','stv_01']
    comb['luxury_stuff'] = comb[luxury_stuff].sum(axis=1)
    comb['not_luxury_stuff'] = comb[not_luxury_stuff].sum(axis=1)
    comb['a_luxury_stuff'] = comb[luxury_stuff].mean(axis=1)
    comb['a_not_luxury_stuff'] = comb[not_luxury_stuff].mean(axis=1)

    # Separating the train and test datasets.
    train = comb[comb.separator == 0]
    test = comb[comb.separator == 1]

    train.drop('separator', axis = 1, inplace = True)
    test.drop('separator', axis = 1, inplace = True)

    # The columns dropped were those that from the feature importance of the baseline model, were of least importance and just added noise to the model.

    X = train.drop(columns=['ward', 'dw_13', 'dw_12', 'lan_13', 'psa_03'])
    y = target.copy()
    tes = test.drop(['ward', 'dw_13', 'dw_12', 'lan_13', 'psa_03'], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2020)

    return X, y, tes, X_train, X_test, y_train, y_test

    
