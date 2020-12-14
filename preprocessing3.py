# Import the libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


def preprocess(train, test):
    train.drop(columns=['dw_12', 'dw_13', 'lan_13'], inplace=True)
    test.drop(columns=['dw_12', 'dw_13', 'lan_13'], inplace=True)
    train_len = len(train)
    data=pd.concat([train,test])
    data['rich'] = data['car_01']+data['stv_01']+data['psa_01']+data['dw_02']+data['lln_00']
    data['poor'] = data['car_00'] +data['stv_00']+data['psa_00']+data['dw_01']+data['lln_01']
    data['household_size'] =data['total_individuals'] / data['total_households']
    columns=data.drop(["target_pct_vunerable","ward"],1).columns

    data_km=data[columns].copy()

    data_km["total_households"]/=data_km["total_households"].max()
    data_km["total_individuals"]/=data_km["total_individuals"].max()

    km=KMeans(15,random_state=2019)
    data["cluster"]=km.fit_predict(data_km[columns])

    train = data[:train_len]
    test = data[train_len:]

    _id = test['ward']
    test.drop(columns=['target_pct_vunerable','ward'], inplace=True)
    train.drop(columns=['ward'], inplace=True)

    train['total_households'] = np.log10(train['total_households'])
    test['total_households'] = np.log10(test['total_households'])

    train['total_individuals'] = np.log10(train['total_individuals'])
    test['total_individuals'] = np.log10(test['total_individuals'])

    X = train.drop(columns=['target_pct_vunerable'])
    y = train['target_pct_vunerable']

    col = ['car_00', 'car_01', 'dw_00', 'dw_01', 'dw_02', 'dw_03', 'dw_04',
           'lan_08', 'lan_09', 'lan_10', 'lan_11', 'lan_12', 'lan_14', 'lgt_00',
           'dw_05', 'dw_06', 'dw_07', 'dw_08', 'dw_09', 'dw_10', 'dw_11', 'lan_00',
           'lan_01', 'lan_02', 'lan_03', 'lan_04', 'lan_05', 'lan_06', 'lan_07',
           'lln_00', 'lln_01', 'pg_00', 'pg_01', 'pg_02', 'pg_03', 'pg_04',
           'psa_00', 'psa_01', 'psa_02', 'psa_03', 'psa_04', 'stv_00', 'stv_01',
            'rich', 'poor']

    X[col] = X[col].round(2)
    test[col] = test[col].round(2)

    categorical_features_indices = np.where(X.dtypes != np.float)[0]; categorical_features_indices

    return X, y, categorical_features_indices, test, _id
