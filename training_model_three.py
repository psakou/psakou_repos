import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('catboost')

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit
import numpy as np
import pandas as pd


def learner(X, y, test, categorical_features_indices, _id):
    testsplit_store=[]
    test_store=[]
    fold=KFold(n_splits=15, shuffle=True, random_state=123456)
    i=1
    for train_index, test_index in fold.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        cat = CatBoostRegressor(n_estimators=10000,eval_metric='RMSE', learning_rate=0.0801032, random_seed= 123456, l2_leaf_reg=4, use_best_model=True)
        cat.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=300,verbose=100, cat_features=categorical_features_indices)
        predict = cat.predict(X_test)
        print("err: ",np.sqrt(mean_squared_error(y_test,predict)))
        testsplit_store.append(np.sqrt(mean_squared_error(y_test,predict)))
        pred = cat.predict(test)
        test_store.append(pred)

    print("score:", np.mean(testsplit_store))

    submit_prep = {"ward": _id, 'target_pct_vunerable': np.mean(test_store, 0)}
    submission = pd.DataFrame(data = submit_prep)

    return submission
