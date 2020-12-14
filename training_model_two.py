import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('catboost')
install('rgf-python')


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor,\
                        HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from rgf.sklearn import RGFRegressor
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def learner(X, y, tes, X_train, X_test, y_train, y_test):

    # In stacking, the most important thing is model diversification. from linear, SVM, KNN and Decision trees and many variations of them.
    # The variations are different values of key parameters of each model.
    # While we did not have the time to tune parameters of each model, except the meta learner Catboost, educated guesses on
    # the parameters were made to have as much variability as possible.

    estimators_1 = [
        ('xgb', XGBRegressor(random_state=2020, objective ='reg:squarederror', learning_rate=0.05)),
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=2020)),
        ('lgb', LGBMRegressor(learning_rate=0.2, random_state=2020)),
        ('svr', SVR(degree=2)),
        ('lasso', Lasso(random_state=2020)),
        ('RGF', RGFRegressor()),
        ('kneiba', KNeighborsRegressor(n_neighbors=4)),
        ('cat', CatBoostRegressor(logging_level='Silent', random_state=2020))
    ]

    predictions_1 = StackingRegressor(estimators=estimators_1, final_estimator=CatBoostRegressor(logging_level='Silent', depth=6, bagging_temperature=5, random_state=2020)).fit(X_train, y_train).predict(tes)

    estimators_2 = [
        ('xgb', XGBRegressor(objective ='reg:squarederror', learning_rate=0.2, random_state=2020)),
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=2020)),
        ('lgb', LGBMRegressor(learning_rate=0.05, random_state=2020)),
        ('svr', SVR(degree=5)),
        ('RGF', RGFRegressor()),
        ('lasso', Lasso(random_state=2020)),
        ('kneiba', KNeighborsRegressor(n_neighbors=6)),
        ('cat', CatBoostRegressor(logging_level='Silent', random_state=2020))
    ]

    predictions_2 = StackingRegressor(estimators=estimators_2, final_estimator=CatBoostRegressor(logging_level='Silent', depth=6, bagging_temperature=5, random_state=2020)).fit(X_train, y_train).predict(tes)

    predictions_cat_1 = CatBoostRegressor(logging_level='Silent', depth=6, bagging_temperature=5, random_state=2020).fit(X_train, y_train).predict(tes)


    # Further averaging, blending and retraining to generalise well
    # While the ratios are greater than one, it still works a treat. This is definitely one of the parameters to tune to achieve great results.
    stack = [x*0.56 + y*0.51 for x, y in zip(predictions_1, predictions_2)]
    stack_2 = [x*0.56 + y*0.51 for x, y in zip(stack, predictions_cat_1)]

    X,y = tes.copy(), stack_2
    preds_ridge = Ridge(random_state=2020).fit(X, y).predict(X)

    # We added a new feature to the test dataset, where we clustered the wards to 150 clusters, then used Catboost's encoder to encode the clusters.
    X['cluster'] = KMeans(150, random_state=2020).fit(X).predict(X)
    preds_cat = CatBoostRegressor(random_state=2020, verbose = False, depth=6, bagging_temperature=5, cat_features=['cluster']).fit(X, y).predict(X)

    # blended the Ridge and Catboost predictions.
    final_blend_2 = [x*0.2 +y*0.8 for x, y in zip(preds_ridge, preds_cat)]

    # Clipping the values from between 0 - 90 was also important as we know that the target variable is between 0 to 100.
    final_blend_2 = np.clip(final_blend_2, a_min=0, a_max=90)

    # Applying regularization to the final blend by substracting a constant from the predictions and clipping again.
    exp = final_blend_2 - 0.48
    exp = np.clip(exp, a_min=0, a_max=90)

    ## Retraining predictions

    # Retraining on the test data by using the prediction of the stacked regressors as our target.
    # We also added the clusters but had to manually mean encode the clusters to the target variable as LinearRegression cannot encode categorical variables.
    X = tes.copy()

    X['cluster'] = KMeans(150, random_state=2020).fit(X).predict(X)
    X['target'] = exp
    X['encoded'] = X['cluster'].map(X.groupby('cluster')['target'].mean())
    y=X.target
    X=X.drop(['cluster', 'target'], 1)
    preds_1 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.7 + LinearRegression().fit(X, y).predict(X)*0.3
    preds_2 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.5 + LinearRegression().fit(X, y).predict(X)*0.5
    preds_3 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.6 + LinearRegression().fit(X, y).predict(X)*0.4

    final = [x*0.3 + y*0.3 + z*0.4 for x, y, z in zip(preds_1, preds_2, preds_3)]

    ## Further retraining of predictions

    # Retraining again this time using Regularized Greedy Forests and Catboost.
    X['final'] = final
    y = X.final
    X = X.drop('final', 1)
    preds_1 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.7 + RGFRegressor().fit(X, y).predict(X)*0.3
    preds_2 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.5 + RGFRegressor().fit(X, y).predict(X)*0.5
    preds_3 = CatBoostRegressor(verbose = False, random_state=2020).fit(X,y).predict(X)*0.6 + RGFRegressor().fit(X, y).predict(X)*0.4

    final2 = [x*0.3 + y*0.3 + z*0.4 for x, y, z in zip(preds_1, preds_2, preds_3)]


    return final2
