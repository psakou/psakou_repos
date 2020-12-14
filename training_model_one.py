from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

## defining the metric
def metric(x,y):
  return np.sqrt(mean_squared_error(x,y))


def learner(train, target, test):

    kf = KFold(n_splits=4,shuffle=False)
    xgb = XGBRegressor(n_estimators=50000,random_state=42,max_depth=5,learning_rate=0.03888)
    scores = []
    pred_test = np.zeros(len(test))
    for (train_index,test_index) in kf.split(train,target):
      X_train,X_test = train.iloc[train_index],train.iloc[test_index]
      y_train,y_test = target.iloc[train_index],target.iloc[test_index]
      xgb.fit(X_train,y_train,early_stopping_rounds=500,eval_set=[(X_test,y_test)],eval_metric='rmse')
      scores.append(metric(xgb.predict(X_test),y_test))
      pred_test+=xgb.predict(test)

    return xgb, pred_test, scores
