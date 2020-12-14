import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def viz(model, train):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,train.columns)), columns=['Value','Feature'])

    plt.figure(figsize=(30, 20))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('XGB Features (avg over folds)')
    plt.tight_layout()
    plt.show()

def submit(submission, pred_test, name):

    submission['target_pct_vunerable'] = np.absolute(pred_test/4)
    submission.to_csv(f'{name}.csv',index=False)

    return submission

def submit_clip(submission, a_min, a_max, name):

    submission['target_pct_vunerable'] = np.clip(submission['target_pct_vunerable'], a_min=a_min, a_max=a_max)
    submission.to_csv(f'{name}_clipped_{a_min}_{a_max}.csv',index=False)

    return submission
