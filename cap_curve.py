#!/usr/bin/env python
# coding: utf-8

# Import Library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

clf= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
clf.fit(X_train, y_train)

roc_auc_value=roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])  
fpr_rf,tpr_rf,thresholds_rf=roc_curve(y_train, clf.predict_proba(X_train)[:,1])

plt.plot(fpr_rf,tpr_rf,'b', label='User Model')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve. AUC ="+str(np.round(roc_auc_value,4)))
plt.grid(True)
plt.legend(loc = 'lower right')
plt.savefig('ROC.png')
plt.show()


#CAP curve
from matplotlib import cm
from scipy import integrate
def capcurve(y_values, y_proba):
    num_sum = np.sum(y_values)
    num_count = len(y_values)
    rate_val = float(num_sum) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_val,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values,y_proba]
    y_cap_df = pd.DataFrame(data=y_cap)

    y_cap_df = y_cap_df.sort_values([1], ascending=False)
    y_cap_df = y_cap_df.reset_index(drop=True)

    yy = np.cumsum(y_cap_df[0]) / float(num_sum)
    yy = np.append([0], yy[0:num_count-1]) 

    percent = 0.5
    row_index = np.trunc(num_count * percent)
    row_index = row_index.astype(np.int32)

    sigma_perfect = 1 * xx[num_sum - 1 ] / 2 + (xx[num_count - 1] - xx[num_sum]) * 1
    sigma_model = integrate.simps(yy,xx)
    sigma_random = integrate.simps(xx,xx)

    gini_value = (sigma_model - sigma_random) / (sigma_perfect - sigma_random)

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    #ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='b', label='User Model')
    ax.plot(xx,xx, linestyle='dashed', color='r', label='Random Model')
    
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.title("CAP Curve. Gini index ="+str(np.round(gini_value,4)))
    plt.xlabel('% of the data')
    plt.ylabel('% of bad')
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.savefig('CAP.png')
    plt.show()


y_pred_proba = clf.predict_proba(X=X_train)
capcurve(y_values=y_train, y_proba=y_pred_proba[:,1])


