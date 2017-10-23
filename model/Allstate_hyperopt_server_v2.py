'''
Hyperparameter tuning process in a cluster. 
'''
import numpy as np
import scipy as sp
from pandas import DataFrame, Series
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import os,time,datetime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import KFold

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


def eval_error(ypred, dtrain):
    ytrue = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(ypred), np.exp(ytrue))

# def logregobj(ypred, dtrain):
#     ytrue = dtrain.get_label()
#     x = ypred - ytrue
#     grad = np.tanh(x)
#     hess = 1.0 - grad * grad
#     return grad, hess

def logregobj(ypred, dtrain):
    ytrue = dtrain.get_label()
    constant = 2
    x = ypred - ytrue
    grad = constant * x / (abs(x) + constant)
    hess = constant ** 2 / (abs(x) + constant) ** 2
    return grad, hess



def xgb_hp_obj(params):
    
    print '\nNew run of hyperparameter optimization:\n'

    params['min_child_weight'] = int(params['min_child_weight'])
    params['colsample_bytree'] = float('{0:.2f}'.format(params['colsample_bytree']))
    params['subsample'] = float('{0:.2f}'.format(params['subsample']))
    params['gamma'] = float('{0:.2f}'.format(params['gamma']))
    params['alpha'] = float('{0:.1f}'.format(params['alpha']))
    params['eta'] = float('{0:.3f}'.format(params['eta']))

    print params   # this is in log space. Remember to take log back for learning rate eta.
    
    cv_result = xgb.cv(params, xgtrain_cv, 
                       obj= logregobj, 
                       feval= eval_error, 
                       num_boost_round = num_rounds, 
                       nfold = nfolds, 
                       seed = random_state, 
                       early_stopping_rounds = early_stopping)
    
    cv_MAE = cv_result['test-mae-mean'].values[-1]
    
    print '\ncv_MAE: %.4f\n' %cv_MAE
    
    with open('xgb_hyperopt_1127.txt', 'a') as file:
        
        file.write('cv_MAE = %.4f\t' %cv_MAE)

        for key in params.iterkeys():
            file.write('%s = %s\t'%(key,str(params.get(key))))
            
        file.write('\n')
    
    return {'loss': cv_MAE, 'status': STATUS_OK}
    
    
def xgb_hp_evaluate(trials):

    params = {}
    params['min_child_weight'] = hp.quniform('min_child_weight',10, 200, 1)
    params['colsample_bytree'] = hp.quniform('colsample_bytree', 0.5, 0.99, 0.05)
    #params['max_depth'] = hp.quniform('max_depth',4, 12, 1)
    params['max_depth'] = hp.randint('max_depth', 8) + 5
    params['subsample'] = hp.quniform('subsample', 0.7, 0.9, 0.05)
    params['gamma'] = hp.quniform('gamma', 0.01, 2, 0.05)
    params['alpha'] = hp.quniform('alpha', 0, 5, 0.1)
    params['eta'] = 10 ** hp.uniform('eta', -1, 0)   # this is in log space. Remember to take log back.
    
    params['silent'] = 1
    params['verbose_eval'] = False

    best = fmin(xgb_hp_obj, params, algo = tpe.suggest, trials = trials, max_evals = 4)
    
    print '\n\nThe best parameter is: '
    print best

if __name__ == '__main__':
    

    # record the time of the excution
    start_time = time.time()
    print 'Start.....\n'

    xtrain  = pd.read_csv('X_train_OHE.csv')
    ytrain = pd.read_csv('Y_train.csv')
    xtest  = pd.read_csv('X_test_OHE.csv')

    test = pd.read_csv('test.csv')
    ids = test['id']

    train_col = xtrain.columns 
    test_col = xtest.columns
    common_col = test_col & train_col
    # test_only = test_col.difference(common_col)
    # train_only = train_col.difference(common_col)

    xtrain_com = xtrain[common_col]
    xtest_com = xtest[common_col]

    print 'dimension of xtrain: {}'.format(xtrain_com.shape)
    print 'dimension of xtest: {}'.format(xtest_com.shape)
    print 'dimension of ytrain: {}'.format(ytrain.shape)

    # The set of parameters

    shift = 200
    ytrain_xgb= np.log(ytrain + shift)
    xtrain_xgb= xtrain_com.as_matrix()
    xtest_xgb= xtest_com.as_matrix()

    # Xgboost model and hyperopt optimization with the parameters

    num_rounds = 10
    random_state = 2016
    early_stopping = 5
    nfolds = 3

    xgtrain_cv = xgb.DMatrix(xtrain_xgb, label = ytrain_xgb)

    #Trials object where the history of search will be stored
    trials = Trials()

    start_time = time.time()
    xgb_hp_evaluate(trials)
    print ("\n\n\nfinished in %f seconds" %(time.time() - start_time)) 
