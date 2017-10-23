'''
Implementation of bayes optimization process in a cluster. 
'''

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import sklearn as sk
import os,time,datetime

import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold



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


def xgb_evaluate(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha,eta):

    params = {}
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = max(min(colsample_bytree,1),0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample,1),0)
    params['gamma'] = max(gamma,0)
    params['alpha'] = max(alpha,0)
    params['eta'] = max(eta,0)
    
    params['silent'] = 1
    params['verbose_eval'] = True
    params['seed'] = random_state
    params['early_stopping_rounds'] = early_stopping
    
#   cv_result = xgb.cv(params, xgtrain, num_boost_round = num_rounds, nfold = nfolds, 
#                        seed = random_state, early_stopping_rounds = early_stopping)
    
    cv_result = xgb.cv(params, xgtrain_cv, obj= logregobj, feval= eval_error, num_boost_round = num_rounds, nfold = nfolds, 
                       seed = random_state, early_stopping_rounds = early_stopping)
    
    
    return -cv_result['test-mae-mean'].values[-1]

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
    test_only = test_col - common_col
    train_only = train_col - common_col

    xtrain_com = xtrain[common_col]
    xtest_com = xtest[common_col]

    # params = {}
    # params['min_child_weight'] = 137
    # params['colsample_bytree'] = 0.6017
    # params['max_depth'] = 9
    # params['subsample'] = 0.8614
    # params['gamma'] = 0.6569
    # params['alpha'] = 1
    # params['eta'] = 0.02
    # params['silent'] = 1
    # params['verbose_eval'] = True
    # params['seed'] = random_state
    # params['early_stopping_rounds'] = early_stopping


    print 'dimension of xtrain: {}'.format(xtrain_com.shape)
    print 'dimension of xtest: {}'.format(xtest_com.shape)
    print 'dimension of ytrain: {}'.format(ytrain.shape)

    # The set of parameters

    shift = 200
    ytrain_xgb= np.log(ytrain + shift)
    xtrain_xgb= xtrain_com.as_matrix()
    xtest_xgb= xtest_com.as_matrix()

    # Xgboost model and bayesian optimization the parameter

    num_rounds = 4000
    random_state = 2016
    early_stopping = 25
    nfolds = 3

    xgtrain_cv = xgb.DMatrix(xtrain_xgb, label = ytrain_xgb)

    num_iter = 30
    init_points = 5

    print 'start xgboost bayesian optimization \n'

    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight':(10,200),   
                                                'colsample_bytree': (0.5,0.99),
                                                'max_depth': (4,12),     # max_depth : 4-12
                                                'subsample': (0.7, 0.9),
                                                'gamma' : (0.01,2),
                                                'alpha' : (0,10),
                                                'eta': (0.001,0.2),
                                                })

    xgbBO.maximize(init_points = init_points, n_iter = num_iter)
    print ("\n\n\nfinished in %f seconds" %(time.time() - start_time)) 
    print('Xgboost: %f' %xgbBO.res['max']['max_val'])