
''''
Implementation of Neural Network.
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from sklearn.cross_validation import train_test_split
import time
from keras.callbacks import EarlyStopping


## Batch generators 
## Keras wrapper

def batch_generator(X, y, batch_size, shuffle):

    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


## neural net,total 2 hidden layers
## Model parameters can be further optimized to improve the performance.
def nn_model():
    model = Sequential()
    model.add(Dense(512, input_dim = X_train.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    
    model.add(Dense(256, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    
#   model.add(Dense(128, init = 'he_normal'))
#   model.add(PReLU())
#   model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

if __name__ == '__main__':


	## read data
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	## set test loss to NaN
	test['loss'] = np.nan

	## response and IDs
	y = train['loss'].values
	id_train = train['id'].values
	id_test = test['id'].values

	## stack train test
	ntrain = train.shape[0]
	tr_te = pd.concat((train, test), axis = 0)

	## Preprocessing and transforming to sparse data
	sparse_data = []

	f_cat = [f for f in tr_te.columns if 'cat' in f]
	for f in f_cat:
	    dummy = pd.get_dummies(tr_te[f].astype('category'))
	    tmp = csr_matrix(dummy)
	    sparse_data.append(tmp)

	f_num = [f for f in tr_te.columns if 'cont' in f]
	scaler = StandardScaler()
	tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
	sparse_data.append(tmp)

	del(tr_te, train, test)

	## sparse train and test data
	xtr_te = hstack(sparse_data, format = 'csr')
	xtrain = xtr_te[:ntrain, :]
	xtest = xtr_te[ntrain:, :]

	print('Dim train', xtrain.shape)
	print('Dim test', xtest.shape)

	del(xtr_te, sparse_data, tmp)

	#Split data for validation
	X_train, X_val, y_train, y_val = train_test_split(xtrain, y, train_size=.80, random_state = 111)


	## define early stop callbacks
	early_stop = EarlyStopping(monitor='mae',patience=0,verbose=0,mode='auto')


	# calling neural network models
	model = nn_model()
	fit = model.fit_generator(generator = batch_generator(X_train, y_train, 128, True),
                         nb_epoch = 5,
                         samples_per_epoch = xtrain.shape[0],
                         validation_data = (X_val.todense(), y_val),
                         callbacks = [early_stop]
                         )


	pred_test = np.zeros(xtest.shape[0])
	pred = np.zeros(X_val.shape[0])
	for i in range(5):
	    print ("Training model %d" % (i+1))
	    model=create_model(num_of_feature)
	    fit= model.fit_generator(generator=batch_generator(X_train, y_train, 128, True),
	                            nb_epoch=5,
	                            samples_per_epoch=X_train.shape[0]
	                            )
	    pred +=model.predict_generator(generator = batch_generator(X_val,128,False))
	    preds_test=preds_test+model.predict_generator(generator=batch_generatorp(xtest, 128, False), val_samples=xtest.shape[0])
	pred_test += model.predict_generator(generator = batch_generatorp(xtest, 128, False), val_samples = xtest.shape[0])[:,0]
	preds_test = preds_test/5
	pred/=5
	print('Total - MAE:', mean_absolute_error(y_val, pred))


	## train predictions
	# df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
	# df.to_csv('preds_oob.csv', index = False)

	## test predictions
	# pred_test /= (nfolds*nbags)
	df = pd.DataFrame({'id': id_test, 'loss': pred_test})
	df.to_csv('submission_keras_first_try%f.csv'%time.time(), index = False)