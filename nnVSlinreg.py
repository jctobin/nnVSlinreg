import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]


###Linear Regression with sklearn
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_Y_train)

"""
Takes less <0.1 seconds
In [46]: regr.coef_
Out[46]: array([ 938.23786125])

In [47]: regr.intercept_
Out[47]: 152.91886182616167
"""

###Linear Regression with keras
# Initialize Network
model = Sequential()
model.add(Dense(1, input_dim=1,init='uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='sgd')

model.fit(diabetes_X_train, diabetes_Y_train, nb_epoch=20000, batch_size=64,verbose=False)

"""
Took about 100 seconds on my chromebook without using multiple CPUs
In [68]: model.get_weights()
Out[68]: 
[array([[ 936.47363281]], dtype=float32),
 array([ 152.80149841], dtype=float32)]
"""

#Make lines and plot for both
w1,w0 = model.get_weights()
tt = np.linspace(np.min(diabetes_X[:, 0]), np.max(diabetes_X[:, 0]), 10)
nn_line = w0+w1*tt
lreg_line = regr.intercept_+regr.coef_*tt 

plt.plot(diabetes_X[:,0],diabetes['target'],'kx',tt,lreg_line,'r-',tt,nn_line[0],'b--')
plt.show()
