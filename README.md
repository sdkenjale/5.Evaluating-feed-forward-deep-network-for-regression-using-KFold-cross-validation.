# 5.Evaluating-feed-forward-deep-network-for-regression-using-KFold-cross-validation.
# A) Aim: Evaluating feed forward deep network for regression using KFold
# cross validation.
# Code:

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate sample data
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1)
# Define KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize list to store evaluation metrics
eval_metrics = []
# Iterate through each fold
for train_index, test_index in kfold.split(X):
	# Split data into training and testing sets
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	# Define and compile model
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=10))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])
	# Fit model to training data
	model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
	# Evaluate model on testing data
	eval_metrics.append(model.evaluate(X_test, y_test))

# Print average evaluation metrics across all folds
print("Average evaluation metrics:")
print("Loss:", np.mean([m[0] for m in eval_metrics]))
print("MAE:", np.mean([m[1] for m in eval_metrics]))

# B) Aim: Evaluating feed forward deep network for multiclass Classification
# using KFold cross-validation.
# Code:

import pandas
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# Import to_categorical directly from tensorflow.keras.utils
from tensorflow.keras.utils import to_categorical

# loading dataset
# Tell pandas to treat the first row as a header
df = pandas.read_csv('C:/Users/admin/Documents/block/pythonProject1/flowers.csv',header=0)
print(df)
# splitting dataset into input and output variables
# Adjust column indexing to start from 0
X = df.iloc[:, 0:4].astype(float)
y = df.iloc[:, 4]
# print(X)
# print(y)
# encoding string output into numeric output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print(encoded_y)
# Use to_categorical from tensorflow.keras.utils
dummy_Y = to_categorical(encoded_y)
print(dummy_Y)


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = baseline_model()
estimator.fit(X, dummy_Y, epochs=100, shuffle=True)
action = estimator.predict(X)
for i in range(25):
	print(dummy_Y[i])
	print('^^^^^^^^^^^^^^^^^^^^^^')
for i in range(25):
	print(action[i])
