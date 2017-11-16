import pandas as pd
import numpy as np
from itertools import combinations
from random import randrange
import matplotlib.pyplot as plt
import scipy.special as sps


# Please change the location of the input file
hockey_data = pd.read_csv(filepath_or_buffer="D:/ml assignment/Linear Regression/preprocessed_datasets.csv")

# Data Preprocessing
hockey_data = hockey_data.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'Country', 'Overall', 'sum_7yr_GP'], axis=1)

train = hockey_data[hockey_data.DraftYear.isin(['2004', '2005', '2006'])]
test = hockey_data[hockey_data.DraftYear.isin(['2007'])]

del train['DraftYear']
del test['DraftYear']

train_target = train['GP_greater_than_0']
del train['GP_greater_than_0']
test_target = test['GP_greater_than_0']
del test['GP_greater_than_0']
train = pd.get_dummies(train, columns=['country_group', 'Position'])
test = pd.get_dummies(test, columns=['country_group', 'Position'])

discrete_columns = ['country_group', 'Position','country_group_CAN', 'country_group_EURO',
       'country_group_USA', 'Position_C', 'Position_D', 'Position_L',
       'Position_R']

# Standardizing the values of the data frame
for i in train.columns:
    if i not in discrete_columns:
        mean = np.mean(train[i])
        std = np.std(train[i])
        if std == 0:
            train.drop(i, axis=1, inplace=True)
        else:
            train[i] = (train[i] - mean) / std
    train['X'] = 1

for i in test.columns:
    if i not in discrete_columns:
        mean = np.mean(test[i])
        std = np.std(test[i])
        if std == 0:
            test.drop(i, axis=1, inplace=True)
        else:
            test[i] = (test[i] - mean) / std
    test['X'] = 1
    test_col = test.columns.tolist()
    test_col = test_col[-1:] + test_col[:-1]
    test = test[test_col]

#Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

train = np.random.permutation(train)

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
#etas = [0.1, 0.05, 0.01]

n_train = train_target.size
# Error values over all iterations.
all_errors = dict()

for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    e_all = []

    for iter in range(0, max_iter):
        for n in range(0, n_train):
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(train[n, :], w))

            # Gradient of the error, using Assignment result
            grad_e = ((y - train_target[n]) * train[n, :])/n

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w = w - grad_e*eta

        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(train, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(train_target, np.log(y)) + np.multiply((1 - train_target), np.log(1 - y)))
        e_all.append(e)

        # Print some information.
        print('eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w.T))

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    all_errors[eta] = e_all

# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')
plt.axis([0, max_iter, 0.2, 0.7])
plt.legend()
plt.show()
