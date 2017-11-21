import pandas as pd
import numpy as np
import scipy as sp
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

train = train.sample(frac = 1)
test = test.sample(frac = 1)

train = pd.get_dummies(train, columns=['country_group', 'Position'])
test = pd.get_dummies(test, columns=['country_group', 'Position'])



discrete_columns = ['country_group', 'Position','country_group_CAN', 'country_group_EURO',
       'country_group_USA', 'Position_C', 'Position_D', 'Position_L',
       'Position_R','GP_greater_than_0']


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
    train_col = train.columns.tolist()
    train_col = train_col[-1:] + train_col[:-1]
    train = train[train_col]

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

target_encode ={'yes':1,'no':0}
train['GP_greater_than_0'] = train['GP_greater_than_0'].map(target_encode)
test['GP_greater_than_0'] = test['GP_greater_than_0'].map(target_encode)

test_target = test['GP_greater_than_0']
Test_target = test_target.values

train = train.values
train = np.delete(train, np.s_[1,4], 1)

test = test.values
test_val = test[:,15]
test = np.delete(test, np.s_[1,4,15],1)


np.random.seed(650)
data = np.random.permutation(train) # get random perm here)  # get random perm here

#Data matrix, with column of ones at end.
X = data # all rows, col 0 to 3 here (except target col)
X = np.delete(X,15,axis = 1)

# Target values, 0 for class 1, 1 for class 2.
t = data[:,15]


# #Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 300
tol = 0.00001

#train = np.random.permutation(train)

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
#etas = [0.1, 0.05, 0.01]

n_train = t.size
# Error values over all iterations.
all_errors = dict()

for eta in etas:  # for each @ for set of @s

    # Initialize w.
    w = np.zeros(20)
    w = np.insert(w,0,0.1,axis =0)
    e_all = []

    for iter in range(0, max_iter):  # 0 - 500
        for n in range(0, n_train):  # size of the training data (rows)
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(X[n, :], w))  # [a,b] is (a row, b col )

            # Gradient of the error, using Assignment result
            grad_e = (y - t[n]) * X[n, :] / n_train

            w = w - grad_e * eta

        y = sps.expit(np.dot(X, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
        e_all.append(e)

        # Print some information.
        print('eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w.T))

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    all_errors[eta] = e_all

correct_prediction = 0
rowcount = test.shape[0]

w_opt = w.T
counter = sps.expit(np.matmul(test,w_opt))
index = 0
for i in np.nditer(counter):
    if i > 0.5:
        val = 1
    else:
        val = 0
    if val == test_val.item(index):
        correct_prediction = correct_prediction+1
        index = index+1

print("Accuracy : \t" + str((correct_prediction / rowcount) * 100) + '%')
print("no of incorrect predictions : \t" + str(rowcount - correct_prediction))
# Plot error over iterations for all etas
# plt.figure(10)
# plt.rcParams.update({'font.size': 15})
# for eta in sorted(all_errors):
#     plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))
#
# plt.ylabel('Negative log likelihood')
# plt.title('Training logistic regression with SGD')
# plt.xlabel('Epoch')
# plt.axis([0, max_iter, 0.2, 0.7])
# plt.legend()
# plt.show()
