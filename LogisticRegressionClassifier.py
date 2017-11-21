import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import pandas as pd

hockey_data = pd.read_csv('D:/ml assignment/Linear Regression/preprocessed_datasets.csv')

# Data Preprocessing
hockey_data.drop(['id','PlayerName','Country','sum_7yr_TOI','Overall','sum_7yr_GP',],axis = 1,inplace = True)

# Train Data Processing
train = hockey_data[hockey_data['DraftYear'].isin([2004,2005,2006])]
target_encode ={'yes':1,'no':0}
train['GP_greater_than_0'] = train['GP_greater_than_0'].map(target_encode)
train.drop(['DraftYear'],axis = 1,inplace = True)
train = pd.get_dummies(train, columns= ['country_group','Position'])

for i in train.columns:
    if i not in ['GP_greater_than_0', 'country_group_CAN', 'country_group_EURO', 'country_group_USA', 'Position_C',
                 'Position_D', 'Position_L', 'Position_R']:
        mean = np.mean(train[i])
        sd = np.std(train[i])
        if sd == 0:
            train.drop(i, axis=1, inplace=True)
        else:
            train[i] = (train[i] - mean) / sd
train['X'] = 1
cols = train.columns.tolist()
train = train[[cols[-1]] + cols[:-1]]

# Test Data Processing
test = hockey_data[hockey_data['DraftYear'] == 2007]
test['GP_greater_than_0'] = test['GP_greater_than_0'].map(target_encode)
test.drop(['DraftYear'],axis = 1,inplace = True)
test = pd.get_dummies(test, columns= ['country_group','Position'])

for i in test.columns:
    if i not in ['GP_greater_than_0', 'country_group_CAN', 'country_group_EURO', 'country_group_USA', 'Position_C',
                 'Position_D', 'Position_L', 'Position_R']:
        mean = np.mean(test[i])
        sd = np.std(test[i])
        if sd == 0:
            test.drop(i, axis=1, inplace=True)
        else:
            test[i] = (test[i] - mean) / sd
test['X'] = 1
cols = test.columns.tolist()
df = test[[cols[-1]] + cols[:-1]]
test_target = test['GP_greater_than_0']
test_target = test_target.values

# Train Feature Engineering
train = train.values
test = test.values
test = np.delete(test,16,axis = 1)
max_iter = 300
tol = 0.00001  # difference in loss

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
np.random.seed(650)
data = np.random.permutation(train) # get random perm here)  # get random perm here

#Data matrix, with column of ones at end.
X = data # all rows, col 0 to 3 here (except target col)
X = np.delete(X,16,axis = 1)

# Target values, 0 for class 1, 1 for class 2.
t = data[:,16]
n_train = t.size

# Error values over all iterations.
all_errors = dict()
for eta in etas:  # for each @ for set of @s

    # Initialize w.
    w = np.zeros(22)
    w = np.insert(w,0,0.1,axis =0)
    e_all = []

    for iter in range(0, max_iter):  # 0 - 500
        for n in range(0, n_train):  # size of the training data (rows)
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(X[n, :], w))  # [a,b] is (a row, b col )

            # Gradient of the error, using Assignment result
            grad_e = ((y - t[n]) * X[n, :])/n_train

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

# Estimating Accuracy for Test Data
correct_predictions = 0
rowcount = test.shape[0]
weight = w.T
predicted_y = sps.expit(np.dot(test,weight))
index = 0
for i in np.nditer(predicted_y):
    if i > 0.5:
        val = 1
    else:
        val = 0
    if val == test_target.item(index):
        correct_predictions = correct_predictions+1
        index = index+1

print("Negative Log Likelihood :" + str(e))
print("Accuracy : \t" + str((correct_predictions / rowcount) * 100) + '%')
print("No of Correct Predictions : \t" + str(correct_predictions))
print("No of Incorrect predictions : \t" + str(rowcount - correct_predictions))

# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))

plt.ylabel('Negative log-likelihood Value')
plt.title('Training Logistic Regression - SGD')
plt.xlabel('Epoch Values')
plt.axis([0, max_iter, 0.58, 0.7])
plt.legend()
plt.show()