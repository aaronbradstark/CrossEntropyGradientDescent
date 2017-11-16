import pandas as pd
import numpy as np
from itertools import combinations
from random import randrange
import matplotlib.pyplot as plt

# Please change the location of the input file
hockey_data = pd.read_csv(filepath_or_buffer="D:/ml assignment/Linear Regression/preprocessed_datasets.csv")

# Data Preprocessing
hockey_data = hockey_data.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'Country', 'Overall', 'GP_greater_than_0'], axis=1)

train = hockey_data[hockey_data.DraftYear.isin(['2004', '2005', '2006'])]
test = hockey_data[hockey_data.DraftYear.isin(['2007'])]

del train['DraftYear']
del test['DraftYear']

sum_7yr_GP_train = train['sum_7yr_GP']
del train['sum_7yr_GP']
sum_7yr_GP_test = test['sum_7yr_GP']
del test['sum_7yr_GP']
train = pd.get_dummies(train, columns=['country_group', 'Position'])
test = pd.get_dummies(test, columns=['country_group', 'Position'])


# Quadratic Interaction Terms
def add_interaction_terms():
    for c1, c2 in combinations(train.columns, 2):
        train['{0}*{1}'.format(c1, c2)] = train[c1] * train[c2]

    for c1, c2 in combinations(test.columns, 2):
        test['{0}*{1}'.format(c1, c2)] = test[c1] * test[c2]


add_interaction_terms()
# Standardizing the values of the data frame
for i in train.columns:
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


def calculate_weight(lam, X, Y):
    identity = np.identity(245)
    inverse = np.linalg.pinv((lam * identity) + np.matmul(X.T, X))
    y_val = np.matmul(X.T, Y)
    weight = np.matmul(inverse, y_val)
    return weight


def calculate_squared_error(weight, data, data_target):
    weight_train = np.matmul(data, weight)
    error = np.subtract(weight_train, data_target)
    square_error = np.matmul(error.T, error)
    sum_sqr_error = np.sum(square_error) / 2
    return sum_sqr_error


# Creating Cross Validation Sets

def perform_cross_validation():
    folds = round(train.shape[0] / 10)
    lambda_frame = pd.DataFrame()
    for i in range(10):
        err_list = []
        counter = folds * i
        counter_2 = counter + folds
        test_target = sum_7yr_GP_train[counter:counter_2]
        train_target = sum_7yr_GP_train[~sum_7yr_GP_train.index.isin(test_target.index)]
        test_tmp = train[counter:counter_2]
        train_tmp = train[~train.index.isin(test_tmp.index)]
        for lam in lambda_values:
            weight = calculate_weight(lam, train_tmp, train_target)
            squared_error = calculate_squared_error(weight, test_tmp, test_target)
            err_list += [squared_error]
        lambda_frame[lambda_frame.shape[1]] = err_list
    learning_list = lambda_frame.T
    error_list = []
    return learning_list, error_list


def surf_learning_list(learning_list, error_list):
    for i in range(learning_list.shape[1]):
        error_list += [np.mean(learning_list[i])]
    mean_frame = pd.DataFrame([error_list])
    learning_frame = pd.concat([learning_list, mean_frame], ignore_index=True)

    position = learning_frame.shape[0]
    learning_col = learning_frame.iloc[position - 1].idxmin(axis=1)
    min_error = learning_frame.iloc[position - 1][learning_col]
    best_lambda = lambda_values[learning_col]
    return error_list, min_error, best_lambda


def calculate_test_error(error_list, min_error, best_lambda):
    test_error = []

    for i in lambda_values:
        test_weight = calculate_weight(i, train, sum_7yr_GP_train)
        test_err_val = calculate_squared_error(test_weight, test, sum_7yr_GP_test)
        test_error += [test_err_val]

    test_lowest_error = min(test_error)
    best_test_lambda = lambda_values[test_error.index(test_lowest_error)]
    validationErrorSet = error_list
    plot_graphs(validationErrorSet, test_error, best_lambda, min_error, test_lowest_error, best_test_lambda)
    print(validationErrorSet)
    print(test_error)


def plot_graphs(validationErrorSet, test_error, best_lambda, min_error, test_lowest_error, best_test_lambda):
    learning_parameter = "Best Lambda Value for Validation Set: " + str(best_lambda)
    error = "Error at Best Lambda Value " + str(min_error)
    print(learning_parameter, error)
    plt.semilogx(lambda_values, validationErrorSet, '-x', label='Validation error')
    plt.semilogx(lambda_values, test_error, '-x', label='Test error')
    plt.semilogx(best_lambda, min_error, marker='o', color='r', label="Best Labmda")
    plt.semilogx(best_test_lambda, test_lowest_error, marker='o', color='g', label="Best Test Labmda")
    plt.ylabel('Sum Squared Error')
    plt.legend()
    plt.xlabel('Lambda')
    plt.show()


def perform_linear_regression():
    learning_list, error_list = perform_cross_validation()
    error_list, min_error, best_lambda = surf_learning_list(learning_list, error_list)
    calculate_test_error(error_list, min_error, best_lambda)


lambda_values = [0, 0.01, 0.1, 1, 10, 100, 1000]
perform_linear_regression()
