import numpy as np
from sklearn.datasets.california_housing import fetch_california_housing
from scipy.io import loadmat

def data_loader_cal_housing(i):

    cal_housing = fetch_california_housing()

    # normalize the features
    mu = np.mean(cal_housing.data.astype(np.float32), 0)
    sigma = np.std(cal_housing.data.astype(np.float32), 0)
    cal_housing.data = (cal_housing.data.astype(np.float32) - mu) / (sigma + 0.000001)


    intercept = np.ones((cal_housing.data.shape[0], 1))
    X = np.concatenate((cal_housing.data, intercept), axis=1)


    # randomly shuffle data with seed i
    np.random.seed(i)
    perm = np.random.permutation(len(cal_housing.target))


    return X[perm], cal_housing.target[perm]


def data_loader_abalone(i):
    x = []
    y = []
    with open("data/abalone.data", 'r') as f:
        lines = f.readlines()
        for sample in lines:
            sample = sample.strip().split(',')
            y.append(float(sample[-1]))
            sample[0] = 1 if sample[0] == 'M' else 0
            x.append(sample[:-1])
    x = np.array(x)
    y = np.array(y)

    # normalize the features
    mu = np.mean(x.astype(np.float32), 0)
    sigma = np.std(x.astype(np.float32), 0)
    x = (x.astype(np.float32) - mu) / (sigma + 0.000001)

    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((x, intercept), axis=1)

    # randomly shuffle data with seed i
    np.random.seed(i)
    perm = np.random.permutation(len(y))

    return x[perm], y[perm]

def data_loader_drug(i):
    x = loadmat('data/qsar.mat')
    x_train = x['X_train']
    x_test = x['X_test']

    y_train = x['y_train']
    y_test = x['y_test']

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).flatten()

    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((x, intercept), axis=1)

    np.random.seed(i)
    perm = np.random.permutation(len(y))

    print(len(x), len(x[0]))

    return  x[perm], y[perm]


