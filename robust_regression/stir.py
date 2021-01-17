import os, sys
import numpy as np
import argparse
from scipy import stats

from create_datasets import data_loader_cal_housing, data_loader_abalone, data_loader_drug


#### code for Consistent Robust Regression https://papers.nips.cc/paper/6806-consistent-robust-regression.pdf ####

def HT(v, k):
    # only keep k largest elements (k corrupted samples)
    not_top_k_index = (abs(v)).argsort()[:len(v)-k]
    v[not_top_k_index] = 0
    return v

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',
                        help='run how many times',
                        type=int,
                        default=5)
    parser.add_argument('--dataset',
                        help="abalone or cal_housing dataset or drug",
                        type=str,
                        default='drug')
    parser.add_argument('--obj',
                        help='objective: crr',
                        type=str,
                        default='crr')
    parser.add_argument('--corrupt',
                        help='whether crete outliers',
                        type=int,
                        default=1)
    parser.add_argument('--iters',
                        help='increase M for how many times',
                        type=int,
                        default=100)
    parser.add_argument('--M',
                        help='initial truncation parameter',
                        type=float,
                        default=0.001)
    parser.add_argument('--eta',
                        help='truncation parameter increment by eta times each iteration',
                        type=float,
                        default=10)
    parser.add_argument('--noise',
                        help='noise ratio',
                        type=float,
                        default=0.2)

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    obj = parsed['obj']
    M = parsed['M']
    corrupt = parsed['corrupt']
    eta = parsed['eta']
    dataset = parsed['dataset']
    bad_ratio=parsed['noise']
    iters=parsed['iters']
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    train_errors = []
    test_errors = []
    val_errors = []

    for i in range(num_trials):
        if dataset == 'cal_housing':
            all_data, all_labels = data_loader_cal_housing(i)  # have been randomly shuffled using seed i
        elif dataset == 'abalone':
            all_data, all_labels = data_loader_abalone(i)
        elif dataset == 'drug':
            all_data, all_labels = data_loader_drug(i)
        print("{} total data points".format(len(all_labels)))

        train_X, train_y = all_data[:int(len(all_labels)*0.8)], all_labels[:int(len(all_labels)*0.8)]
        val_X, val_y = all_data[int(len(all_labels)*0.8):int(len(all_labels)*0.9)], all_labels[int(len(all_labels)*0.8):int(len(all_labels)*0.9)]
        test_X, test_y = all_data[int(len(all_labels)*0.9):], all_labels[int(len(all_labels)*0.9):]

        if corrupt == 1:
            train_y[:int(len(train_y)*bad_ratio)] = np.random.normal(5, 5, int(len(train_y)*bad_ratio))
            val_y[:int(len(val_y)*bad_ratio)] = np.random.normal(5, 5, int(len(val_y)*bad_ratio))

        theta = np.zeros(len(train_X[0]))
        theta_old = np.ones(len(train_X[0])) * 100
        M = parsed['M']
        for it in range(iters):
            while np.linalg.norm(theta_old-theta, ord=2) > 2 / (eta * M):
                theta_old = theta
                residual = np.dot(train_X, theta) - train_y
                s = np.minimum(1.0/abs(residual), np.ones(len(train_y)) * M)
                SS = np.diag(s)
                tmp = np.dot(train_X.T, SS)
                theta = np.dot(
                    np.dot(np.linalg.inv(np.dot(tmp, train_X) + 1e-20*np.identity(len(train_X[0]))), tmp),
                    train_y)
                # theta = theta - (0.2 / (M * len(train_y))) * np.dot(np.dot(train_X.T, SS), residual)
            M = M * eta
            print(M)
            train_error = np.sqrt(np.mean((np.dot(train_X, theta) - train_y) ** 2))
            print("iter {}, training error {} ".format(it, train_error))
            test_error = np.sqrt(np.mean((np.dot(test_X, theta) - test_y) ** 2))
            print("iter {}, test error {} ".format(it, test_error))
            val_error = np.sqrt(np.mean((np.dot(val_X, theta) - val_y) ** 2))
            print("iter {}, val error {} ".format(it, val_error))

        train_error = np.sqrt(np.mean((np.dot(train_X, theta) - train_y) ** 2))
        train_errors.append(train_error)
        print("trial {}, training error {} ".format(i, train_error))
        test_error = np.sqrt(np.mean((np.dot(test_X, theta) - test_y) ** 2))
        test_errors.append(test_error)
        print("trial {}, test error {} ".format(i, test_error))
        val_error = np.sqrt(np.mean((np.dot(val_X, theta) - val_y) ** 2))
        val_errors.append(val_error)
        print("trial {}, val error {} ".format(i, val_error))


    print("avg train error {}".format(np.mean(np.array(train_errors))))
    print("se train error {}".format(stats.sem(np.array(train_errors))))
    print("avg test error {}".format(np.mean(np.array(test_errors))))
    print("se test error {}".format(stats.sem(np.array(test_errors))))
    print("avg val error {}".format(np.mean(np.array(val_errors))))
    print("se val error {}".format(stats.sem(np.array(val_errors))))


if __name__ == '__main__':
    main()




