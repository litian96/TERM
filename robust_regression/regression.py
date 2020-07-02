import os, sys
import numpy as np
import argparse
import time
from scipy import stats

import cvxpy as cp

from create_datasets import data_loader_cal_housing, data_loader_abalone, data_loader_drug


def compute_gradients_l1(theta, X, y, gamma):  # L1 loss with a L2 regularizer (as in the paper)
    grad = (np.dot(X.T, np.sign(np.dot(X, theta) - y))) / y.size + gamma * theta
    return grad


def compute_gradients_huber(theta, X, y, delta, gamma):
    pred = np.dot(X, theta)
    ind = np.where(abs(pred-y) <= delta)[0]
    grad1 = np.dot(X[ind].T, pred[ind]-y[ind])
    ind_ = np.where(abs(pred-y) > delta)[0]
    grad2 = np.dot(delta * X[ind_].T, np.sign(pred[ind_] - y[ind_]))
    return (grad1 + grad2) / y.size + gamma * theta


def compute_gradients_tilting(theta, X, y, t):  # our objective
    loss = (np.dot(X, theta) - y) ** 2
    if t > 0:
        max_l = max(loss)
        loss = loss - max_l

    grad = (np.dot(np.multiply(np.exp(loss * t), (np.dot(X, theta) - y)).T, X)) / y.size
    ZZ = np.mean(np.exp(t * loss))
    return grad / (ZZ)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',
                        help='run how many times',
                        type=int,
                        default=5)
    parser.add_argument('--obj',
                        help='objective: l1, l2, huber, tilting',
                        type=str,
                        default='original')
    parser.add_argument('--t',
                        help='value of t for tilting',
                        type=float,
                        default=-2.0)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_iters',
                        help='how many iterations of gradient descent',
                        type=int,
                        default=10000)
    parser.add_argument('--corrupt',
                        help='whether crete outliers',
                        type=int,
                        default=0)
    parser.add_argument('--gamma',
                        help='regularization parameter',
                        type=float,
                        default=0.00000001)
    parser.add_argument('--delta',
                        help='threshold for huber loss',
                        type=float,
                        default=1.0)
    parser.add_argument('--dataset',
                        help="dataset",
                        type=str,
                        default='drug')
    parser.add_argument('--cvxpy',
                        help='if using cvxpy to solve the objective, or own implementation',
                        type=int,
                        default=0)
    parser.add_argument('--noise',
                        help='noise ratio',
                        type=float,
                        default=0.0)
    parser.add_argument('--oracle',
                        help='if invoke genie ERM',
                        type=int,
                        default=0)

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    obj = parsed['obj']
    t = parsed['t']
    num_iters = parsed['num_iters']
    corrupt = parsed['corrupt']
    lr = parsed['lr']
    gamma = parsed['gamma']
    delta = parsed['delta']
    dataset = parsed['dataset']
    cvxpy = parsed['cvxpy']
    bad_ratio=parsed['noise']
    oracle=parsed['oracle']
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

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

        train_X, train_y = all_data[:int(len(all_labels)*0.8)], \
                            all_labels[:int(len(all_labels)*0.8)]
        val_X, val_y = all_data[int(len(all_labels)*0.8):int(len(all_labels)*0.9)], \
                            all_labels[int(len(all_labels)*0.8):int(len(all_labels)*0.9)]
        test_X, test_y = all_data[int(len(all_labels) * 0.9):], \
                            all_labels[int(len(all_labels) * 0.9):]

        theta = np.zeros(len(train_X[0]))

        if oracle == 1:
            corrupt = 0
            train_X = train_X[int(len(train_y) * bad_ratio):]
            train_y = train_y[int(len(train_y) * bad_ratio):]
            print("len train_X: ", len(train_X))

        if corrupt == 1:
            train_y[:int(len(train_y)*bad_ratio)] = np.random.normal(5, 5, int(len(train_y) * bad_ratio))
            val_y[:int(len(val_y)*bad_ratio)] = np.random.normal(5, 5, int(len(val_y) * bad_ratio))


        if obj == 'l2':
            if cvxpy:
                print("use cvxpy")
                theta = cp.Variable(len(train_X[0]))
                objective = cp.Minimize(1.0/len(train_y) * cp.sum_squares(train_X * theta - train_y) + gamma * cp.sum_squares(theta))
                prob = cp.Problem(objective)
                # The optimal objective value is returned by `prob.solve()`.
                result = prob.solve()
                theta = theta.value
                print(theta)
            else:
                theta = np.dot(
                    np.dot(np.linalg.inv(np.dot(train_X.T, train_X) + gamma * np.identity(len(train_X[0]))), train_X.T),
                    train_y)
        else:
            if cvxpy:
                print("use cvxpy")
                if obj == 'l1':
                    theta = cp.Variable(len(train_X[0]))
                    objective = cp.Minimize(
                        1.0/len(train_y) * cp.norm(train_X * theta - train_y, 1) + gamma * cp.sum_squares(theta))
                    prob = cp.Problem(objective)
                    # The optimal objective value is returned by `prob.solve()`.
                    result = prob.solve()
                    theta = theta.value
                    print(theta)
                elif obj == 'huber':
                    theta = cp.Variable(len(train_X[0]))
                    objective = cp.Minimize(
                        1.0/len(train_y) * cp.sum(cp.huber(train_X * theta - train_y, delta)) + gamma * cp.sum_squares(theta))
                    prob = cp.Problem(objective)
                    result = prob.solve()
                    theta = theta.value
                    print(theta)
                elif obj == 'tilting':
                    theta = cp.Variable(len(train_X[0]))
                    objective = cp.Minimize(cp.sum(cp.exp(t * ((train_X @ theta - train_y) ** 2))))
                    prob = cp.Problem(objective)
                    result = prob.solve(verbose=True, max_iters=500)
                    theta = theta.value
            else:
                for j in range(num_iters):
                    if obj == 'l1':
                        grads_theta = compute_gradients_l1(theta, train_X, train_y, gamma)
                    elif obj == 'tilting':
                        grads_theta = compute_gradients_tilting(theta, train_X, train_y, t)
                    elif obj == 'huber':
                        grads_theta = compute_gradients_huber(theta, train_X, train_y, delta, gamma)

                    if np.linalg.norm(grads_theta, ord=2) < 1e-10:
                        break
                    theta = theta - lr * grads_theta
                    if j % 1000 == 0:
                        train_error = np.sqrt(np.mean((np.dot(train_X, theta) - train_y) ** 2))
                        loss = (np.dot(train_X, theta) - train_y) ** 2
                        print("training error: ", train_error)

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

