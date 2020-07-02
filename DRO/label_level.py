import os, sys
import numpy as np
import argparse
import time
from scipy import stats

from simple_projections import project_onto_chi_square_ball
from create_datasets import data_loader


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((X, intercept), axis=1)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def loss(h, y):
    return -y * np.log(h) - (1 - y) * np.log(1 - h)


def compute_gradients_vanilla(theta, X, y):  # the vanilla
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, h-y) / y.size
    return gradient


# minimax / distributionally robust opt
# w is the weight vector (p_i in paper) of each sample
def compute_gradients_dro(theta, X, y, w):
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, np.multiply(w, h - y))  # because of w, no need to divided by y.size
    return gradient


def compute_gradients_tilting(theta, X_1, y_1, X_2, y_2, t):  # our objective

    h_1 = predict_prob(theta, X_1)
    h_2 = predict_prob(theta, X_2)
    gradient1 = np.dot(X_1.T, h_1 - y_1)
    gradient2 = np.dot(X_2.T, h_2 - y_2)
    l_1 = np.mean(loss(h_1, y_1))
    l_2 = np.mean(loss(h_2, y_2))
    l_max = max(l_1, l_2)

    gradient = (np.exp((l_1-l_max) * t) * gradient1 + np.exp((l_2-l_max) * t) * gradient2) / (y_1.size+y_2.size)
    ZZ = (np.exp(t * (l_1-l_max)) + np.exp(t * (l_2-l_max)))/2
    return gradient / ZZ


def predict_prob(theta, X):
    return sigmoid(np.dot(X, theta))


def predict(theta, X, threshold):
    # advanced ERM
    res = predict_prob(theta, X)
    return (res - threshold + 0.5).round()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',
                        help='run how many times',
                        type=int,
                        default=5)
    parser.add_argument('--obj',
                        help='objective: original, minimax, tilting',
                        type=str,
                        default='original')
    parser.add_argument('--t',
                        help='value of t for tilting',
                        type=float,
                        default=1.0)
    parser.add_argument('--rho',
                        help='value of rho for minimax (distributionally robust opt work)',
                        type=float,
                        default=1)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_iters',
                        help='how many iterations of gradient descent',
                        type=int,
                        default=20000)
    parser.add_argument('--corrupt',
                        help='whether create outliers',
                        type=int,
                        default=0)
    parser.add_argument('--eval_interval',
                        help='eval every how many iterations (of SGD or GD)',
                        type=int,
                        default=1000)
    parser.add_argument('--threshold',
                        help='decision boundary for ERM',
                        type=float,
                        default=0.5)

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    obj = parsed['obj']
    rho = parsed['rho']
    t = parsed['t']
    num_iters = parsed['num_iters']
    lr = parsed['lr']
    corrupt = parsed['corrupt']
    interval = parsed['eval_interval']
    threshold = parsed['threshold']
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    train_rare = []
    test_rare = []
    train_common = []
    test_common = []

    begin = time.time()

    for i in range(num_trials):
        all_data, all_labels = data_loader("data/hiv1/raw", i)  # have been randomly shuffled using seed i
        print("{} total data points".format(len(all_labels)))
        train_X, train_y = all_data[:int(len(all_labels)*0.8)], all_labels[:int(len(all_labels)*0.8)]
        if corrupt == 1:
            np.random.seed(i)
            train_y[:2000] = 1
        val_X, val_y = all_data[int(len(all_labels) * 0.8):int(len(all_labels) * 0.9)], all_labels[int(len(all_labels) * 0.8):int(len(all_labels) * 0.9)]
        test_X, test_y = all_data[int(len(all_labels)*0.9):], all_labels[int(len(all_labels)*0.9):]
        if corrupt == 1:
            np.random.seed(i)
            val_y[:125] = 1
        theta = np.zeros(20 * 8)

        class1 = np.where(train_y == 1)[0]
        class2 = np.where(train_y == 0)[0]
        X_1, y_1 = train_X[class1], train_y[class1]
        X_2, y_2 = train_X[class2], train_y[class2]

        for j in range(num_iters):
            if obj == 'minimax':
                h = predict_prob(theta, train_X)
                loss_vector = loss(h, train_y)
                p = project_onto_chi_square_ball(loss_vector, rho)
                grads_theta = compute_gradients_dro(theta, train_X, train_y, p)
            elif obj == 'tilting':
                grads_theta = compute_gradients_tilting(theta, X_1, y_1, X_2, y_2, t)
            else:
                grads_theta = compute_gradients_vanilla(theta, train_X, train_y)
            if np.linalg.norm(grads_theta, ord=2) < 1e-60:
                break

            theta = theta - lr * grads_theta
            if j % interval == 0:
                #rare_sample_train = np.where(train_y == 1)[0]
                preds_rare = predict(theta, train_X, threshold)
                train_accuracy = (preds_rare == train_y).mean()
                print("training accuracy: ", train_accuracy)

        preds_train = predict(theta, train_X, threshold)
        preds_test = predict(theta, test_X, threshold)
        preds_val = predict(theta, val_X, threshold)

        train_accuracy = (preds_train == train_y).mean()
        train_accuracies.append(train_accuracy)

        test_accuracy = (preds_test == test_y).mean()
        test_accuracies.append(test_accuracy)

        val_accuracy = (preds_val == val_y).mean()
        val_accuracies.append(val_accuracy)

        rare_sample_train = np.where(train_y == 1)[0]
        preds_rare = preds_train[rare_sample_train]
        rare_train_accuracy = (preds_rare == train_y[rare_sample_train]).mean()
        train_rare.append(rare_train_accuracy)

        rare_sample_test = np.where(test_y == 1)[0]
        preds_rare = preds_test[rare_sample_test]
        rare_test_accuracy = (preds_rare == test_y[rare_sample_test]).mean()
        test_rare.append(rare_test_accuracy)

        common_sample_train = np.where(train_y == 0)[0]
        preds_common = preds_train[common_sample_train]
        common_train_accuracy = (preds_common == 0).mean()
        train_common.append(common_train_accuracy)

        common_sample_test = np.where(test_y == 0)[0]
        preds_common = preds_test[common_sample_test]
        common_test_accuracy = (preds_common == 0).mean()
        test_common.append(common_test_accuracy)

        h = predict_prob(theta, train_X)
        loss_vector_train = loss(h, train_y)

        h = predict_prob(theta, test_X)
        loss_vector_test = loss(h, test_y)

        print("trial {}, training accuracy on the rare class {}".format(i, rare_train_accuracy))
        print("trial {}, testing accuracy on the rare class {}".format(i, rare_test_accuracy))

        print("trial {}, training accuracy on the common class {}".format(i, common_train_accuracy))
        print("trial {}, testing accuracy on the common class {}".format(i, common_test_accuracy))

        print("trial {}, overall training accuracy {}".format(i, train_accuracy))
        print("trial {}, overall testing accuracy {}".format(i, test_accuracy))
        print("trial {}, overall validation accuracy {}".format(i, val_accuracy))

    end = time.time()
    print("avg time (s): {}".format((end-begin)/num_trials))
    print("avg training accuracy on the rare class {}".format(np.mean(np.array(train_rare))))
    print("std training accuracy on the rare class {}".format(np.std(np.array(train_rare))))
    print("avg testing accuracy on the rare class {}".format(np.mean(np.array(test_rare))))
    print("std testing accuracy on the rare class {}".format(np.std(np.array(test_rare))))

    print("avg training accuracy on the common class {}".format(np.mean(np.array(train_common))))
    print("std training accuracy on the common class {}".format(np.std(np.array(train_common))))
    print("avg testing accuracy on the common class {}".format(np.mean(np.array(test_common))))
    print("std testing accuracy on the common class {}".format(np.std(np.array(test_common))))

    print("avg overall training accuracy {}".format(np.mean(np.array(train_accuracies))))
    print("std overall training accuracy {}".format(np.std(np.array(train_accuracies))))
    print("avg overall testing accuracy {}".format(np.mean(np.array(test_accuracies))))
    print("std overall testing accuracy {}".format(np.std(np.array(test_accuracies))))
    print("avg overall validation accuracy {}".format(np.mean(np.array(val_accuracies))))


if __name__ == '__main__':
    main()