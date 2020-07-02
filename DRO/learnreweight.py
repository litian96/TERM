import os, sys
import numpy as np
import argparse
import time
from scipy import stats

from simple_projections import project_onto_chi_square_ball
from create_datasets import data_loader_hiv


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

def compute_gradients_individual(theta, X, y):
    h = predict_prob(theta, X)
    gradient = np.multiply(X, np.transpose([h-y]))

    return gradient

def predict_prob(theta, X):
    return sigmoid(np.dot(X, theta))


def predict(theta, X):
    return predict_prob(theta, X).round()


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
    parser.add_argument('--t_in',
                        help='value of t for across sample tilting',
                        type=float,
                        default=1.0)
    parser.add_argument('--t_out',
                        help='value of t for across class tilting',
                        type=float,
                        default=1.0)
    parser.add_argument('--rho',
                        help='value of rho for the distributionally robust opt work',
                        type=float,
                        default=1)
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
    parser.add_argument('--num_corrupted',
                        help='how many data points are outliers/noisy',
                        type=int,
                        default=1500)
    parser.add_argument('--eval_interval',
                        help='eval every how many iterations (of SGD or GD)',
                        type=int,
                        default=1000)
    parser.add_argument('--imbalance',
                        help='whether to make it imbalanced',
                        type=int,
                        default=0)
    parser.add_argument('--gamma',
                        help='gamma for focal loss',
                        type=float,
                        default=0.2)
    parser.add_argument('--dataset',
                        help='dataset (hiv1 or adult)',
                        type=str,
                        default='adult')


    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    obj = parsed['obj']
    rho = parsed['rho']
    t_in = parsed['t_in']
    t_out = parsed['t_out']
    num_iters = parsed['num_iters']
    lr = parsed['lr']
    corrupt = parsed['corrupt']
    num_corrupted = parsed['num_corrupted']
    interval = parsed['eval_interval']
    imbalance = parsed['imbalance']
    gamma = parsed['gamma']
    dataset = parsed['dataset']
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    train_rare = []
    test_rare = []
    train_common = []
    test_common = []

    begin = time.time()

    for i in range(num_trials):
        all_data, all_labels = data_loader_hiv("data/hiv1/raw", i)  # have been randomly shuffled using seed i
        print("{} total data points, {} training samples, {}% rare".format(len(all_labels), int(0.8*(len(all_labels))),
                                                                          len(np.where(all_labels==1)[0])/len(all_labels)))

        train_X, train_y = all_data[:int(len(all_labels)*0.8)], all_labels[:int(len(all_labels)*0.8)]
        val_X, val_y = all_data[int(len(all_labels) * 0.8):int(len(all_labels) * 0.9)], all_labels[
                                                                                        int(len(all_labels) * 0.8):int(
                                                                                            len(all_labels) * 0.9)]
        test_X, test_y = all_data[int(len(all_labels) * 0.9):], all_labels[int(len(all_labels) * 0.9):]

        # 10 samples for each class
        rare_keep_ind = np.where(val_y == 1)[0][:10]
        common_keep_ind = np.where(val_y == 0)[0][:10]
        val_X, val_y = np.concatenate((val_X[common_keep_ind], val_X[rare_keep_ind]), axis=0), \
                   np.concatenate((val_y[common_keep_ind], val_y[rare_keep_ind]), axis=0)


        theta = np.zeros(len(train_X[0]))

        for j in range(num_iters):
            grads_train_individual = compute_gradients_individual(theta, train_X, train_y)
            grads_train = np.mean(grads_train_individual, axis=0)
            grads_val = compute_gradients_vanilla(theta - lr * grads_train, val_X, val_y)
            gradients_w = -1 * np.dot(grads_train_individual, grads_val)
            w = np.maximum(np.ones(len(train_X))/len(train_X)-0.001*gradients_w, 0)
            w = w / np.sum(w)
            theta = theta - lr * np.average(grads_train_individual, axis=0, weights=w)

            if j % interval == 0:
                if corrupt == 0:
                    rare_sample_train = np.where(train_y == 1)[0]
                    preds_rare = predict(theta, train_X)[rare_sample_train]
                    train_accuracy = (preds_rare == train_y[rare_sample_train]).mean()
                    print("rare training accuracy: ", train_accuracy)
                else:
                    preds_rare = predict(theta, train_X)
                    train_accuracy = (preds_rare == train_y).mean()
                    print("training accuracy: ", train_accuracy)


        preds_train = predict(theta, train_X)
        preds_test = predict(theta, test_X)
        preds_val = predict(theta, val_X)

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

        print("trial {}, training accuracy on the rare class {}".format(i, rare_train_accuracy))
        print("trial {}, testing accuracy on the rare class {}".format(i, rare_test_accuracy))

        print("trial {}, training accuracy on the common class {}".format(i, common_train_accuracy))
        print("trial {}, testing accuracy on the common class {}".format(i, common_test_accuracy))

        print("trial {}, overall training accuracy {}".format(i, train_accuracy))
        print("trial {}, overall testing accuracy {}".format(i, test_accuracy))


    print("avg training accuracy on the rare class {}".format(np.mean(np.array(train_rare))))
    print("se training accuracy on the rare class {}".format(stats.sem(np.array(train_rare))))
    print("avg testing accuracy on the rare class {}".format(np.mean(np.array(test_rare))))
    print("se testing accuracy on the rare class {}".format(stats.sem(np.array(test_rare))))

    print("avg training accuracy on the common class {}".format(np.mean(np.array(train_common))))
    print("se training accuracy on the common class {}".format(stats.sem(np.array(train_common))))
    print("avg testing accuracy on the common class {}".format(np.mean(np.array(test_common))))
    print("se testing accuracy on the common class {}".format(stats.sem(np.array(test_common))))

    print("avg overall training accuracy {}".format(np.mean(np.array(train_accuracies))))
    print("se overall training accuracy {}".format(stats.sem(train_accuracies)))
    print("avg overall testing accuracy {}".format(np.mean(np.array(test_accuracies))))
    print("se overall testing accuracy {}".format(stats.sem(test_accuracies)))

    print("avg overall validation accuracy {}".format(np.mean(np.array(val_accuracies))))


if __name__ == '__main__':
    main()


