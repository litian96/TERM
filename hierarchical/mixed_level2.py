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
    gradient = np.dot(X.T, np.multiply(w, h - y))
    return gradient


def compute_gradients_tilting(theta, Xs, ys, t_in, t_out, num_corrupted):

    def compute_gradients_inner_tilting(theta, X, y, t):
        # inside each annotator
        class1 = np.where(y == 1)[0]
        class2 = np.where(y == 0)[0]
        X_1, y_1 = X[class1], y[class1]
        X_2, y_2 = X[class2], y[class2]
        h_1 = predict_prob(theta, X_1)
        h_2 = predict_prob(theta, X_2)
        gradient1 = np.dot(X_1.T, h_1 - y_1)
        gradient2 = np.dot(X_2.T, h_2 - y_2)
        l_1 = np.mean(loss(h_1, y_1))
        l_2 = np.mean(loss(h_2, y_2))
        l_max = max(l_1, l_2)
        gradient = (np.exp((l_1 - l_max) * t) * gradient1 + np.exp((l_2 - l_max) * t) * gradient2)
        ZZ = len(y_1) * np.exp(t * (l_1 - l_max)) + len(y_2) * np.exp(t * (l_2 - l_max))
        return gradient / ZZ, np.array([len(y_1) * np.exp(t * l_1), len(y_2) * np.exp(t * l_2)])


    def compute_inner_obj(t_in, l, num_sample):
        return 1/t_in * np.log(1/num_sample * (l[0] + l[1]))


    gradients = []
    inner_objs = []
    for i in range(len(Xs)):
        g, l = compute_gradients_inner_tilting(theta, Xs[i], ys[i], t_in)
        gradients.append(g)
        inner_objs.append(compute_inner_obj(t_in, l, len(ys[i])))

    l_max = max(inner_objs)
    total_samples = ys[0].size
    final_g = np.exp((inner_objs[0]-l_max) * t_out) * gradients[0] * ys[0].size
    ZZ = np.exp(t_out * (inner_objs[0] - l_max)) * ys[0].size
    for i in range(1, len(ys)):
        final_g += np.exp((inner_objs[i]-l_max) * t_out) * gradients[i] * ys[i].size
        total_samples += ys[i].size
        ZZ += np.exp(t_out * (inner_objs[i] - l_max)) * ys[i].size

    return final_g / ZZ


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
                        help='value of rho for minimax (distributionally robust opt work)',
                        type=float,
                        default=1)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.5)
    parser.add_argument('--num_iters',
                        help='how many iterations of gradient descent',
                        type=int,
                        default=15000)
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

    for i in range(num_trials):
        all_data, all_labels = data_loader("data/hiv1/raw", i)  # have been randomly shuffled using seed i
        print("{} total data points".format(len(all_labels)))
        train_X, train_y = all_data[:int(len(all_labels)*0.8)], all_labels[:int(len(all_labels)*0.8)]
        val_X, val_y = all_data[int(len(all_labels) * 0.8):int(len(all_labels) * 0.9)], all_labels[
                                                                                        int(len(all_labels) * 0.8):int(
                                                                                            len(all_labels) * 0.9)]
        test_X, test_y = all_data[int(len(all_labels) * 0.9):], all_labels[int(len(all_labels) * 0.9):]

        if imbalance == 1:
            len_common = len(np.where(train_y == 0)[0])
            len_rare = int(len_common * 0.05)
            rare_keep_ind = np.where(train_y == 1)[0][:len_rare]
            common_keep_ind = np.where(train_y == 0)[0]
            train_X, train_y = np.concatenate((train_X[common_keep_ind], train_X[rare_keep_ind]), axis=0), \
                   np.concatenate((train_y[common_keep_ind], train_y[rare_keep_ind]), axis=0)

            np.random.seed(123456 + i)
            perm = np.random.permutation(len(train_y))
            train_X, train_y = train_X[perm], train_y[perm]


        if corrupt == 1:
            np.random.seed(i)
            print(num_corrupted, "corrupted")
            num_common = len(np.where(train_y == 0)[0])
            num_rare = len(np.where(train_y == 1)[0])
            train_y[:num_corrupted] = np.random.choice([0, 1], num_corrupted,
                                                       p=[num_common / len(train_y), num_rare / len(train_y)])


        theta = np.zeros(len(train_X[0]))

        Xs = []
        ys = []

        for anno in range(3):
            Xs.append(train_X[int(num_corrupted/3)*anno : int(num_corrupted/3)*(anno+1)])
            ys.append(train_y[int(num_corrupted/3)*anno : int(num_corrupted/3)*(anno+1)])
        each_hw = int((len(train_X) - num_corrupted) / 7)
        for anno in range(7):
            Xs.append(train_X[num_corrupted+anno*each_hw : num_corrupted+(anno+1)*each_hw])
            ys.append(train_y[num_corrupted+anno*each_hw : num_corrupted+(anno+1)*each_hw])

        for j in range(num_iters):
            if obj == 'minimax':
                h = predict_prob(theta, train_X)
                loss_vector = loss(h, train_y)
                p = project_onto_chi_square_ball(loss_vector, rho)
                grads_theta = compute_gradients_dro(theta, train_X, train_y, p)
            elif obj == 'tilting':
                grads_theta = compute_gradients_tilting(theta, Xs, ys, t_in, t_out, num_corrupted)
            else:
                grads_theta = compute_gradients_vanilla(theta, train_X, train_y)

            if np.linalg.norm(grads_theta, ord=2) < 1e-60:
                break

            theta = theta - lr * grads_theta
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

    end = time.time()

    #print("avg training accuracy on the rare class {}".format(np.mean(np.array(train_rare))))
    print("avg testing accuracy on the rare class {}".format(np.mean(np.array(test_rare))))
    print("se testing accuracy on the rare class {}".format(stats.sem(np.array(test_rare))))

    #print("avg training accuracy on the common class {}".format(np.mean(np.array(train_common))))
    print("avg testing accuracy on the common class {}".format(np.mean(np.array(test_common))))
    print("se testing accuracy on the common class {}".format(stats.sem(np.array(test_common))))

    #print("avg overall training accuracy {}".format(np.mean(np.array(train_accuracies))))
    print("avg overall testing accuracy {}".format(np.mean(np.array(test_accuracies))))
    print("se overall testing accuracy {}".format(stats.sem(np.array(test_accuracies))))

    print("avg overall validation accuracy {}".format(np.mean(np.array(val_accuracies))))


if __name__ == '__main__':
    main()


