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

# gradients for generalized cross entropy loss: https://arxiv.org/pdf/1805.07836.pdf
def compute_gradients_gce(theta, X0, X1, y0, y1, q):
    h0 = predict_prob(theta, X0)  # the probabilities of being label 1 for training samples (X, 0)
    h1 = predict_prob(theta, X1)  # the probabilities of being label 1 for training samples (X, 1)
    gradient_y_0 = np.dot(X0.T, np.multiply(h0, np.power(1-h0, q)))
    gradient_y_1 = np.dot(X1.T, np.multiply(h1-1, np.power(h1, q)))
    gradient = (gradient_y_0 + gradient_y_1) / (y0.size + y1.size)

    return gradient


# gradients for focal loss
def compute_gradients_focal(theta, X0, X1, y0, y1, gamma):
    h0 = predict_prob(theta, X0)
    h1 = predict_prob(theta, X1)
    w_y_0 = np.power(h0, gamma)
    gradient_y_0 = 0.3 * np.dot(X0.T, np.multiply(w_y_0, h0 - y0 - gamma * (1-h0) * np.log(1-h0)))
    w_y_1 = np.power(1-h1, gamma)
    gradient_y_1 = 0.7 * np.dot(X1.T, np.multiply(w_y_1, h1 - y1 + gamma * h1 * np.log(h1)))
    gradient = (gradient_y_0 + gradient_y_1)/(y0.size+y1.size)

    return gradient



def compute_gradients_dro(theta, X, y, w):
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, np.multiply(w, h - y))
    return gradient


def compute_gradients_tilting(theta, X_1, y_1, X_2, y_2, t_in, t_out):

    def compute_gradients_inner_tilting(theta, X, y, t):
        h = predict_prob(theta, X)
        l = loss(h, y)
        if t > 0:
            l_max = max(l)
        else:
            l_max = 0

        gradient = np.dot(X.T, np.multiply(np.exp((l - l_max) * t), h - y))
        ZZ = np.mean(np.exp(t * (l - l_max)))
        return gradient / ZZ, l


    def compute_inner_obj(t_in, l):
        return (1/t_in) * np.log(np.mean(np.exp(t_in*l)))

    gradient1, l1 = compute_gradients_inner_tilting(theta, X_1, y_1, t_in)
    gradient2, l2 = compute_gradients_inner_tilting(theta, X_2, y_2, t_in)

    # the weight is e^{t*loss}/sum, where loss is the (cross sample) tilted loss of the classes
    l_1, l_2 = compute_inner_obj(t_in, l1), compute_inner_obj(t_in, l2)
    l_max = max(l_1, l_2)

    gradient = (np.exp((l_1 - l_max) * t_out) * gradient1  + \
                                                np.exp((l_2 - l_max) * t_out) * gradient2)
    ZZ = len(y_1) * np.exp(t_out * (l_1 - l_max)) + len(y_2) * np.exp(t_out * (l_2 - l_max))

    return gradient / ZZ


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
                        help='objective: erm, dro, tilting, etc',
                        type=str,
                        default='erm')
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
                        default=0.1)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=1)
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
    parser.add_argument('--q',
                        help='the q parameter for generalized cross entropy loss, \in (0,1]',
                        type=float,
                        default=0.2)

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
    q = parsed['q']
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
        all_data, all_labels = data_loader("data/hiv1/raw", i)  # have been randomly shuffled using seed i
        print("{} total data points, {} training samples, {}% rare".format(len(all_labels), int(0.8*(len(all_labels))),
                                                                          len(np.where(all_labels==1)[0])/len(all_labels)))

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
            num_common = len(np.where(train_y==0)[0])
            num_rare = len(np.where(train_y==1)[0])
            train_y[:num_corrupted] = np.random.choice([0, 1], num_corrupted, p=[num_common/len(train_y), num_rare/len(train_y)])
            val_y[:int(num_corrupted*0.1)] = np.random.choice([0, 1], int(0.1*num_corrupted),
                                                       p=[num_common / len(train_y), num_rare / len(train_y)])


        theta = np.zeros(len(train_X[0]))

        class1 = np.where(train_y == 1)[0]
        class2 = np.where(train_y == 0)[0]
        X_1, y_1 = train_X[class1], train_y[class1]
        X_0, y_0 = train_X[class2], train_y[class2]

        for j in range(num_iters):
            if obj == 'dro':
                h = predict_prob(theta, train_X)
                loss_vector = loss(h, train_y)
                p = project_onto_chi_square_ball(loss_vector, rho)
                grads_theta = compute_gradients_dro(theta, train_X, train_y, p)
            elif obj == 'tilting':
                grads_theta = compute_gradients_tilting(theta, X_1, y_1, X_0, y_0, t_in, t_out)
            elif obj == 'erm':
                grads_theta = compute_gradients_vanilla(theta, train_X, train_y)
            elif obj == 'fl':
                grads_theta = compute_gradients_focal(theta, X_0, X_1, y_0, y_1, gamma)
            elif obj == 'gce':
                grads_theta = compute_gradients_gce(theta, X_0, X_1, y_0, y_1, q)

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
    print("se overall testing accuracy {}".format(stats.sem(test_accuracies)))

    print("avg overall validation accuracy {}".format(np.mean(np.array(val_accuracies))))


if __name__ == '__main__':
    main()


