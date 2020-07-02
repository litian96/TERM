import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients

        estimates = 0


        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test()  # have set the latest model for all clients
                num_train, num_correct_train = self.train_error()
                num_val, num_correct_val = self.validate()
                tqdm.write('At round {} testing accuracy: {}'.format(i,
                                                                     np.divide(np.array(num_correct_test),
                                                                         np.array(num_test))))
                print('average accuracy: ', np.sum(np.array(num_correct_test)) / np.sum(np.array(num_test)))


            if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
                test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
                np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
                #train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
                #np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
                #validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
                #np.savetxt(self.output + "_" + str(i) + "_validation.csv", validation_accuracies, delimiter=",")

            indices, selected_clients = self.select_clients(i, pk, num_clients=self.clients_per_round)

            print('devices', indices)

            selected_clients = selected_clients.tolist()
            csolns = []
            updates = []
            losses = []
            old_weights = []

            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = c.get_params()
                loss = c.get_loss()
                losses.append(loss)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                update = [(u - v) * 1.0 for u, v in zip(weights_before, soln[1])]
                updates.append(update)
                old_weights.append(weights_before)


            max_l = max(losses)
            new_ = np.mean(np.exp(self.t * np.array((losses-max_l))))
            estimates = estimates * 0.5 + new_ * 0.5
            weights = np.exp(self.t * np.array((losses - max_l))) / (estimates * self.clients_per_round)

            for idx, u in enumerate(updates):
                updates[idx] = [layer_weight * weights[idx] for layer_weight in updates[idx]]
                new_weights = []
                len_layer = len(updates[idx])
                for l in range(len_layer):
                    new_weights.append(old_weights[idx][l] - updates[idx][l])
                csolns.append(new_weights)

            # update models
            self.latest_model = self.aggregate(csolns)





