import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.model_utils import gen_epoch, gen_batch
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.AdagradOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        batches = {}
        for c in self.clients:
            batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2)

        print("finished generating data")

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                num_train_overall, num_correct_train_overall = self.train_error()
                num_test, num_correct_test = self.test()  # have set the latest model for all clients
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(
                    np.array(num_correct_train_overall)) * 1.0 / np.sum(np.array(num_train_overall))))
                tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(
                    np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))))

            csolns = []
            for c in self.clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                batch = next(batches[c])
                solns, grads, loss = c.solve_sgd(batch)
                if i % 50 == 0:
                    print(i, loss)
                csolns.append(solns)

            if np.linalg.norm(process_grad(grads[1]), ord=2) < 1e-4:
                break
            # aggregate
            self.latest_model = self.aggregate(csolns)





