from functools import reduce
import numpy as np

from models.base import MachineLearning

class NeuralNetwork(MachineLearning):
    """ TODO: work on mini-batch / stochastic gradient descent
              dynamic learning rate
    """

    # list of units of each hidden layer
    hidden_layers = []

    theta_shapes = None

    # flatten order defines how the training make thetas to a 1D array
    # in python / numpy implementation, default is 'C'
    # [[1, 2, 3], [4, 5, 6]] => [1, 2, 3, 4, 5, 6]
    # in octave, default is 'F'. this is used for matching coursera machine learning course assignments
    # [[1, 2, 3], [4, 5, 6]] => [1, 4, 2, 5, 3, 6]
    flatten_order = 'C'

    def init_theta(self, thetas=None, option='random', theta_range=0.05, seed=None):
        """ initialize the thetas (i.e. weights) for neural network training

            parameters
            ----------------------------------------
            thetas:      list of np.array denote each layer's initial weights, default None
                         - if not None, we use the given thetas to train, we also imply the hidden layers attributes

            option:      str, either 'random', 'zeros', 'ones'
                         - only applicable if thetas is not set (i.e. thetas is None)
                         - 'random': use the given seed to generate random thetas in the range of [-theta_range, theta_range]
                         - 'zeros': all thetas are 0, this is not allowed in neural network
                         - 'ones': all thetas are 1, this is not allowed in neural network

            theta_range: float, default 0.05
                         - used when theta initialization option is 'random'
                         - generate random thetas in the range of [-0.05, 0.05] when theta_range is 0.05

            seed:        integer or RandomState in numpy
                         - set the random seed generator
        """
        if thetas is None:
            if option not in ['zeros', 'ones', 'random']:
                raise Exception('wrong type of theta initialization option')

            if option in ['zeros', 'ones']:
                raise Exception('using constant initial theta cannot work')

            if option == 'random':
                if seed is None:
                    raise Exception('you must supply the seed argument')

                np.random.seed(seed)

                in_sizes = [self.X.shape[1]] + [n+1 for n in self.hidden_layers]
                out_sizes = self.hidden_layers + [self.y.shape[1]]
                self.theta_shapes = list(zip(out_sizes, in_sizes))

                required_n_theta = sum([o*i for (o, i) in self.theta_shapes])
                self.theta = np.random.random_sample((required_n_theta, 1)) * theta_range * 2 - theta_range

            return
        
        # or we make our own theta
        self.theta_shapes = [theta.shape for theta in thetas]
        # flatten the thetas
        self.theta = np.array(
            reduce(lambda a,b: a+b, map(lambda theta: theta.flatten(self.flatten_order).tolist(), thetas))
        )
        # infer hidden layers from thetas
        self.hidden_layers = [r for (r, c) in self.theta_shapes[:-1]]

    def reshape_theta(self, theta, theta_shapes, order='C'):
        """ Convert the 1D theta array back to thetas per layer
        """

        thetas = []

        for layer_idx, (out_size, in_size) in enumerate(self.theta_shapes):

            theta_begin_idx = sum([o*i for (o, i) in self.theta_shapes[:layer_idx]])
            theta_end_idx = theta_begin_idx + in_size * out_size

            theta_layer = theta[theta_begin_idx:theta_end_idx].reshape((out_size, in_size), order=order)

            thetas.append(theta_layer)

        return thetas

    def train_checked(self):
        if not self.hidden_layers:
            print('There has to be at least 1 hidden layer, please set self.hidden_layers')
            return False

        return True

    def calc_cost_gradient(self, theta, X, y, lambda_):

        m = len(X)
        a_s = [X]
        # first z is dummy
        z_s = [None]
        # include input layer and output layer, so +2
        total_layers = len(self.hidden_layers) + 2

        thetas = self.reshape_theta(theta, self.theta_shapes, self.flatten_order)

        for idx, hidden_layer in enumerate(self.hidden_layers):
            z = a_s[idx].dot(thetas[idx].T)
            z_s.append(z)
            a = np.hstack([np.ones((m, 1)), self.sigmoid(z)])
            a_s.append(a)

        a = self.sigmoid(a_s[-1].dot(thetas[-1].T))
        a_s.append(a)

        # calc cost
        unreg_cost = (1.0/m) * np.sum(-y * np.log(a_s[-1]) - (1 - y) * np.log(1 - a_s[-1]))

        # Add regularized error. Drop the bias terms in the 1st columns.
        reg_cost = 0.0
        for theta_layer in thetas:
            reg_cost = reg_cost + (lambda_ / (2.0*m)) * np.sum( np.square(theta_layer[:, 1:]) )

        # backpropagate
        # d_s[0] is no use, it's just for better layer indexing when we notate terms
        d_s = [0] * total_layers
        d_s[-1] = a_s[-1] - y

        for layer_idx in range(len(self.hidden_layers), 0, -1):

            # if it's not the output layer, d_s has 1 extra column which is redundant for backward propagation
            if layer_idx + 1 == total_layers - 1:
                d_next_layer = d_s[layer_idx+1]
            else:
                d_next_layer = d_s[layer_idx+1][:, 1:]

            d = d_next_layer.dot(thetas[layer_idx]) * np.hstack(
                [np.ones((m, 1)), self.sigmoid_gradient(z_s[layer_idx])]
            )

            d_s[layer_idx] = d

        theta_grads = [0] * len(thetas)
        theta_grad = (1.0 / m) * d_s[-1].T.dot(a_s[-2])
        theta_grads[-1] = theta_grad

        for layer_idx in range(len(self.hidden_layers), 0, -1):

            theta_grad = (1.0 / m) * d_s[layer_idx][:, 1:].T.dot(a_s[layer_idx-1])
            theta_grads[layer_idx-1] = theta_grad

        # regularization
        for idx, theta_layer in enumerate(thetas):
            theta_copy = np.copy(theta_layer)
            theta_copy[:, 0] = 0
            theta_grads[idx] = theta_grads[idx] + (lambda_ / float(m)) * theta_copy

        # flatten the theta gradient
        theta_gradient = np.array(
            reduce(lambda a,b: a+b, map(lambda theta_grad: theta_grad.flatten(self.flatten_order).tolist(), theta_grads))
        )

        return unreg_cost, reg_cost, theta_gradient


    def predict_function(self, X, add_ones=True):

        if add_ones:
            X = np.hstack([np.ones((len(X), 1)), X])

        m = len(X)
        thetas = self.reshape_theta(self.result['optimal']['theta'], self.theta_shapes, self.flatten_order)

        # keep doing forward propagation to get the final neural network output as prediction
        pred = self.sigmoid(X.dot(thetas[0].T))

        for theta in thetas[1:]:
            pred = self.sigmoid( np.hstack([np.ones((m, 1)), pred]).dot(theta.T) )

        return pred