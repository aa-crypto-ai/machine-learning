import numpy as np
import scipy.optimize as optimize

class MachineLearning:
    """ a general class for machine learning algorithm
        it provides the infrastructure of the algorithm
        
        it allows 2 training algorithms, "raw" or "fmin_bfgs"

        -- "gd" --
        own implementation of the Gradient Descent algoirthm, more efficient than the fmin_bfgs here

        -- "fmin_bfgs" --
        use the scipy function fmin_bfgs
        however, to evaulate the training algorithm, we need to re-compute the cost / gradient in the callback function,
        making it very inefficient

        To use this class,
        1. override self.calc_cost_gradient(theta, X, y, l) to return a tuple (cost, gradient)
        2. override self.predict_function(X, add_ones=True) to return the prediction result
        
        # sample use of the class
        ml = MachineLearning()
        ml.set_data(X, y, add_ones=True)
        ml.init_theta('random', seed=123)
        ml.train(alg='gd')
        print(ml.result)
        y = ml.predict_function(X, add_ones=True)

        TODO:
        1. implement the stopping criteria
        2. allow using validation / test dataset
        3. store the training result into a file
        
    """
    alg_options = ['gd', 'minibatch_gd', 'fmin_bfgs']

    def __init__(self):
        self.X = None
        self.y = None

        self.lambda_ = 0
        self.theta = None
        self.alpha = 0.3
        self.batch_size = None

        # below to be implemented
        self.stopping_criteria = {
            'cost_change': 1e-10,
            'max_iteration': 999999,
        }
        self.cost_history = []

        # control whether to save theta and gradient history during training
        self.save_history = False
        self.theta_history = []
        self.gradient_history = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z));

    @staticmethod
    def sigmoid_gradient(z):
        s = MachineLearning.sigmoid(z);
        grad = s * (1 - s)
        return grad

    def set_data(self, X, y, add_ones=True):
        """ set the data used for training
            X:        np array of shape (m, n), m is no. of samples, n is no. of features
            y:        np array of shape m
            add_ones: boolean, default True
                      set if we need to add a "one" column to denote the intercept term / bias term
        """
        self.X_raw = X
        if add_ones:
            self.X = np.hstack([np.ones((len(X), 1)), X])
        else:
            self.X = self.X_raw
        self.y = y

    def init_theta(self, theta=None, option='zeros', theta_range=0.05, seed=None):
        """ initialize the thetas (i.e. weights) for neural network training

            parameters
            ----------------------------------------
            theta:       np.array denoting initial weights, default None
                         - if not None, we use the given theta to train

            option:      str, either 'random', 'zeros', 'ones'
                         - only applicable if theta is not set (i.e. theta is None)
                         - 'random': use the given seed to generate random thetas in the range of [-theta_range, theta_range]
                         - 'zeros': all thetas are 0
                         - 'ones': all thetas are 1

            theta_range: float, default 0.05
                         - used when theta initialization option is 'random'
                         - generate random thetas in the range of [-0.05, 0.05] when theta_range is 0.05

            seed:        integer or RandomState in numpy
                         - set the random seed generator for randomization
        """

        if theta is None:

            if option not in ['zeros', 'ones', 'random']:
                raise Exception('wrong type of theta initialization option')

            if option == 'zeros':
                self.theta = np.zeros((self.X.shape[1], 1))
                return

            if option == 'ones':
                self.theta = np.ones((self.X.shape[1], 1))
                return

            if option == 'random':
                if seed is None:
                    raise Exception('you must supply the seed argument')

                np.random.seed(seed)
                self.theta = np.random.random_sample((self.X.shape[1], 1)) * theta_range * 2 - theta_range
                return

        self.theta = theta


    def clear_history(self):
        """ clear training history (cost, theta, gradient histories)
        """
        print('Clearing previous train history')
        self.cost_history = []
        self.theta_history = []
        self.gradient_history = []


    def calc_cost_gradient(self, theta, X, y, l):
        """ the function to return tuple (cost, gradient)
            function signature: self.calc_cost_gradient(theta, X, y, l)
        """
        raise Exception('Please override this function')

    def cost_function(self, theta, X, y, l):
        """ the function to return the cost for scipy fmin_bfgs
            when using scipy fmin_bfgs, it could have been more efficient by using a separate cost function / gradient function
            but now we favour the use of "gd" training algorithm
        """
        unreg_cost, reg_cost, gradient = self.calc_cost_gradient(theta, X, y, l)
        return unreg_cost + reg_cost

    def gradient_function(self, theta, X, y, l):
        """ the function to return the gradient for scipy fmin_bfgs
            when using scipy fmin_bfgs, it could have been more efficient by using a separate cost function / gradient function
            but now we favour the use of "gd" training algorithm
        """
        unreg_cost, reg_cost, gradient = self.calc_cost_gradient(theta, X, y, l)
        return gradient

    def predict_function(self, X, add_ones=True):
        """ the function to return the prediction result by using the trained theta
        
            need to override this for every subclass
            signature: self.prediction_function(X, add_ones=True)
            return:    y (np array of shape (m, n))

            parameters
            --------------------------------
            X:        np array of shape (m, n)
                      - m is no. of samples
                      - n is no. of features
            add_ones: boolean, default True
                      - whether to add the "one" column (intercept / bias) to X
        """
        raise Exception('Please override this function')

    def train_checked(self):
        """ other criteria to be checked before running the training
            return False if something is not properly check in the subclass
        """
        return True

    def train(self, alg='gd'):
        """ start training the data
            
            either use algorithm "gd" or "fmin_bfgs"
            
            return None

            results are stored in self.result as a dictionary (even if user interrupts)
            {
                'optimal': {
                    'theta': current_theta,
                    'cost': current_cost,
                    'gradient': current_gradient,
                },
                'n_func_calls': len(cost_history),
                'n_grad_calls': len(gradient_history),
                'exit_msg': exit_msg,
                'error': error,  # True or False
            }
            
            -------------------------------------------
            "gd"
            -------------------------------------------
            Gradient descent: the preferred algorithm here
            Advantages:
            - the entire training algorithm is written in scratch without machine learning packages
            - can fully customize what we want during the training
            Disadvantages:
            - learning rate alpha needs to be set carefully
            - the training itself is not an optimized algorithm compared to others e.g. fmin_bfgs
            
            -------------------------------------------
            "fmin_bfgs"
            -------------------------------------------
            not the preferred algorithm here because it's very inefficient
            it use scipy.optimize.fmin_bfgs function
            
            Advantages:
            - no need to set learning rate alpha
            - the algorithm takes care of how to optimize theta more efficiently
            Disadvantages:
            - using callback to get cost value during the training for evaluation makes the training inefficient
            
        """

        if self.X is None or self.y is None:
            raise Exception('set X and y data first by running self.set_data(X, y, add_ones=True)')

        if self.theta is None:
            raise Exception('set initial theta first by running self.init_theta(option)')

        if self.cost_history or self.theta_history:
            raise Exception('previous training history exists, run self.clear_history() first if you want to clear it')

        if alg not in self.alg_options:
            raise Exception('choose algorithm: either ' + ' or '.join(self.alg_options))

        if alg == 'minibatch_gd' and self.batch_size is None:
            raise Exception('set self.batch_size for mini batch gradient descent')

        if not self.train_checked():
            raise Exception('Training not properly initialized')

        if alg in ['gd', 'minibatch_gd']:
            if alg == 'minibatch_gd':
                np.random.seed(123)

            while True:

                try:
                    if alg == 'gd':
                        batches_X = [self.X]
                        batches_y = [self.y]
                    elif alg == 'minibatch_gd':
                        # random indices
                        remaining_indices = list(range(len(self.X)))
                        batch_indices = []

                        while remaining_indices and len(remaining_indices) >= self.batch_size:
                            indices = np.random.choice(remaining_indices, self.batch_size, replace=False)
                            remaining_indices = [i for i in remaining_indices if i not in indices]
                            batch_indices.append(indices)
                        if remaining_indices:
                            batch_indices.append(remaining_indices)

                        batches_X = [self.X[indices] for indices in batch_indices]
                        batches_y = [self.y[indices] for indices in batch_indices]

                    cost = 0.0
                    for epoch, (batch_X, batch_y) in enumerate(zip(batches_X, batches_y)):

                        batch_unreg_cost, batch_reg_cost, batch_gradient = self.calc_cost_gradient(self.theta, batch_X, batch_y, self.lambda_)
                        # handle unregularized cost and regularized cost separately to ensure comparability of cost using grad descent and stochastic grad descent
                        batch_cost = batch_unreg_cost * len(batch_X) + (batch_reg_cost * len(batch_X)) * (float(len(batch_X)) / len(self.X))
                        cost += batch_cost
                        self.theta = self.theta.reshape(batch_gradient.shape) - (self.alpha * batch_gradient)

                    cost = cost / float(len(self.X))

                    if self.save_history:
                        self.gradient_history.append(batch_gradient)
                        self.theta_history.append(self.theta)

                    if self.cost_history:
                        cost_improvement = self.cost_history[-1] - cost
                        if cost_improvement <= 0:
                            exit_msg = 'cost didn\'t decrease!'
                            error = True
                            # break

                        if cost_improvement <= 1e-10:
                            exit_msg = 'cost decreases very little!'
                            error = False
                            # break

                    if len(self.cost_history) % 500 == 0:
                        print('Run iterations: %d, Cost: %.5f' % (len(self.cost_history), cost))

                    self.cost_history.append(cost)

                except KeyboardInterrupt:
                    exit_msg = 'Keyboard Interrupt'
                    error = False
                    break

            self.result = {
                'optimal': {
                    'theta': self.theta,
                    'cost': self.cost_history[-1],
                    'gradient': batch_gradient,
                },
                'n_func_calls': len(self.cost_history),
                'exit_msg': exit_msg,
                'error': error,
            }

        if alg == 'fmin_bfgs':

            def callback(theta):
                """ For storing training history: cost
                    this is inefficient as the same cost has to be re-computed,
                    but there are no other possible methods yet
                """
                unreg_cost, reg_cost, gradient = self.calc_cost_gradient(theta, self.X, self.y, self.lambda_)

                self.cost_history.append(unreg_cost + reg_cost)
                if self.save_history:
                    self.theta_history.append(theta)
                    self.gradient_history.append(gradient)

            # need to store the first cost when theta = initial theta first
            unreg_cost, reg_cost, gradient = self.calc_cost_gradient(self.theta, self.X, self.y, self.lambda_)
            cost = unreg_cost + reg_cost
            self.cost_history = [cost]
            if self.save_history:
                self.gradient_history = [gradient]
                self.theta_history = [self.theta]

            args = {}
            if self.stopping_criteria['max_iteration'] is not None:
                args['maxiter'] = self.stopping_criteria['max_iteration']

            try:
                optimize_result = optimize.fmin_bfgs(
                    f=self.cost_function,
                    x0=self.theta,
                    fprime=self.gradient_function,
                    args=(self.X, self.y, self.lambda_),
                    callback=callback,
                    retall=True,
                    full_output=True,
                    **args
                )


                xopt, fopt, gopt, bopt, n_func_calls, n_grad_calls, warnflag, theta_history = optimize_result

                
                if warnflag == 0:
                    error = False
                    exit_msg = None

                if warnflag == 1:
                    exit_msg = 'Maximum number of iterations exceeded.'
                    error = False

                if warnflag == 2:
                    exit_msg = 'Gradient and/or function calls not changing.'
                    error = True

                if warnflag > 2:
                    error = True
                    exit_msg = 'Unknown error'

                self.theta_history = theta_history
                self.result = {
                    'optimal': {
                        'theta': xopt,
                        'cost': fopt,
                        'gradient': gopt,
                    },
                    'n_func_calls': n_func_calls,
                    'exit_msg': warnflag,
                    'error': error,
                }


            
            except KeyboardInterrupt:

                self.result = {
                    'optimal': {
                        'theta': self.theta,
                        'cost': self.cost_history[-1],
                        'gradient': gopt,
                    },
                    'n_func_calls': len(self.cost_history),
                    'exit_msg': 'Keyboard Interrupt',
                    'error': False,
                }