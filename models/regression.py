import numpy as np

from models.base import MachineLearning

class LogisticRegression(MachineLearning):

    def calc_cost_gradient(self, theta, X, y, lambda_):
        m, n = X.shape
        theta_ = theta.reshape((n,1))
        y_ = y.reshape((m,1))
        pred = X.dot(theta_)
        s = self.sigmoid(pred)
        cost1 = np.log(s)
        cost2 = np.log(1 - s)
        cost1 = cost1.reshape((m,1))
        cost2 = cost2.reshape((m,1))
        total_cost = y_ * cost1 + (1 - y_) * cost2

        J = -((np.sum(total_cost))/m)
        # add regularization to the cost
        J = J + float(lambda_) / (2 * m) * theta_[1:].T.dot(theta_[1:])

        grad = ( (X.T).dot(s - y_) ) / m
        # add regularization to the gradient
        grad[1:] = grad[1:] + float(lambda_) / m * theta_[1:]

        return J, grad.flatten()

    def predict_function(self, X, add_ones=True):

        if add_ones:
            X = np.hstack([np.ones((len(X), 1)), X])

        theta = self.result['optimal']['theta']
        result = X.dot(theta)

        return self.sigmoid(result)


class LinearRegression(MachineLearning):

    def calc_cost_gradient(self, theta, X, y, lambda_):
        m, n = X.shape
        theta_ = theta.reshape((n, 1))
        y_ = y.reshape((m, 1))
        pred = X.dot(theta_)

        # cost with regularization
        J = 1.0 / (2*m) * np.sum(np.power(pred - y, 2)) + float(lambda_) / (2*m) * theta_[1:].T.dot(theta_[1:])

        grad = 1.0 / m * (X.T).dot( pred - y )
        # add regularization to the gradient
        grad[1:] = grad[1:] + float(lambda_) / m * theta_[1:]

        return J, grad.flatten()

    def predict_function(self, X, add_ones=True):

        if add_ones:
            X = np.hstack([np.ones((len(X), 1)), X])

        theta = self.result['optimal']['theta']
        return X.dot(theta)