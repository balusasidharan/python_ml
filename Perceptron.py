import numpy as np

class Perceptron(object):
    eta: float
    n_iter : int
    random_state : int

    w_ : 1d-array
    errors : list

    def __int__(self,eta=0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit (self, x, y):
