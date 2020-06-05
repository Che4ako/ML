import time
import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse as sparse
from numpy.linalg import norm
import math


class GDClassifier:
    """
    Класс логистической регрессии, обучающийся обычным градиентным спуском
    Выбор шага происходит следующим образом:
    step = step_alpha / k^step_beta, где k - номер шага
    """
    def __init__(self, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, l2_coef=0.0, **kwargs):
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.l2_coef = l2_coef
        self.history = {'time': [], 'func': []}
        self.w = None
        """
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Критерий остановки: |f(x_{k+1}) - f(x_{k})| < tolerance
        l2_coef - коэффициент l2-регуляризации
        max_iter - максимальное число итераций       
        """

    #значение Loss-функции
    def func(self, X, y, w):
        if sparse.issparse(X):
            return np.logaddexp(np.zeros(X.shape[0]), 
                                  - (X.dot(w) * y)).sum() / X.shape[0] + self.l2_coef * 0.5 * norm(w) ** 2
        else:
            return np.logaddexp(np.zeros(X.shape[0]), 
                                  - np.dot(X, w) * y).sum() / X.shape[0] + self.l2_coef * 0.5 * norm(w) ** 2
    
    #градиент Loss-функции
    def grad(self, X, y, w):
        if sparse.issparse(X):
            return - (X.T).dot(expit(- X.dot(w) * y) * y) / X.shape[0] + self.l2_coef * w
        else:
            return - np.dot(X.T, expit(- np.dot(X, w) * y) * y) / X.shape[0] + self.l2_coef * w
        
    #метод fit - минимизация функционала градиентным спуском
    """
    Обучение метода по выборке X с ответами y
    X - scipy.sparse.csr_matrix или двумерный numpy.array
    y - одномерный numpy array
    w_0 - начальное приближение в методе
    """
       
    def fit(self, X, y, w_0=None, trace=False):
        previous_time = time.monotonic()
        self.history = {'time': [], 'func': []}
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        self.history['time'].append(0)
        self.history['func'].append(self.func(X, y, w_0))
        
        self.w = w_0 - self.step_alpha * self.grad(X, y, w_0)
        self.history['func'].append(self.func(X, y, self.w))
        self.history['time'].append(time.monotonic() - previous_time)
        previous_time = time.monotonic()
        if abs(self.history['func'][1] - self.history['func'][0]) < self.tolerance:
            if trace == True:
                return self.history
            else:
                return
        else:
            for k in range(2, self.max_iter+1):
                #у нас уже была сделана 1 итерация вне цикла + в цикле будет от 2 до 1000 включительно,
                #т.е. итого 1+999 итераций
                self.w = self.w - (self.step_alpha / (k ** self.step_beta)) * self.grad(X, y, self.w)
                self.history['func'].append(self.func(X, y, self.w))
                self.history['time'].append(time.monotonic() - previous_time)
                previous_time = time.monotonic()
                if abs(self.history['func'][k] - self.history['func'][k-1]) < self.tolerance:
                    break
            if trace == True:
                return self.history
            else:
                return            
    #получение меток ответов на выборке X      
    def predict(self, X):
        if sparse.issparse(X):
            R = np.sign(X.dot(self.w))
            R[R == 0.] = 1
            return R
        else:
            R = np.sign(np.dot(X, self.w))
            R[R == 0.] = 1
            return R
    #метод predict_proba
    def predict_proba(self, X):
        if sparse.issparse(X):
            a = ((1 + np.exp(- X.dot(self.w))) ** -1).reshape(X.shape[0], 1)
            b = ((1 + np.exp(X.dot(self.w))) ** -1).reshape(X.shape[0], 1)
            return np.hstack((b, a))
        else:
            a = ((1 + np.exp(np.dot(- X, self.w))) ** -1).reshape(X.shape[0], 1)
            b = ((1 + np.exp(np.dot(X, self.w))) ** -1).reshape(X.shape[0], 1)
            return np.hstack((b, a))

    #методы, которые можно вызывать после обучения и получения значения self.w    
    def get_objective(self, X, y):
        return self.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        return self.grad(X, y, self.w)
    
    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):
    """
    Класс стохастического градиентного спуска
    batch_size - размер подвыборки, по которой считается градиент
    """
    
    def __init__(self, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, l2_coef=0.0, **kwargs):
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.l2_coef = l2_coef
        self.history = {'epoch_num': [], 'time': [], 'func': [], 'weights_diff': []}
        self.w = None
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1.0):
        np.random.seed(self.random_seed)
        previous_time = time.monotonic()
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        self.w = w_0
        previous_weight = self.w
        self.history = {'epoch_num': [], 'time': [], 'func': [], 'weights': [], 'weights_diff': []}
        self.history['epoch_num'].append(0)
        self.history['time'].append(0)
        self.history['func'].append(self.func(X, y, w_0))
        self.history['weights_diff'].append(norm(w_0) ** 2)
        self.history['weights'].append(w_0)
        epoch_number = 0
        
        for k in range(1, self.max_iter+1):
            s = np.arange(X.shape[0])
            np.random.shuffle(s)
            iter_numbers = math.ceil(X.shape[0] / self.batch_size)
            for i in range(iter_numbers):
                #это работает с csr_matrix тоже
                X_trunc = X[s[i*self.batch_size:(i+1)*self.batch_size]]
                y_trunc = y[s[i*self.batch_size:(i+1)*self.batch_size]]
                self.w = self.w - (self.step_alpha / (k ** self.step_beta)) * self.grad(X_trunc, y_trunc, self.w)
                epoch_number += X_trunc.shape[0]
                if epoch_number / X.shape[0] - self.history['epoch_num'][-1] >= log_freq:
                    self.history['epoch_num'].append(epoch_number / X.shape[0])
                    self.history['time'].append(time.monotonic() - previous_time)
                    previous_time = time.monotonic()
                    self.history['func'].append(self.func(X, y, self.w))
                    self.history['weights_diff'].append(norm(self.w - previous_weight) ** 2)
                    self.history['weights'].append(self.w)
                    previous_weight = self.w
                    if abs(self.history['func'][-1] - self.history['func'][-2]) < self.tolerance:
                        break
            #это относится к циклу по "k" 
            if (len(self.history['func']) >= 2 and
                   abs(self.history['func'][-1] - self.history['func'][-2]) < self.tolerance):
                break
        if trace == True:
            return self.history
        else:
            return
