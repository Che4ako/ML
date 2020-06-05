import time
import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse as sparse
from numpy.linalg import norm
import math


class SVMClassifier:

    def __init__(self, step_alpha=1, step_beta=0, tolerance=1e-5, 
                 max_iter=1000, l2_coef=1.0, C_coef=1.0, **kwargs):
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.l2_coef = l2_coef
        self.C_coef = C_coef
        self.history = {'weights': [], 'func': []}
        self.w = None
        self.b = None

    #вычисление значения функционала, соответсвующего безусловной задачи оптимизации для SVM
    #параметр self.C_coef нужно подбирать отдельно для каждой задачи
    #см. подробности https://github.com/esokolov/ml-course-hse/blob/master/2016-fall/lecture-notes/lecture05-linclass.pdf
    #и в лекции Воронцова по SVM
    def func(self, X, y, w, b):
        #print(np.dot(X, w) + b)
        #print(y.shape)
        return self.C_coef * (np.maximum(1.0 - (np.dot(X, w) + b) * y, 
                          np.zeros(X.shape[0]))).sum() / X.shape[0] + self.l2_coef * 0.5 * norm(w) ** 2
    
    #вычисление градиента по весам
    def grad_w(self, X, y, w, b):
        gradient_coefficients = ((np.dot(X, w) + b) * y <= 1.0)
        #print(gradient_coefficients.sum())
        return (self.l2_coef * w - self.C_coef * 
                ((X * y[:, np.newaxis] * gradient_coefficients[:, np.newaxis]).T).sum(axis=1) 
                / X.shape[0]
                )
    
    #вычисление производной по параметру сдвига гиперплоскости
    def grad_b(self, X, y, w, b):
        gradient_coefficients = ((np.dot(X, w) + b) * y <= 1.0)
        return - self.C_coef * (y * gradient_coefficients).sum() / X.shape[0]
        
    def fit(self, X, y, w_0=None, b_0=None, trace=False):
        if w_0 is None:
            w_0 = np.ones(X.shape[1]) / X.shape[1]
            #w_0 = np.zeros(X.shape[1])
        if b_0 is None:
            b_0 = 0.0
        self.history['func'].append(self.func(X, y, w_0, b_0))
        
        self.w = w_0 - self.step_alpha * self.grad_w(X, y, w_0, b_0)
        self.b = b_0 - self.step_alpha * self.grad_b(X, y, w_0, b_0)
        self.history['func'].append(self.func(X, y, self.w, self.b))
        self.history['weights'].append(self.get_weights())

        if abs(self.history['func'][1] - self.history['func'][0]) < self.tolerance:
            return
        else:
            #градиентный спуск для минимизации функционала
            for k in range(2, self.max_iter+1):
                w_curr = self.w.copy()
                b_curr = self.b
                self.w = w_curr - (self.step_alpha / (k ** self.step_beta)) * self.grad_w(X, y, w_curr, b_curr)
                self.b = b_curr - (self.step_alpha / (k ** self.step_beta)) * self.grad_b(X, y, w_curr, b_curr)
                self.history['func'].append(self.func(X, y, self.w, self.b))
                self.history['weights'].append(self.get_weights())
                if abs(self.history['func'][k] - self.history['func'][k-1]) < self.tolerance:
                    return           
        
    def predict(self, X):
        R = np.sign(np.dot(X, self.w) + self.b)
        R[R == 0.] = 1
        return R

    #значение функционала    
    def get_objective(self, X, y):
        return self.func(X, y, self.w, self.b)
    
    #посмотреть веса обученя
    def get_weights(self):
        return np.hstack((self.w, self.b))
    
    
