class RegressionTree():

    def __init__(self, max_depth=3, min_size=10):
        
        self.max_depth = max_depth
        self.min_size = min_size
        #значение в поддереве (среднее по всем листам)
        self.value = 0
        # номер лучшего признака
        self.feature_idx = -1
        # значение лучшего признака
        self.feature_threshold = 0
        #правый и левый потомоки
        self.left = None
        self.right = None
        
    def fit(self, X, y):
        
        # начальное значение - среднее значение y
        self.value = y.mean()
        base_error = ((y - self.value) ** 2).sum()
        error = base_error
        
        #одно из условий прекращения обучения: достигли максимальной глубины
        if self.max_depth <= 1:
            return
        
        left_value, right_value = 0.0, 0.0
        
        #перебираю всевозможные признаки и трешхолды
        for feat in range(X.shape[1]):
            
            prev_error1, prev_error2 = base_error, 0 
            idxs = np.argsort(X[:, feat])
            
            #более быстрый пересчет ошибки
            mean1, mean2 = y.mean(), 0
            sm1, sm2 = y.sum(), 0
            
            N = X.shape[0]
            N1, N2 = N, 0
            thres = 1
            
            while thres < N - 1:
                N1 -= 1
                N2 += 1

                idx = idxs[thres]
                x = X[idx, feat]
                
                delta1 = (sm1 - y[idx]) * 1.0 / N1 - mean1
                delta2 = (sm2 + y[idx]) * 1.0 / N2 - mean2
                sm1 -= y[idx]
                sm2 += y[idx]
                
                # пересчитываем ошибки за O(1)
                prev_error1 += (delta1**2) * N1 
                prev_error1 -= (y[idx] - mean1)**2 
                prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1)
                mean1 = sm1/N1
                
                prev_error2 += (delta2**2) * N2 
                prev_error2 += (y[idx] - mean2)**2 
                prev_error2 -= 2 * delta2 * (sm2 - mean2 * N2)
                mean2 = sm2/N2
                
                if thres < N - 1 and np.abs(x - X[idxs[thres + 1], feat]) < 1e-5:
                    thres += 1
                    continue
                
                #условия разделения: уменьшение ошибки и в кажом листе будет >= min_size
                if (prev_error1 + prev_error2 < error):
                    if (min(N1,N2) > self.min_size):
                        self.feature_idx, self.feature_threshold = feat, x
                        left_value, right_value = mean1, mean2
                        error = prev_error1 + prev_error2
                                     
                thres += 1
 
        #если не нашли разбияения
        if self.feature_idx == -1:
            return
        
        self.left = RegressionTree(self.max_depth - 1)
        self.right = RegressionTree(self.max_depth - 1)
        
        idxs_l = (X[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (X[:, self.feature_idx] <= self.feature_threshold)
    
        self.left.fit(X[idxs_l, :], y[idxs_l])
        self.right.fit(X[idxs_r, :], y[idxs_r])
        
    def __predict(self, x):
        if self.feature_idx == -1:
            return self.value
        
        if x[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)
        
    def predict(self, X):
        y = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])
            
        return y


class GradientBoosting():
    
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, 
                 random_state=17, min_size = 5):
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0]])
        self.min_size = min_size
        self.loss_by_iter = []
        self.trees = []
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        b = self.initialization(y)
        
        prediction = b.copy()
            
        for t in range(self.n_estimators):  
            
            if t == 0:
                resid = y
            else:
                #вектор антиградиента
                resid = (y - prediction)
            
            tree = RegressionTree(max_depth=self.max_depth, min_size = self.min_size)
            tree.fit(X, resid)
            b = tree.predict(X).reshape([X.shape[0]])
            self.trees.append(tree)
            #учитываем learning_rate, чтобы увеличить обобщающую способность
            prediction += self.learning_rate * b
            
            if t > 0:
                self.loss_by_iter.append(mse(y,prediction))
                   
        return self
    
    def predict(self, X):
        
        # сначала прогноз – это просто вектор из средних значений ответов на обучении
        pred = np.ones([X.shape[0]]) * np.mean(self.y)
        # добавляем прогнозы деревьев с учётом learning_rate при обучении
        for t in range(self.n_estimators):
            pred += self.learning_rate * self.trees[t].predict(X).reshape([X.shape[0]])
            
        return pred
