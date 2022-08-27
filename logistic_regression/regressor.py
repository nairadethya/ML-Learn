import numpy as np

class LogisticRegression:

    def __init__(self):
        self.coef_ = 1
        self.bias = 0
    
    def train(self, X, y, learning_rate=0.001):
        feature, num = X.shape
        self.coef_ = np.ones(feature)
        
        
        for i in range(100000):
            bias_update = (1/num) * np.sum(((np.dot(self.coef_, X) + self.bias) - y))
            weight_update = (1/num) * np.sum(((np.dot(self.coef_, X) + self.bias) - y) * X)
            self.bias = self.bias - learning_rate * bias_update
            self.coef_ = self.coef_ - learning_rate * weight_update
    
    def test(self,X, y):
        feature, num = X.shape
        y_predicted = (np.dot(self.coef_, X) + self.bias).reshape(3,1)
        
        print("Accuracy: " + str(np.sum(abs(y - y_predicted))/num))
        