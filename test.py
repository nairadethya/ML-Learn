from logistic_regression import LogisticRegression
import numpy as np

# from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    X = np.array([[1,2,3],[2,3,4],[4,5,6]])
    y = np.array([[1],[2],[1]]).reshape(3,1)
 
    l = LogisticRegression()
    l.train(X,y)
    print(l.coef_)
    l.test(X,y)

    # lr = LogisticRegression()
    # lr.fit(X,y)
    # print(lr.coef_)