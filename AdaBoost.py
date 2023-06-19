import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def compute_error(y, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier
    '''

    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)


def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier.
    (Referedd to as alpha in chapter 10.1 of The Elements of Statistical Learning)
    '''

    return np.log((1 - error) / error)


def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration.
    '''  
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))


class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.weak_learners = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, M=100):
        self.alphas = []
        self.training_errors = []
        self.M = M

        for m in range(0, M):
            if m==0:
                # initialize all weights to 1/N
                w_i = np.ones(len(y)) * 1/len(y)
            else:
                w_i = update_weights(w_i, alpha, y, y_pred)

            # Fit weak classifier and predict labels
            decision_tree = DecisionTreeClassifier(max_depth=1)     # Stump
            decision_tree.fit(X,
                              y,
                              sample_weight=w_i)
            y_pred = decision_tree.predict(X)

            self.weak_learners.append(decision_tree)

            # Compute error
            error = compute_error(y, y_pred, w_i)
            self.training_errors.append(error)

            # Compute alpha
            alpha = compute_alpha(error)
            self.alphas.append(alpha)


    def predict(self, X):
        weak_preds = pd.DataFrame(index=range(len(X)),
                                  columns=range(self.M))
        
        for m in range(self.M):
            y_pred_m = self.weak_learners[m].predict(X)*self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m

        y_pred = (1*np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
    
    def error_rates(self, X, y):
        self.prediction_errors = []

        for m in range(self.M):
            y_pred_m = self.weak_learners[m].predict(X)
            error = compute_error(y=y,
                                  y_pred=y_pred_m,
                                  w_i=np.ones(len(y)))
            self.prediction_errors.append(error)
