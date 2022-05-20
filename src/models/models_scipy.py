import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import NearestNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

def sign_log_transform(x):
    return np.sign(x) * np.log(np.abs(x))

def sign_exp_transform(x):
    return np.sign(x) * np.exp(-np.abs(x))



class GaussianProcessRegressorModel:

    def __init__(self, kernel) -> None:
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8)
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

class RBFInterpolatorModel:

    def __init__(self, degree=1) -> None:
        self.x_scaler = StandardScaler()
        self.degree = degree
        
    def fit(self, x_train, y_train):
        x_train = self.x_scaler.fit_transform(x_train)
        self.model = RBFInterpolator(x_train, 
                      y_train,
                      degree=self.degree,
                      smoothing=0.000001)

    def predict(self, x):
        x = self.x_scaler.transform(x)
        return self.model(x)


class NearestNDInterpolatorModel:

    def fit(self, x_train, y_train):
        self.model = NearestNDInterpolator(x_train, y_train)

    def predict(self, x_test):
        return self.model(x_test)






