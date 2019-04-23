import numpy as np
import pandas as pd
import seaborn
from numpy import size, log, pi, sum, diff, array, zeros, diag, mat, asarray, sqrt, copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import scipy.optimize as opt

class Garch(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.__model = model

    @property
    def model(self):
        return self.__model
    
    
    def fit(self, X, begVals, method = "Nelder-Mead", jac=None, hess=None, hessp=None, bounds=None,
            constraints=(), tol=None, callback=None, options=None):
        
        self.X_ = X
        self.begVals = begVals
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
        
        self.__model = self.__model(self.begVals, self.X_)
        minimization = self.__model.minimize(self.method, self.jac, self.hess, self.hessp, 
                                             self.bounds, self.constraints, self.tol, 
                                             self.callback, self.options)
        
        self.output = GarchOutput(minimization)

        return self.output
    
    
    def predict(self, steps = 1):
    # Ensure that instance has been fitted
        check_is_fitted(self, "X_")
        self.steps = steps
        est_params = self.output.params
        
        prediction = self.__model.forecast(self.steps, est_params)
        
        return prediction
    
class GarchOutput():
    def __init__(self, results):
        self.__results = results
        self.__params = results["x"]
        
    @property
    def results(self):
        return self.__results
    @property
    def params(self):
        return self.__params

    
### Add properties to get additional elements of the opt.minimize output
### When different types of garch are used, how will this effect the parameter outputs?
### Raise errors if fails to converge?


### Different Types of Garch Models

class vanilla_garch():
    def __init__(self, params, data):
        self.__params = params
        self.__data = data
    
    @property
    def params(self):
        return self.__params
    @property
    def data(self):
        return self.__data
    

    def likelihood(self, params, data, sigma_last = False):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        beta = params[3]

        T = size(data, 0)
        eps = data - mu

        sigma2 = np.empty(T)

        sigma2[0] = omega/(1 - alpha - beta)

        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]

        lls = 0.5 * (log(2 * pi) + log(sigma2) + eps**2/sigma2)
        ll = sum(lls)

        if sigma_last is True:
            results = [sigma2[-1], eps[-1]]
        else:
            results = ll


        return results
    
    def minimize(self, method, jac=None, hess=None, hessp=None, bounds=None, constraints=(), 
                 tol=None, callback=None, options=None):
        
        results = opt.minimize(fun = self.likelihood, x0 = self.__params, args = self.__data, method = method,
                               jac = jac, hess = hess, hessp = hessp, bounds = bounds, constraints = constraints, tol = tol, 
                               callback = callback, options = options)
        return results
    

    def forecast(self, steps, est_params):
        est_mu = est_params[0]
        est_omega = est_params[1]
        est_alpha = est_params[2]
        est_beta = est_params[3]
        init_sigma2, init_eps = self.likelihood(self.__params, self.__data, sigma_last = True)
        
        
        forecast_values = np.empty(steps)
        forecast_values[0] = est_omega + est_alpha * init_eps**2 + est_beta * init_sigma2
        for t in range(1,steps):
            forecast_values[t] = est_omega + forecast_values[t-1] * (est_alpha + est_beta)
        
        return forecast_values
        
        
class gjr_garch():
    def __init__(self, params, data,sigma_last = False):
        self.__params = params
        self.__data = data
        self.sigma_last = sigma_last
        
    @property
    def params(self):
        return self.__params
    @property
    def data(self):
        return self.__data
        
    def likelihood(self, params, data, sigma_last = False):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        beta = params[3]
        gamma = params[4]

        T = size(data, 0)
        eps = data - mu

        sigma2 = np.empty(T)

        sigma2[0] = omega/(1 - alpha - beta - 0.5 * gamma)

        for t in range(1, T):
            if eps[t-1] >= 0:
                sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            else:
                sigma2[t] = omega + alpha * eps[t-1]**2 + gamma * eps[t-1]**2 + beta * sigma2[t-1] 
        

        lls = 0.5 * (log(2 * pi) + log(sigma2) + eps**2/sigma2)
        ll = sum(lls)

        if sigma_last is True:
            results = [sigma2[-1], eps[-1]]
        else:
            results = ll

        return results
    
    def minimize(self, method, jac=None, hess=None, hessp=None, bounds=None, constraints=(), 
                 tol=None, callback=None, options=None):
        results = opt.minimize(fun = self.likelihood, x0 = self.__params, args = self.__data, method = method,
                               jac = jac, hess = hess, hessp = hessp, bounds = bounds, constraints = constraints, tol = tol, 
                               callback = callback, options = options)
        return results
    
### Check with Dr. Brough    
    def forecast(self, steps, est_params):
        est_mu = est_params[0]
        est_omega = est_params[1]
        est_alpha = est_params[2]
        est_beta = est_params[3]
        est_gamma = est_params[4]
        init_sigma2, init_eps = self.likelihood(self.__params, self.__data, sigma_last = True)
        
        
        forecast_values = np.empty(steps)
        if init_eps >= 0:
            forecast_values[0] = est_omega + est_alpha * init_eps**2 + est_beta * init_sigma2
        else:
            forecast_values[0] = est_omega + (est_alpha + est_gamma) * init_eps**2 + est_beta * init_sigma2
            
        for t in range(1,steps):
            forecast_values[t] = est_omega + forecast_values[t-1] * (est_alpha + est_beta + 0.5*est_gamma)
        
        return forecast_values