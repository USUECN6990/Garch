# import packages test
from src.Garch import Garch
from src.Garch import vanilla_garch, gjr_garch
import pandas as pd
import numpy as np

df = pd.read_csv("data/dem2gbp.csv")
r = df.DEM2GBP.values[:749] 

begVals = np.array([0.0, 0.045, .23, .64])
finfo = np.finfo(np.float64)
bounds = [(-10, 10), (finfo.eps, 2 * r.var()), (0.0, 1.0), (0.0, 1.0)]


# Examples using "vanilla_garch" class -----------------------------------

obj1 = Garch(vanilla_garch) # instantiate object

# fit 1 Using SLSQP
obj1.fit(r,begVals,'SLSQP', bounds = bounds)

# SLSQP Results
print(obj1.results) # results from minimize function (more verbose than .params)

#SLSQP Estimated Parameters
print(obj1.params) # fitted parameters

# predict sigma squared
obj1.predict(steps = 10) # predicting 10 time steps into the future

# fit 2 Using Nelder-Mead (Which is default)
obj2 = Garch(vanilla_garch)

# fit
obj2.fit(r, begVals, method = 'Nelder-Mead')

# Nelder-Mead results
print(obj2.results)

# Nelder-Mead Estimated Parameters
print(obj2.params)

# prediction
print(obj2.predict(steps = 10))

# Examples using "gjr_garch" class ----------------------------------------
begVals2 = np.array([0.0, 0.045, .23, .64, 0.01]) 

obj_gjr = Garch(gjr_garch)

# fit model
obj_gjr.fit(r,begVals2,method = 'Nelder-Mead')

# examine results from fitting
print(obj_gjr.results)
print(obj_gjr.params)

# make predictions
obj_gjr.predict(steps = 10)
