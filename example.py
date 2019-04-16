# import packages test
from src.Garch import Garch
from src.Garch import vanilla_garch
import pandas as pd
import numpy as np

df = pd.read_csv("data/dem2gbp.csv")
r = df.DEM2GBP.values[:749] 

begVals = np.array([0.0, 0.045, .23, .64])
finfo = np.finfo(np.float64)
bounds = [(-10, 10), (finfo.eps, 2 * r.var()), (0.0, 1.0), (0.0, 1.0)]

# fit 1 Using SLSQP
fit1 = Garch(vanilla_garch).fit(r,begVals,'SLSQP', bounds = bounds)

# SLSQP Results
fit1.results

#SLSQP Estimated Parameters
fit1.params

# fit 2 Using Nelder-Mead (Which is default)
fit2 = Garch(vanilla_garch).fit(r, begVals, method = 'Nelder-Mead')

# Nelder-Mead results
fit2.results

# Nelder-Mead Estimated Parameters
fit2.params
