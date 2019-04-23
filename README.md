# GARCH

Final Group Project for ECN6990--Implementing the GARCH models using python. 

## Overview:

To instantiate an object of this class, classes of specific types of garch models (described below) were used. The instantiated object can be used to fit the model and make predictions. Thus the `GARCH` class is loosely based on classes that are used in the `sklearn` pipeline, and has both `fit()` and `predict` methods. 

See example.py for example code. 

## Description of classes used when instantiating GARCH object: 

The `GARCH` class must take a specific type of GARCH model class as an argument in order to instantiate. Two kinds of GARCH model classes were created (`vanilla_garch` and `gjr_garch`, described below). However, the `GARCH` class was designed to be flexible and  additional classes of specific types of GARCH models can be created and used with it. 

If a new class for a type of GARCH model is developed it must contain `likelihood()`, `minimize()`, and `forecast()` methods. 
The `likelihood()`, method needs to include a argument called "sigma_last" (default is false). The purpose of this argument is that if false, the `likelihood()` method returns a number (the log likelihood), or if true, it returns estimated values of &sigma;<sup>2</sup> and &epsilon;<sub>t-1</sub>.

### vanilla_garch

&sigma;<sup>2</sup><sub>t</sub> = &omega; + &alpha;&epsilon;<sub>t-1</sub><sup>2</sup> + &beta;&sigma;<sup>2</sup><sub>t-1</sub>

Alexander, C. (2008). Market risk analysis, practical financial econometrics (Vol. 2). John Wiley & Sons. (See page 136)

Sheppard, K. (2018). Introduction to python for econometrics, statistics and data analysis. Self-published, University of Oxford, version, 2. (See page 349)


### gjr_garch

&sigma;<sup>2</sup><sub>t</sub> = &omega; + &alpha;&epsilon;<sup>2</sup><sub>t-1</sub> + &gamma;&epsilon;<sup>2</sup><sub>t-1</sub>I<sub>(&epsilon;<sub>t-1</sub> < 0)</sub> + &beta;&sigma;<sup>2</sup><sub>t-1</sub>

Alexander, C. (2008). Market risk analysis, practical financial econometrics (Vol. 2). John Wiley & Sons. (See page 150)

## How to instantiate an object

Create an object with the chosen GARCH parameter inside. 

Example: 
`obj1 = Garch(vanilla_garch) `
This will instantiate the object for use of type vanilla garch. 

## How to fit model

The `fit()` method was modeled after SciKit-Learn. 
The `fit()` method requires two arguments X (array of the data to which GARCH model is being fit) and begVals (initial parameter values needed by GARCH model, used as starting values in the minimize function). 'begVals' is an array of length four (&mu;, &omega;, &alpha;, &beta;) for `vanilla_garch` or five (&mu;, &omega;, &alpha;, &beta;, &gamma;) for `gjr_garch`. The method argument is the algorithm used to find optimized maximum likelihood parameters (default value is "Nelder-Mead"). Additional, optional arguments to be passed to the minimize function are allowed, see the scipy.optimize.minimize documentation. 

Example:

`obj1.fit(data, begVals = [`&mu;, &omega;, &alpha;, &beta;`])`

After running the `fit()` method, `obj1.params` will return the fitted parameters, and `obj1.results` returns the output for opt.minimize.


## Prediction

The `predict()` method takes one argument (steps), which is the number of time steps that &sigma;<sup>2</sup> is predicted for. 

Example: `obj.fit(steps = 5)`, predicting 5 time steps (returns array of length 5).




