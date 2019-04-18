# GARCH

Final Group Project for ECN6990--Implementing the GARCH models using python. 

## Overview:

We created a `GARCH` class. To instantiate an object of this class we use classes of specific types of garch models (described below). The instantiated object can be used to fit the model and make predictions. Thus the `GARCH` class is loosely based on classes that are used in the `sklearn` pipeline, and has both `fit()` and `predict` methods. 

See example.py for example code. 

## Description of classes used when instantiating GARCH object: 

The `GARCH` class must take a specific type of GARCH model class as an argument in order to instantiate. We have created two kinds of GARCH model classes to use (`vanilla_garch` and `gjr_garch`, described below). However, the `GARCH` class was designed to be flexible and  additional classes of specific types of GARCH models can be created and used with it. 

If a new class for a type of GARCH model is developed in must contain `likelihood`, `minimize` and `forecast` methods. 
The `likelihood`, method needs to include a argument called "sigma_last" (default to false). The purpose of this argument is that if false the `likelihood` method returns a number (the log likelihood) or if true it returns estimated &sigma;<sup>2</sup> and &epsilon;<sub>t-1</sub>

### vanilla_garch

&sigma;<sup>2</sup><sub>t</sub> = &omega; + &alpha;&epsilon;<sub>t-1</sub><sup>2</sup> + &beta;&sigma;<sup>2</sup><sub>t-1</sub>

Alexander, C. (2008). Market risk analysis, practical financial econometrics (Vol. 2). John Wiley & Sons. (See page 136)

Sheppard, K. (2018). Introduction to python for econometrics, statistics and data analysis. Self-published, University of Oxford, version, 2. (See page 349)


### gjr_garch

&sigma;<sup>2</sup><sub>t</sub> = &omega; + &alpha;&epsilon;<sup>2</sup><sub>t-1</sub> + &gamma;&epsilon;<sup>2</sup><sub>t-1</sub>I<sub>(&epsilon;<sub>t-1</sub> < 0)</sub> + &beta;&sigma;<sup>2</sup><sub>t-1</sub>

Alexander, C. (2008). Market risk analysis, practical financial econometrics (Vol. 2). John Wiley & Sons. (See page 150)


## How to instantiate an object

...

## How to fit model

Description of arguments needed...

## Prediction

How model prediction works...


