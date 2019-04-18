# GARCH
Testing <sub>subscript</sub> and <sup>superscript</sup>
Final Group Project for ECN6990--Implementing the GARCH models using python. 

## Overview:

We created a `GARCH` class. To instantiate an object of this class we use classes of specific types of garch models (described below). The instantiated object can be used to fit the model and make predictions. Thus the `GARCH` class is loosely based on classes that are used in the `sklearn` pipeline, and has both `fit()` and `predict` methods. 

See example.py for example code. 

## Description of classes used when instantiating GARCH object: 

The `GARCH` class must take a specific type of GARCH model class as an argument in order to instantiate. We have created two kinds of GARCH model classes to use (`vanilla_garch` and `gjr_garch`, described below). However, the `GARCH` class was designed to be flexible and  additional classes of specific types of GARCH models can be created and used with it. 

If a new class for a type of GARCH model is developed in must contain `likelihood`, `minimize` and `forecast` methods. 
The `likelihood`, method needs to include a argument called "sigma_last" (default to false). The purpose of this argument is that if false the `likelihood` method returns a number (the log likelihood) or if true it returns estimated sigma squared and &epsilon;<sub>t-1</sub>

### vanilla_garch

citation here

### gjr_garch

citation here

## How to instantiate an object

...

## How to fit model

Description of arguments needed...

## Prediction

How model prediction works...


