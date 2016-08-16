# BayesianOptimization

A reference class implementation of Bayesian optimization of gaussian process built in R on top of GPfit. Bayesian optimization is a method to find the global maximum (or minimum) of expensive cost functions. This implemention uses the Expected Improvement acquisition function and is capable of handling noisy data when using the nug.thres parameter.

Also includes hyperparameter tuning of an extreme gradient boosting machine used in a Kaggle competition.

An example of the two dimensional plot created by the reference class is included below.

![](http://i.imgur.com/RFAdKfC.gif)
