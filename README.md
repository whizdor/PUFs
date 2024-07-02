# Companion Arbiter PUFs.

This repository contains the assignments for the course CS771: Introduction to Machine Learning, taken by Prof. Purushottam Kar, during 2024 Semester - II. 

## Overview

The assignment focuses on developing a linear model for a Companion Arbiter PUF (Physically Unclonable Function) problem and experimenting with various machine learning models and hyperparameters.

## Contents

1. Mathematical derivation of a linear model for the Companion Arbiter PUF problem
2. Python implementation of the model
3. Experimental analysis of different models and hyperparameters

## Implementation Details

The main implementation is in Python, using the following libraries:
- NumPy
- scikit-learn
- SciPy (specifically, the khatri_rao function)

The core functions implemented are:
- `my_fit`: Trains the model on the given data
- `my_map`: Maps the input challenges to the feature space

## Models Explored

1. LinearSVC
2. Logistic Regression
3. Ridge Regression

## Hyperparameter Analysis

The project includes an extensive analysis of various hyperparameters, including:
- Loss function (hinge vs squared hinge)
- Regularization parameter (C)
- Tolerance (tol)
- Penalty (l1 vs l2)

## Results

The best performance was achieved using Logistic Regression with the following hyperparameters:
- C: 125
- solver: liblinear
- penalty: l2
- tolerance: 0.0025

This configuration achieved an accuracy of 0.9935.

## Authors

This assignment has been done by:

- Aditi Khandelia
- Kushagra Srivastava
- Mahaarajan J
- Ruthvik Tunuguntala
