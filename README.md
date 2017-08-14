[![PyPI](https://img.shields.io/pypi/v/analyzefit.svg)](https://pypi.python.org/pypi/analyzefit/)[![Build Status](https://travis-ci.org/wsmorgan/analyzefit.svg?branch=master)](https://travis-ci.org/wsmorgan/analyzefit)[![codecov](https://codecov.io/gh/wsmorgan/analyzefit/branch/master/graph/badge.svg)](https://codecov.io/gh/wsmorgan/analyzefit)[![Code Health](https://landscape.io/github/wsmorgan/analyzefit/master/landscape.svg?style=flat)](https://landscape.io/github/wsmorgan/analyzefit/master)

# analyzefit

Analyze fit is a python package that performs standard analysis on the
fit of a regression model. The analysis class validate method will
create a residuals vs fitted plot, a quantile plot, a spread location
plot, and a leverage plot for the model provided as well as print the
accuracy scores for any metric the user likes. For example:

![alt_text](../master/support/images/validation.png)

If a detailed plot is desired then the plots can also be generated
individually using the methods res_vs_fit, quantile, spread_loc, and
leverage respectively. By default when the plots are created
individually they are rendered in an interactive inverontment using
the bokeh plotting package. For example:

![alt text](../master/support/images/interactive.pdf)

This allows the user to determine which points the model is failing to
predict.

Full API Documentation available at: [github pages](https://wsmorgan.github.io/analysefit/).

## Installing the code

To install analyzefit you may either pip install:

```
pip install analyzefit
```

or clone this repository and install manually:

```
python setup.py install
```

# Validating a Model

To use analyze fit simply pass the feature matrix, target values, and
the model to the analysis class then call the validate method, (or any
other plotting method). For example:

```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from analyzefit import Analysis

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
X = df.iloc[:,:-1].values
y = df[["MEDV"]].values
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(X_train,y_train)

an = Analysis(X_train, y_train, slr)
an.validate()

an.validate(X=X_test, y=y_test, metric=[mean_squared_error, r2_score])

an.res_vs_fit()

an.quantile()

an.spread_loc()

an.leverage()
```

## Python Packages Used

- numpy

- matplotlib

- bokeh

- sklearn
