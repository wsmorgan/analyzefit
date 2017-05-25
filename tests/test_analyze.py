"""Tests the manipulation of the data used in the regression model for the generation of plots.
"""

import pytest
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def test_Validate():
    """Test that Validate function works correctly"""

    from analyzefit.analyze import analysis

    X = np.random.random_sample((10,3))*5
    y = np.random.random_sample((10,1))

    slr = LinearRegression()
    slr.fit(X,y)

    an = analysis(X,y,slr)

    accuracy = an.Validate(testing=True)

    val = mean_squared_error(y,slr.predict(X))

    assert np.allclose(accuracy,val)

    accuracy = an.Validate(testing=True,X=X,y=y,metric=mean_squared_error)
    
    assert np.allclose(accuracy,val)

    accuracy = an.Validate(testing=True,metric=[mean_squared_error,r2_score])
    val = [mean_squared_error(y,slr.predict(X)),r2_score(y,slr.predict(X))]
    
    assert np.allclose(accuracy,val)
    
    with pytest.raises(ValueError):
        an.Validate(X=[1,2,3])

def test_init():
    """Test that the class gets initialized correctly."""

    from analyzefit.analyze import analysis

    X = np.random.random_sample((10,3))*5
    y = np.random.random_sample((10,1))

    slr = LinearRegression()
    slr.fit(X,y)

    an = analysis(X,y,slr, predict = "predict")

    with pytest.raises(AttributeError):
        an = analysis(X,y,slr,predict="Stuff")
    
