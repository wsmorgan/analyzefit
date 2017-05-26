"""Tests the manipulation of the data used in the regression model for the generation of plots.
"""

import pytest
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from analyzefit.analyze import analysis

X = np.array([[  1.62864000e+00,   0.00000000e+00,   2.18900000e+01,
                 0.00000000e+00,   6.24000000e-01,   5.01900000e+00,
                 1.00000000e+02,   1.43940000e+00,   4.00000000e+00,
                 4.37000000e+02,   2.12000000e+01,   3.96900000e+02,
                 3.44100000e+01],
              [  1.14600000e-01,   2.00000000e+01,   6.96000000e+00,
                 0.00000000e+00,   4.64000000e-01,   6.53800000e+00,
                 5.87000000e+01,   3.91750000e+00,   3.00000000e+00,
                 2.23000000e+02,   1.86000000e+01,   3.94960000e+02,
                 7.73000000e+00],
              [  5.57780000e-01,   0.00000000e+00,   2.18900000e+01,
                 0.00000000e+00,   6.24000000e-01,   6.33500000e+00,
                 9.82000000e+01,   2.11070000e+00,   4.00000000e+00,
                 4.37000000e+02,   2.12000000e+01,   3.94670000e+02,
                 1.69600000e+01],
              [  6.46600000e-02,   7.00000000e+01,   2.24000000e+00,
                 0.00000000e+00,   4.00000000e-01,   6.34500000e+00,
                 2.01000000e+01,   7.82780000e+00,   5.00000000e+00,
                 3.58000000e+02,   1.48000000e+01,   3.68240000e+02,
                 4.97000000e+00],
              [  9.29900000e-02,   0.00000000e+00,   2.56500000e+01,
                 0.00000000e+00,   5.81000000e-01,   5.96100000e+00,
                 9.29000000e+01,   2.08690000e+00,   2.00000000e+00,
                 1.88000000e+02,   1.91000000e+01,   3.78090000e+02,
                 1.79300000e+01],
              [  1.23247000e+00,   0.00000000e+00,   8.14000000e+00,
                 0.00000000e+00,   5.38000000e-01,   6.14200000e+00,
                 9.17000000e+01,   3.97690000e+00,   4.00000000e+00,
                 3.07000000e+02,   2.10000000e+01,   3.96900000e+02,
                 1.87200000e+01],
              [  1.35540000e-01,   1.25000000e+01,   6.07000000e+00,
                 0.00000000e+00,   4.09000000e-01,   5.59400000e+00,
                 3.68000000e+01,   6.49800000e+00,   4.00000000e+00,
                 3.45000000e+02,   1.89000000e+01,   3.96900000e+02,
                 1.30900000e+01],
              [  1.25179000e+00,   0.00000000e+00,   8.14000000e+00,
                 0.00000000e+00,   5.38000000e-01,   5.57000000e+00,
                 9.81000000e+01,   3.79790000e+00,   4.00000000e+00,
                 3.07000000e+02,   2.10000000e+01,   3.76570000e+02,
                 2.10200000e+01],
              [  1.51772000e+01,   0.00000000e+00,   1.81000000e+01,
                 0.00000000e+00,   7.40000000e-01,   6.15200000e+00,
                 1.00000000e+02,   1.91420000e+00,   2.40000000e+01,
                 6.66000000e+02,   2.02000000e+01,   9.32000000e+00,
                 2.64500000e+01],
              [  6.37960000e-01,   0.00000000e+00,   8.14000000e+00,
                 0.00000000e+00,   5.38000000e-01,   6.09600000e+00,
                 8.45000000e+01,   4.46190000e+00,   4.00000000e+00,
                 3.07000000e+02,   2.10000000e+01,   3.80020000e+02,
                 1.02600000e+01]])

y = np.array([14.4,  24.4,  18.1,  22.5,  20.5,  15.2,  17.4,  13.6,   8.7,  18.2])

slr = LinearRegression()
slr.fit(X,y)

an = analysis(X, y, slr, testing=True)

def test_Validate():
    """Test that Validate function works correctly"""

    accuracy = an.validate(testing=True)

    val = mean_squared_error(y, slr.predict(X))

    assert np.allclose(accuracy,val)

    accuracy = an.validate(testing=True, X=X, y=y, metric=mean_squared_error)
    
    assert np.allclose(accuracy,val)

    accuracy = an.validate(testing=True, metric=[mean_squared_error, r2_score])
    val = [mean_squared_error(y, slr.predict(X)), r2_score(y, slr.predict(X))]
    
    assert np.allclose(accuracy,val)
    
    with pytest.raises(ValueError):
        an.validate(X=[1,2,3])

def test_init():
    """Test that the class gets initialized correctly."""

    from analyzefit.analyze import analysis

    an2 = analysis(X, y, slr, predict = "predict", testing=True)

    with pytest.raises(AttributeError):
        an3 = analysis(X, y, slr, predict="Stuff", testing=True)
    
def test_res_fit():
    """Tests that the res_vs_fit method returns an object."""

    assert an.res_vs_fit(interact=False, show=False) is not None
    assert an.res_vs_fit(show=False, y=list(y), X=X) is not None

def test_quantile():
    """Tests that the quantile method returns an object."""

    assert an.quantile(interact=False, show=False) is not None
    assert an.quantile(dist=[1,2,3,4,5,6,7,8,9,10], show=False) is not None
    assert an.quantile(dist="uniform", show=False) is not None

    with pytest.raises(ValueError):
        an.quantile(dist=[1], show=False)

def test_spread_loc():
    """Tests that the spread_loc method returns an object."""

    assert an.spread_loc(interact=False,show=False) is not None
    assert an.spread_loc(show=False) is not None
    with pytest.raises(ValueError):
        an.spread_loc(X=[1,2,3])
    with pytest.raises(ValueError):
        an.spread_loc(pred=[1,2,3], y=[2])
    
def test_leverage():
    """Tests that the spread_loc method returns an object."""

    assert an.leverage(interact=False, show=False) is not None
    assert an.leverage(X=X.tolist(), y=y, show=False) is not None
    
