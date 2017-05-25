# Revision History for "analyzefit"

## Revision 0.3.1
- Impletemnting unit tests.
- Fixed setup.py so pip install works.
- Fixed bug in Hat Matrix (forgot to add column of 1's) and Cook's
  Distance (number of features was off by one).
- Fixed bug where the passed in user supplied data wasn't getting used
  in the full visualization plots.
- Removed base.py and msg.py since neither is used.

## Revision 0.3.0
- Implemented the full validation function which makes all 4 plots and
  displays the accuracy metric.
- Improved documentation of methods used.
- Removed code duplication in several subroutines.

## Revision 0.2.0
- Implemented the Residuals vs Leverage plot.

## Revision 0.1.0
- Fixed the line in the quantile plot so that it is a linear fit of
  the data and the distribution.
- Initial implementation of the Spread-location plot added. Need to
  add line that splits data set in half.

## Revision 0.0.3
-Finished implementing Quantile plots. The dashed line going through
 them still needs some work though.

## Revision 0.0.2
- Started implementing Quantile plots.

## Revision 0.0.1
- Initial implementation of the residual vs fitted values plot.

## Revision 0.0.0

- Initial commit to repo.
