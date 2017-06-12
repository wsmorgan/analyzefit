import matplotlib
import os

if os.system != "nt": #pragma: no cover
    matplotlib.use("Agg")


from analyzefit.analyze import Analysis
