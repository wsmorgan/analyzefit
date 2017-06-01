import matplotlib
import os

if os.system != "nt": #pragma: no cover
    matplotlib.use("Agg")
