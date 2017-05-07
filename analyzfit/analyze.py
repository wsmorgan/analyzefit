"""The main class for the analysis of a given fit."""
import mpld3
import numpy as np
import matplotlib.pyplot as plt
from plotting import scatter_with_hover, scatter

class analysis(object):
    """The main class for the analysis of a given fit.

    Args:
        model (object): The fitting model (the model must have a predict method).
        X (numpy ndarray): The X valuse to be used for plots.
        y (numpy array): The y values to be used for plots.
        predict (str): The name of the method that is equivalent to the 
            sklearn predict function. Default = 'predict'.

    """

    def __init__(self, X,y,model, predict=None):
        self.X = X
        self.y = y
        if predict ==None:
            pred = getattr(model,"predict",None)
        else:
            pred = getattr(model,predict,None)
        if callable(pred):
            self.model = model
        else:
            raise AttributeError("The fitting model must have a callable method "
                                 "for making predictions.")
        self.predictions = self.model.predict(self.X)
        self._run_from_ipython()

    def _run_from_ipython(self):
        try:
            __IPYTHON__
            self._in_ipython  = True
        except NameError:
            self._in_ipython = False
    
    def res_vs_fit(self,X=None,y=None,interact=True,show=True):
        """Makes the residual vs fitted values plot.
        
        Args:
            X (optional, numpy ndarray): The dataset to make the plot for
                if different than the dataset used to initialize the method.
            y (optional, numpy array): The target values to make the plot for
                if different than the dataset used to initialize the method.
            interact (optional, bool): True if the plot is to be interactive.
        """
        if X !=None and y!=None:
            pred = self.model.predict(X)
        elif X!=None or y!=None:
            raise ValueError("In order to make a plot for a diferent data set "
                             "than the set initially passed to the function "
                             "both fitting data (X) and target data (y) must "
                             "be provided.")
        else:
            pred = self.predictions
            y = self.y

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        res = y-pred

        if interact:
            if show:
                scatter_with_hover(pred,res,in_notebook=self._in_ipython,title="Residues vs Predictions",
                                   x_label="Predictions",y_label="Residues")
            else:
                return scatter_with_hover(pred,res,in_notebook=self._in_ipython,title="Residues vs Predictions",
                                          x_label="Predictions",y_label="Residues")

        else:
            if show:
                scatter(pred,res,title="Residues vs Predictions",x_label="Predictions",y_label="Residues")
            else:
                return scatter(pred,res,title="Residues vs Predictions",x_label="Predictions",y_label="Residues", show=show)
                
