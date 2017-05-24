"""The main class for the analysis of a given fit."""

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

    def __init__(self, X, y, model, predict=None):
        self.X = X
        self.y = y
        if predict ==None:
            pred = getattr(model, "predict", None)
        else:
            pred = getattr(model, predict, None)
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
    
    def res_vs_fit(self, X=None, y=None, interact=True, show=True, ax=None, title=None):
        """Makes the residual vs fitted values plot.
        
        Args:
            X (optional, numpy.ndarray): The dataset to make the plot for
                if different than the dataset used to initialize the method.
            y (optional, numpy.array): The target values to make the plot for
                if different than the dataset used to initialize the method.
            interact (optional, bool): True if the plot is to be interactive.
            show (optional, bool): True if plot is to be displayed.
            ax (optional, matplotlib axis object): The subplot on which to 
                drow the plot.
            title (optional, str): The title of the plot.
        """
        from manipulate import residual
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

        if title is None:
            title = "Residue vs Predictions"

        res = residual(y,pred)

        if interact:
            from bokeh.plotting import figure
            from bokeh.plotting import show as show_fig
            from bokeh.models import HoverTool

            hover = HoverTool(tooltips=[("entry#", "@label"),])
            fig = figure(tools=['box_zoom', 'reset', hover], title=title,
                         x_range=[min(pred), max(pred)], y_range=[min(res), max(res)])

            fig = scatter_with_hover(pred, res, in_notebook=self._in_ipython,
                                     x_label="Predictions", y_label="Residues",
                                     show_plt=False, fig=fig)
            if show:
                show_fig(fig)
            else:
                return fig
            
        else:
            if show:
                scatter(pred, res, title=title, x_label="Predictions",
                        y_label="Residues")                    
            else:
                if ax is None:
                    return scatter(pred, res, title=title,
                                   x_label="Predictions", y_label="Residues", show_plt=False)
                        
                else:
                    return scatter(pred, res, title=title,
                                   x_label="Predictions", y_label="Residues",
                                   show_plt=False, ax=ax)

    def quantile(self, data=None, dist=None, interact=True, show=True, title=None, ax=None):
        """Makes a quantile plot of the predictions against the desired distribution.

        Args:
            data (optional numpy.array, optional): The user supplied data for the quantile plot.
                If None then the model predictions will be used.
            dist (optional str or numpy.array): The distribution to be compared to. Either 
                'Normal', 'Uniform', or a numpy array of the user defined distribution.
            interact (optional, bool): True if the plot is to be interactive.
            show (option, bool): True if plot is to be displayed.
            ax (optional, matplotlib axis object): The subplot on which to 
                drow the plot.
            title (optional, str): The title of the plot.
        """

        if data == None:
            data = sorted(self.predictions)
        else:
            data = sorted(data)

        if dist is None or str(dist).lower() == "normal":
            dist = sorted(np.random.normal(np.mean(data),np.std(data),len(data)))
        elif str(dist).lower() == "uniform":
            dist = sorted(np.random.uniform(min(data),max(data),len(data)))
        else:
            if not type(dist) is np.ndarray:
                raise ValueError("The user provided distribution must be a numpy array")
            elif len(dist) != len(data):
                raise ValueError("The user provided distribution must have the same "
                                 "size as the input data or number of predictions.")
            else:
                dist = sorted(dist)

        if title is None:
            title = "Quantile"
        
        if interact:
            from bokeh.plotting import figure
            from bokeh.plotting import show as show_fig
            from bokeh.models import HoverTool

            hover = HoverTool(tooltips=[("entry#", "@label"),])
            fig = figure(tools=['box_zoom', 'reset',hover],title=title,
                         x_range=[min(dist), max(dist)],y_range=[min(data), max(data)])
            
            fig = scatter_with_hover(dist, data, in_notebook=self._in_ipython,
                                    fig=fig, x_label="Distribution",
                                     y_label="Predictions", show_plt=False)
            fig.line(dist, np.poly1d(np.polyfit(dist, data, 1))(dist),
                     line_dash="dashed", line_width=2)
            if show:
                show_fig(fig)
            else:
                return fig

        else:
            if ax is None:
                fig = scatter(dist, data, title=title, x_label="Distribution",
                              y_label="Predictions", show_plt=False)            
                plt.plot(dist, np.poly1d(np.polyfit(dist, data,1))(dist),
                         c="k", linestyle="--", lw=2)
                plt.xlim([min(dist), max(dist)])
                plt.ylim([min(data), max(data)])
            else:
                fig = scatter(dist, data, title=title, x_label="Distribution",
                              y_label="Predictions", show_plt=False,ax=ax)            
                fig.plot(dist, np.poly1d(np.polyfit(dist, data,1))(dist),
                         c="k", linestyle="--", lw=2)
                fig.set_xlim([min(dist), max(dist)])
                fig.set_ylim([min(data), max(data)])
                
            if show:
                plt.show()
            else:
                return fig

    def Spread_Loc(self,X=None,y=None,interact=True,show=True,title=None,ax=None):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (optional, numpy ndarray): The dataset to make the plot for
                if different than the dataset used to initialize the method.
            y (optional, numpy array): The target values to make the plot for
                if different than the dataset used to initialize the method.
            interact (optional, bool): True if the plot is to be interactive.
            show (option, bool): True if plot is to be displayed.
            ax (optional, matplotlib axis object): The subplot on which to 
                drow the plot.
            title (optional, str): The title of the plot.
        """

        from manipulate import std_residuals

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

        if title is None:
            title = "Spread-Location"

        root_stres = np.sqrt(np.absolute(std_residuals(y,pred)))

        if interact:
            from bokeh.plotting import figure
            from bokeh.plotting import show as show_fig
            from bokeh.models import HoverTool

            hover = HoverTool(tooltips=[("entry#", "@label"),])
            fig = figure(tools=['box_zoom', 'reset', hover], title=title,
                         x_range=[min(pred), max(pred)],
                         y_range=[min(root_stres), max(root_stres)])
            
            fig = scatter_with_hover(pred, root_stres, in_notebook=self._in_ipython,
                                     fig=fig, x_label="Fitted Values",
                                     y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False)

            if show:
                show_fig(fig)
            else:
                return fig

        else:
            if ax is None:
                fig = scatter(pred, root_stres, title=title,
                              x_label="Fitted Values",
                              y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False)

                plt.xlim([min(pred), max(pred)])
                plt.ylim([min(root_stres), max(root_stres)])
            else:
                fig = scatter(pred, root_stres, title=title,
                              x_label="Fitted Values",
                              y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False,ax=ax)

                fig.set_xlim([min(pred), max(pred)])
                fig.set_ylim([min(root_stres), max(root_stres)])
                
            if show:
                plt.show()
            else:
                return fig

    def Leverage(self,X=None,y=None,interact=True,show=True,title=None, ax=None):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (optional, numpy ndarray): The dataset to make the plot for
                if different than the dataset used to initialize the method.
            y (optional, numpy array): The target values to make the plot for
                if different than the dataset used to initialize the method.
            interact (optional, bool): True if the plot is to be interactive.
            show (option, bool): True if plot is to be displayed.
            ax (optional, matplotlib axis object): The subplot on which to 
                drow the plot.
            title (optional, str): The title of the plot.
        """

        from manipulate import std_residuals, cooks_dist, hat_diags

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
            X = self.X

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if title is None:
            title = "Residual vs Leverage"

        stres = std_residuals(y,pred)

        c_dist = cooks_dist(y,pred,X)
        h_diags = hat_diags(X)
        
        if interact:
            from bokeh.plotting import figure
            from bokeh.plotting import show as show_fig
            from bokeh.models import HoverTool

            hover = HoverTool(tooltips=[("entry#", "@label"),])
            fig = figure(tools=['box_zoom', 'reset', hover], title=title,
                         x_range=[min([min(c_dist), min(h_diags)]),
                                  max([max(c_dist), max(h_diags)])],
                         y_range=[min(stres), max(stres)])

            fig = scatter_with_hover(c_dist, stres, in_notebook=self._in_ipython,
                                     show_plt=False, name="cooks distance")

            fig = scatter_with_hover(h_diags, stres, in_notebook=self._in_ipython,
                                     fig=fig, y_label="Standardized Residuals",
                                     show_plt=False, color="yellow", name="influence")
            if show:
                show_fig(fig)
            else:
                return fig

        else:
            if ax is None:
                fig = plt.figure()
                fig = scatter(c_dist, stres, title="Residual vs Leverage plot",
                              y_label="Standardized Residuals", label="Cooks Distance",
                              show_plt=False, fig=fig)            
                fig = scatter(h_diags, stres, title="Residual vs Leverage plot",
                              label="Influence", show_plt=False, fig=fig)
                plt.xlim([min([min(c_dist), min(h_diags)]), max([max(c_dist), max(h_diags)])])
                plt.ylim([min(stres), max(stres)])
            else:
                fig = scatter(c_dist, stres, title="Residual vs Leverage plot",
                              y_label="Standardized Residuals", label="Cooks Distance",
                              show_plt=False, ax=ax)            
                fig = scatter(h_diags, stres, title="Residual vs Leverage plot",
                              label="Influence", show_plt=False, ax=ax)
                fig.set_xlim([min([min(c_dist), min(h_diags)]),
                              max([max(c_dist), max(h_diags)])])
                fig.set_ylim([min(stres), max(stres)])
                
            plt.legend()
            if show:
                plt.show()
            else:
                return fig

    def Validate(self,X=None,y=None,pred=None,metric=None):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (optional, numpy ndarray): The dataset to make the plot for
                if different than the dataset used to initialize the method.
            y (optional, numpy array): The target values to make the plot for
                if different than the dataset used to initialize the method.
            pred (optional, numpy array): The predicted values to make the plot for
                if y and X are different than the dataset used to initialize the method.
            metric (option, function or list of functions): The functions used to
                determine how accurate the fit is.
        """

        if not X is None and not y is None and pred is None:
            pred = self.model.predict(X)
        elif not X is None or not y is None:
            raise ValueError("In order to make a plot for a diferent data set "
                             "than the set initially passed to the function "
                             "both fitting data (X) and target data (y) must "
                             "be provided.")
        else:
            pred = self.predictions
            y = self.y
            X = self.X


        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        ax1 = self.res_vs_fit(show=False, interact=False, ax=ax1)
        ax2 = self.quantile(show=False, interact=False, ax=ax2)
        ax3 = self.Spread_Loc(show=False, interact=False, ax=ax3)
        ax4 = self.Leverage(show=False, interact=False, ax=ax4)
        
        fig.tight_layout()
        plt.show()

        if metric is None:
            from sklearn.metrics import mean_squared_error
            print("Prediction error for {1} metric: {0:.2f} ".format(mean_squared_error(y, pred),
                                                                     "mean squared error"))
        else:
            if isinstance(metric,list):
                for m in metric:
                    print("Prediction error for {1} metric: {0:.2f} ".format(m(y, pred),
                                                                             m.__name__))
            else:
                print("Prediction error for {1} metric: {0:.2f} ".format(metric(y, pred),
                                                                         metric.__name__))
