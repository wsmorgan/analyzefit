"""The main class for the analysis of a given fit."""
import numpy as np
import matplotlib.pyplot as plt
from analyzefit.plotting import scatter_with_hover, scatter

class Analysis(object):
    """The main class for the analysis of a given fit.

    Args:
        model (object): The fitting model (the model must have a predict method).

        X (numpy.ndarray): The X valuse to be used for plots.

        predict (str): The name of the method that is equivalent to the sklearn predict 
          function. Default = 'predict'.

        y (numpy.ndarray): The y values to be used for plots.

    Attributes:
        validate (object): Creates a residual vs fitted plot, a quatile plot, a 
          spread vs location plot, and a leverage plot and prints the accuracy
          score to the screen.
        res_vs_fit (object): Creates a plot of the residuals vs the fittted 
          values in an interactive bokeh figure.
        quantile (object): Creates a quantile plot for the fitted values in an 
          interactive bokeh figure.
        spread_loc (object): Creates a plot of the spread in residuals vs the fitted 
          values in an interactive bokeh figure.
        leverage (object): Creates a plot of the cooks distance and the influence vs 
          the standardized residuals in an interactive bokeh figure.

    Examples:
        The following examples show how to validate the fit of sklearn's LinearRegression
        on the housing dataset. It shows how to generate each of the plots that can be used
        to verify the accuracy af a fit.

        >>>> import pandas as pd

        >>>> import numpy as np

        >>>> from sklearn.linear_model import LinearRegression

        >>>> from sklearn.cross_validation import train_test_split

        >>>> from sklearn.metrics import mean_squared_error, r2_score

        >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

        >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

        >>>> X = df.iloc[:,:-1].values

        >>>> y = df[["MEDV"]].values

        >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

        >>>>          test_size=0.3, random_state=0)

        >>>> slr = LinearRegression()

        >>>> slr.fit(X_train,y_train)

        >>>> an = analyze.analysis(X_train, y_train, slr)

        >>>> an.validate()

        >>>> an.validate(X=X_test, y=y_test, metrics=[mean_squared_error, r2_score)

        >>>> an.res_vs_fit()

        >>>> an.quantile()

        >>>> an.spread_loc()

        >>>> an.leverage()
    """

    def __init__(self, X, y, model, predict=None, testing=False):
        """Initial setup of model.
        Args:
            model (object): The fitting model (the model must have a predict method).

            X (numpy.ndarray): The X valuse to be used for plots.

            y (numpy.ndarray): The y values to be used for plots.

            predict (str): The name of the method that is equivalent to the 
                sklearn predict function. Default = 'predict'.
        
            testing (bool, optional): True if unit testing.

        Raises:
            AttributeError: if the model object does not have a prediction attribute.
        """
            
        if predict is None:
            pred = getattr(model, "predict", None)
        else:
            pred = getattr(model, predict, None)
        if callable(pred):
            self.model = model
        else:
            raise AttributeError("The fitting model must have a callable method "
                                 "for making predictions.")
        
        self.X, self.y, self.predictions = self._check_input(X, y)
        self._run_from_ipython()
        self._testing = testing

    def _run_from_ipython(self):
        try: #pragma: no cover
            __IPYTHON__
            self._in_ipython  = True
        except NameError:
            self._in_ipython = False

    def _check_input(self,X,y,pred=None):
        """Checks if the input provided by the user will work for the 
            validatio and plots.

        Args:
            X (numpy.ndarray or None): The feature matrix.
            y (numpy.ndarray or None): The correct target values.
            pred (numpy.ndarray or None): The predictions made by the model.
        
        Returns:
            X (numpy.ndarray or None): The feature matrix.
            y (numpy.ndarray or None): The correct target values.
            pred (numpy.ndarray or None): The predictions made by the model.
        
        Raises:
            ValueError: if the user suplied values of X and y have different
              lengths.
            ValueError: if the target values are given but neither the feature matrix
              of predictions are supplied, or if the feature matrix is given but target 
              values are not, or if predictions are given but not target values.
        """
        if X is not None and y is not None and pred is None:
            pred = self.model.predict(X)
        elif (X is not None and y is None) or ((
                y is not None and X is None) and (y is not None and pred is None)) or (
                    pred is not None and y is None):
            raise ValueError("In order to make a plot for a diferent data set "
                             "than the set initially passed to the function "
                             "two sets of data must be passed in. Either the "
                             "new prediction (pred) set and the true values (y), or the "
                             "new features matrix (X) and the true values(y).")
        if pred is None:
            pred = self.predictions
        if y is None:
            y = self.y

        if len(y) != len(pred) or (X is not None and len(X) != len(y)):
            raise ValueError("The number of predictions, target values, and featuers "
                             "must be the same.")

        if X is None:
            X = self.X
            
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(X,np.ndarray):
            X = np.array(X)

        return X, y, pred
    
    def res_vs_fit(self, X=None, y=None, pred=None, interact=True, show=True, ax=None,
                   title=None):
        """Makes the residual vs fitted values plot.
        
        Args:
            X (numpy.ndarray, optional): The dataset to make the plot for
              if different than the dataset used to initialize the method.

            y (numpy.ndarray, optional): The target values to make the plot for
              if different than the dataset used to initialize the method.

            pred (numpy.ndarray, optional): The predicted values to make the plot for
              if y and X are different than the dataset used to initialize the method.

            interact (bool, optional): True if the plot is to be interactive.

            show (bool, optional): True if plot is to be displayed.

            ax (matplotlib.axes._subplots.AxesSubplot, optional): The subplot on which to 
              drow the plot.

            title (str, optional): The title of the plot.

        Returns:
            fig (matplotlib.figure.Figure or bokeh.plotting.figure): An 
              object containing the plot if show=False.

        Examples:
            >>>> import pandas as pd

            >>>> import numpy as np

            >>>> from sklearn.linear_model import LinearRegression

            >>>> from sklearn.cross_validation import train_test_split

            >>>> from sklearn.metrics import mean_squared_error, r2_score

            >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

            >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

            >>>> X = df.iloc[:,:-1].values

            >>>> y = df[["MEDV"]].values

            >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

            >>>>          test_size=0.3, random_state=0)

            >>>> slr = LinearRegression()

            >>>> slr.fit(X_train,y_train)

            >>>> an = analyze.analysis(X_train, y_train, slr)

            >>>> an.res_vs_fit()

            >>>> an.res_vs_fit(X=X_test,y=y_test,title="Test values")

            >>>> an.res_vs_fit(pred=slr.predict(X_test),y=y_test,title="Test values")
        """

        from analyzefit.manipulate import residual

        X, y, pred = self._check_input(X, y, pred=pred)
        
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
            if show: #pragma: no cover
                show_fig(fig)
            else:
                return fig
            
        else:
            if show: #pragma: no cover
                scatter(pred, res, title=title, x_label="Predictions",
                        y_label="Residues")                    
            else:
                if ax is None and not self._testing: #pragma: no cover
                        return scatter(pred, res, title=title,
                                       x_label="Predictions", y_label="Residues", show_plt=False)
                        
                elif not self._testing or ax is not None:
                    return scatter(pred, res, title=title,
                                   x_label="Predictions", y_label="Residues",
                                   show_plt=False, ax=ax)
                    

    def quantile(self, data=None, dist=None, interact=True, show=True, title=None, ax=None):
        """Makes a quantile plot of the predictions against the desired distribution.

        Args:
            data (numpy.ndarray, optional): The user supplied data for the quantile plot.
              If None then the model predictions will be used.

            dist (str or numpy.ndarray, optional): The distribution to be compared to. Either 
              'Normal', 'Uniform', or a numpy array of the user defined distribution.

            interact (bool, optional): True if the plot is to be interactive.

            show (bool, optional): True if plot is to be displayed.

            ax (matplotlib.axes._subplots.AxesSubplot, optional): The subplot on which to 
              drow the plot.

            title (str, optional): The title of the plot.
        
        Rasises:
            ValueError: if data and the distribution are of different lengths.

        Returns:
            fig (matplotlib.figure.Figure or bokeh.plotting.figure): An 
              object containing the plot if show=False.

        Examples:
            >>>> import pandas as pd

            >>>> import numpy as np

            >>>> from sklearn.linear_model import LinearRegression

            >>>> from sklearn.cross_validation import train_test_split

            >>>> from sklearn.metrics import mean_squared_error, r2_score

            >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

            >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

            >>>> X = df.iloc[:,:-1].values

            >>>> y = df[["MEDV"]].values

            >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

            >>>>          test_size=0.3, random_state=0)

            >>>> slr = LinearRegression()

            >>>> slr.fit(X_train,y_train)

            >>>> an = analyze.analysis(X_train, y_train, slr)

            >>>> an.quantile()

            >>>> an.quantile(data=y_test,dist="uniform",title="Test values vs uniform distribution")

            >>>> an.quantile(data=y_test,dist=np.random.samples((len(y_test))))
        """

        if data is None:
            data = self.predictions
            
        if dist is None or str(dist).lower() == "normal":
            dist = np.random.normal(np.mean(data),np.std(data),len(data))
        elif str(dist).lower() == "uniform":
            dist = np.random.uniform(min(data),max(data),len(data))
        else:
            if type(dist) is not np.ndarray:
                dist = np.array(dist)
        if len(dist) != len(data):
            raise ValueError("The user provided distribution must have the same "
                             "size as the input data or number of predictions.")
            
        dist = sorted(dist)
        data = sorted(data)
        
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
            if show: #pragma: no cover
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

                poly_fit = np.polyfit(dist,data,1)
                if len(poly_fit.shape) == 1:
                    fig.plot(dist, np.poly1d(np.polyfit(dist, data,1))(dist),
                             c="k", linestyle="--", lw=2)
                fig.set_xlim([min(dist), max(dist)])
                fig.set_ylim([min(data), max(data)])
                
            if show: #pragma: no cover
                plt.show()
            elif not self._testing or ax is not None:
                return fig
            else: 
                plt.close()

    def spread_loc(self, X=None, y=None, pred=None, interact=True, show=True,
                   title=None, ax=None):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (numpy.ndarray, optional): The dataset to make the plot for
              if different than the dataset used to initialize the method.

            y (numpy.ndarray, optional): The target values to make the plot for
              if different than the dataset used to initialize the method.

            pred (numpy.ndarray, optional): The predicted values to make the plot for
              if y and X are different than the dataset used to initialize the method.

            interact (bool, optional): True if the plot is to be interactive.

            show (bool, optional): True if plot is to be displayed.

            ax (matplotlib.axes._subplots.AxesSubplot, optional): The subplot on which to 
              drow the plot.

            title (str, optional): The title of the plot.

        Returns:
            fig (matplotlib.figure.Figure or bokeh.plotting.figure): An 
              object containing the plot if show=False.

        Examples:
            >>>> import pandas as pd

            >>>> import numpy as np

            >>>> from sklearn.linear_model import LinearRegression

            >>>> from sklearn.cross_validation import train_test_split

            >>>> from sklearn.metrics import mean_squared_error, r2_score

            >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

            >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

            >>>> X = df.iloc[:,:-1].values

            >>>> y = df[["MEDV"]].values

            >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

            >>>>          test_size=0.3, random_state=0)

            >>>> slr = LinearRegression()

            >>>> slr.fit(X_train,y_train)

            >>>> an = analyze.analysis(X_train, y_train, slr)

            >>>> an.spread_loc()

            >>>> an.spread_loc(X=X_test,y=y_test,title="Test values")

            >>>> an.spread_loc(pred=slr.predict(X_test),y=y_test,title="Test values")
        """

        from analyzefit.manipulate import std_residuals

        X, y, pred = self._check_input(X, y, pred=pred)

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
                                     fig=fig, x_label="Predictions",
                                     y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False)

            if show: #pragma: no cover
                show_fig(fig)
            else:
                return fig

        else:
            if ax is None:
                fig = scatter(pred, root_stres, title=title,
                              x_label="Predictions",
                              y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False)

                plt.xlim([min(pred), max(pred)])
                plt.ylim([min(root_stres), max(root_stres)])
            else:
                fig = scatter(pred, root_stres, title=title,
                              x_label="Predictions",
                              y_label=r'$\sqrt{Standardized Residuals}$', show_plt=False,ax=ax)

                fig.set_xlim([min(pred), max(pred)])
                fig.set_ylim([min(root_stres), max(root_stres)])
                
            if show: #pragma: no cover
                plt.show()
            elif not self._testing or ax is not None:
                return fig
            else:
                plt.close()

    def leverage(self, X=None, y=None, pred=None, interact=True, show=True,
                 title=None, ax=None):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (numpy.ndarray, optional): The dataset to make the plot for
              if different than the dataset used to initialize the method.

            y (numpy.ndarray, optional): The target values to make the plot for
              if different than the dataset used to initialize the method.

            pred (numpy.ndarray, optional): The predicted values to make the plot for
              if y and X are different than the dataset used to initialize the method.

            interact (bool, optional): True if the plot is to be interactive.

            show (bool, optional): True if plot is to be displayed.

            ax (matplotlib.axes._subplots.AxesSubplot, optional): The subplot on which to 
              drow the plot.

            title (str, optional): The title of the plot.

        Rasises:
            ValueError: if the number of predictions is not the same as the number of 
              target values or if the number of rows in the feature matrix is not the
              same as the number of targets.

        Returns:
            fig (matplotlib.figure.Figure or bokeh.plotting.figure): An 
              object containing the plot if show=False.

        Examples:
            >>>> import pandas as pd

            >>>> import numpy as np

            >>>> from sklearn.linear_model import LinearRegression

            >>>> from sklearn.cross_validation import train_test_split

            >>>> from sklearn.metrics import mean_squared_error, r2_score

            >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

            >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

            >>>> X = df.iloc[:,:-1].values

            >>>> y = df[["MEDV"]].values

            >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

            >>>>          test_size=0.3, random_state=0)

            >>>> slr = LinearRegression()

            >>>> slr.fit(X_train,y_train)

            >>>> an = analyze.analysis(X_train, y_train, slr)

            >>>> an.leverage()

            >>>> an.leverage(X=X_test,y=y_test,title="Test values")

            >>>> an.leverage(X=X_test,pred=slr.predict(X_test),y=y_test,title="Test values")
        """

        from analyzefit.manipulate import std_residuals, cooks_dist, hat_diags

        X, y, pred = self._check_input(X, y, pred=pred)

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
            if show: #pragma: no cover
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
            if show: #pragma: no cover
                plt.show()
            elif not self._testing or ax is not None:
                return fig
            else:
                plt.close()

    def validate(self, X=None, y=None, pred=None, dist=None, metric=None, testing=False):
        """The spread-location, or scale-location, plot of the data.

        Args:
            X (numpy.ndarray, optional): The dataset to make the plot for
              if different than the dataset used to initialize the method.

            y (numpy.ndarray, optional): The target values to make the plot for
              if different than the dataset used to initialize the method.

            pred (numpy.ndarray, optional): The predicted values to make the plot for
              if y and X are different than the dataset used to initialize the method.

            dist (str or numpy.ndarray, optional): The distribution to be compared to. Either 
              'Normal', 'Uniform', or a numpy array of the user defined distribution.

            metric (function or list of functions, optional): The functions used to
              determine how accurate the fit is.

            testing (bool, optional): True if this is a unit test.

        Returns:
            score (list of float): The scores from each of the metrics of in testing mode.

        Examples:
            >>>> import pandas as pd

            >>>> import numpy as np

            >>>> from sklearn.linear_model import LinearRegression

            >>>> from sklearn.cross_validation import train_test_split

            >>>> from sklearn.metrics import mean_squared_error, r2_score

            >>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep="\s+")

            >>>> df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

            >>>> X = df.iloc[:,:-1].values

            >>>> y = df[["MEDV"]].values

            >>>> X_train, X_test, y_train, y_test = train_test_split(X, y, 

            >>>>          test_size=0.3, random_state=0)

            >>>> slr = LinearRegression()

            >>>> slr.fit(X_train,y_train)

            >>>> an = analyze.analysis(X_train, y_train, slr)

            >>>> an.validate()

            >>>> an.validate(X=X_test, y=y_test, metrics=[mean_squared_error, r2_score)
        """

        X, y, pred = self._check_input(X, y, pred=pred)

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        ax1 = self.res_vs_fit(y=y, pred=pred, show=False, interact=False, ax=ax1)
        ax2 = self.quantile(data=pred, dist=dist, show=False, interact=False, ax=ax2)
        ax3 = self.spread_loc(y=y, pred=pred, show=False, interact=False, ax=ax3)
        ax4 = self.leverage(y=y, X=X, pred=pred, show=False, interact=False, ax=ax4)
        
        fig.tight_layout()
        if not testing: #pragma: no cover
            plt.show()
        else:  #pragma: no cover
            plt.close()

        if metric is None:
            from sklearn.metrics import mean_squared_error
            if not testing: #pragma: no cover
                print("Prediction error for {1} metric: {0:.2f} ".format(mean_squared_error(y, pred),
                                                                         "mean squared error"))
            else:
                return mean_squared_error(y,pred)
        else:
            if isinstance(metric,list):
                if not testing: #pragma: no cover
                    for m in metric:
                        print("Prediction error for {1} metric: {0:.2f} ".format(m(y, pred),
                                                                                 m.__name__))
                else:
                    return [m(y,pred) for m in metric]    
            else:
                if not testing: #pragma: no cover
                    print("Prediction error for {1} metric: {0:.2f} ".format(metric(y, pred),
                                                                             metric.__name__))
                else:
                    return metric(y,pred)
