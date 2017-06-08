from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool
from bokeh.io import output_notebook
from collections import OrderedDict
import matplotlib.pyplot as plt 
import numpy as np

def scatter_with_hover(x, y, in_notebook=True, show_plt=True,
                       fig=None, name=None, marker='o',
                       fig_width=500, fig_height=500, x_label=None,
                       y_label=None, title=None, color="blue",**kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic
    tooltips showing columns from `df`. Modified from: 
    http://blog.rtwilson.com/bokeh-plots-with-dataframe-based-tooltips/

    Args:
        x (numpy.ndarray): The data for the x-axis.
        y (numpy.ndarray): The data for the y-axis.

        fig (bokeh.plotting.Figure, optional): Figure on which to plot 
          (if not given then a new figure will be created)

        name (str, optional): Series name to give to the scattered data
        marker (str, optional): Name of marker to use for scatter plot

    Returns:
        fig (bokeh.plotting.Figure): Figure (the same as given, or the newly created figure) 
            if show is False
    """
    # Make it so it works for ipython.
    if in_notebook: #pragma: no cover
        output_notebook()
    # insert the correct hover identifier.
    hover = HoverTool(tooltips=[("entry#", "@label"),])
    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        # if title is None:
        #     fig = figure(width=fig_width, height=fig_height, tools=['box_zoom', 'reset',hover])
        # else:
        fig = figure(width=fig_width, height=fig_height,
                     tools=['box_zoom', 'reset',hover],title=title)
    # We're getting data from the given dataframe
    source = ColumnDataSource(data=dict(x=x,y=y,label=range(1,len(x)+1)))

    # Actually do the scatter plot - the easy bit
    # (other keyword arguments will be passed to this function)
    fig.scatter('x', 'y', source=source, marker=marker,color=color,name=name)

    if x_label is not None:
        fig.xaxis.axis_label = x_label
    if y_label is not None:
        fig.yaxis.axis_label = y_label
    if show_plt: # pragma: no cover
        show(fig)
    else:
        return(fig)

def scatter(x,y,show_plt=True, x_label=None, y_label=None, label=None,
            title=None,fig=None,ax=None):
    """Make a standard matplotlib style scatter plot.

    Args:
        x (numpy.ndarray): The data for the x-axis.
        y (numpy.ndarray): The data for the y-axis.
        show (bool, optional): True if plot is to be shown.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        label (str, optional): The data trend label.
        title (str, optional): The plot title.
        fig (matplotlib.figure.Figure, optional): An initial figure to add points too.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A subplot object to plot on.

    Returns:
        fig (matplotlib object): Returns the matplotlib object if show = False.
    """
    if fig is None and ax is None:
        fig = plt.figure()
    elif ax is None:
        fig = fig
    else:
        ax = ax

    if not isinstance(x,np.ndarray):
        x = np.array(x)
    if not isinstance(y,np.ndarray):
        y = np.array(y)

    if ax is None:
        if label is not None:
            plt.scatter(x,y,label=label)
        else:
            plt.scatter(x,y)
            
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        if title is not None:
            plt.title(title)
    else:
        if label is not None:
            ax.scatter(x,y,label=label)
        else:
            ax.scatter(x,y)

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)

        return ax
            
    if show_plt is True: #pragma: no cover
        plt.show()
    else:
        return fig
