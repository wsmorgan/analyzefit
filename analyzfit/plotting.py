from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool
from bokeh.io import output_notebook
from collections import OrderedDict
import matplotlib.pyplot as plt 

def scatter_with_hover(x, y, in_notebook=True, show_plt=True,
                       fig=None, name=None, marker='o',
                       fig_width=500, fig_height=500, x_label=None,
                       y_label=None, title=None, color="blue",**kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic
    tooltips showing columns from `df`. Modified from: 
    http://blog.rtwilson.com/bokeh-plots-with-dataframe-based-tooltips/

    Args:
        x (numpy array): The data for the x-axis.
        y (numpy array): The data for the y-axis.
        fig (optional, bokeh.plotting.Figure): Figure on which to plot 
            (if not given then a new figure will be created)
        name (optional, str): Series name to give to the scattered data
        marker (optional, str): Name of marker to use for scatter plot

    Returns:
        fig (bokeh.plotting.Figure): Figure (the same as given, or the newly created figure) 
            if show is False
    """
    # Make it so it works for ipython.
    if in_notebook:
        output_notebook()
    # insert the correct hover identifier.
    hover = HoverTool(tooltips=[("entry#", "@label"),])
    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        if title is None:
            fig = figure(width=fig_width, height=fig_height, tools=['box_zoom', 'reset',hover])
        else:
            fig = figure(width=fig_width, height=fig_height,
                         tools=['box_zoom', 'reset',hover],title=title)
    # We're getting data from the given dataframe
    source = ColumnDataSource(data=dict(x=x,y=y,label=range(1,len(x)+1)))

    # Actually do the scatter plot - the easy bit
    # (other keyword arguments will be passed to this function)
    fig.scatter('x', 'y', source=source, marker=marker,color=color,name=name)

    if not x_label is None:
        fig.xaxis.axis_label = x_label
    if not y_label is None:
        fig.yaxis.axis_label = y_label
    if show_plt:
        show(fig)
    else:
        return(fig)

def scatter(x,y,show_plt=True, x_label=None, y_label=None, label=None,
            title=None,fig=None,ax=None):
    """Make a standard matplotlib style scatter plot.

    Args:
        x (numpy array): The data for the x-axis.
        y (numpy array): The data for the y-axis.
        show (optional, bool): True if plot is to be shown.
        x_label (optional, str): The x-axis label.
        y_label (optional, str): The y-axis label.
        label (optional, str): The data trend label.
        title (optional, str): The plot title.
        fig (optional, matplotlib pyplot object): An initial figure to add points too.
        ax (optional, matplotlib axis object): A subplot object to plot on.

    Returns:
        fig (matplotlib pyplot object): Returns the matplotlib object if show = False.
    """
    if fig is None and ax is None:
        fig = plt.figure()
    elif ax is None:
        fig = fig
    else:
        ax = ax

    if ax is None:
        if not label is None:
            plt.scatter(x,y,label=label)
        else:
            plt.scatter(x,y)
            
        if not x_label is None:
            plt.xlabel(x_label)
        if not y_label is None:
            plt.ylabel(y_label)
        if not title is None:
            plt.title(title)
    else:
        if not label is None:
            ax.scatter(x,y,label=label)
        else:
            ax.scatter(x,y)

        if not x_label is None:
            ax.set_xlabel(x_label)
        if not y_label is None:
            ax.set_ylabel(y_label)
        if not title is None:
            ax.set_title(title)

        return ax
            
    if show_plt is True:
        plt.show()
    else:
        return fig
