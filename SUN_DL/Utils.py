# This python file contains the Utils of the framework

# Importing the libraries--------------------------------------------
import os
import inspect
import logging
import collections
from IPython import display
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

# configure the logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
# Define the Utils-------------------------------------------------
# Define Hyperparameters class to help manage the hyperparameters
class Hyperparameters:
    def save_hyperparameters(self, ignore=[]):  # ignore argument recieve the name of argument want ignore
        """Save the function arguments into the class attributes"""
        caller_frame = inspect.currentframe().f_back   # this frame's caller
        _, _, _, locals_vars = inspect.getargvalues(caller_frame)  # get the argument of caller_frame
        self.hyperparameters = {k:v for k, v in locals_vars.items() if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hyperparameters.items():  # save the parameter to the class
            setattr(self, k, v)
        logging.info('Finish saving arguments to {}'.format(self))

# Define ProgressBoard class to plot data points in animation
## utils functions
def use_svg_display():
    """Use the svg format to display a plot in Jupyter"""
    backend_inline.set_matplotlib_formats('svg')

class ProgressBoard(Hyperparameters):
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                xscale='linear', yscale='linear', ls=['-', '--', '-.', ':'],
                colors=['C0', 'C1', 'C2', 'C3'], fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()  # save the argument in one-line (quite easy!!!)
        os.environ['KMP_DUPLICATE_LIB_OK']='True'  # solve the problem of matplotlib

    def draw(self, x, y, label, every_n = 1):
        """
        (x, y): point to plot
        label: for example train_acc, test_acc and so on
        every_n: control the plot frequency
        """
        Point = collections.namedtuple('Point', ['x', 'y'])  # create an named tuple to save points
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()  # raw_points saves the points to construct lines
            self.data = collections.OrderedDict()  # data save the lines to plot
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]  # fetch the points collection we want to add
        points.append(Point(x, y))
        if len(points) != every_n:  # check whethter we should plot
            return
        mean = lambda x: sum(x) / len(x)
        line = self.data[label]  # fetch the line we want to plot
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:  # check whether we need to display
            return
        use_svg_display()
        if self.fig is None:  # create the figure if it hasn't
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                        linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = "x"
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
        

# Define the test class
class test_class(Hyperparameters):
    def __init__(self, test1 = 1, test2 = 2, test3 = 3):
        super().__init__()
        self.save_hyperparameters(ignore=["test3"])

def main():
    test1 = test_class()

if __name__ == "__main__":
    main()