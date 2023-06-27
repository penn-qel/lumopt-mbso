import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from lumopt.utilities.plotter import Plotter, SnapShots

class MyPlotter(Plotter):
    def __init__(self, movie = True, plot_history = True, plot_fields = True):
        self.plot_history = plot_history
        self.plot_fields = plot_fields

        if self.plot_fields:
            if plot_history:
                self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
            else:
                self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        else:
            self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10)) #Changed this line as jank fix

        ## Flatten the axes because it is difficult to keep track otherwise:
        l = list(map(list, zip(*self.ax)))
        self.ax = list(flatten(l))

        self.fig.show()
        self.movie = movie
        if movie:
            metadata = dict(title = 'Optimization', artist='lumopt', comment = 'Continuous adjoint optimization')
            self.writer = SnapShots(fps = 2, metadata = metadata)
        self.iter = 0

    def draw_and_save(self):
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        fname = 'optimization' + str(self.iter) + '.png'
        self.fig.savefig(fname)
        print("Saved frame")
        self.iter += 1
