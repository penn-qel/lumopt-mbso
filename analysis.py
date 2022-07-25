########################################################
# This defines and implements a set of functions for post-analysis of simulations
# Author: Amelia Klein
########################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from lumopt.optimization import Optimization
from lumopt.utilities.simulation import Simulation

def plot_elliptical_surface(x, y, rx, ry, phi, figsize = None, dpi = None):
    '''Plots geometry of given paramters'''
    maxr = max(np.amax(rx), np.amax(ry))
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1, 1, 1)
    for i, xval in enumerate(x):
        ellipse = patches.Ellipse((xval, y[i]), 2*rx[i], 2*ry[i], angle = phi[i], facecolor='black')
        ax.add_patch(ellipse)
    ax.set_title('Geometry')
    ax.set_xlim(min(x) - maxr, max(x) + maxr)
    ax.set_ylim(min(y) - maxr, max(y) + maxr)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    return fig

def transmission_vs_NA(opt, sim, figsize = None, dpi = None):
    NArange = np.arange(100.0)/100
    T = np.zeros(NArange.shape)

    def create_NA_boundary(NA):
        def boundary(kx, ky):
            return np.square(kx) + np.square(ky) < NA**2

    for i, NA in enumerate(NArange):
        opt.geometry.kboundary_func = create_NA_boundary(NA)
        T[i] = opt.geometry.get_fom(sim)

    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(NArange, T)
    ax.xlabel('Collection NA')
    ax.ylabel('Transmission efficiency')
    plt.savefig('transmissionvsNA.png')
    plt.close()


def plot_geom_hist(opt, figsize = None, dpi = None):
    os.mkdir('./geoms')
    #Iterate through param history and plot frames
    for i, params in enumerate(opt.params_hist):
        scaled_params = opt.geometry.get_from_params(params)
        x = scaled_params[0] + opt.geometry.init_x
        y = scaled_params[1] + opt.geometry.init_y
        rx = scaled_params[2]
        ry = scaled_params[3]
        phi = scaled_params[4]
        filename = './geoms/geom_' + str(i) + '.png'
        fig = plot_elliptical_surface(x*1e6, y*1e6, rx*1e6, ry*1e6, phi, figsize = figsize, dpi = dpi)
        plt.savefig(filename)
        plt.close(fig)

    np.save('params_hist', opt.params_hist)

#To be called directly after simulation runs. No further simulations needed
def process_3D_simulation(opt, figsize = None, dpi = None, geom_hist = False, trans_vs_NA = False):

    if geom_hist:
        plot_geom_hist(opt, figsize, dpi)

    sim = Simulation('./', opt.use_var_fdtd, hide_fdtd_cad = True)
    sim.load('forward_0')
    if trans_vs_NA:
        transmission_vs_NA(opt, sim, figsize, dpi)

    #Clears data from final simulation files
    sim.remove_data_and_save()
    sim.load('adjoint_0')
    sim.remove_data_and_save()
    return

#For post-analysis of specific monitors from command line
def main():
    #Todo
    sim = Simulation('./', use_var_fdtd = False, hide_fdtd_cad = True)
    sim.load('forward_0')
    
if __name__ == "__main__":
    main()
