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
    ax.set_xlabel('Collection NA')
    ax.set_ylabel('Transmission efficiency')
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


def plot_grad_hist(opt, figsize = None, dpi = None):
    '''Plots gradient history of each parameter type'''
    #Iterate through grad history
    N = len(opt.grad_hist)
    dx_mean, dy_mean, drx_mean, dry_mean, dphi_mean = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    dx_std, dy_std, drx_std, dry_std, dphi_std = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i, grads in enumerate(opt.grad_hist):
        if opt.geometry.pillars_rotate:
            dx, dy, drx, dry, dphi = np.split(np.abs(grads), 5)
        else:
            dx, dy, drx, dry = np.split(np.abs(grads), 4)

        dx_mean[i] = np.mean(dx)
        dx_std[i] = np.std(dx)
        dy_mean[i] = np.mean(dy)
        dy_std[i] = np.std(dy)
        drx_mean[i] = np.mean(drx)
        drx_std[i] = np.std(drx)
        dry_mean[i] = np.mean(dry)
        dry_std[i] = np.std(dry)
        if opt.geometry.pillars_rotate:
            dphi_mean[i] = np.mean(dphi)
            dphi_std[i] = np.std(dphi)


    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1,1,1)
    x = np.arange(N)
    ax.errorbar(x, dx_mean, dx_std, label = 'dx')
    ax.errorbar(x, dy_mean, dy_std, label = 'dy')
    ax.errorbar(x, drx_mean, drx_std, label = 'drx')
    ax.errorbar(x, dry_mean, dry_std, label = 'dry')
    if opt.geometry.pillars_rotate:
        ax.errorbar(x, dphi_mean, dphi_std, label = 'dphi')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient magnitude')
    ax.legend()
    plt.savefig('grad_hist.png')
    plt.close(fig)

    np.savez('grad_hist_log', opt.grad_hist)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1,1,1)
    x = np.arange(N)
    ax.errorbar(x, dx_mean, dx_std, label = 'dx')
    ax.errorbar(x, dy_mean, dy_std, label = 'dy')
    ax.errorbar(x, drx_mean, drx_std, label = 'drx')
    ax.errorbar(x, dry_mean, dry_std, label = 'dry')
    if opt.geometry.pillars_rotate:
        ax.errorbar(x, dphi_mean, dphi_std, label = 'dphi')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient magnitude')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('grad_hist_log.png')
    plt.close(fig)

    grad_hist = np.vstack(opt.grad_hist)
    dx, dy, drx, dry, dphi = np.split(np.abs(grad_hist), 5)

    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient magnitude')
    ax1.set_title('d/dx history')
    ax1.boxplot(dx)

    ax2 = fig.add_subplot(5,1,2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient magnitude')
    ax2.set_title('d/dy history')
    ax2.boxplot(dy)   

    ax3 = fig.add_subplot(5,1,3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient magnitude')
    ax3.set_title('d/drx history')
    ax3.boxplot(drx)   

    ax4 = fig.add_subplot(5,1,4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Gradient magnitude')
    ax4.set_title('d/dry history')
    ax4.boxplot(dry)   

    ax5 = fig.add_subplot(5,1,5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Gradient magnitude')
    ax5.set_title('d/dphi history')
    ax5.boxplot(dphi)

    plt.savefig('grad_hist_boxplots.png')
    plt.close(fig)      




#To be called directly after simulation runs. No further simulations needed
def process_3D_simulation(opt, figsize = None, dpi = None, geom_hist = True, grad_hist = True, trans_vs_NA = False):

    if geom_hist:
        plot_geom_hist(opt, figsize, dpi)

    if grad_hist:
        plot_grad_hist(opt, figsize, dpi)

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
