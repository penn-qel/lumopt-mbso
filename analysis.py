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
        return boundary

    for i, NA in enumerate(NArange):
        opt.fom.kboundary_func = create_NA_boundary(NA)
        T[i] = opt.fom.get_fom(sim)

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
    dx, dy, drx, dry, dphi = np.split(np.abs(grad_hist.transpose()), 5)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient magnitude')
    ax1.set_title('d/dx history')
    ax1.boxplot(dx)
    ax1.set_yscale('log')
    plt.savefig('grad_history_dx.png')
    plt.close(fig1)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient magnitude')
    ax2.set_title('d/dy history')
    ax2.boxplot(dy)   
    ax2.set_yscale('log')
    plt.savefig('grad_history_dy.png')
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient magnitude')
    ax3.set_title('d/drx history')
    ax3.boxplot(drx)  
    ax3.set_yscale('log')
    plt.savefig('grad_history_drx.png')
    plt.close(fig3)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Gradient magnitude')
    ax4.set_title('d/dry history')
    ax4.boxplot(dry)
    ax4.set_yscale('log')
    plt.savefig('grad_history_dry.png')
    plt.close(fig4)

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1,1,1)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Gradient magnitude')
    ax5.set_title('d/dphi history')
    ax5.boxplot(dphi)
    ax5.set_yscale('log')
    plt.savefig('grad_history_dphi.png')
    plt.close(fig5)

#Gets array of constraints from dictionary and set of parameters
def get_constraints(params, constraints_dict):
    constraint_fun = constraints_dict['fun']
    return constraint_fun(params)

#Counts number of constraints within a tolerance. Returns count and locations
def count_violated_constraints(params, constraint_dict, tol = 0):
    cons = get_constraints(params, constraint_dict)
    locations = np.nonzero((cons - tol) < 0.0)
    return locations[0].size

def constraint_hist(params_hist, constraint_dict):
    iterations = len(params_hist)
    violations_hist, tol1_hist, tol2_hist, tol3_hist = np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)
    for i in range(iterations):
        violations_hist[i] = count_violated_constraints(params_hist[i], constraint_dict)
        tol1_hist[i] = count_violated_constraints(params_hist[i], constraint_dict, 0.1e-9)
        tol2_hist[i] = count_violated_constraints(params_hist[i], constraint_dict, 1e-9)
        tol3_hist[i] = count_violated_constraints(params_hist[i], constraint_dict, 10e-9)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of violated constraints')
    ax.plot(violations_hist, label = 'dx = 0')
    ax.plot(tol1_hist, label = 'dx = 0.1nm')
    ax.plot(tol2_hist, label = 'dx = 1 nm')
    ax.plot(tol3_hist, label = 'dx = 10 nm')
    ax.legend()
    plt.savefig('constraint_hist.png')
    plt.close(fig)

def constraint_report(opt, tol = 1e-9):
    final_params = opt.params_hist[-1]
    violations = count_violated_constraints(final_params, opt.optimizer.constraints)
    tolerance = count_violated_constraints(final_params, opt.optimizer.constraints, tol)
    print("There are {} violated constraints".format(violations))
    print("There are {} constraints within tolerance of {} nm".format(tolerance, tol*1e9))
    constraint_hist(opt.params_hist, opt.optimizer.constraints)


#To be called directly after simulation runs. No further simulations needed
def process_3D_simulation(opt, figsize = None, dpi = None, geom_hist = True, grad_hist = True, do_constraint_report = True, trans_vs_NA = False):

    if geom_hist:
        plot_geom_hist(opt, figsize, dpi)

    if grad_hist:
        plot_grad_hist(opt, figsize, dpi)

    if do_constraint_report:
        constraint_report(opt)

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
