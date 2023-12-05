########################################################
# Class: analysis.py
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
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from utils.ffthelpers import propagate_fields

def transmission_vs_NA(fom, name = 'forward_0', figsize = None, dpi = None, use_var_fdtd = False):
    '''Creates plot of transmission vs NA for ktransmissionfom simulations'''
    NArange = np.arange(100.0)/100
    T = np.zeros(NArange.shape)

    sim = Simulation('./', use_var_fdtd, hide_fdtd_cad = True)
    sim.load(name)

    fom.enter_analysis()

    def create_NA_boundary(NA):
        def boundary(kx, ky):
            return np.square(kx) + np.square(ky) < NA**2
        return boundary

    for i, NA in enumerate(NArange):
        fom.set_kboundary_func(create_NA_boundary(NA))
        T[i] = fom.get_fom(sim)

    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(NArange, T)
    ax.set_xlabel('Collection NA')
    ax.set_ylabel('Transmission efficiency')
    plt.savefig('transmissionvsNA.png')
    plt.close()

def plot_geom_hist(params_hist, geometry, show_constraints = False, constraints = None, figsize = None, dpi = None):
    '''Creates set of geometry plots at each iteration'''
    if not show_constraints:
        os.mkdir('./geoms')
    else:
        os.mkdir('./geoms_const')
    #Iterate through param history and plot frames
    for i, params in enumerate(params_hist):
        geometry.update_geometry(params)
        if not show_constraints:
            filename = './geoms/geom_' + str(i) + '.png'
            cons = None
        else:
            filename = './geoms_const/geom_' + str(i) + '.png'
            cons = constraints
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        geometry.plot(ax, constrained=cons)
        ax.set_title('Iteration ' + str(i))
        plt.savefig(filename)
        plt.close(fig)

def plot_grad_hist(grad_hist, constraints, params_hist, cut_constrained = False, figsize = None, dpi = None):
    '''Plots gradient history of each parameter type'''
    #Iterate through grad history
    N = len(grad_hist)
    paramslist = ['x', 'y', 'rx', 'ry', 'phi']
    #List of 5 empty lists, each will correspond to a variable
    grads = [[], [], [], [], []]
    for i, grad in enumerate(grad_hist):
        dx, dy, drx, dry, dphi = np.split(np.abs(grad), 5)
        if cut_constrained:
            cons = constraints.identify_constrained_pillars(params_hist[i])
            dx = np.delete(dx, list(cons))
            dy = np.delete(dy, list(cons))
            drx = np.delete(drx, list(cons))
            dry = np.delete(dry, list(cons))
            dphi = np.delete(dphi, list(cons))
        grads[0].append(dx)
        grads[1].append(dy)
        grads[2].append(drx)
        grads[3].append(dry)
        grads[4].append(dphi)

    #Creates subfolder to store plots
    if cut_constrained:
        dir_name = './grad_hist_cons'
    else:
        dir_name = './grad hist'
    os.mkdir(dir_name)

    for i, param in enumerate(paramslist):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient magnitude')
        ax.set_title('d/d' + param + ' history')
        ax.boxplot(grads[i])
        ax.set_yscale('log')
        plt.savefig(dir_name + '/d' + param + '.png')
        plt.close(fig)

def plot_constraint_hist(params_hist, constraints):
    '''Makes and saves a plot showing # of violated constraints per iteration with various tolerances'''
    iterations = len(params_hist)
    violations_hist, tol1_hist, tol2_hist, tol3_hist, tol4_hist = np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)
    for i in range(iterations):
        violations_hist[i] = constraints.count_violated_constraints(params_hist[i])
        tol1_hist[i] = constraints.count_violated_constraints(params_hist[i], 0.1e-9)
        tol2_hist[i] = constraints.count_violated_constraints(params_hist[i], 1e-9)
        tol3_hist[i] = constraints.count_violated_constraints(params_hist[i], -0.1e-9)
        tol4_hist[i] = constraints.count_violated_constraints(params_hist[i], -1e-9)

    #Plot number of constraints violated per iteration at various tolerances
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of violated constraints')
    ax.plot(violations_hist, label = 'dx = 0')
    ax.plot(tol1_hist, label = 'dx = 0.1nm')
    ax.plot(tol2_hist, label = 'dx = 1 nm')
    ax.plot(tol3_hist, label = 'dx = -0.1 nm')
    ax.plot(tol4_hist, label = 'dx = -1 nm')
    ax.legend()
    plt.savefig('constraint_hist.png')
    plt.close(fig)

def print_constraint_report(params_hist, constraints, tol = 1e-9):
    '''Prints to command line # of violated constraints as of final iteration'''
    final_params = params_hist[-1]
    violations = constraints.count_violated_constraints(final_params)
    tolerance = constraints.count_violated_constraints(final_params, tol)
    print("There are {} violated constraints".format(violations))
    print("There are {} constraints within tolerance of {} nm".format(tolerance, tol*1e9))

def savehist(opt):
    '''Saves parameter hist and gradient hist to np files'''
    np.savez('params_hist', opt.params_hist)
    np.savez('grad_hist', opt.grad_hist)
    np.savez('fom_hist', opt.fom_hist)

def clear_savedata(use_var_fdtd=False):
    '''Clears simulation data from hard drive. Should be run when analysis is complete'''
    sim = Simulation('./', use_var_fdtd, hide_fdtd_cad = True)
    sim.load('forward_0')
    #Clears data from final simulation files
    sim.remove_data_and_save()
    sim.load('adjoint_0')
    sim.remove_data_and_save()


def get_field_from_monitor(monitor_name, use_var_fdtd = False):
    '''Gets field from given analysis monitor'''

    #Loads and clears current data from simulation
    sim = Simulation('./', use_var_fdtd, hide_fdtd_cad = True)
    sim.load('forward_0')
    sim.fdtd.switchtolayout()

    #Turn off opt fields monitor and on analysis monitor. Only needed for calculating gradient
    sim.fdtd.setnamed('opt_fields', 'enabled', False)
    sim.fdtd.setnamed(monitor_name, 'enabled', True)
    
    #Runs simulation and treturns fields
    print('Running analysis simulation for monitor: ' + monitor_name)
    sim.fdtd.run()
    fields = get_fields(sim.fdtd, monitor_name = monitor_name, field_result_name = monitor_name + '_fields', get_eps = False, get_D = False, get_H = False, nointerpolation = False)
    sim.fdtd.switchtolayout()
    sim.fdtd.setnamed(monitor_name, 'enabled', False)
    return fields

def plot_2D_field(ax, field, x, y, cmap):
    '''Plots a n x m x 3 matrix of n in one dimension and n in other'''
    X, Y = np.meshgrid(x, y)
    fieldmag = np.abs(field[:,:,0])**2 + np.abs(field[:,:,1])**2 + np.abs(field[:,:,2])**2
    im = ax.pcolormesh(X*1e6, Y*1e6, np.transpose(fieldmag), cmap = plt.get_cmap(cmap), shading = 'nearest')
    plt.colorbar(im, ax=ax)

def plot_2D_field_from_monitor(monitor_name, wavelengths, cmap, norm_axis = 'z', prop_dist = None):
    '''Plots 2D field from analysis monitor at each wavelength'''
    fields = get_field_from_monitor(monitor_name)
    if prop_dist is not None:
        fields.E, fields.H = propagate_fields(fields, prop_dist)
    dir_name = './' + monitor_name + '_plots'
    os.mkdir(dir_name)
    for indx, wl in enumerate(wavelengths):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        if norm_axis == 'x':
            plot_2D_field(ax, fields.E[0,:,:,indx,:], fields.y, fields.z, cmap)
            ax.set_xlabel('y (um)')
            ax.set_ylabel('z (um)')
        if norm_axis == 'y':
            plot_2D_field(ax, fields.E[:,0,:,indx,:], fields.x, fields.z, cmap)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('z (um)')
        if norm_axis == 'z':
            plot_2D_field(ax, fields.E[:,:,0,indx,:], fields.x, fields.y, cmap)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
        ax.set_title('|E|^2 at wl of {:.0f} nm'.format(wl*1e9))
        plt.savefig('{}/{:.0f}.png'.format(dir_name, wl*1e9))
        plt.close(fig)

def plot_xy_field_from_monitor(monitor_name, wavelengths, cmap, prop_dist = None):
    '''Plots 2D xy field from analysis monitor at each wavelength. Wrapper to general function'''
    plot_2D_field_from_monitor(monitor_name, wavelengths, cmap, norm_axis = 'z', prop_dist = prop_dist)

def plot_xz_field_from_monitor(monitor_name, wavelengths, cmap, prop_dist = None):
    '''Plots 2D xz field from analysis monitor at each wavelength. Wrapper to general function'''
    plot_2D_field_from_monitor(monitor_name, wavelengths, cmap, norm_axis = 'y', prop_dist = prop_dist)

def plot_yz_field_from_monitor(monitor_name, wavelengths, cmap, prop_dist = None):
    '''Plots 2D yz field from analysis monitor at each wavelength. Wrapper to general function'''
    plot_2D_field_from_monitor(monitor_name, wavelengths, cmap, norm_axis = 'x', prop_dist = prop_dist)