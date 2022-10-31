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

class Analysis(object):
    """
        :param opt:             Handle to Optimization object
        :param constraints:     Handle to Constraints object
    """

    def __init__(self, opt, constraints):
        '''Class constructor'''
        self.opt = opt
        self.constraints = constraints

    def transmission_vs_NA(self, figsize = None, dpi = None):
        '''Creates plot of transmission vs NA for ktransmissionfom simulations'''
        NArange = np.arange(100.0)/100
        T = np.zeros(NArange.shape)

        sim = Simulation('./', self.opt.use_var_fdtd, hide_fdtd_cad = True)
        sim.load('forward_0')

        def create_NA_boundary(NA):
            def boundary(kx, ky):
                return np.square(kx) + np.square(ky) < NA**2
            return boundary

        for i, NA in enumerate(NArange):
            self.opt.fom.kboundary_func = create_NA_boundary(NA)
            T[i] = self.opt.fom.get_fom(sim)

        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(NArange, T)
        ax.set_xlabel('Collection NA')
        ax.set_ylabel('Transmission efficiency')
        plt.savefig('transmissionvsNA.png')
        plt.close()

    def plot_geom_hist(self, show_constraints = False, figsize = None, dpi = None):
        '''Creates set of geometry plots at each iteration'''
        if not show_constraints:
            os.mkdir('./geoms')
        else:
            os.mkdir('./geoms_const')
        #Iterate through param history and plot frames
        for i, params in enumerate(self.opt.params_hist):
            scaled_params = self.opt.geometry.get_from_params(params)
            x = scaled_params[0] + self.opt.geometry.init_x
            y = scaled_params[1] + self.opt.geometry.init_y
            rx = scaled_params[2]
            ry = scaled_params[3]
            phi = scaled_params[4]
            if not show_constraints:
                filename = './geoms/geom_' + str(i) + '.png'
                cons = None
            else:
                filename = './geoms_const/geom_' + str(i) + '.png'
                cons = self.constraints.identify_constrained_pillars(params)
            fig = self.plot_elliptical_surface(x*1e6, y*1e6, rx*1e6, ry*1e6, phi, constrained = cons, figsize = figsize, dpi = dpi)
            plt.savefig(filename)
            plt.close(fig)

    def plot_grad_hist(self, cut_constrained = False, figsize = None, dpi = None):
        '''Plots gradient history of each parameter type'''
        #Iterate through grad history
        N = len(self.opt.grad_hist)
        paramslist = ['x', 'y', 'rx', 'ry', 'phi']
        #List of 5 empty lists, each will correspond to a variable
        grads = [[], [], [], [], []]
        for i, grad in enumerate(self.opt.grad_hist):
            dx, dy, drx, dry, dphi = np.split(np.abs(grad), 5)
            if cut_constrained:
                cons = self.constraints.identify_constrained_pillars(self.opt.params_hist[i])
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

        for i, param in enumerate(paramslist):
            if cut_constrained:
                title = 'grad_history_cons_d' + param + '.png'
            else:
                title = 'grad_history_d' + param + '.png'
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Gradient magnitude')
            ax.set_title('d/d' + param + ' history')
            ax.boxplot(grads[i])
            ax.set_yscale('log')
            plt.savefig(title)
            plt.close(fig)

    def plot_constraint_hist(self):
        '''Makes and saves a plot showing # of violated constraints per iteration with various tolerances'''
        iterations = len(self.opt.params_hist)
        violations_hist, tol1_hist, tol2_hist, tol3_hist, tol4_hist = np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)
        for i in range(iterations):
            violations_hist[i] = self.constraints.count_violated_constraints(self.opt.params_hist[i])
            tol1_hist[i] = self.constraints.count_violated_constraints(self.opt.params_hist[i], 0.1e-9)
            tol2_hist[i] = self.constraints.count_violated_constraints(self.opt.params_hist[i], 1e-9)
            tol3_hist[i] = self.constraints.count_violated_constraints(self.opt.params_hist[i], -0.1e-9)
            tol4_hist[i] = self.constraints.count_violated_constraints(self.opt.params_hist[i], -1e-9)

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

    def print_constraint_report(self, tol = 1e-9):
        '''Prints to command line # of violated constraints as of final iteration'''
        final_params = self.opt.params_hist[-1]
        violations = self.constraints.count_violated_constraints(final_params)
        tolerance = self.constraints.count_violated_constraints(final_params, tol)
        print("There are {} violated constraints".format(violations))
        print("There are {} constraints within tolerance of {} nm".format(tolerance, tol*1e9))

    def savehist(self):
        '''Saves parameter hist and gradient hist to np file'''
        np.savez('params_hist', self.opt.params_hist)
        np.savez('grad_hist', self.opt.grad_hist)

    def clear_savedata(self):
        '''Clears simulation data from hard drive. Should be run when analysis is complete'''
        sim = Simulation('./', self.opt.use_var_fdtd, hide_fdtd_cad = True)
        sim.load('forward_0')
        #Clears data from final simulation files
        sim.remove_data_and_save()
        sim.load('adjoint_0')
        sim.remove_data_and_save()

    def plot_elliptical_surface(self, x, y, rx, ry, phi, constrained = None, figsize = None, dpi = None):
        '''Plots geometry of given paramter set. Optionally colors in constraints'''
        maxr = max(np.amax(rx), np.amax(ry))
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(1, 1, 1)
        for i, xval in enumerate(x):
            color = 'black'
            if constrained is not None and i in constrained:
                color = 'red'
            ellipse = patches.Ellipse((xval, y[i]), 2*rx[i], 2*ry[i], angle = phi[i], facecolor=color)
            ax.add_patch(ellipse)
        ax.set_title('Geometry')
        ax.set_xlim(min(x) - maxr, max(x) + maxr)
        ax.set_ylim(min(y) - maxr, max(y) + maxr)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return fig
