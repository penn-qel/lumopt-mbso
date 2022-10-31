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

    def plot_grad_hist(self, figsize = None, dpi = None):
        '''Plots gradient history of each parameter type'''
        #Iterate through grad history
        N = len(self.opt.grad_hist)
        dx_mean, dy_mean, drx_mean, dry_mean, dphi_mean = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        dx_std, dy_std, drx_std, dry_std, dphi_std = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        for i, grads in enumerate(self.opt.grad_hist):
            if self.opt.geometry.pillars_rotate:
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
            if self.opt.geometry.pillars_rotate:
                dphi_mean[i] = np.mean(dphi)
                dphi_std[i] = np.std(dphi)


        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(1,1,1)
        x = np.arange(N)
        ax.errorbar(x, dx_mean, dx_std, label = 'dx')
        ax.errorbar(x, dy_mean, dy_std, label = 'dy')
        ax.errorbar(x, drx_mean, drx_std, label = 'drx')
        ax.errorbar(x, dry_mean, dry_std, label = 'dry')
        if self.opt.geometry.pillars_rotate:
            ax.errorbar(x, dphi_mean, dphi_std, label = 'dphi')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient magnitude')
        ax.legend()
        plt.savefig('grad_hist.png')
        plt.close(fig)

        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(1,1,1)
        x = np.arange(N)
        ax.errorbar(x, dx_mean, dx_std, label = 'dx')
        ax.errorbar(x, dy_mean, dy_std, label = 'dy')
        ax.errorbar(x, drx_mean, drx_std, label = 'drx')
        ax.errorbar(x, dry_mean, dry_std, label = 'dry')
        if self.opt.geometry.pillars_rotate:
            ax.errorbar(x, dphi_mean, dphi_std, label = 'dphi')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient magnitude')
        ax.set_yscale('log')
        ax.legend()
        plt.savefig('grad_hist_log.png')
        plt.close(fig)

        grad_hist = np.vstack(self.opt.grad_hist)
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
