################################################
# Script: metasurface.py

# Description: This script replaces the optimizer wrapper to allow for linear constraints
# Author: Amelia Klein
###############################################

import numpy as np
import scipy as sp
import scipy.optimize as spo

from lumopt.optimizers.generic_optimizers import ScipyOptimizers

class ConstrainedOptimizer(ScipyOptimizers):
	'''Wrapper for the constained optimizers (eg SQP) in Scipy's optimization package

	Parameters
	------------
	:param max_iter:  			Number of maximum iterations; each can make multile FOM/gradient evaluations
	:param method: 				String with chosen minimization algorithm
	:param constraints: 		List of dictionaries defining constraints for SLSQP
	:param scaling_factor 		scalar or vector of same length as optimization parameters, typically used to scale
								magnitudes between zero and one
	:param scale_initial_gradient_to: 	enforces rescaling of gradient to change optimization parameters by at least this much,
										the default value of zero disables automatically scaling
	:param penalty_fun: 		penalty function to be added to the figure of merit: it must be a function
								that takes a vector with the optimization parameters and returns a single value
	:param penalty_jac:  		gradient of the penalty function; must be a function that takes a vector with the optimization
								parameters and returns a vector of the same length
	:param ftol:				tolerance parameter 'ftol' which allows to stop optimization when changes in FOM are less than this
	:param concurrent_adjoint: 	Boolean flag telling whether or not adjoint should be solved concurrently with forward sim

	'''
	def __init__(self, max_iter, method = 'SLSQP', constraints = None, scaling_factor = 1.0, pgtol = 1.0e-5, ftol = 1.0e-12, scale_initial_gradient_to = 0, penalty_fun = None, penalty_jac = None, concurrent_adjoint = True):
		super().__init__(max_iter = max_iter,
						method = method,
						scaling_factor = scaling_factor,
						pgtol = pgtol,
						ftol = ftol,
						scale_initial_gradient_to = scale_initial_gradient_to,
						penalty_fun = penalty_fun,
						penalty_jac = penalty_jac)
		self.constraints = constraints
		self.concurrent_adjoint = concurrent_adjoint

	def run(self):
		res = spo.minimize(fun = self.callable_fom,
							x0 = self.start_point,
							jac = self.callable_jac,
							bounds = self.bounds,
							callback = self.callback,
							options = {'maxiter':self.max_iter, 'disp':True, 'gtol':self.pgtol, 'ftol':self.ftol},
							method = self.method,
							constraints = self.constraints)
		res.x /= self.scaling_factor
		res.fun = -res.fun
		if hasattr(res, 'jac'):
			res.jac = -res.jac*self.scaling_factor
		return res


	def concurrent_adjoint_solves(self):
		return (self.method in ['L-BFGS-B', 'BFGS', 'SLSQP']) and self.concurrent_adjoint
	'''def reset_start_params(self, start_params, scale_initial_gradient_to):
		#This function is responsible for scaling the start params and the FOM
		#This bypasses the FOM scaling if is passed as an input
		self.scale_initial_gradient_to = scale_initial_gradient_to
		self.start_point = start_params * self.scaling_factor
		if self.fom_scaling_factor is None:
			if scale_initial_gradient_to != 0.0:
				if self.bounds is None:
					raise UserWarning("bounds are required to scale the initial gradient.")
				self.auto_detect_scaling(scale_initial_gradient_to)
			else:
				self.fom_scaling_factor = 1'''