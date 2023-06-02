###################################################################
# Class: zoned_moving_metasurface3D.py

# Description: this class defines a geometry object corresponding to a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints.
# Allows selecting a certain zone to be passed as actual parameters, while the rest is held constant
# Author: Amelia Klein
###################################################################

import numpy as np
import scipy as sp
import scipy.constants
import lumapi
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.utilities.wavelengths import Wavelengths
from interpolate_fields import interpolate_fields
from moving_metasurface3D import MovingMetasurface3D

class ZonedMovingMetasurface3D(MovingMetasurface3D):
    """
        :param posx:            Array of shape (N,) defining initial x-coordinates of pillar centers
        :param posy:            Array of shape (N,) defining initial y-coordinates of pillar centers
        :param rx:              Array of shape (N,) defining initial x-axis radius of each pillar
        :param ry:              Array of shape (N,) defining initial y-axis radius of each pillar
        :param phi:             Array of shape (N,) defining intial phi-rotation of each pillar in degrees
        :param pillars_rotate:  Boolean determining if pillar rotation is an optimization variable
        :param z:               z-position of bottom of metasurface
        :param h:               height of metasurface pillars
        :param height_precision:Number of points along height of each pillar used to calculate gradient
        :param angle_precision: Number of points along circumference of pillar used to calculate gradient
        :param eps_in:          Permittivity of pillars
        :param eps_out:         Permittivity of surrounding material
        :param dx:              step size for computing the figure of merit gradient using permittivity perturbations.
        :param inner:			Radius of inner ring of active area. Default 0
        :param outer:			Radius of outer ring of active area. Default infinity.
    """

    def __init__(self, posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, inner = 0, outer = np.inf, phi = None, pillars_rotate = True, height_precision = 10, angle_precision = 20, scaling_factor = 1, phi_scaling = 1/180, limit_nearest_neighbor_cons = True, make_meshgrid = False, dx = 10e-9, params_debug = False):
        super().__init__(posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, phi, pillars_rotate, height_precision, angle_precision, scaling_factor, phi_scaling, limit_nearest_neighbor_cons, make_meshgrid, dx, params_debug)
        
        if not self.pillars_rotate:
        	raise UserWarning("Non-rotating pillars not currently supported")

        self.set_active_pillars(inner, outer)

        #Recalculate bounds. Not sure if this is necessary but don't know how behavior would work so in case
        self.bounds = self.calculate_bounds()

    def set_active_pillars(self, inner, outer):
    	'''Gets list True/False of which pillars are active in current region'''
    	if inner > outer:
    		raise UserWarning("Inner radius must be smaller than outer radius")
    	if inner < 0 or outer < 0:
    		raise UserWarning("Inner and outer radii should be positive")
    	self.inner = inner
    	self.outer = outer
        dist = np.sqrt(np.square(self.init_x) + np.square(self.init_y))
        self.active = = dist >= inner and dist < outer
        if np.count_nonzero(self.active) == 0:
        	raise UserWarning("No pillars fall within active region")
        return self.active

    def get_from_params(self, params):
    	'''Retrieves correctly scaled parameters from list of params. Slots in active parameters to entire geometry'''

    	#Gets active optimization parameters only
    	offset_x_active, offset_y_active, rx_active, ry_active, phi_active = super().get_from_params(params)

    	#Gets copy of current internal parameters
    	offset_x = np.copy(self.offset_x)
    	offset_y = np.copy(self.offset_y)
    	rx = np.copy(self.rx)
    	ry = np.copy(self.ry)
    	phi = np.copy(self.phi)

    	#Updates internal parameters
    	offset_x[active] = offset_x_active
    	offset_y[active] = offset_y_active
    	rx[active] = rx_active
    	ry[active] = ry_active
    	phi[active] = phi_active

    	return offset_x, offset_y, rx, ry, phi

    def get_current_params(self):
    	'''Returns list of optimization params as single array'''
    	s1 = self.scaling_factor
    	s2 = self.phi_scaling
    	return np.concatenate((self.offset_x[active]*s1, self.offset_y[active]*s1, self.rx[active]*s1, self.ry[active]*s1, self.phi[active]*s2))

    def calculate_bounds(self):
        '''Calculates bounds given the minimum feature size'''
        '''Bounds should be [min_feature_size/2, inf] for radiii and [-inf, inf] for offsets'''
        radius_bounds = [(self.min_feature_size*self.scaling_factor/2, np.inf)]*self.rx[valid].size*2
        offset_bounds = [(-np.inf, np.inf)]*self.rx[valid].size*2
        phi_bounds = [(-np.inf, np.inf)]*(self.rx[valid].size)
        if self.pillars_rotate:
            return (offset_bounds + radius_bounds + phi_bounds)
        else:
            return (offset_bounds + radius_bounds)

    def calculate_gradient(self, gradient_fields):
    	'''Calculates gradient at each wavelength with respect to all parameters'''

    	#Calculates whole gradient of all parameters
    	total_deriv = super().calculate_gradient(gradient_fields)
    	deriv_x, deriv_y, deriv_rx, deriv_ry, deriv_phi = np.split(total_deriv, 5)

    	#Reconcatenates with only active portion
    	return np.concatenate(deriv_x[active,:], deriv_y[active,:], deriv_rx[active,:], deriv_ry[active,:], deriv_phi[active,:])