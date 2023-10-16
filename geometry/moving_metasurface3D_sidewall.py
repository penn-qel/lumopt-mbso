###################################################################
# Class: moving_metasurface3D_sidewall.py

# Description: this class defines a geometry object corresponding to a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints.
# Implements pillars as etched at a particular sidewall angle
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
from lumopt.utilities.gradients import GradientFields
from lumopt.utilities.simulation import Simulation
from geometry.moving_metasurface3D import MovingMetasurface3D
from utils.interpolate_fields import interpolate_fields
from utils.get_fields_from_cad import get_fields_from_cad

class MovingMetasurface3DSidewall(MovingMetasurface3D):
    """Defines object consisting of array of elliptical pillars, where axes lengths, positions, and
    rotations are all free optimization variables. Pillars are made with a sidewall angle.

    Not currently set up to do actual optimizations, but intended instead for use in post-analysis

    Parameters
    ----------------
        :param posx:            Array of shape (N,) defining initial x-coordinates of pillar centers
        :param posy:            Array of shape (N,) defining initial y-coordinates of pillar centers
        :param rx:              Array of shape (N,) defining initial x-axis radius of each pillar
        :param ry:              Array of shape (N,) defining initial y-axis radius of each pillar
        :param min_feature_size:    Scalar that determines minimum pillar diameter and spacing.
        :param z:               z-position of bottom of metasurface
        :param h:               height of metasurface pillars
        :param eps_in:          Permittivity of pillars
        :param eps_out:         Permittivity of surrounding material

    Optional kwargs
    -----------------
        :kwarg sidewall_angle:  Angle in degrees of sidewall etch. Default 90
        :kwarg sidwall_points:  Number of points used to generate sidewall. Subtract 1 to get number of layers. Default to height_precision

    Inherited kwargs
    ---------------
        :kwarg phi:             Array of shape (N,) defining intial phi-rotation of each pillar in degrees. Defaults to all 0
        :kwarg height_precision:Number of points along height of each pillar used to calculate gradient. Default 10
        :kwarg angle_precision: Number of points along circumference of pillar used to calculate gradient. Default 20
        :kwarg pillars_rotate:  Boolean determining if pillar rotation is an optimization variable. Default True
        :kwarg scaling_factor:  Factor to scale all position and radius parameters. Default 1
        :kwarg phi_scaling:     Scaling factor to scale rotation parameters. Default 1/180
        :kwarg limit_nearest_neighbor_cons: Flag to limit constraints to nearest neighbors in initial grid. Default True
        :kwarg make_meshgrid:   Flag to automatically make meshgrid of input x and y points. Default False.
        :kwarg dx:              Step size for computing FOM gradient using permittivity perturbations. Default 10e-9
        :kwarg params_debug:    Flag for a debug mode to print parameters on updates. Default False.
    """

    def __init__(self, posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, **kwargs):

        super().__init__(posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, **kwargs)

        #Unpack kwargs
        self.sidewall_angle = kwargs.get('sidewall_angle', 90)
        sidewall_points = kwargs.get('sidewall_res', None)

        if sidewall_points is None:
            self.sidewall_points = self.height_precision
        else:
            self.sidewall_points = sidewall_points

    def add_geo(self, sim, params, only_update):
        '''Adds geometry to a Lumerical simulation as stack of layers to approximate sidewall angle.
        This can be time consuming for many layers. A faster way to implement this would be to combine
        it into a single structure group so Lumerical can handle it all in one operation'''

        if params is None:
            params = self.get_current_params()
        sim.fdtd.switchtolayout()

        #Get composite parameters
        offset_x, offset_y, rx, ry, phi = self.get_from_params(params)
        ravg = (rx + ry)/2

        #Get list of z values that represent tops of each chunk
        zvals = np.linspace(self.z + self.h, self.z, self.sidewall_points)
        heights = -1*np.diff(zvals)

        #Iterate over each chunk
        for i, h in enumerate(heights):
            #Bottom z
            z0 = zvals[i+1]

            #Total distance from top
            dh = i*h

            #Radius scaling factor
            scale_factor = (ravg-dh/np.tan(np.radians(self.sidewall_angle)))/ravg

            #Get modified parameters for layer
            params = self.get_scaled_params(offset_x, offset_y, rx*scale_factor, ry*scale_factor, phi)

            #Add layer to simulation
            self.add_geo_impl(sim, only_update, params, h, z0, groupname = 'Pillars' + str(i))


    def calculate_gradients(self, gradient_fields):
        raise Exception("Gradient calculation not implemented")
        #Could actually probably do this pretty easily using code from super(). Just need to redo
        #section where I pick which points to use.