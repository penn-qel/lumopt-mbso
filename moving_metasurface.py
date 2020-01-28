################################################
# Script: moving_metasurface.py

# Description: This script defines a geometry object corresponding to a metasurface not bound to a
# constrained grid, but able to move around within fabrication constraints
# Author: Amelia Klein
###############################################

import sys
import numpy as np
import scipy as sp
import lumapi
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.utilities.wavelengths import Wavelengths

class MovingMetasurface2D(Geometry):
    """
        Defines a 2D metasurface composed of an array of pillars with given height
    
        :param posx:                Array of shape (N,1) defining initial x-coordinates of each pillar
        :param widths:              Array of shape (N,1) defining initial withds of each pillar
        :param min_feature_size:    Minimum pillar width and minimum gap between pillars
        :param z:                   z-position of bottom of metasurface 
        :param h:                   height of metasurface
        :param height_precision:    Number of points along height of pillar to calculate gradient
        :param eps_in:              Permittivity of the pillars
        :param eps_out:             Permittivity of the material around the pillars
        :param dx:                  Step size for computing FOM gradient using permittivity perturbations

    """

    def __init__(self, posx, init_widths, min_feature_size, y, h, eps_in, eps_out, height_precision = 10, dx = 1.0e-9, scaling_factor = 1):
        self.init_pos = posx
        self.widths = init_widths
        self.offsets = np.zeros(posx.size)
        self.y = float(y)
        self.h = float(h)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.height_precision = int(height_precision)

        if self.h <= 0:
            raise UserWarning("pillar height must be positive.")

        self.dx = float(dx)
        if self.dx < 0.0:
            raise UserWarning("step size must be positive.")
        self.gradients = list()

        self.min_feature_size = float(min_feature_size)
        self.scaling_factor = scaling_factor
        self.bounds = self.calculate_bounds()

    def add_geo(self, sim, params, only_update):
        ''' Adds the geometry to a Lumerical simulation'''

        groupname = 'Pillars'
        if params is None:
            widths = self.widths
            offsets = self.offsets
        else:
            offsets, widths = MovingMetasurface2D.split_params(params, self.scaling_factor)

        #Saves current data to a .mat file so that structure group script can access it
        #Building many objects within a structure group script is MUCH faster than individually
        self.save_to_mat(widths, self.init_pos + offsets)
        sim.fdtd.switchtolayout()

        if not only_update:
            sim.fdtd.addstructuregroup()
            sim.fdtd.set('name', groupname)
            #sim.fdtd.adduserprop('index', 0, self.n)
            self.counter = 0
            sim.fdtd.adduserprop('counter', 0, 0)
            sim.fdtd.set('x', 0)
            sim.fdtd.set('y', self.y)
        self.counter += 1
        sim.fdtd.select(groupname)
        self.create_script(sim, groupname, only_update)
        #Change trivial parameter to force structure to update
        sim.fdtd.set('counter', self.counter)

    def update_geometry(self, params, sim = None):
        '''Sets the widths. Allow option of also setting positions in future?'''
        self.offsets, self.widths = MovingMetasurface2D.split_params(params, self.scaling_factor)

    def calculate_gradients(self, gradient_fields):
        '''Gradients are calculated by increasing the radius of each pillar'''
        posx = self.offsets + self.init_pos
        width_gradients = list()
        pos_gradients = list()
        for i, w in enumerate(self.widths):
            width_gradients.append(self.pillar_derivative(posx[i], w, gradient_fields))
            pos_gradients.append(self.pos_derivative(posx[i], w, gradient_fields))
        self.gradients.append(np.array(pos_gradients + width_gradients))
        return self.gradients[-1]

    def get_current_params(self):
        return MovingMetasurface2D.combine_params(self.offsets, self.widths, self.scaling_factor)

    def save_to_mat(self, widths, pos):
        '''Saves core parameters to .mat file'''
        scipy.io.savemat('params.mat', mdict={'x': pos, 'height': self.h, 'widths': widths})

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Writes structure group script'''
        struct_script = ('deleteall;\n'
            'data = matlabload("params.mat");\n'
            'for(i=1:length(x)) {\n'
                'addrect;\n'
                'set("name", "pillar_"+num2str(i));\n'
                'set("x", x(i));\n'
                'set("y min", 0);\n'
                'set("y max", height);\n'
                'set("x span", widths(i));\n')

        sim.fdtd.select(groupname)
        if self.eps_in.mesh_order:
            struct_script += 'set("override mesh order from material database", true);\nset("mesh order", mesh_order);\n'
            if not only_update:
                sim.fdtd.adduserprop('mesh_order', 0, self.eps_in.mesh_order)

        if self.eps_in.name == str('<Object defined dielectric>'):
            struct_script += 'set("index", index);\n}'
            if not only_update:
                sim.fdtd.adduserprop('index', 0, np.sqrt(self.eps_in.base_epsilon))
            sim.fdtd.set('script', struct_script)
        else:
            struct_script += 'set("material", mat);\n}'
            if not only_update:
                sim.fdtd.adduserprop('mat', 5, self.eps_in.name)

                #Set permittivity parameter in material (normally done in set_script function)
                self.eps_in.wavelengths = Material.get_wavelengths(sim)
                freq_array = sp.constants.speed_of_light / self.eps_in.wavelengths.asarray()
                fdtd_index = sim.fdtd.getfdtdindex(self.eps_in.name, freq_array, float(freq_array.min()), float(freq_array.max()))
                self.eps_in.permittivity = np.asarray(np.power(fdtd_index, 2)).flatten()
            sim.fdtd.set('script', struct_script)


    def plot(self, ax):
        x = (self.offsets + self.init_pos)*1e6
        w = self.widths.copy()*1e6
        maxwidth = np.amax(w)
        ax.clear()
        for indx, rad in enumerate(w):
            rect = patches.Rectangle((x[indx] - w[indx]/2, 0), w[indx], self.h*1e6, facecolor='black')
            ax.add_patch(rect)
        ax.set_title('Geometry')
        ax.set_ylim(0, self.h*1e6)
        ax.set_xlim(min(x) - maxwidth/2, max(x) + maxwidth/2)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return True

    def pillar_derivative(self, x0, w, gradient_fields):
        '''Calculates derivative for a particular pillar width'''

        #Parameterize surface by z
        yv = np.linspace(self.y,self.y + self.h, self.height_precision)

        integrand_fun = gradient_fields.boundary_perturbation_integrand()
        wavelengths = gradient_fields.forward_fields.wl
        eps_in = self.eps_in.get_eps(wavelengths)
        eps_out = self.eps_out.get_eps(wavelengths)

        #Create a list of derivatives, calculated for each wavelength
        derivs = list()
        for idx, wl in enumerate(wavelengths):
            #Integrate across surface of pillar for each wavelength
            integrand_per_wl = np.zeros(yv.size)
            for i, y in enumerate(yv):
                #Calculate for each edge
                norm1 = np.array([1, 0, 0])
                norm2 = np.array([-1, 0, 0])
                integrandright = integrand_fun(x0+w/2,y,0,wl,norm1,eps_in[idx], eps_out[idx])
                integrandleft = integrand_fun(x0-w/2,y,0,wl,norm2, eps_in[idx], eps_out[idx])
                integrand_per_wl[i] = (integrandleft + integrandright) / 2
            #Perform integral for each wavelength
            derivs.append(np.trapz(y = integrand_per_wl, x = yv))

        return np.array(derivs).flatten()

    def pos_derivative(self, x0, w, gradient_fields):
        '''Calculates derivative for a particular pillar offset'''

        #Parameterize surface by z
        yv = np.linspace(self.y,self.y + self.h, self.height_precision)

        integrand_fun = gradient_fields.boundary_perturbation_integrand()
        wavelengths = gradient_fields.forward_fields.wl
        eps_in = self.eps_in.get_eps(wavelengths)
        eps_out = self.eps_out.get_eps(wavelengths)

        #Create a list of derivatives, calculated for each wavelength
        derivs = list()
        for idx, wl in enumerate(wavelengths):
            #Integrate across surface of pillar for each wavelength
            integrand_per_wl = np.zeros(yv.size)
            for i, y in enumerate(yv):
                #Calculate for each edge
                norm1 = np.array([1, 0, 0])
                norm2 = np.array([1, 0, 0])
                integrandright = integrand_fun(x0+w/2,y,0,wl,norm1,eps_in[idx], eps_out[idx])
                integrandleft = integrand_fun(x0-w/2,y,0,wl,norm2, eps_in[idx], eps_out[idx])
                integrand_per_wl[i] = (integrandleft + integrandright) / 2
            #Perform integral for each wavelength
            derivs.append(np.trapz(y = integrand_per_wl, x = yv))

        return np.array(derivs).flatten()

    def calculate_bounds(self):
        '''Calculates the bounds given the minimum feature size'''
        '''Bounds should be [min_feature_size, inf] for widths, [-inf, inf] for offsets'''

        width_bounds = [(self.min_feature_size*self.scaling_factor, np.inf)] * self.widths.size
        offset_bounds = [(-np.inf, np.inf)] * self.init_pos.size
        return (offset_bounds + width_bounds)

    def build_constraints(self):
        '''Builds the constraint objects given the minimum feature size to ensure minimum gap between structures'''

        #Calculates inequality value for two pillars (constraint requires result is greater than zero)
        def general_constraint(widths, pos, i, j, min_feature_size):
            return (pos[j] - pos[i]) - 0.5*(widths[j] + widths[i]) - min_feature_size
            #return (pos[j] - pos[i]) - 0.5*(widths[j] + widths[i]) - min_feature_size

        #Returns Jacobian for a particular constraint
        def general_jacobian(params, i, j):
            num_objects = params.size//2
            jac = np.zeros(params.size)

            jac[i] = -1
            jac[j] = 1
            jac[i+num_objects] = -0.5
            jac[j+num_objects] = -0.5

            return jac

        cons = []
        #Iterate across all pillars
        for i, value in enumerate(self.init_pos):
            #Skip last element
            if i == self.init_pos.size - 2:
                break

            #Define callable constraint function for optimization algorithm
            def constraint(x):
                #First half of array is offsets, second half is widths
                offsets, widths = np.split(x,2)
                return general_constraint(widths, offsets + self.init_pos, i, i+1, self.min_feature_size)

            #Callable jacobian for this constraint
            def jacobian(x):
                return general_jacobian(x, i, i+1)

            #Add to list of constraints
            cons.append({'type':'ineq', 'fun':constraint, 'jac': jacobian})

        return cons

    @staticmethod
    def split_params(params, scaling_factor = 1):
        return np.split(params/scaling_factor, 2)

    @staticmethod
    def combine_params(offsets, widths, scaling_factor = 1):
        return np.concatenate((offsets, widths))*scaling_factor
        
