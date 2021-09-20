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

from datetime import datetime

class MovingMetasurface2D(Geometry):
    """
        Defines a 2D metasurface composed of an array of pillars with given height
    
        :param posx:                Array of shape (N,1) defining initial x-coordinates of each pillar
        :param widths:              Array of shape (N,1) defining initial withds of each pillar
        :param min_feature_size:    Minimum pillar width and minimum gap between pillars
        :param y:                   y-position of bottom of metasurface 
        :param h:                   height of metasurface
        :param height_precision:    Number of points along height of pillar to calculate gradient
        :param eps_in:              Permittivity of the pillars
        :param eps_out:             Permittivity of the material around the pillars
        :param dx:                  Step size for computing FOM gradient using permittivity perturbations

    """

    def __init__(self, posx, init_widths, min_feature_size, y, h, eps_in, eps_out, height_precision = 10, dx = 1.0e-9, scaling_factor = 1, simulation_span = 100e-6):
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
        self.simulation_span = float(simulation_span)

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

        simulation_right = self.simulation_span / 2
        simulation_left = -simulation_right

        eps_in = self.eps_in.get_eps(gradient_fields.forward_fields.wl)
        eps_out = self.eps_out.get_eps(gradient_fields.forward_fields.wl)
        eps_0 = sp.constants.epsilon_0
        wl = gradient_fields.forward_fields.wl

        start_time = datetime.now()
        #Creates ordered arrays of x and y coordinates
        x = np.stack(((self.offsets+self.init_pos-self.widths/2),(self.offsets + self.init_pos + self.widths/2) ), axis = -1).flatten()
        y = np.linspace(self.y, self.y + self.h, self.height_precision) 
        xv, yv = np.meshgrid(x, y, indexing ='ij')

        Ef, Df = MovingMetasurface2D.interpolate_fields(xv, yv, 0, gradient_fields.forward_fields)
        Ea, Da = MovingMetasurface2D.interpolate_fields(xv, yv, 0, gradient_fields.adjoint_fields)

        integrand = 2*np.real(eps_0*(eps_in - eps_out)*np.sum(Ef[:,:,:,1:]*Ea[:,:,:,1:], axis=-1) + 1/eps_0 *(1.0/eps_out - 1.0/eps_in)*Df[:,:,:,0]*Da[:,:,:,0])
        lines = np.trapz(integrand, x = y, axis=1)
        lines = np.reshape(lines, (x.size//2, 2, wl.size))

        #Directionality based on pointing OUT of the boundary or INWARD
        width_deriv = lines[:,0,:] + lines[:,1,:]
        pos_deriv = -lines[:,0,:] + lines[:,1,:]

        total_deriv = np.concatenate((pos_deriv, width_deriv))
        self.gradients.append(total_deriv)
        return total_deriv

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


    def calculate_bounds(self):
        '''Calculates the bounds given the minimum feature size'''
        '''Bounds should be [min_feature_size, inf] for widths, [-inf, inf] for offsets'''

        width_bounds = [(self.min_feature_size*self.scaling_factor, np.inf)] * self.widths.size
        offset_bounds = [(-np.inf, np.inf)] * self.init_pos.size
        return (offset_bounds + width_bounds)

    def build_constraints(self):
        '''Builds the constraint objects given the minimum feature size to ensure minimum gap between structures'''
        cons = []
        #Iterate across all pillars
        for i, value in enumerate(self.init_pos):
            #Skip last element
            if i == self.init_pos.size - 2:
                break

            #Add to list of constraints
            cons.append(
                {'type':'ineq', 
                'fun':lambda x,i: MovingMetasurface2D.general_constraint(x[len(x)//2:], x[:len(x)//2] + self.init_pos*self.scaling_factor, i, i+1, self.min_feature_size*self.scaling_factor), 
                'jac':lambda x,i: MovingMetasurface2D.general_jacobian(x, i, i+1),
                'args': (i,)})

        return cons

    @staticmethod
    def split_params(params, scaling_factor = 1):
        return np.split(params/scaling_factor, 2)

    @staticmethod
    def combine_params(offsets, widths, scaling_factor = 1):
        return np.concatenate((offsets, widths))*scaling_factor

    #Calculates inequality value for two pillars (constraint requires result is greater than zero)
    @staticmethod
    def general_constraint(widths, pos, i, j, min_feature_size):
        return (pos[j] - pos[i]) - 0.5*(widths[j] + widths[i]) - min_feature_size

    #Returns Jacobian for a particular constraint
    @staticmethod
    def general_jacobian(params, i, j):
        num_objects = params.size//2
        jac = np.zeros(params.size)

        jac[i] = -1
        jac[j] = 1
        jac[i+num_objects] = -0.5
        jac[j+num_objects] = -0.5

        return jac

    #Takes in ij meshgrid of points to evaluate and returns interpolated E and D fields at those points
    @staticmethod
    def interpolate_fields(x, y, z, fields):
        #fields.x, fields.y, fields.z, fields.E, fields.D, fields.wl are relevant terms
        #E a 5D matrix in form x:y:z:wl:vector where vector = 0,1,2 for x,y,z components

        
        #Finds meshgrid indices of upper bound x,y,z locations in array
        nx = x.shape[0]
        ny = x.shape[1]

        xi, yi = np.meshgrid(np.searchsorted(fields.x, x[:,0]), np.searchsorted(fields.y, y[0,:]), indexing = 'ij')
        x1 = (fields.x[xi-1]).flatten()
        x2 = (fields.x[xi]).flatten()
        y1 = (fields.y[yi-1]).flatten()
        y2 = (fields.y[yi]).flatten()
        xi = xi.flatten()
        yi = yi.flatten()
        x = x.flatten()
        y = y.flatten()

        #Calculates rectangle areas for bilinear interpolation (Wikipedia)
        denom = (x2-x1)*(y2-y1)
        w11 = ((x2-x)*(y2-y)/denom).reshape(x.size, 1, 1)
        w12 = ((x2-x)*(y-y1)/denom).reshape(x.size, 1, 1)
        w21 = ((x-x1)*(y2-y)/denom).reshape(x.size, 1, 1)
        w22 = ((x-x1)*(y-y1)/denom).reshape(x.size, 1, 1)
        
        E = fields.E[xi-1,yi-1,z,:,:]*w11 + fields.E[xi-1,yi,z,:,:]*w12 + fields.E[xi,yi-1,z,:,:]*w21 + fields.E[xi,yi,z,:,:]*w22
        D = fields.D[xi-1,yi-1,z,:,:]*w11 + fields.D[xi-1,yi,z,:,:]*w12 + fields.D[xi,yi-1,z,:,:]*w21 + fields.D[xi,yi,z,:,:]*w22
        return E.reshape(nx, ny, fields.wl.size, 3), D.reshape(nx, ny, fields.wl.size, 3)
        


class MovingMetasurfaceAnnulus(MovingMetasurface2D):

    """
    Defines a 3D metasurface composed of concentric rings defined by their widths and positions

    :param posr:                Array of shape (N,1) defining initial central r-coordinates of each ring
    :param widths:              Array of shape (N,1) defining initial widths of each ring
    :param min_feature_size:    Minimum ring width and minimum gap between ring
    :param z:                   z-position of bottom of metasurface 
    :param h:                   height of metasurface
    :param height_precision:    Number of points along height of ring to calculate gradient
    :param eps_in:              Permittivity of the rings
    :param eps_out:             Permittivity of the material around the rings
    :param dx:                  Step size for computing FOM gradient using permittivity perturbations

    """

    def __init__(self, posr, init_widths, min_feature_size, z, h, eps_in, eps_out, height_precision = 10, dx = 1.0e-9, scaling_factor = 1, simulation_diameter = 100e-6):
        super().__init__(posx = posr,
                         init_widths = init_widths,
                         min_feature_size = min_feature_size,
                         y = z,
                         h = h,
                         eps_in = eps_in,
                         eps_out = eps_out,
                         height_precision = height_precision,
                         dx = dx,
                         scaling_factor = scaling_factor,
                         simulation_span = simulation_diameter)
        self.z = self.y

    def add_geo(self, sim, params, only_update):
        ''' Adds the geometry to a Lumerical simulation'''

        groupname = 'Rings'
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
            sim.fdtd.set('y', 0)
            sim.fdtd.set('z', self.z)
        self.counter += 1
        sim.fdtd.select(groupname)
        self.create_script(sim, groupname, only_update)
        #Change trivial parameter to force structure to update
        sim.fdtd.set('counter', self.counter)

    def save_to_mat(self, widths, pos):
        '''Saves core parameters to .mat file'''
        scipy.io.savemat('params.mat', mdict={'r': pos, 'height': self.h, 'widths': widths})

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Writes structure group script'''
        struct_script = ('deleteall;\n'
            'data = matlabload("params.mat");\n'
            'for(i=1:length(r)) {\n'
                'addring;\n'
                'set("name", "ring_"+num2str(i));\n'
                'set("x", 0);\n'
                'set("y", 0);\n'
                'set("inner radius", r(i)-widths(i)/2);\n'
                'set("outer radius", r(i)+widths(i)/2);\n'
                'set("z min", 0);\n'
                'set("z max", height);\n'
                'set("theta start", 0);\n'
                'set("theta stop", 0);\n')

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

    def calculate_gradients(self, gradient_fields):
        raise UserWarning("Explicit gradient calculation not implemented. Use deps")

    def build_constraints(self):
    	#Same constraints as before, with one additional one ensuring that the innermost radius is positive
    	cons = super().build_constraints()
    	def jacobian(x):
    		jac = np.zeros(x.size)
    		jac[0] = 1
    		jac[x.size//2] = -0.5
    		return jac

    	cons.append(
    		{'type':'ineq',
    		'fun':lambda x: x[0] + self.init_pos[0]*self.scaling_factor - x[len(x)//2] / 2,
    		'jac':lambda x:jacobian(x)})

    	return cons

