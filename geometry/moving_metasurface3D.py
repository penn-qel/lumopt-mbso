###################################################################
# Class: moving_metasurface3D.py

# Description: this class defines a geometry object corresponding to a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints
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
from utils.interpolate_fields import interpolate_fields
from utils.get_fields_from_cad import get_fields_from_cad

class MovingMetasurface3D(Geometry):
    """Defines object consisting of array of elliptical pillars, where axes lengths, positions, and
    rotations are all free optimization variables.

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
        
        #Unpack kwargs
        phi = kwargs.get('phi', None)
        pillars_rotate = kwargs.get('pillars_rotate', True)
        height_precision = kwargs.get('height_precision', 10)
        angle_precision = kwargs.get('angle_precision', 20)
        scaling_factor = kwargs.get('scaling_factor', 1)
        phi_scaling = kwargs.get('phi_scaling', 1/180)
        limit_nearest_neighbor_cons = kwargs.get('limit_nearest_neighbor_cons', True)
        make_meshgrid = kwargs.get('make_meshgrid', False)
        dx = kwargs.get('dx', 10e-9)
        params_debug = kwargs.get('params_debug', False)


        self.init_x = posx.flatten()
        self.init_y = posy.flatten()
        self.rx = rx.flatten()
        self.ry = ry.flatten()
        
        #Option for constructing meshgrid out of x and y arrays automatically
        if make_meshgrid:
            x0, y0 = np.meshgrid(self.init_x, self.init_y, indexing='ij')
            rx0, ry0 = np.meshgrid(self.rx, self.ry, indexing = 'ij')
            self.init_x = x0.flatten()
            self.init_y = y0.flatten()
            self.rx = rx0.flatten()
            self.ry = ry0.flatten()

        self.offset_x = np.zeros(self.init_x.size).flatten()
        self.offset_y = np.zeros(self.init_x.size).flatten()
        self.z = float(z)
        self.h = float(h)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.height_precision = int(height_precision)
        self.angle_precision = int(angle_precision)

        if phi is None:
            self.phi = np.zeros(self.rx.size).flatten()
        else:
            self.phi = phi.flatten()

        if self.h <= 0:
            raise UserWarning("pillar height must be positive.")

        if not(self.init_x.size == self.init_y.size == self.rx.size == self.ry.size == self.phi.size):
            raise UserWarning('Initial parameter arrays must have same shape (N,)')

        self.pillars_rotate = pillars_rotate
        self.min_feature_size = float(min_feature_size)
        self.scaling_factor = scaling_factor
        self.phi_scaling = phi_scaling
        self.dx = dx

        self.bounds = self.calculate_bounds()
        self.limit_nearest_neighbor_cons = limit_nearest_neighbor_cons
        self.params_debug = params_debug

        #Assert we have a square grid for nearest neighbor calculations by checking N_pillars is a perfect square
        if self.limit_nearest_neighbor_cons:
            if make_meshgrid:
                self.grid_shape = (posx.size, posy.size)
            else:
                N = int(np.sqrt(posx.size) + 0.5)
                if posx.size == N**2:
                    self.grid_shape = (N, N)
                else:
                    raise UserWarning("Must do built-in meshgrid or use a perfect square of pillars when constraining nearest neighbors only")

    def add_geo_impl(self, sim, only_update, params, h, z, groupname = 'Pillars'):
        '''Called by add_geo. Implements actual pushing of parameters to Lumerical'''

        offset_x, offset_y, rx, ry, phi = self.get_from_params(params)

        if not only_update:
            sim.fdtd.addstructuregroup()
            sim.fdtd.set('name', groupname)
            sim.fdtd.set('x', 0)
            sim.fdtd.set('y', 0)
            sim.fdtd.set('z', 0)

            #Set parameters as user props
            sim.fdtd.adduserprop('posx', 6, offset_x + self.init_x)
            sim.fdtd.adduserprop('posy', 6, offset_y + self.init_y)
            sim.fdtd.adduserprop('rx', 6, rx)
            sim.fdtd.adduserprop('ry', 6, ry)
            sim.fdtd.adduserprop('phi', 6, phi)
            sim.fdtd.adduserprop('height', 0, h)
            sim.fdtd.adduserprop('z0', 0, z)

        sim.fdtd.select(groupname)
        sim.fdtd.set('posx', offset_x + self.init_x)
        sim.fdtd.set('posy', offset_y + self.init_y)
        sim.fdtd.set('rx', rx)
        sim.fdtd.set('ry', ry)
        sim.fdtd.set('phi', phi)
        self.create_script(sim, groupname, only_update)

    def add_geo(self, sim, params, only_update):
        '''Adds the geometry to a Lumerical simulation'''

        if params is None:
            #Uses interally saved parameters
            params = self.get_current_params()
        sim.fdtd.switchtolayout()

        self.add_geo_impl(sim, only_update, params, self.h, self.z)

    def update_geometry(self, params, sim = None):
        '''Updates internal values of parameters according to input'''
        self.offset_x, self.offset_y, self.rx, self.ry, self.phi = self.get_from_params(params)
        if self.params_debug:
            self.print_current_params()

    def calculate_gradients(self, gradient_fields):
        '''Calculates gradient at each wavelength with respect to all parameters'''

        #Ellipse described by x = x0 + rx*cos(theta), y = y0 + ry*sin(theta) 
        eps_in = self.eps_in.get_eps(gradient_fields.forward_fields.wl)
        eps_out = self.eps_out.get_eps(gradient_fields.forward_fields.wl)
        eps_0 = sp.constants.epsilon_0
        wl = gradient_fields.forward_fields.wl
        phi = np.radians(self.phi).reshape((self.phi.size, 1, 1))

        z = np.linspace(self.z, self.z + self.h, self.height_precision)
        theta = np.linspace(0, 2*np.pi, self.angle_precision)
        pillars = np.arange(self.rx.size)
        pillarv, thetav, zv = np.meshgrid(pillars, theta, z, indexing = 'ij')

        u = self.rx[pillarv]*np.cos(thetav)
        v = self.ry[pillarv]*np.sin(thetav)
        xv = self.offset_x[pillarv] + self.init_x[pillarv] + u*np.cos(phi) - v*np.sin(phi)
        yv = self.offset_y[pillarv] + self.init_y[pillarv] + u*np.sin(phi) + v*np.cos(phi)

        Ef, Df = interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.forward_fields)
        Ea, Da = interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.adjoint_fields)

        Ef = Ef.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Df = Df.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Ea = Ea.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Da = Da.reshape(pillars.size, theta.size, z.size, wl.size, 3)


        #Calculate surface normal vectors
        phiv = np.reshape(phi, (phi.size, 1, 1))
        nx = (self.ry[pillarv]*np.cos(thetav)*np.cos(phiv) - self.rx[pillarv]*np.sin(thetav)*np.sin(phiv)).reshape(pillars.size, theta.size, z.size, 1)
        ny = (self.rx[pillarv]*np.sin(thetav)*np.cos(phiv) + self.ry[pillarv]*np.cos(thetav)*np.sin(phiv)).reshape(pillars.size, theta.size, z.size, 1)

        nlength = np.sqrt(np.square(nx) + np.square(ny))
        nx = nx/nlength
        ny = ny/nlength

        normal = np.zeros((pillars.size, theta.size, z.size, 1, 3))
        normal[:,:,:,:,0] = nx
        normal[:,:,:,:,1] = ny

        def project(a,n):
            return np.expand_dims(np.sum(a*n, axis=-1), axis = 4) * n

        Dfperp = project(Df, normal)
        Daperp = project(Da, normal)
        Efpar = Ef - project(Ef, normal)
        Eapar = Ea - project(Ea, normal)

        #Calculates derivatives of level set, scaled to use opt parameters rather than real ones
        A = (xv - self.offset_x[pillarv] - self.init_x[pillarv])*np.cos(phiv) + (yv - self.offset_y[pillarv] - self.init_y[pillarv])*np.sin(phiv)
        B = -(xv - self.offset_x[pillarv] - self.init_x[pillarv])*np.sin(phiv) + (yv - self.offset_y[pillarv] - self.init_y[pillarv])*np.cos(phiv)
        d_dx = np.expand_dims((-2*A*np.cos(phi)/np.power(self.rx[pillarv], 2) + 2*B*np.sin(phi)/np.power(self.ry[pillarv], 2))/self.scaling_factor, axis=3)
        d_dy = np.expand_dims((-2*A*np.sin(phi)/np.power(self.rx[pillarv], 2) - 2*B*np.cos(phi)/np.power(self.ry[pillarv], 2))/self.scaling_factor, axis=3)
        d_drx = np.expand_dims((-2*np.power(A, 2)/np.power(self.rx[pillarv], 3))/self.scaling_factor, axis=3)
        d_dry = np.expand_dims((-2*np.power(B, 2)/np.power(self.rx[pillarv], 3))/self.scaling_factor, axis=3)
        d_dphi = np.expand_dims((-2*A*B*(np.power(self.rx[pillarv],2) - np.power(self.ry[pillarv],2))/(np.power(self.rx[pillarv],2)*np.power(self.ry[pillarv],2)))/self.phi_scaling, axis=3)

        if self.pillars_rotate:
            grad_mag = np.sqrt(d_dx**2 + d_dy**2 + d_drx**2 + d_dry**2 + d_dphi**2)
        else:
            grad_mag = np.sqrt(d_dx**2 + d_dy**2 + d_drx**2 + d_dry**2)

        #Calculates integrand according to Eq 5.40 in Owen Miller's thesis. Multiplies by arc length parameter in order to integrate over parameterized ellipse
        integrand = -2/grad_mag*np.real(eps_0 * (eps_in - eps_out) *np.sum(Efpar*Eapar, axis=-1) + (1/eps_out - 1/eps_in) /eps_0 * np.sum(Dfperp*Daperp, axis=-1))
        curve_integrand = integrand * np.expand_dims(np.sqrt(np.square(self.rx[pillarv]*np.sin(thetav)) + np.square(self.ry[pillarv]*np.cos(thetav))), axis = 3)

        deriv_x = np.trapz(np.trapz(d_dx*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_y = np.trapz(np.trapz(d_dy*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_rx = np.trapz(np.trapz(d_drx*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_ry = np.trapz(np.trapz(d_dry*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_phi = np.trapz(np.trapz(d_dphi*curve_integrand, z, axis = 2), theta, axis = 1)
        
        if self.pillars_rotate:
            total_deriv = np.concatenate((deriv_x, deriv_y, deriv_rx, deriv_ry, deriv_phi))
        else:
            total_deriv = np.concatenate((deriv_x, deriv_y, deriv_rx, deriv_ry))
        return total_deriv

    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        '''Semi hack to reduce memory usage of gradient calculation. Actual calculation of gradients still done in Python
        but only one instance of field data exists at a time'''

        #Store scaling weights in CAD
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        #Pull and delete fields from CAD
        forward_fields = get_fields_from_cad(sim.fdtd,
                            field_result_name = forward_fields,
                            get_eps = True,
                            get_D = True,
                            get_H = False,
                            nointerpolation = True,
                            clear_result = True)
        adjoint_fields = get_fields_from_cad(sim.fdtd,
                            field_result_name = adjoint_fields,
                            get_eps = True,
                            get_D = True,
                            get_H = False,
                            nointerpolation = True,
                            clear_result = True)
        
        adjoint_fields.scale(3, wl_scaling_factor)
        grad = self.calculate_gradients(GradientFields(forward_fields, adjoint_fields))

        #Scale by wavelength and reshape for compatability with gradient wavelength integral on cad expectations
        total_deriv = np.reshape(grad, (grad.shape[0], 1, grad.shape[1]))

        #Stores result back in CAD
        lumapi.putMatrix(sim.fdtd.handle, 'total_deriv', total_deriv)

        #Return name of result in CAD
        return 'total_deriv'

    def get_scaled_params(self, offset_x, offset_y, rx, ry, phi = None):
        '''Retrieves correctly scaled individual parameter values'''
        s1 = self.scaling_factor
        s2 = self.phi_scaling
        if self.pillars_rotate:
            return np.concatenate((offset_x*s1, offset_y*s1, rx*s1, ry*s1, phi*s2))
        else:
            return np.concatenate((offset_x*s1, offset_y*s1, rx*s1, ry*s1))


    def get_current_params(self):
        '''Returns list of params as single array'''
        return self.get_scaled_params(self.offset_x, self.offset_y, self.rx, self.ry, self.phi)

    def get_from_params(self, params):
        '''Retrieves correctly scaled individual parameter values from list of params'''
        if self.pillars_rotate:
            offset_x, offset_y, rx, ry, phi = np.split(params, 5)
        else:
            offset_x, offset_y, rx, ry = np.split(params, 4)
            phi = self.phi*self.phi_scaling
        s1 = self.scaling_factor
        s2 = self.phi_scaling
        return offset_x/s1, offset_y/s1, rx/s1, ry/s1, phi/s2


    def plot(self, ax):
        '''Plots current geometry'''
        x = (self.offset_x + self.init_x)*1e6
        y = (self.offset_y + self.init_y)*1e6
        rx = self.rx.copy()*1e6
        ry = self.ry.copy()*1e6
        maxr = max(np.amax(rx), np.amax(ry))
        ax.clear()
        for i, xval in enumerate(x):
            ellipse = patches.Ellipse((xval, y[i]), 2*rx[i], 2*ry[i], angle = self.phi[i], facecolor='black')
            ax.add_patch(ellipse)
        ax.set_title('Geometry')
        ax.set_xlim(min(x) - maxr, max(x) + maxr)
        ax.set_ylim(min(y) - maxr, max(y) + maxr)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return True

    def calculate_bounds(self):
        '''Calculates bounds given the minimum feature size'''
        '''Bounds should be [min_feature_size/2, inf] for radiii and [-inf, inf] for offsets'''
        radius_bounds = [(self.min_feature_size*self.scaling_factor/2, np.inf)]*self.rx.size*2
        offset_bounds = [(-np.inf, np.inf)]*self.rx.size*2
        phi_bounds = [(-np.inf, np.inf)]*(self.rx.size)
        if self.pillars_rotate:
            return (offset_bounds + radius_bounds + phi_bounds)
        else:
            return (offset_bounds + radius_bounds)
           

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Creates Lumerical script for structure group'''
        struct_script = ('deleteall;\n'
                'for(i=1:length(posx)) {\n'
                    'addcircle;\n'
                    'set("name", "pillar_"+num2str(i));\n'
                    'set("make ellipsoid", true);\n'
                    'set("first axis", "z");\n'
                    'set("rotation 1", phi(i));\n'
                    'set("x", posx(i));\n'
                    'set("y", posy(i));\n'
                    'set("z min", z0);\n'
                    'set("z max", z0+height);\n'
                    'set("radius", rx(i));\n'
                    'set("radius 2", ry(i));\n')

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

    def print_current_params(self, scaled = False):
        params = self.get_current_params()
        if self.pillars_rotate:
            offset_x, offset_y, rx, ry, phi = np.split(params, 5)
        else:
            offset_x, offset_y, rx, ry, phi = np.split(params, 4)
        print('x offset: ')
        print(offset_x) if scaled else print(self.offset_x)
        print('y offset: ')
        print(offset_y) if scaled else print(self.offset_y)
        print('rx:')
        print(rx) if scaled else print(self.rx)
        print('ry:')
        print(ry) if scaled else print(self.ry)
        if self.pillars_rotate:
            print('phi:')
            print(phi) if scaled else print(self.phi)

    @staticmethod
    def get_params_from_existing_simulation(filename, get_wavelengths = False):
        '''Opens an existing .fsp file that was used in an optimization and retrieves geometry 
        Returns parameters as a dict'''

        #Open simulation file
        sim = Simulation('./', use_var_fdtd = False, hide_fdtd_cad = True)
        sim.load(filename)

        #Create dict to store parameters
        params = dict()

        #Retrieve parameters
        sim.fdtd.select('Pillars')
        params['posx'] = sim.fdtd.get('posx').flatten()
        params['posy'] = sim.fdtd.get('posy').flatten()
        params['rx'] = sim.fdtd.get('rx').flatten()
        params['ry'] = sim.fdtd.get('ry').flatten()
        params['phi'] = sim.fdtd.get('phi').flatten()
        params['z'] = sim.fdtd.get('z0')
        params['h'] = sim.fdtd.get('height')

        if get_wavelengths:
            f = sim.fdtd.getglobalmonitor('custom frequency samples').flatten()
            params['wl'] = scipy.constants.speed_of_light/f

        sim.fdtd.close()

        return params

    @staticmethod
    def create_from_existing_simulation(filename, get_wavelengths = False):
        '''Creates a geometry object based on an existing structure saved in a .fsp sim file.
        For use with analysis only, as will put dummy parameters for inputs related to optimization'''

        params = MovingMetasurface3D.get_params_from_existing_simulation(filename, get_wavelengths)

        geom = MovingMetasurface3D(posx = params['posx'], posy = params['posy'], rx = params['rx'], ry = params['ry'], 
            min_feature_size = 0, z = params['z'], h = params['h'], eps_in = 0, eps_out = 2.4**2, 
            phi = params['phi'], scaling_factor = 1, phi_scaling = 1, limit_nearest_neighbor_cons = False)

        if get_wavelengths:
            return geom, params['wl']
        else:
            return geom