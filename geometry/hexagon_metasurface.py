###################################################################
# Class: hexagon_metasurface.py

# Description: this class defines a geometry object corresponding to a 3D
# metasurface of hexagons allows to move around within fabrication constraints
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

class HexagonMetasurface(Geometry):
    """Defines object consisting of array of elliptical pillars, where axes lengths, positions, and
    rotations are all free optimization variables.

    Parameters
    ----------------
        :param posx:            Array of shape (N,) defining initial x-coordinates of pillar centers
        :param posy:            Array of shape (N,) defining initial y-coordinates of pillar centers
        :param r:               Array of shape (N,) defining radius of circle that hexagon inscribes
        :param min_feature_size:    Scalar that determines minimum hexagon diameter and spacing.
        :param z:               z-position of bottom of metasurface
        :param h:               height of metasurface pillars
        :param eps_in:          Permittivity inside hexagons
        :param eps_out:         Permittivity of surrounding material

    Optional kwargs
    ---------------
        :kwarg height_precision:Number of points along height of each pillar used to calculate gradient. Default 10
        :kwarg edge_precision: Number of points along each edge of hexagon used to calculate gradient. Default 4
        :kwarg scaling_factor:  Factor to scale all position and radius parameters. Default 1
        :kwarg limit_nearest_neighbor_cons: Flag to limit constraints to nearest neighbors in initial grid. Default True
        :kwarg make_meshgrid:   Flag to automatically make meshgrid of input x and y points. Default False.
        :kwarg dx:              Step size for computing FOM gradient using permittivity perturbations. Default 10e-9
        :kwarg params_debug:    Flag for a debug mode to print parameters on updates. Default False.
    """

    def __init__(self, posx, posy, r, min_feature_size, z, h, eps_in, eps_out, **kwargs):
        
        #Unpack kwargs
        height_precision = kwargs.get('height_precision', 10)
        edge_precision = kwargs.get('edge_precision', 4)
        scaling_factor = kwargs.get('scaling_factor', 1)
        limit_nearest_neighbor_cons = kwargs.get('limit_nearest_neighbor_cons', True)
        make_meshgrid = kwargs.get('make_meshgrid', False)
        dx = kwargs.get('dx', 10e-9)
        params_debug = kwargs.get('params_debug', False)


        self.init_x = posx.flatten()
        self.init_y = posy.flatten()
        self.r = r.flatten()
        
        #Option for constructing meshgrid out of x and y arrays automatically
        if make_meshgrid:
            x0, y0 = np.meshgrid(self.init_x, self.init_y, indexing='ij')
            r1, r2 = np.meshgrid(self.r, self.r, indexing = 'ij')
            self.init_x = x0.flatten()
            self.init_y = y0.flatten()
            self.r = r1.flatten()

        self.offset_x = np.zeros(self.init_x.size).flatten()
        self.offset_y = np.zeros(self.init_x.size).flatten()
        self.z = float(z)
        self.h = float(h)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.height_precision = int(height_precision)
        self.edge_precision = int(edge_precision)

        if self.h <= 0:
            raise UserWarning("pillar height must be positive.")

        if not(self.init_x.size == self.init_y.size == self.r.size):
            raise UserWarning('Initial parameter arrays must have same shape (N,)')

        self.min_feature_size = float(min_feature_size)
        self.scaling_factor = scaling_factor
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

        offset_x, offset_y, r = self.get_from_params(params)
        points = self.get_vertex_matrix(params)

        if not only_update:
            sim.fdtd.addstructuregroup()
            sim.fdtd.set('name', groupname)
            sim.fdtd.set('x', 0)
            sim.fdtd.set('y', 0)
            sim.fdtd.set('z', 0)
            sim.fdtd.adduserprop('posx', 6, offset_x + self.init_x)
            sim.fdtd.adduserprop('posy', 6, offset_y + self.init_y)
            sim.fdtd.adduserprop('r', 6, r)
            sim.fdtd.adduserprop('points', 6, points)
            sim.fdtd.adduserprop('height', 0, h)
            sim.fdtd.adduserprop('z0', 0, z)

        sim.fdtd.select(groupname)
        sim.fdtd.set('posx', offset_x + self.init_x)
        sim.fdtd.set('posy', offset_y + self.init_y)
        sim.fdtd.set('r', r)
        sim.fdtd.set('points', points)

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
        self.offset_x, self.offset_y, self.r = self.get_from_params(params)
        if self.params_debug:
            self.print_current_params()

    def calculate_gradients(self, gradient_fields):
        '''Calculates gradient at each wavelength with respect to all parameters'''
 
        #Split into six segments, parameterized by pairs of vertices
        eps_in = self.eps_in.get_eps(gradient_fields.forward_fields.wl)
        eps_out = self.eps_out.get_eps(gradient_fields.forward_fields.wl)
        eps_0 = sp.constants.epsilon_0
        wl = gradient_fields.forward_fields.wl

        z = np.linspace(self.z, self.z + self.h, self.height_precision)

        #parameterize line segments for each hex. NOTE: do we include endpoints??
        t = np.linspace(0,1,self.edge_precision+1, endpoint=False)
        t = t[1:]

        hexes = np.arange(self.r.size)
        segments = np.arange(6)

        #Nx6xedge_precisionxheight_precision matrix of points
        hexv, segmentv, tv, zv = np.meshgrid(hexes, segments, t, z, indexing='ij')

        x0 = self.offset_x[hexv] + self.init_x[hexv]
        y0 = self.offset_y[hexv] + self.init_y[hexv]
        r = self.r[hexv]

        #Calculate (x,y) for all points
        xv, yv = HexagonMetasurface.get_hex_point(x0, y0, r, segmentv, tv)

        #Get fields
        Ef, Df = interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.forward_fields)
        Ea, Da = interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.adjoint_fields)

        fieldshape = (hexes.size, 6, t.size, z.size, wl.size, 3)
        Ef = Ef.reshape(fieldshape)
        Df = Df.reshape(fieldshape)
        Ea = Ea.reshape(fieldshape)
        Da = Da.reshape(fieldshape)

        #Get normal vectors as (Nx6xt.sizexz.size) matrix
        midx, midy = HexagonMetasurface.get_hex_point(x0, y0, r, segmentv, 0.5)
        nx = midx - x0 
        ny = midy - y0

        #Normalize normal vectors
        nlength = np.sqrt(np.square(nx) + np.square(ny))
        nx = (nx/nlength).reshape(hexes.size,6,t.size,z.size,1)
        ny = (ny/nlength).reshape(hexes.size,6,t.size,z.size,1)

        normal = np.zeros((hexes.size, 6, t.size, z.size, 1, 3))
        normal[:,:,:,:,:,0] = nx
        normal[:,:,:,:,:,1] = ny

        #Separate parallel and perpendicular fields
        def project(a,n):
            return np.expand_dims(np.sum(a*n, axis=-1), axis = 5) * n

        Dfperp = project(Df, normal)
        Daperp = project(Da, normal)
        Efpar = Ef - project(Ef, normal)
        Eapar = Ea - project(Ea, normal)

        #Factors to multiply for each derivative (dot product of normal vector with perturbation vector)
        d_dx = normal[:,:,:,:,:,0]
        d_dy = normal[:,:,:,:,:,1]
        d_dr = 1

        #Adjoint integrand
        integrand = 2*np.real(eps_0*(eps_in-eps_out)*np.sum(Efpar*Eapar, axis=-1)+ (1/eps_out - 1/eps_in)/eps_0*np.sum(Dfperp*Daperp, axis=-1))
        #Multiply by arc length of segment (r)
        curve_integrand = integrand*np.expand_dims(r, axis=4)

        #Integrate over z and t, sum over segments
        deriv_x = np.sum(np.trapz(np.trapz(d_dx*curve_integrand, z, axis=3),t,axis=2),axis=1)
        deriv_y = np.sum(np.trapz(np.trapz(d_dy*curve_integrand, z, axis=3),t,axis=2),axis=1)
        deriv_r = np.sum(np.trapz(np.trapz(d_dr*curve_integrand, z, axis=3),t,axis=2),axis=1)

        return np.concatenate((deriv_x, deriv_y, deriv_r))/self.scaling_factor


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

    def get_scaled_params(self, offset_x, offset_y, r):
        '''Retrieves correctly scaled individual parameter values'''
        s = self.scaling_factor
        return np.concatenate((offset_x*s, offset_y*s, r*s))


    def get_current_params(self):
        '''Returns list of params as single array'''
        return self.get_scaled_params(self.offset_x, self.offset_y, self.r)

    def get_from_params(self, params):
        '''Retrieves correctly scaled individual parameter values from list of params'''
        offset_x, offset_y, r = np.split(params, 3)
        s = self.scaling_factor
        return offset_x/s, offset_y/s, r/s


    def plot(self, ax):
        '''Plots current geometry'''
        x = (self.offset_x + self.init_x)*1e6
        y = (self.offset_y + self.init_y)*1e6
        r = self.r.copy()*1e6
        maxr = np.amax(r)
        ax.clear()
        for i, xval in enumerate(x):
            hexagon = patches.RegularPolygon((xval, y[i]), 6, radius=r[i], orientation = np.pi/2, facecolor='black')
            ax.add_patch(hexagon)
        ax.set_title('Geometry')
        ax.set_xlim(min(x) - maxr, max(x) + maxr)
        ax.set_ylim(min(y) - maxr, max(y) + maxr)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return True

    def calculate_bounds(self):
        '''Calculates bounds given the minimum feature size'''
        '''Bounds should be [min_feature_size/2, inf] for radiii and [-inf, inf] for offsets'''
        radius_bounds = [(self.min_feature_size*self.scaling_factor/2, np.inf)]*self.r.size
        offset_bounds = [(-np.inf, np.inf)]*self.r.size*2
        return (offset_bounds + radius_bounds)
           

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Creates Lumerical script for structure group'''
        struct_script = ('deleteall;\n'
                'for(i=1:length(posx)) {\n'
                    'addpoly;\n'
                    'set("name", "pillar_"+num2str(i));\n'
                    'set("x", 0);\n'
                    'set("y", 0);\n'
                    'set("z min", z0);\n'
                    'set("z max", z0+height);\n'
                    'vertices = reshape(points(i,:,:), [6,2]);\n'
                    'set("vertices", vertices);\n')

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
        offset_x, offset_y, r = np.split(params, 3)
        print('x offset: ')
        print(offset_x) if scaled else print(self.offset_x)
        print('y offset: ')
        print(offset_y) if scaled else print(self.offset_y)
        print('r:')
        print(r) if scaled else print(self.r)

    def get_vertex_matrix(self, params):
        '''Get Nx6x2 matrix of all vertex points for all hexagons'''
        offset_x, offset_y, r = self.get_from_params(params)
        x0 = offset_x + self.init_x
        y0 = offset_y + self.init_y

        #Pre-allocate Nx6x2 array of points
        points = np.empty((offset_x.size,6,2))

        #Iterate over each vertex
        for i in range(6):
            points[:,i,:] = np.stack(HexagonMetasurface.get_vertex(x0,y0,r,i), axis=-1)

        return points

    @staticmethod
    def get_vertex(x0, y0, r, i):
        '''Returns (x,y) for ith vertex of hexagons, i = 0,1,2,3,4,5'''
        angle = np.radians(60)
        return x0 + r*np.cos(i*angle), y0 + r*np.sin(i*angle)

    @staticmethod
    def get_hex_point(x0, y0, r, i, t):
        '''Returns (x,y) on segment between i and i+1 vertex parameterized by 0<t<1'''
        
        #Get endpoints
        x1, y1 = HexagonMetasurface.get_vertex(x0,y0,r,i)
        x2, y2 = HexagonMetasurface.get_vertex(x0,y0,r,i+1)

        #Parametric equation of line segment
        return x1 + (x2 - x1)*t, y1 + (y2 - y1)*t

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
        params['r'] = sim.fdtd.get('r').flatten()
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

        params = HexagonMetasurface.get_params_from_existing_simulation(filename, get_wavelengths)

        geom = HexagonMetasurface(posx = params['posx'], posy = params['posy'], r = params['r'], 
            min_feature_size = 0, z = params['z'], h = params['h'], eps_in = 1, eps_out = 2.4**2, 
            scaling_factor = 1, limit_nearest_neighbor_cons = False)

        if get_wavelengths:
            return geom, params['wl']
        else:
            return geom