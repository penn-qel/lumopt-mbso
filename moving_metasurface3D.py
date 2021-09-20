###################################################################
# Class: moving_metasurface3D.py

# Description: this class defines a geometry object corresponding to a 3D
# metasurface of ellipctical pillars allowed to move around within fabrication constraints
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

class MovingMetasurface3D(Geometry):
    """
        :param posx:            Array of shape (N,) defining initial x-coordinates of pillar centers
        :param posy:            Array of shape (N,) defining initial y-coordinates of pillar centers
        :param rx:              Array of shape (N,) defining initial x-axis radius of each pillar
        :param ry:              Array fo shape (N,) defining initial y-axis radius of each pillar
        :param z:               z-position of bottom of metasurface
        :param h:               height of metasurface pillars
        :param height_precision:Number of points along height of each pillar used to calculate gradient
        :param angle_precision: Number of points along circumference of pillar used to calculate gradient
        :param eps_in:          Permittivity of pillars
        :param eps_out:         Permittivity of surrounding material
    """

    def __init__(self, posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, height_precision = 10, angle_precision = 20, scaling_factor = 1):
        self.init_x = posx.flatten()
        self.init_y = posy.flatten()
        self.rx = rx.flatten()
        self.ry = ry.flatten()
        self.offset_x = np.zeros(posx.size).flatten()
        self.offset_y = np.zeros(posy.size).flatten()
        self.z = float(z)
        self.h = float(h)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.height_precision = int(height_precision)
        self.angle_precision = int(angle_precision)

        if self.h <= 0:
            raise UserWarning("pillar height must be positive.")

        if not(posx.size == posy.size == rx.size == ry.size):
            raise UserWarning('Initial parameter arrays must have same shape (N,)')

        self.gradients = list()
        self.min_feature_size = float(min_feature_size)
        self.scaling_factor = scaling_factor

        self.bounds = self.calculate_bounds()

    def add_geo(self, sim, params, only_update):
        '''Adds the geometry to a Lumerical simulation'''

        groupname = 'Pillars'
        if params is None:
            offset_x = self.offset_x
            offset_y = self.offset_y
            rx = self.rx
            ry = self.ry
        else:
            offset_x, offset_y, rx, ry = np.split(params/self.scaling_factor, 4)

        scipy.io.savemat('params.mat', mdict={'x': offset_x + self.init_x, 'y': offset_y + self.init_y, 'rx': rx, 'ry': ry, 'height': self.h, 'z': self.z})
        sim.fdtd.switchtolayout()

        if not only_update:
            sim.fdtd.addstructuregroup()
            sim.fdtd.set('name', groupname)
            self.counter = 0
            sim.fdtd.adduserprop('counter', 0, 0)
            sim.fdtd.set('x', 0)
            sim.fdtd.set('y', 0)
            sim.fdtd.set('z', self.z)
        self.counter += 1
        sim.fdtd.select(groupname)
        self.create_script(sim, groupname, only_update)
        sim.fdtd.set('counter', self.counter)

    def update_geometry(self, params, sim = None):
        '''Updates internal values of parameters according to input'''
        self.offset_x, self.offset_y, self.rx, self.ry = np.split(params/self.scaling_factor, 4)

    def calculate_gradients(self, gradient_fields):
        '''Calculates gradient at each wavelength with respect to all parameters'''

        #Ellipse described by x = x0 + rx*cos(theta), y = y0 + ry*sin(theta) 
        eps_in = self.eps_in.get_eps(gradient_fields.forward_fields.wl)
        eps_out = self.eps_out.get_eps(gradient_fields.forward_fields.wl)
        eps_0 = sp.constants.epsilon_0
        wl = gradient_fields.forward_fields.wl

        z = np.linspace(self.z, self.z + self.h, self.height_precision)
        theta = np.linspace(0, 2*np.pi, self.angle_precision)
        pillars = np.arange(self.rx.size)
        pillarv, thetav, zv = np.meshgrid(pillars, theta, z, indexing = 'ij')

        xv = self.offset_x[pillarv] + self.init_x[pillarv] + self.rx[pillarv]*np.cos(thetav)
        yv = self.offset_y[pillarv] + self.init_y[pillarv] + self.ry[pillarv]*np.sin(thetav)

        Ef, Df = self.interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.forward_fields)
        Ea, Da = self.interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.adjoint_fields)

        Ef = Ef.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Df = Df.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Ea = Ea.reshape(pillars.size, theta.size, z.size, wl.size, 3)
        Da = Da.reshape(pillars.size, theta.size, z.size, wl.size, 3)


        #Calculate surface normal vectors
        nx = (self.ry[pillarv]*np.cos(thetav)).reshape(pillars.size, theta.size, z.size, 1)
        ny = (self.rx[pillarv]*np.sin(thetav)).reshape(pillars.size, theta.size, z.size, 1)

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

        #Calculates integrand according to theory. Multiplies by arc length parameter in order to integrate over parameterized ellipse
        integrand = 2*np.real(eps_0 * (eps_in - eps_out) *np.sum(Efpar*Eapar, axis=-1) + (1/eps_out - 1/eps_in) /eps_0 * np.sum(Dfperp*Daperp, axis=-1))
        curve_integrand = integrand * np.expand_dims(np.sqrt(np.square(self.rx[pillarv]*np.sin(thetav)) + np.square(self.ry[pillarv]*np.cos(thetav))), axis = 4)

        x_comp = np.dot(normal, np.array([1,0,0]))
        y_comp = np.dot(normal, np.array([0,1,0]))
        rx_comp = np.absolute(normal[:,:,:,:,0])
        ry_comp = np.absolute(normal[:,:,:,:,1])
        deriv_x = np.trapz(np.trapz(x_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_y = np.trapz(np.trapz(y_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_rx = np.trapz(np.trapz(rx_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_ry = np.trapz(np.trapz(ry_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        
        total_deriv = np.concatenate((deriv_x, deriv_y, deriv_rx, deriv_ry))
        self.gradients.append(total_deriv)
        return total_deriv

    def get_current_params(self):
        '''Returns list of params as single array'''
        return np.concatenate((self.offset_x, self.offset_y, self.rx, self.ry))*self.scaling_factor

    def plot(self, ax):
        '''Plots current geometry'''
        x = (self.offset_x + self.init_x)*1e6
        y = (self.offset_y + self.init_y)*1e6
        rx = self.rx.copy()*1e6
        ry = self.ry.copy()*1e6
        maxr = max(np.amax(rx), np.amax(ry))
        ax.clear()
        for i, xval in enumerate(x):
            ellipse = patches.Ellipse((xval, y[i]), rx[i], ry[i], facecolor='black')
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
        return (offset_bounds + radius_bounds)

    def build_constraints(self):
        '''Builds constraint objects given minimum feature size'''
        '''Constraints such that left + right and top + bottom points of all pillars are min_feat_size apart'''
        '''2N^2 constraints where N is number of pillars'''

        def constraint(params):
            '''Returns (2N^2,) array of constraints'''
            offset_x, offset_y, rx, ry = np.split(params, 4)
            x = offset_x + self.init_x * self.scaling_factor
            y = offset_y + self.init_y * self.scaling_factor
            bound = self.min_feature_size * self.scaling_factor
            N = x.size

            horiz = np.zeros((N,N))
            vert = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    horiz[i,j] = ((x[i] +rx[i]) - (x[j] - rx[j]))**2 + (y[i]-y[j])**2 - bound**2
                    vert[i,j] = (x[i] - x[j])**2 + ((y[i] + ry[i]) - (y[j] - ry[j]))**2 - bound**2
            
            return np.append(horiz.flatten(), vert.flatten())

        def jacobian(params):
            '''Returns (2N^2, 4N) array of jacobian'''
            offset_x, offset_y, rx, ry = np.split(params, 4)
            x = offset_x + self.init_x * self.scaling_factor
            y = offset_y + self.init_y * self.scaling_factor
            N = x.size

            '''Construct sparse matrix a where a[row_ind[k], col_ind[k]] = data[k]'''
            #6 meaningful derivs for each constraint, so 12N^2 nonzero values in 8N^3-element matrix

            def evaluate_index(k):
                '''For given value k in [0, 12N^2) gives corresponding row_ind, col_ind, and data'''
                direction = k//(6*N**2) #0 for horizontal, 1 for vertical
                k = k%(6*N**2)
                
                i = k//(6*N)
                k = k%(6*N)

                j = k//6
                k = k%6

                if direction == 0:
                    if k == 0:
                        #d/dxi
                        val = 2*((x[i] + rx[i]) - (x[j] - rx[j]))
                        col_ind = i
                    elif k == 1:
                        #d/drxi
                        val = 2*((x[i] + rx[i]) - (x[j] - rx[j]))
                        col_ind = 2*N + i
                    elif k == 2:
                        #d/dxj
                        val = -2*((x[i] + rx[i]) - (x[j] - rx[j]))
                        col_ind = j
                    elif k == 3:
                        #d/drxj
                        val = 2*((x[i] + rx[i]) - (x[j] - rx[j]))
                        col_ind = 2*N + j
                    elif k == 4:
                        #d/dyi
                        val = 2*(y[i] - y[j])
                        col_ind = N + i
                    elif k == 5:
                        #d/dyj
                        val = -2*(y[i] - y[j])
                        col_ind = N + j
                    else:
                        raise UserWarning('Unexpected number of values in Jacobian')

                    row_ind = i*N + j

                elif direction == 1:
                    if k == 0:
                        #d/dyi
                        val = 2*((y[i] + ry[i]) - (y[j] - ry[j]))
                        col_ind = N + i
                    elif k == 1:
                        #d/dryi
                        val = 2*((y[i] + ry[i]) - (y[j] - ry[j]))
                        col_ind = 3*N + i
                    elif k == 2:
                        #d/dyj
                        val = -2*((y[i] + ry[i]) - (y[j] - ry[j]))
                        col_ind = N + j
                    elif k == 3:
                        #d/dryj
                        val = 2*((y[i] + ry[i]) - (y[j] - ry[j]))
                        col_ind = 3*N + j
                    elif k == 4:
                        #d/dxi
                        val = 2*(x[i] - x[j])
                        col_ind = i
                    elif k == 5:
                        #d/dxj
                        val = -2*(x[i] - x[j])
                        col_ind = j
                    else:
                        raise UserWarning('Unexpected number of values in Jacobian')

                    row_ind = i*N + j + N**2
                else:
                    raise UserWarning('Unexpected number of values in Jacobian')

                return row_ind, col_ind, val

                
            rows = np.zeros(12*N**2)
            cols = np.zeros(12*N**2)
            vals = np.zeros(12*N**2)
            for i in range(12*N**2):
                rows[i], cols[i], vals[i] = evaluate_index(i)

            return (sp.sparse.csc_matrix((vals,(rows, cols)), shape=(2*N**2, 4*N))).toarray()

        return {'type': 'ineq', 'fun': constraint, 'jac': jacobian} 
           

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Creates Lumerical script for structure group'''
        struct_script = ('deleteall;\n'
                'data = matlabload("params.mat");\n'
                'for(i=1:length(x)) {\n'
                    'addcircle;\n'
                    'set("name", "pillar_"+num2str(i));\n'
                    'set("make ellipsoid", true);\n'
                    'set("x", x(i));\n'
                    'set("y", y(i));\n'
                    'set("z min", z);\n'
                    'set("z max", z+height);\n'
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

     #Takes in arrays of points to evaluate and returns interpolated E and D fields at those points
    @staticmethod
    def interpolate_fields(x, y, z, fields):
        #fields.x, fields.y, fields.z, fields.E, fields.D, fields.wl are relevant terms

        xi = np.searchsorted(fields.x, x).reshape(x.size, 1, 1)
        yi = np.searchsorted(fields.y, y).reshape(x.size, 1, 1)
        zi = np.searchsorted(fields.z, z).reshape(x.size, 1, 1)
        x = x.reshape(x.size, 1, 1)
        y = y.reshape(x.size, 1, 1)
        z = z.reshape(x.size, 1, 1)

        #Follows Wikipedia algorithm for trilinear interpolation
        xd = (x - fields.x[xi-1])/(fields.x[xi] - fields.x[xi-1])
        yd = (y - fields.y[yi-1])/(fields.y[yi] - fields.y[yi-1])
        zd = (z - fields.z[zi-1])/(fields.z[zi] - fields.z[zi-1])

        E00 = fields.E[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.E[xi, yi-1, zi-1,:,:].squeeze()*xd
        E01 = fields.E[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi-1, zi,:,:].squeeze()*xd
        E10 = fields.E[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi-1,:,:].squeeze()*xd
        E11 = fields.E[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi,:,:].squeeze()*xd

        E0 = E00*(1-yd) + E10*yd
        E1 = E01*(1-yd) + E11*yd
        E = E0*(1-zd) + E1*zd

        D00 = fields.D[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.D[xi, yi-1, zi-1,:,:].squeeze()*xd
        D01 = fields.D[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.D[xi, yi-1, zi,:,:].squeeze()*xd
        D10 = fields.D[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.D[xi, yi, zi-1,:,:].squeeze()*xd
        D11 = fields.D[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.D[xi, yi, zi,:,:].squeeze()*xd

        D0 = D00*(1-yd) + D10*yd
        D1 = D01*(1-yd) + D11*yd
        D = D0*(1-zd) + D1*zd

        return E, D 
