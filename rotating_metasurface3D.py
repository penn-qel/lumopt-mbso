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

class MovingMetasurface3D(Geometry):
    """
        :param posx:            Array of shape (N,) defining initial x-coordinates of pillar centers
        :param posy:            Array of shape (N,) defining initial y-coordinates of pillar centers
        :param rx:              Array of shape (N,) defining initial x-axis radius of each pillar
        :param ry:              Array of shape (N,) defining initial y-axis radius of each pillar
        :param phi:             Array of shape (N,) defining intial phi-rotation of each pillar in degrees
        :param z:               z-position of bottom of metasurface
        :param h:               height of metasurface pillars
        :param height_precision:Number of points along height of each pillar used to calculate gradient
        :param angle_precision: Number of points along circumference of pillar used to calculate gradient
        :param eps_in:          Permittivity of pillars
        :param eps_out:         Permittivity of surrounding material
    """

    def __init__(self, posx, posy, rx, ry, phi, min_feature_size, z, h, eps_in, eps_out, height_precision = 10, angle_precision = 20, scaling_factor = 1, phi_scaling = 1):
        self.init_x = posx.flatten()
        self.init_y = posy.flatten()
        self.rx = rx.flatten()
        self.ry = ry.flatten()
        self.phi = phi.flatten()
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
        self.phi_scaling = phi_scaling

        self.bounds = self.calculate_bounds()

    def add_geo(self, sim, params, only_update):
        '''Adds the geometry to a Lumerical simulation'''

        groupname = 'Pillars'
        if params is None:
            offset_x = self.offset_x
            offset_y = self.offset_y
            rx = self.rx
            ry = self.ry
            phi = self.phi
        else:
            offset_x, offset_y, rx, ry, phi = self.get_from_params(params)
        scipy.io.savemat('params.mat', mdict={'x': offset_x + self.init_x, 'y': offset_y + self.init_y, 'rx': rx, 'ry': ry, 'phi': phi, 'height': self.h, 'z': self.z})
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
        self.offset_x, self.offset_y, self.rx, self.ry, self.phi = self.get_from_params(params)

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

        Ef, Df = self.interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.forward_fields)
        Ea, Da = self.interpolate_fields(xv.flatten(), yv.flatten(), zv.flatten(), gradient_fields.adjoint_fields)

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

        #Calculates integrand according to theory. Multiplies by arc length parameter in order to integrate over parameterized ellipse
        integrand = 2*np.real(eps_0 * (eps_in - eps_out) *np.sum(Efpar*Eapar, axis=-1) + (1/eps_out - 1/eps_in) /eps_0 * np.sum(Dfperp*Daperp, axis=-1))
        curve_integrand = integrand * np.expand_dims(np.sqrt(np.square(self.rx[pillarv]*np.sin(thetav)) + np.square(self.ry[pillarv]*np.cos(thetav))), axis = 4)

        x_comp = np.dot(normal, np.array([1,0,0]))
        y_comp = np.dot(normal, np.array([0,1,0]))

        rx_dir = np.zeros((pillars.size, theta.size, z.size, 1, 3))
        rx_dir[:,:,:,0,0] = np.cos(thetav)*np.cos(phiv)
        rx_dir[:,:,:,0,1] = np.cos(thetav)*np.sin(phiv)
        rx_comp = np.sum(normal*rx_dir, axis=-1)
        ry_dir = np.zeros((pillars.size, theta.size, z.size, 1, 3))
        ry_dir[:,:,:,0,0] = -np.sin(thetav)*np.sin(phiv)
        ry_dir[:,:,:,0,1] = np.sin(thetav)*np.cos(phiv)
        ry_comp = np.sum(normal*ry_dir, axis=-1)
        phi_dir = np.zeros((pillars.size, theta.size, z.size, 1, 3))
        phi_dir[:,:,:,0,0] = -self.rx[pillarv]*np.cos(thetav)*np.sin(phiv) - self.ry[pillarv]*np.sin(thetav)*np.cos(thetav)
        phi_dir[:,:,:,0,1] = self.rx[pillarv]*np.cos(thetav)*np.cos(phiv) - self.ry[pillarv]*np.sin(thetav)*np.sin(phiv)
        phi_comp = np.sum(normal*phi_dir, axis=-1)

        deriv_x = np.trapz(np.trapz(x_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_y = np.trapz(np.trapz(y_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_rx = np.trapz(np.trapz(rx_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_ry = np.trapz(np.trapz(ry_comp*curve_integrand, z, axis = 2), theta, axis = 1)
        deriv_phi = np.trapz(np.trapz(phi_comp*curve_integrand, z, axis = 2), theta, axis = 1)*self.scaling_factor/self.phi_scaling
        
        total_deriv = np.concatenate((deriv_x, deriv_y, deriv_rx, deriv_ry, deriv_phi))
        self.gradients.append(total_deriv)
        return total_deriv

    def get_current_params(self):
        '''Returns list of params as single array'''
        s1 = self.scaling_factor
        s2 = self.phi_scaling
        return np.concatenate((self.offset_x*s1, self.offset_y*s1, self.rx*s1, self.ry*s1, self.phi*s2))

    def get_from_params(self, params):
        '''Retrieves correctly scaled individual parameter values from list of params'''
        offset_x, offset_y, rx, ry, phi = np.split(params, 5)
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
        offset_bounds = [(-np.inf, np.inf)]*self.rx.size*3
        return (offset_bounds + radius_bounds)

    def build_constraints(self):
        '''Builds constraint objects given minimum feature size'''
        '''Constraints such that 4 extreme points of all pillars are min_feat_size apart'''
        '''(4N*(4N-1)/2 = 8N^2-2N constraints for N pillars)'''

        def constraint(params):
            '''Returns (8N^2-2N,) array of constraints'''
            offset_x, offset_y, rx, ry, phi = self.get_from_params(params)
            phi = np.radians(phi)
            x = offset_x + self.init_x
            y = offset_y + self.init_y
            bound = self.min_feature_size
            N = x.size
            points_right = np.zeros((N,2))
            points_right[:,0] = x + rx*np.cos(0)*np.cos(phi) - ry*np.sin(0)*np.sin(phi)
            points_right[:,1] = y + rx*np.cos(0)*np.sin(phi) + ry*np.sin(0)*np.cos(phi)

            points_up = np.zeros((N,2))
            points_up[:,0] = x + rx*np.cos(np.pi/2)*np.cos(phi) - ry*np.sin(np.pi/2)*np.sin(phi)
            points_up[:,1] = y + rx*np.cos(np.pi/2)*np.sin(phi) + ry*np.sin(np.pi/2)*np.cos(phi)

            points_left = np.zeros((N,2))
            points_left[:,0] = x + rx*np.cos(np.pi)*np.cos(phi) - ry*np.sin(np.pi)*np.sin(phi)
            points_left[:,1] = y + rx*np.cos(np.pi)*np.sin(phi) + ry*np.sin(np.pi)*np.cos(phi)

            points_down = np.zeros((N,2))
            points_down[:,0] = x + rx*np.cos(np.pi*3/2)*np.cos(phi) - ry*np.sin(np.pi*3/2)*np.sin(phi)
            points_down[:,1] = y + rx*np.cos(np.pi*3/2)*np.sin(phi) + ry*np.sin(np.pi*3/2)*np.cos(phi)

            points = np.concatenate((points_right, points_up, points_left, points_down))
            num_points = points.shape[0]
            cons = np.zeros(num_points*(num_points-1)//2)
            counter = 0

            for i in range(num_points):
                for j in range(i+1, num_points):
                    cons[counter] = (points[i, 0] - points[j, 0])**2 + (points[i,1] - points[j,1])**2 - bound**2
                    counter +=1
            
            return cons

        def jacobian(params):
            '''Returns (8N^2-2N, 5N) array of jacobian'''
            offset_x, offset_y, rx, ry, phi = self.get_from_params(params)
            phi = np.radians(phi)
            x = offset_x + self.init_x
            y = offset_y + self.init_y
            N = x.size

            points_right = np.zeros((N,2))
            points_right[:,0] = x + rx*np.cos(0)*np.cos(phi) - ry*np.sin(0)*np.sin(phi)
            points_right[:,1] = y + rx*np.cos(0)*np.sin(phi) + ry*np.sin(0)*np.cos(phi)
            theta1 = np.zeros(N)

            points_up = np.zeros((N,2))
            points_up[:,0] = x + rx*np.cos(np.pi/2)*np.cos(phi) - ry*np.sin(np.pi/2)*np.sin(phi)
            points_up[:,1] = y + rx*np.cos(np.pi/2)*np.sin(phi) + ry*np.sin(np.pi/2)*np.cos(phi)
            theta2 = np.pi/2*np.ones(N)

            points_left = np.zeros((N,2))
            points_left[:,0] = x + rx*np.cos(np.pi)*np.cos(phi) - ry*np.sin(np.pi)*np.sin(phi)
            points_left[:,1] = y + rx*np.cos(np.pi)*np.sin(phi) + ry*np.sin(np.pi)*np.cos(phi)
            theta3 = np.pi*np.ones(N)

            points_down = np.zeros((N,2))
            points_down[:,0] = x + rx*np.cos(np.pi*3/2)*np.cos(phi) - ry*np.sin(np.pi*3/2)*np.sin(phi)
            points_down[:,1] = y + rx*np.cos(np.pi*3/2)*np.sin(phi) + ry*np.sin(np.pi*3/2)*np.cos(phi)
            theta4 = 3*np.pi/2*np.ones(N)

            points = np.concatenate((points_right, points_up, points_left, points_down))
            theta = np.concatenate((theta1, theta2, theta3, theta4))
            phi = np.concatenate((phi, phi, phi, phi))
            num_points = points.shape[0]

            counter = 0
            jac = np.zeros((num_points*(num_points-1)//2, params.size))
            for i in range(num_points):
                for j in range(i+1, num_points):
                    v1 = 2*(points[i,0] - points[j,0])
                    v2 = 2*(points[i,1] - points[j,1])

                    iv = i%4
                    jv = j%4

                    #d/dxi
                    jac[counter, iv] = v1
                    #d/dxj
                    jac[counter, jv] = -v1
                    #d/dyi
                    jac[counter, iv+N] = v2
                    #d/dyj
                    jac[counter, jv+N] = -v2
                    #d/drxi
                    jac[counter, iv+2*N] = np.cos(theta[i])*np.cos(phi[i])*v1 + np.cos(theta[i])*np.sin(phi[i])*v2
                    #d/drxj
                    jac[counter, jv+2*N] = -np.cos(theta[j])*np.cos(phi[j])*v1 - np.cos(theta[j])*np.sin(phi[j])*v2
                    #d/dryi
                    jac[counter, iv+3*N] = -np.sin(theta[i])*np.sin(phi[i])*v1 + np.sin(theta[i])*np.cos(phi[i])*v2
                    #d/dryj
                    jac[counter, jv+3*N] = np.sin(theta[j])*np.sin(phi[j])*v1 - np.sin(theta[j])*np.cos(phi[j])*v2
                    #d/dphii
                    jac[counter, iv+4*N] = (-1*(rx[iv]*np.cos(theta[i])*np.sin(theta[i]) + ry[iv]*np.sin(theta[i])*np.cos(phi[i]))*v1 + 
                            (rx[iv]*np.cos(theta[i])*np.cos(phi[i]) -ry[iv]*np.sin(theta[i])*np.sin(phi[i])*v2))
                    #d/dphij
                    jac[counter, jv+4*N] = ((rx[jv]*np.cos(theta[j])*np.sin(phi[j]) + ry[jv]*np.sin(theta[j])*np.cos(phi[j]))*v1 + 
                            (-rx[jv]*np.cos(theta[j])*np.cos(phi[j]) + ry[jv]*np.sin(theta[j])*np.sin(phi[j]))*v2)

                    counter +=1

            return jac

        return {'type': 'ineq', 'fun': constraint, 'jac': jacobian} 
           

    def create_script(self, sim, groupname = 'Pillars', only_update = False):
        '''Creates Lumerical script for structure group'''
        struct_script = ('deleteall;\n'
                'data = matlabload("params.mat");\n'
                'for(i=1:length(x)) {\n'
                    'addcircle;\n'
                    'set("name", "pillar_"+num2str(i));\n'
                    'set("make ellipsoid", true);\n'
                    'set("first axis", "z");\n'
                    'set("rotation 1", phi(i));\n'
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
