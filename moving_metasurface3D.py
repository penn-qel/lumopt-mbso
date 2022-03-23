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
from interpolate_fields import interpolate_fields

class MovingMetasurface3D(Geometry):
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
    """

    def __init__(self, posx, posy, rx, ry, min_feature_size, z, h, eps_in, eps_out, phi = None, pillars_rotate = True, height_precision = 10, angle_precision = 20, scaling_factor = 1, phi_scaling = 1, limit_nearest_neighbor_cons = True, make_meshgrid = False):
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
            self.phi = phi

        if self.h <= 0:
            raise UserWarning("pillar height must be positive.")

        if not(self.init_x.size == self.init_y.size == self.rx.size == self.ry.size == self.phi.size):
            raise UserWarning('Initial parameter arrays must have same shape (N,)')

        self.pillars_rotate = pillars_rotate
        self.gradients = list()
        self.min_feature_size = float(min_feature_size)
        self.scaling_factor = scaling_factor
        self.phi_scaling = phi_scaling

        self.bounds = self.calculate_bounds()
        self.limit_nearest_neighbor_cons = limit_nearest_neighbor_cons

        #Assert we have a square grid for nearest neighbor calculations by checking N_pillars is a perfect square
        if self.limit_nearest_neighbor_cons:
            if make_meshgrid:
                self.grid_shape = (posx.size, posy.size)
            else:
                N = int(np.sqrt(self.posx.size) + 0.5)
                if self.posx.size == N**2:
                    self.grid_shape = (N, N)
                else:
                    raise UserWarning("Must do built-in meshgrid or use a perfect square of pillars when constraining nearest neighbors only")

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
        A = (xv - self.offset_x[pillarv] - self.init_x[pillarv])*np.cos(phi) + (yv - self.offset_y[pillarv] - self.init_y[pillarv])*np.sin(phi)
        B = -(xv - self.offset_x[pillarv] - self.init_x[pillarv])*np.sin(phi) + (yv - self.offset_y[pillarv] - self.init_y[pillarv])*np.cos(phi)
        d_dx = np.expand_dims((2*A*np.cos(phi)/np.power(self.rx[pillarv], 2) - 2*B*np.sin(phi)/np.power(self.ry[pillarv], 2))/self.scaling_factor, axis=3)
        d_dy = np.expand_dims((2*A*np.sin(phi)/np.power(self.rx[pillarv], 2) + 2*B*np.cos(phi)/np.power(self.ry[pillarv], 2))/self.scaling_factor, axis=3)
        d_drx = np.expand_dims((-2*np.power(A, 2)/np.power(self.rx[pillarv], 3))/self.scaling_factor, axis=3)
        d_dry = np.expand_dims((-2*np.power(B, 2)/np.power(self.rx[pillarv], 3))/self.scaling_factor, axis=3)
        d_dphi = np.expand_dims((-2*A*B*(1/np.power(self.rx[pillarv], 2) - 1/np.power(self.ry[pillarv], 2)))/self.phi_scaling, axis=3)

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
        self.gradients.append(total_deriv)
        return total_deriv

    def get_current_params(self):
        '''Returns list of params as single array'''
        s1 = self.scaling_factor
        s2 = self.phi_scaling
        if self.pillars_rotate:
            return np.concatenate((self.offset_x*s1, self.offset_y*s1, self.rx*s1, self.ry*s1, self.phi*s2))
        else:
            return np.concatenate((self.offset_x*s1, self.offset_y*s1, self.rx*s1, self.ry*s1))

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

            if not self.limit_nearest_neighbor_cons:
                points = np.concatenate((points_right, points_up, points_left, points_down))
                num_points = points.shape[0]
                cons = np.zeros(num_points*(num_points-1)//2)
                counter = 0

                for i in range(num_points):
                    for j in range(i+1, num_points):
                        cons[counter] = (points[i, 0] - points[j, 0])**2 + (points[i,1] - points[j,1])**2 - bound**2
                        counter +=1
                
                return cons

            #If limited to nearest neighbors, constraint each pillar (i,j) with its (i+1) and (j+1) counterparts
            #Do a constraint for each pair of points (4 per pillar). Total of (Nx-1)*(Ny-1)*16 constraints. For a square, Nx,Ny = sqrt(N)
            Nx = self.grid_shape[0]
            Ny = self.grid_shape[1]
            points = np.stack((points_right, points_up, points_left, points_down), axis=-1).reshape(Nx, Ny, 2, 4)
            cons = np.zeros(16*(2*Nx*Ny - Nx - Ny))
            counter = 0
            for i in np.arange(Nx-1):
                for j in np.arange(Ny-1):
                    for k1 in np.arange(4):
                        for k2 in np.arange(4):
                            #side=0 corresponds to pillar to right, side=1 pillar above
                            for side in np.arange(2):
                                if side == 0:
                                    i2 = i+1
                                    j2 = j
                                    if i == Nx-1:
                                        continue
                                else:
                                    i2 = i
                                    j2 = j+1
                                    if j == Ny-1:
                                        continue

                                cons[counter] = (points[i,j,0,k1] - points[i2,j2,0,k2])**2 - (points[i,j,1,k1] - points[i2,j2,1,k2])**2 - bound**2
                                counter += 1
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


            if not self.limit_nearest_neighbor_cons:
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

                        if self.pillars_rotate:
                            #d/dphii
                            jac[counter, iv+4*N] = (-1*(rx[iv]*np.cos(theta[i])*np.sin(theta[i]) + ry[iv]*np.sin(theta[i])*np.cos(phi[i]))*v1 + 
                                    (rx[iv]*np.cos(theta[i])*np.cos(phi[i]) -ry[iv]*np.sin(theta[i])*np.sin(phi[i])*v2))
                            #d/dphij
                            jac[counter, jv+4*N] = ((rx[jv]*np.cos(theta[j])*np.sin(phi[j]) + ry[jv]*np.sin(theta[j])*np.cos(phi[j]))*v1 + 
                                    (-rx[jv]*np.cos(theta[j])*np.cos(phi[j]) + ry[jv]*np.sin(theta[j])*np.sin(phi[j]))*v2)

                        counter +=1

                return jac

            #Nearest neighbor only option for jacobin
            Nx = self.grid_shape[0]
            Ny = self.grid_shape[1]
            N = x.size
            points = np.stack((points_right, points_up, points_left, points_down), axis=-1).reshape(Nx, Ny, 2, 4)
            theta = np.stack((theta1, theta2, theta3, theta4), axis=-1).reshape(Nx, Ny, 4)
            phi = np.stack((phi, phi, phi, phi), axis=-1).reshape(Nx, Ny, 4)
            jac = np.zeros(16*(2*N-Nx-Ny), params.size)
            counter = 0
            for i in np.arange(Nx):
                for j in np.arange(Ny):
                    pillarnum = i*Ny + j 
                    pillarabovenum = i*Ny + j + 1
                    pillarrightnum = (i+1)*Ny + j
                    for k1 in np.arange(4):
                        for k2 in np.arange(4):
                            for side in np.arange(2):
                                #Start with pillar to right
                                if side==0:
                                    if i == Nx-1:
                                        continue
                                    otherpillar=pillarrightnum
                                    i2 = i+1
                                    j2 = j
                                #Then pillar above
                                else:
                                    if j == Ny-1:
                                        continue
                                    otherpillar=pillarabovenum
                                    i2 = i
                                    j2 = j+1

                                vx = 2*(points[i,j,0,k1] - points[i2,j2,0,k2])
                                vy = 2*(points[i,j,1,k1] - points[i2,j2,1,k2])

                                #d/dxi
                                jac[counter, pillarnum] = vx
                                #d/dxj
                                jac[counter, otherpillar] = -vx
                                #d/dyi
                                jac[counter, pillarnum+N] = vy
                                #d/dyj
                                jac[counter, otherpillar+N] = -vy

                                #d/drxi
                                jac[counter, pillarnum+2*N] = np.cos(theta[i,j,k1])*np.cos(phi[i,j,k1])*v1 + np.cos(theta[i,j,k1])*np.sin(phi[i,j,k1])*v2
                                #d/drxj
                                jac[counter, otherpillar+2*N] = -np.cos(theta[i2,j2,k2])*np.cos(phi[i2,j2,k2])*v1 - np.cos(theta[i2,j2,k2])*np.sin(phi[i2,j2,k2])*v2
                                #d/dryi
                                jac[counter, pillarnum+3*N] = -np.sin(theta[i,j,k1])*np.sin(phi[i,j,k1])*v1 + np.sin(theta[i,j,k1])*np.cos(phi[i,j,k1])*v2
                                #d/dryj
                                jac[counter, otherpillar+3*N] = np.sin(theta[i2,j2,k2])*np.sin(phi[i2,j2,k2])*v1 - np.sin(theta[i2,j2,k2])*np.cos(phi[i2,j2,k2])*v2

                                if self.pillars_rotate:
                                    #d/dphii
                                    jac[counter, pillarnum+4*N] = (-1*(rx[pillarnum]*np.cos(theta[i,j,k1])*np.sin(theta[i,j,k1]) + 
                                                                    ry[pillarnum]*np.sin(theta[i,j,k1])*np.cos(phi[i,j,k1]))*v1 + 
                                                                    (rx[pillarnum]*np.cos(theta[i,j,k1])*np.cos(phi[i,j,k1])-
                                                                        ry[pillarnum]*np.sin(theta[i,j,k1])*np.sin(phi[i,j,k1])*v2))
                                    #d/dphij
                                    jac[counter, otherpillar+4*N] = ((rx[otherpillar]*np.cos(theta[i2,j2,k2])*np.sin(phi[i2,j2,k2]) +
                                                                    ry[otherpillar]*np.sin(theta[i2,j2,k2])*np.cos(phi[i2,j2,k2]))*v1 + 
                                                                    (-rx[otherpillar]*np.cos(theta[i2,j2,k2])*np.cos(phi[i2,j2,k2]) +
                                                                    ry[otherpillar]*np.sin(theta[i2,j2,k2])*np.sin(phi[i2,j2,k2]))*v2)

                                counter += 1

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
