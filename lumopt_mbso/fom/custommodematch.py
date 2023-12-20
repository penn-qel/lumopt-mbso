################################################
# Script: custommodematch.py

# Description: This script defines a FOM object defined by an arbitrary profile
# Author: Amelia Klein
###############################################

import sys
import numpy as np
import scipy as sp
import scipy.constants
import lumapi
import os

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from lumopt_mbso.fom.wavelengthintegrals import fom_wavelength_integral, fom_gradient_wavelength_integral_impl, fom_gradient_wavelength_integral_on_cad_impl
from lumopt_mbso.utils.spatial_integral import spatial_integral

class CustomModeMatch(object):

    """ Calculates the figure of merit from an overlap integral between the fields recorded by a field monitor and the inputted mode.
        
        Parameters
        ----------
        :param monitor_name:   name of the field monitor that records the fields to be used in the mode overlap calculation.
        :param Emodefun:       Function of x,y,z,wl (N,) arrays that returns a (N,3) vector describing the E field mode
        :param Hmodefun:       Function of x,y,z,wl (N,) arrays that returns a (N,3) vector describing the H field mode
        :param material:       Material that FOM is measured at as Material object or single epsilon

        Optional kwargs
        -----------
        :kwarg direction:      direction of propagation ('Forward' or 'Backward') of the mode injected by the source. Default 'Forward'
        :kwarg multi_freq_src: bool flag to enable / disable a multi-frequency mode calculation and injection for the adjoint source. Default False.
        :kwarg target_T_fwd:   function describing the target T_forward vs wavelength. Default lambda wl: np.ones(wl.size)
        :kwarg norm_p:         exponent of the p-norm used to generate the figure of merit; use to generate the FOM. Default 1
        :kwarg target_fom:     A target value for the figure of merit. Default 0
        :kwarg target_T_fwd_weights:    Takes in array of wavelength and returns weights for FOM integral. Default lambda wl: np.ones(wl.size)
        :kwarg use_maxmin:     Boolean to ptimize by maximizing min(F(w)). Default False
    """

    def __init__(self, monitor_name, Emodefun, Hmodefun, material, **kwargs):  
        #Unpack kwargs
        direction = kwargs.get('direction', 'Forward')
        multi_freq_src = kwargs.get('multi_freq_src', False)
        target_T_fwd = kwargs.get("target_T_fwd", lambda wl: np.ones(wl.size))
        norm_p = kwargs.get("norm_p", 1)
        target_fom = kwargs.get("target_fom", 0)
        target_T_fwd_weights = kwargs.get("target_T_fwd_weights", lambda wl: np.ones(wl.size))
        use_maxmin = kwargs.get("use_maxmin", False)


        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('empty monitor name.')
        self.adjoint_source_name = monitor_name + '_adj_src'
        self.target_fom = target_fom
        
        self.direction = str(direction)
        self.multi_freq_src = bool(multi_freq_src)
        if self.direction != 'Forward' and self.direction != 'Backward':
            raise UserWarning('invalid propagation direction.')
        target_T_fwd_result = target_T_fwd(np.linspace(0.1e-6, 10.0e-6, 1000))
        if target_T_fwd_result.size != 1000:
            raise UserWarning('target transmission must return a flat vector with the requested number of wavelength samples.')
        elif np.any(target_T_fwd_result.min() < 0.0) or np.any(target_T_fwd_result.max() > 1.0):
            raise UserWarning('target transmission must always return numbers between zero and one.')
        else:
            self.target_T_fwd = target_T_fwd
        
        target_T_fwd_weights_result = target_T_fwd_weights(np.linspace(0.1e-6, 10.0e-6, 1000))
        
        if target_T_fwd_weights_result.size != 1000:
            raise UserWarning('target transmission weights must return a flat vector with the requested number of wavelength samples.')
        elif np.any(target_T_fwd_weights_result.min() < 0.0):
            raise UserWarning('target transmission weights must always return positive numbers or zero.')
        else:
            self.target_T_fwd_weights = target_T_fwd_weights

        self.norm_p = int(norm_p)
        if self.norm_p < 1:
            raise UserWarning('exponent p for norm must be positive.')

        self.material = material if isinstance(material, Material) else Material(material)

        #Test outputs of these functions!
        self.Emodefun = Emodefun
        self.Hmodefun = Hmodefun

        self.use_maxmin = bool(use_maxmin)

    def initialize(self, sim):
        self.check_monitor_alignment(sim)
        self.wavelengths = ModeMatch.get_wavelengths(sim)
        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'

        #Run forward simulation to get field grid
        print("Running config sim")
        sim.save('temp')
        sim.fdtd.run()
        fields = self.get_fom_fields(sim)
        sim.fdtd.switchtolayout()
        CustomModeMatch.add_adjoint_source(sim, self.monitor_name, self.adjoint_source_name, adjoint_injection_direction, self.multi_freq_src)
        self.import_adjoint_source(sim, fields)
        os.remove('temp.fsp')

    def make_forward_sim(self, sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', False)

    def make_adjoint_sim(self, sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', True)

    def check_monitor_alignment(self, sim):
      
        ## Here, we check that the FOM_monitor is properly aligned with the mesh
        if sim.fdtd.getnamednumber(self.monitor_name) != 1:
            raise UserWarning('monitor could not be found or the specified name is not unique.')
        
        # Get the orientation
        monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')

        if (monitor_type == 'Linear X') or (monitor_type == '2D X-normal'):
            orientation = 'x'
        elif (monitor_type == 'Linear Y') or (monitor_type == '2D Y-normal'):
            orientation = 'y'
        elif (monitor_type == 'Linear Z') or (monitor_type == '2D Z-normal'):
            orientation = 'z'
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')

        monitor_pos = sim.fdtd.getnamed(self.monitor_name, orientation)
        if sim.fdtd.getnamednumber('FDTD') == 1:
            grid = sim.fdtd.getresult('FDTD', orientation)
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            grid = sim.fdtd.getresult('varFDTD', orientation)
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        ## Check if this is exactly aligned with the simulation mesh. It exactly aligns if we find a point
        ## along the grid which is no more than 'tol' away from the position
        tol = 1e-9
        dist_from_nearest_mesh_point = min(abs(grid-monitor_pos))
        if dist_from_nearest_mesh_point > tol:
            print('WARNING: The monitor "{}" is not aligned with the grid. Its distance to the nearest mesh point is {}. This can introduce small phase errors which sometimes result in inaccurate gradients.'.format(self.monitor_name,dist_from_nearest_mesh_point))


    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        normal = ''
        if monitor_type == '2D X-normal':
            geometric_props.extend(['y span','z span'])
            normal = 'x'
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span','z span'])
            normal = 'y'
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span','y span'])
            normal = 'z'
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
            normal = 'y'
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
            normal = 'x'
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props, normal

    def get_fom(self, sim):
        fom_fields = self.get_fom_fields(sim)
        trans_coeff = CustomModeMatch.get_transmission_coefficient(fom_fields, self.get_monitor_normal(sim), self.Emodefun, self.Hmodefun)
        source_power = CustomModeMatch.get_source_power(sim, self.wavelengths)
        #FOM = 1/8 * 1/sourcepower * 1/int(Re(Em x Hm*)dS) * |int(E x Hm* dS + Em* x H dS)|^2
        self.T_fwd_vs_wavelength = np.real(trans_coeff * trans_coeff.conj() / (8 * source_power * self.modepower))
        #A = 1/4 * 1/sourcepower * 1/modepower * conj(int(E x Hm* dS + Em* x H dS))
        self.adjoint_scaling = np.conj(trans_coeff) /(4*source_power * self.modepower)
        self.phase_prefactors = trans_coeff / (4*source_power)

        if self.use_maxmin:
            min_fom = np.amin(self.T_fwd_vs_wavelength)
            self.min_indx = np.where(self.T_fwd_vs_wavelength == min_fom)
            print(self.T_fwd_vs_wavelength)
            return np.array([min_fom.real])
        return fom_wavelength_integral(self.T_fwd_vs_wavelength, self.wavelengths, self.target_T_fwd, self.norm_p, self.target_T_fwd_weights)

    def get_fom_fields(self, sim):
        fom_fields = get_fields(sim.fdtd,
                                monitor_name = self.monitor_name,
                                field_result_name = 'fom_fields',
                                get_eps = False,
                                get_D = False,
                                get_H = True,
                                nointerpolation = False)
        return fom_fields

    def get_adjoint_field_scaling(self, sim):
        omega = 2.0 * np.pi * sp.constants.speed_of_light / self.wavelengths
        adjoint_source_power = CustomModeMatch.get_source_power(sim, self.wavelengths)
        #return self.adjoint_scaling
        return np.conj(self.phase_prefactors)*omega*1j / adjoint_source_power

    def import_adjoint_source(self, sim, fields):
        '''Imports adjoint source profile based off of the custom profile given in declarization'''

        #Get coordinate grid
        xarray = fields.x
        yarray = fields.y
        zarray = fields.z

        eps = self.material.get_eps(self.wavelengths)
        #Use all wavelengths or only center wavelength
        if self.multi_freq_src:
            wavelengths = self.wavelengths
        else:
            wavelengths = np.array([self.wavelengths[int(self.wavelengths.size/2)]])
            eps = np.array([eps[int(self.wavelengths.size/2)]])

        xv, yv, zv, wlv = np.meshgrid(xarray, yarray, zarray, wavelengths, indexing = 'ij')

        #Get normal vector to monitor
        norm = self.get_monitor_normal(sim)

        #Calculate source
        #Esource = (1/epsilon*epsilon0)Hm* x n
        #Hsource = (1/mu0)Em* x n
        #modepower = Re(Em x Hm*) . n (power of FOM mode, not adjoint. But convenient to calculate it once now.)

        Em = self.Emodefun(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((xarray.size, yarray.size, zarray.size, wavelengths.size, 3))
        Hm = self.Hmodefun(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((xarray.size, yarray.size, zarray.size, wavelengths.size, 3))
        Esource = np.cross(np.conj(Hm), norm)/(eps.reshape(1,1,1,wavelengths.size,1)*scipy.constants.epsilon_0)
        Hsource = np.cross(np.conj(Em), norm)/scipy.constants.mu_0

        modepower = np.dot(np.real(np.cross(Em, np.conj(Hm))), norm)
        self.modepower = spatial_integral(modepower, xarray, yarray, zarray)

        #Push field data into adjoint source
        lumapi.putMatrix(sim.fdtd.handle, 'x', xarray)
        lumapi.putMatrix(sim.fdtd.handle, 'y', yarray)
        lumapi.putMatrix(sim.fdtd.handle, 'z', zarray)
        lumapi.putMatrix(sim.fdtd.handle, 'f', scipy.constants.speed_of_light/wavelengths)
        lumapi.putMatrix(sim.fdtd.handle, 'Ex', Esource[:,:,:,:,0])
        lumapi.putMatrix(sim.fdtd.handle, 'Ey', Esource[:,:,:,:,1])
        lumapi.putMatrix(sim.fdtd.handle, 'Ez', Esource[:,:,:,:,2])
        lumapi.putMatrix(sim.fdtd.handle, 'Hx', Hsource[:,:,:,:,0])
        lumapi.putMatrix(sim.fdtd.handle, 'Hy', Hsource[:,:,:,:,1])
        lumapi.putMatrix(sim.fdtd.handle, 'Hz', Hsource[:,:,:,:,2])

        sim.fdtd.eval("EM = rectilineardataset('EM fields', x, y, z);")
        sim.fdtd.eval("EM.addparameter('lambda', c/f, 'f', f);")
        sim.fdtd.eval("EM.addattribute('E', Ex, Ey, Ez);")
        sim.fdtd.eval("EM.addattribute('H', Hx, Hy, Hz);")

        sim.fdtd.select(self.adjoint_source_name)
        dataset = sim.fdtd.getv("EM")
        sim.fdtd.importdataset(dataset)
        if not self.multi_freq_src:
            sim.fdtd.setnamed(self.adjoint_source_name, 'override global source settings', False)

    def get_monitor_normal(self, sim):
        #Returns normal vector based on monitor type and propagation direction
        monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')
        geo_props, normal = CustomModeMatch.cross_section_monitor_props(monitor_type)
        if normal == 'x':
            vector = np.array([1, 0, 0])
        elif normal == 'y':
            vector = np.array([0, 1, 0])
        elif normal == 'z':
            vector = np.array([0, 0, 1])
        else:
            raise UserWarning('Normal vector for monitor could not be found')
        if self.direction == 'Backward':
            vector = vector * -1
        return vector

    @staticmethod
    def get_source_power(sim, wavelengths):
        frequency = sp.constants.speed_of_light / wavelengths
        source_power = sim.fdtd.sourcepower(frequency)
        return np.asarray(source_power).flatten()

    @staticmethod
    def get_transmission_coefficient(fields, normal, Em, Hm):
        norm = normal / np.linalg.norm(normal)
        xarray = fields.x.flatten()
        yarray = fields.y.flatten()
        zarray = fields.z.flatten()
        wlarray = fields.wl.flatten()

        xv, yv, zv, wlv = np.meshgrid(xarray, yarray, zarray, wlarray, indexing = 'ij')
        Emode = Em(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((xarray.size, yarray.size, zarray.size, wlarray.size, 3))
        Hmode = Hm(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((xarray.size, yarray.size, zarray.size, wlarray.size, 3))

        S1 = np.cross(fields.E, np.conj(Hmode))
        S2 = np.cross(np.conj(Emode), fields.H)
        T = np.dot(S1, norm) + np.dot(S2, norm)

        if zarray.size > 1:
            T = np.trapz(T, zarray, axis = 2)
        if yarray.size > 1:
            T = np.trapz(T, yarray, axis = 1)
        if xarray.size > 1:
            T = np.trapz(T, xarray, axis = 0)

        return T.flatten()
    
    @staticmethod
    def add_adjoint_source(sim, monitor_name, source_name, direction, multi_freq_source):
    #Adds adjoint source object to simulation, but does not import field profile
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addimportedsource()
        else:
            raise UserWarning('no FDTD solver object could be found')
        sim.fdtd.set('name', source_name)
        sim.fdtd.select(source_name)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = CustomModeMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.setnamed(source_name, 'direction', direction)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency field profile', multi_freq_source)

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        assert np.allclose(wl, self.wavelengths)

        if self.use_maxmin:
            assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
            wl = wl[self.min_indx]
            T_fwd_partial_deriv = -1.0 *np.sign(self.T_fwd_vs_wavelength[self.min_indx] - self.target_T_fwd(wl)) * (T_fwd_partial_derivs_vs_wl[:,self.min_indx]).flatten()
            return T_fwd_partial_deriv.flatten().real
        return fom_gradient_wavelength_integral_impl(self.T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, self.target_T_fwd(wl).flatten(), self.wavelengths, self.norm_p, self.target_T_fwd_weights(wl).flatten())

    def fom_gradient_wavelength_integral_on_cad(self, sim, grad_var_name, wl):
        assert np.allclose(wl, self.wavelengths)
        return fom_gradient_wavelength_integral_on_cad_impl(sim, grad_var_name, self.T_fwd_vs_wavelength, self.target_T_fwd(wl).flatten(), wl, self.norm_p, self.target_T_fwd_weights(wl).flatten())
