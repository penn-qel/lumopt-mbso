################################################
# Script: transmissionfom.py

# Description: This script defines a FOM object based on the transmission through some portion of a DFT monitor
# Author: Amelia Klein
###############################################

import sys
import numpy as np
import scipy as sp
import scipy.constants
import lumapi
import time

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from wavelengthintegrals import fom_wavelength_integral, fom_gradient_wavelength_integral_impl
from spatial_integral import spatial_integral

class TransmissionFom(object):
    """Calculates the figure of merit by integrating the Poynting vector through a portion of the monitor. 
    The transmission objective is assumed to be over a spot with a certain position and radius

    Parameters
    -----------
    :param monitor_name:    name of the monitor that records the fields to calculate the FOM
    :param direction:       direction of propagation ('Forward' or 'Backward') of the source mode
    :param multi_freq_src:  bool flag to enable / disable multi-frequency source calculation for adjoint
    :param target_T_fwd:    function describing the target T_forward vs wavelength
    :param boundary_func:   function defining boundary for integral. Returns 1 if within region, 0 if outside
    :param norm_p:          exponent of the p-norm used to generate the FOM
    :param target_fom:      A target value for the FOM for printing/plotting distance of current design from target
    :param use_maxmin:      Boolean that triggers FOM/gradient calculations based on the worst-performing frequency, rather than average
    """

    def __init__(self, monitor_name, direction = 'Forward', boundary_func = lambda x, y, z: np.ones(x.size), multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size), target_T_fwd_weights = lambda wl: np.ones(wl.size), norm_p = 1, target_fom = 0, use_maxmin = False):
        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('empty monitor name')
        self.adjoint_source_name = monitor_name + '_adj_src'
        self.target_fom = target_fom
        self.direction = str(direction)
        self.multi_freq_src = bool(multi_freq_src)
        self.boundary_func = boundary_func
        if self.direction != 'Forward' and self.direction != 'Backward':
            raise UserWarning('invalid propagation direction')
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
        self.use_maxmin = bool(use_maxmin)
        if self.use_maxmin:
            raise UserWarning('maxmin formulation not currently supported')
        #self.fom_fields = None

    def initialize(self, sim):
        self.check_monitor_alignment(sim)

        self.wavelengths = ModeMatch.get_wavelengths(sim)
        TransmissionFom.add_index_monitor(sim, self.monitor_name, self.wavelengths.size)

        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'
        TransmissionFom.add_adjoint_source(sim, self.monitor_name, self.adjoint_source_name, adjoint_injection_direction, self.multi_freq_src)

    def make_forward_sim(self, sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', False)

    def make_adjoint_sim(self,sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', True)
        self.import_adjoint_source(sim)

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

    def get_fom(self, sim):
        self.fom_fields = self.get_fom_fields(sim)
        self.source_power = ModeMatch.get_source_power(sim, self.wavelengths)
        norm = self.get_monitor_normal(sim)

        power_vs_wl = TransmissionFom.calculate_transmission_vs_wl(self.fom_fields, norm, self.boundary_func)
        self.T_fwd_vs_wavelength = power_vs_wl / self.source_power

        #Use only minimum element as figure of merit
        if self.use_maxmin:
            min_fom = np.amin(self.T_fwd_vs_wavelength)
            self.min_indx = np.where(self.T_fwd_vs_wavelength == min_fom)
            return np.array([min_fom.real])

        #Use existing FOM calculator
        fom = fom_wavelength_integral(self.T_fwd_vs_wavelength, self.wavelengths, self.target_T_fwd, self.norm_p, self.target_T_fwd_weights)
        return fom

    def get_adjoint_field_scaling(self, sim):
        omega = 2.0 * np.pi * sp.constants.speed_of_light / self.wavelengths
        adjoint_source_power = ModeMatch.get_source_power(sim, self.wavelengths)
        return 1j*omega/np.sqrt(adjoint_source_power)

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        #Use same implementation as ModeMatch
        assert np.allclose(wl, self.wavelengths)

        #Return derivative of minimum wavelength only
        if self.use_maxmin:
            assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
            return (T_fwd_partial_derivs_vs_wl[:, self.min_indx]).flatten()

        #print(T_fwd_partial_derivs_vs_wl)
        return fom_gradient_wavelength_integral_impl(self.T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, self.target_T_fwd(wl).flatten(), self.wavelengths, self.norm_p, self.target_T_fwd_weights(wl).flatten())

    def get_fom_fields(self, sim):
        fom_fields = get_fields(sim.fdtd,
                            monitor_name = self.monitor_name,
                            field_result_name = 'fom_fields',
                            get_eps = True,
                            get_D = False,
                            get_H = True,
                            nointerpolation = False)
        return fom_fields

    def get_monitor_normal(self, sim):
        #Returns normal vector based on monitor type and propagation direction
        monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
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

    def import_adjoint_source(self, sim):
        #Imports data to define adjoint source profile
        norm = self.get_monitor_normal(sim)
        if self.multi_freq_src:
            wavelengths = self.wavelengths
        else:
            if self.use_maxmin:
                wavelengths = np.array([self.wavelengths[self.min_indx]])
            else:
                wavelengths = np.array([self.wavelengths[int(self.wavelengths.size/2)]])
        

        if self.fom_fields is None:
            raise UserWarning("Attempt to calculate gradient before first forward sim. Try setting scale_gradient_to=0")

        xarray = self.fom_fields.x
        yarray = self.fom_fields.y
        zarray = self.fom_fields.z

        xv, yv, zv, wlv = np.meshgrid(xarray, yarray, zarray, wavelengths, indexing='ij')
        weights = self.boundary_func(xv.flatten(), yv.flatten(), zv.flatten()).reshape((xarray.size, yarray.size, zarray.size, wavelengths.size, 1))
        eps = self.fom_fields.eps
        E = self.fom_fields.E
        H = self.fom_fields.H

        #Calculate source
        Esource = weights*np.conj(E)
        Hsource = -1*weights*np.conj(H)

        if not self.multi_freq_src:
            Esource = Esource[:,:,:,int(self.wavelengths.size/2),np.newaxis,:]
            Hsource = Hsource[:,:,:,int(self.wavelengths.size/2),np.newaxis,:]
        power = -0.5*np.dot(np.real(np.cross(Esource, np.conj(Hsource))), norm)

        self.calc_adjoint_power = spatial_integral(power, xarray, yarray, zarray)
        if not self.multi_freq_src:
            val = self.calc_adjoint_power
            self.calc_adjoint_power = val*np.ones(self.wavelengths.size)

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
        sim.fdtd.eval("matlabsave('sourcefile.mat', EM);")
        sim.fdtd.select(self.adjoint_source_name)
        sim.fdtd.importdataset("sourcefile.mat") #Is there a way to define a source without saving to .mat?
        if not self.multi_freq_src:
            sim.fdtd.setnamed(self.adjoint_source_name, 'override global source settings', False)

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
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.setnamed(source_name, 'direction', direction)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency field profile', multi_freq_source)
            #if multi_freq_source:
                #sim.fdtd.setnamed(source_name, 'frequency points', sim.fdtd.getglobalmonitor('frequency points'))

    @staticmethod
    def calculate_transmission_vs_wl(fom_fields, normal, boundary_func):
        norm = normal / np.linalg.norm(normal)
        xarray = fom_fields.x.flatten()
        yarray = fom_fields.y.flatten()
        zarray = fom_fields.z.flatten()
        wlarray = fom_fields.wl.flatten()
        xv, yv, zv = np.meshgrid(xarray, yarray, zarray, indexing='ij')
        E = fom_fields.E
        H = fom_fields.H

        power = 0.5*np.dot(np.real(np.cross(E, np.conj(H))), norm)
        weights = boundary_func(xv.flatten(), yv.flatten(), zv.flatten()).reshape((xarray.size, yarray.size, zarray.size, 1))
        integrand = power*weights

        return spatial_integral(integrand, xarray, yarray, zarray)

    @staticmethod
    def add_index_monitor(sim, monitor_name, frequency_points):
        sim.fdtd.select(monitor_name)
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(monitor_name))
        index_monitor_name = monitor_name + '_index'
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addindex()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addeffectiveindex()
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        sim.fdtd.set('name', index_monitor_name)
        sim.fdtd.setnamed(index_monitor_name, 'override global monitor settings', True)
        #sim.fdtd.setnamed(index_monitor_name, 'frequency points', frequency_points)
        sim.fdtd.setnamed(index_monitor_name, 'record conformal mesh when possible', True)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geometric_props = ['monitor type']
        props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        geometric_props.extend(props)
        for prop_name in geometric_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            if prop_val == "Linear X":
                sim.fdtd.setnamed(index_monitor_name, prop_name, "2D Z-normal")
                sim.fdtd.setnamed(index_monitor_name, "y span", 0)
                continue
            elif prop_val == "Linear Y":
                sim.fdtd.setnamed(index_monitor_name, prop_name, "2D Z-normal")
                sim.fdtd.setnamed(index_monitor_name, "x span", 0)
                continue
            sim.fdtd.setnamed(index_monitor_name, prop_name, prop_val)
        sim.fdtd.setnamed(index_monitor_name, 'spatial interpolation', 'none')

    @staticmethod
    def callable_spot_boundary_func(r, x0 = 0, y0 = 0, z0 = 0, axis = 'z'):
        #Returns a boundary function that defines a spot with radius r around a central coordinate
        #with normal axis. Can be used as constructor input for FOM object.
        if axis == 'x':
            def spot_boundary(x,y,z):
                return np.sqrt(np.power(y-y0, 2) + np.power(z-z0, 2)) <= r
        elif axis == 'y':
            def spot_boundary(x,y,z):
                return np.sqrt(np.power(x-x0, 2) + np.power(z-z0, 2)) <= r
        elif axis == 'z':
            def spot_boundary(x,y,z):
                return np.sqrt(np.power(x-x0, 2) + np.power(y-y0, 2)) <= r
        else:
            raise UserWarning("Axis should be 'x', 'y' or 'z'")

        return spot_boundary

    @staticmethod
    def callable_spot_boundary_func_2D(r, x0 = 0, y0 = 0, axis = 'y'):
        if axis == 'x':
            def spot_boundary(x,y,z):
                return np.abs(y - y0) <= r
        elif axis == 'y':
            def spot_boundary(x,y,z):
                return np.abs(x - x0) <= r
        else:
            raise UserWarning("Axis should be 'x' or 'y'")
        return spot_boundary