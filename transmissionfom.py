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

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.lumerical_methods.lumerical_scripts import get_fields

class TransmissionFom(object):
	"""Calculates the figure of merit by integrating the Poynting vector through a portion of the monitor. 
	The transmission objective is assumed to be over a spot with a certain position and radius

	Parameters
	-----------
	:param monitor_name: 	name of the monitor that records the fields to calculate the FOM
	:param direction: 		direction of propagation ('Forward' or 'Backward') of the source mode
	:param multi_freq_src: 	bool flag to enable / disable multi-frequency source calculation for adjoint
	:param target_T_fwd: 	function describing the target T_forward vs wavelength
	:param boundary_func: 	function defining boundary for integral. Returns 1 if within region, 0 if outside
	:param spot_center: 	(x,y) point describing center position of desired focal spot
	:param spot_radius: 	radius of spot to be integrated over. If radius is bigger than monitor size, entire monitor will be used
	:param norm_p: 			exponent of the p-norm used to generate the FOM
	:param target_fom: 		A target value for the FOM for printing/plotting distance of current design from target
	:param use_maxmin: 		Boolean that triggers FOM/gradient calculations based on the worst-performing frequency, rather than average
	"""

	def __init__(self, monitor_name, direction = 'Forward', boundary_func = lambda x, y, z: 1, multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1, target_fom = 0, use_maxmin = False):
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
		self.norm_p = int(norm_p)
		if self.norm_p < 1:
			raise UserWarning('exponent p for norm must be positive.')
		self.use_maxmin = bool(use_maxmin)

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
		fom = ModeMatch.fom_wavelength_integral(self.T_fwd_vs_wavelength, self.wavelengths, self.target_T_fwd, self.norm_p)
		return fom

	def get_adjoint_field_scaling(self, sim):
		#Scaling factor is 1. Adjoint source itself is calculated exactly
		return np.ones(self.T_fwd_vs_wavelength.size)

	def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
		#Use same implementation as ModeMatch
		assert np.allclose(wl, self.wavelengths)

		#Return derivative of minimum wavelength only
		if self.use_maxmin:
			assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
			return (T_fwd_partial_derivs_vs_wl[:, self.min_indx]).flatten()

		#print(T_fwd_partial_derivs_vs_wl)
		return ModeMatch.fom_gradient_wavelength_integral_impl(self.T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, self.target_T_fwd(wl).flatten(), self.wavelengths, self.norm_p)

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
		xarray = self.fom_fields.x
		yarray = self.fom_fields.y
		zarray = self.fom_fields.z

		if self.multi_freq_src:
			wavelengths = self.wavelengths
		else:
			if self.use_maxmin:
				wavelengths = np.array([self.wavelengths[self.min_indx]])
			else:
				wavelengths = np.array([self.wavelengths[int(self.wavelengths.size/2)]])


		#Field dimension: 0 (x-axis), 1 (y-axis), 2 (z-axis), 3 (frequency), 4 (vector component)
		Esource = np.zeros((xarray.size, yarray.size, zarray.size, wavelengths.size, 3), dtype = np.complex128)
		Hsource = np.zeros((xarray.size, yarray.size, zarray.size, wavelengths.size, 3), dtype = np.complex128)
		eps = self.fom_fields.eps
		#wl_indx = int(self.wavelengths.size/2)
		#wl = self.wavelengths[wl_indx]

		#Get normal vector
		norm = self.get_monitor_normal(sim)

		#Calculate source
		#Esource = 1/(2*epsilon*epsilon_0*sourcepower)H* x n
		#Hsource = 1/(2*mu0*sourcepower)E* x n
		for idx, wl in enumerate(wavelengths):
			for i, x in enumerate(xarray):
				for j, y in enumerate(yarray):
					for k, z in enumerate(zarray):
						Esource[i,j,k,idx,:] = self.boundary_func(x,y,z)*np.cross(np.conj(self.fom_fields.getHfield(x,y,z,wl)),norm)/(self.fom_fields.eps[i,j,k,idx]*2*self.source_power[idx]*scipy.constants.epsilon_0)
						Hsource[i,j,k,idx,:] = self.boundary_func(x,y,z)*np.cross(np.conj(self.fom_fields.getfield(x,y,z,wl)),norm)/(2*self.source_power[idx]*scipy.constants.mu_0)

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
		sim.fdtd.importdataset("sourcefile.mat")


		'''	def import_adjoint_source(self, sim):
		#Imports data to define adjoint source profile
		if self.multi_freq_src:
			print('WARNING: Multi-frequency source not implemented. Using center wavelength only')

		xarray = self.fom_fields.x
		yarray = self.fom_fields.y
		zarray = self.fom_fields.z

		#Field dimension: 0 (x-axis), 1 (y-axis), 2 (z-axis), 3 (frequency), 4 (vector component)
		Esource = np.zeros((xarray.size, yarray.size, zarray.size, 3), dtype = np.complex128)
		Hsource = np.zeros((xarray.size, yarray.size, zarray.size, 3), dtype = np.complex128)
		eps = self.fom_fields.eps
		wl_indx = int(self.wavelengths.size/2)
		wl = self.wavelengths[wl_indx]

		#Get normal vector
		norm = self.get_monitor_normal(sim)

		#Calculate source
		#Esource = 1/(2*epsilon*epsilon_0*sourcepower)H* x n
		#Hsource = 1/(2*mu0*sourcepower)E* x n
		for i, x in enumerate(xarray):
			for j, y in enumerate(yarray):
				for k, z in enumerate(zarray):
					Esource[i,j,k,:] = self.boundary_func(x,y,z)*np.cross(np.conj(self.fom_fields.getHfield(x,y,z,wl)),norm)/(self.fom_fields.eps[i,j,k,wl_indx]*2*self.source_power[wl_indx]*scipy.constants.epsilon_0)
					Hsource[i,j,k,:] = self.boundary_func(x,y,z)*np.cross(np.conj(self.fom_fields.getfield(x,y,z,wl)),norm)/(2*self.source_power[wl_indx]*scipy.constants.mu_0)

		lumapi.putMatrix(sim.fdtd.handle, 'x', xarray)
		lumapi.putMatrix(sim.fdtd.handle, 'y', yarray)
		lumapi.putMatrix(sim.fdtd.handle, 'z', zarray)
		lumapi.putMatrix(sim.fdtd.handle, 'Ex', Esource[:,:,:,0])
		lumapi.putMatrix(sim.fdtd.handle, 'Ey', Esource[:,:,:,1])
		lumapi.putMatrix(sim.fdtd.handle, 'Ez', Esource[:,:,:,2])
		lumapi.putMatrix(sim.fdtd.handle, 'Hx', Hsource[:,:,:,0])
		lumapi.putMatrix(sim.fdtd.handle, 'Hy', Hsource[:,:,:,1])
		lumapi.putMatrix(sim.fdtd.handle, 'Hz', Hsource[:,:,:,2])

		sim.fdtd.eval("EM = rectilineardataset('EM fields', x, y, z);")
		sim.fdtd.eval("EM.addattribute('E', Ex, Ey, Ez);")
		sim.fdtd.eval("EM.addattribute('H', Hx, Hy, Hz);")
		sim.fdtd.eval("matlabsave('sourcefile.mat', EM);")
		sim.fdtd.select(self.adjoint_source_name)
		sim.fdtd.importdataset("sourcefile.mat")'''



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

		boundary_weight_vs_pos = TransmissionFom.boundary_weights_vs_pos(xarray, yarray, zarray, boundary_func)

		#Solve per wavelength
		fom_vs_wl = np.zeros(wlarray.size)
		for idx, wl in enumerate(wlarray):
			#Calculate power at each position
			fom_vs_pos = np.zeros((xarray.size, yarray.size, zarray.size))
			for i, x in enumerate(xarray):
				for j, y in enumerate(yarray):
					for k, z in enumerate(zarray):
						fom_vs_pos[i, j, k] = TransmissionFom.calculate_power(x, y, z, wl, norm, fom_fields)
			#Factor in boundary of desired integral region
			bounded_fom_vs_pos = fom_vs_pos * boundary_weight_vs_pos

			#Integrate, but skipping over dimensions that are single-valued.
			if zarray.size > 1:
				fom_vs_xy = np.trapz(y = bounded_fom_vs_pos, x = zarray)
			else:
				fom_vs_xy = bounded_fom_vs_pos.squeeze()
			if yarray.size > 1:
				fom_vs_x = np.trapz(y = fom_vs_xy, x = yarray)
			else:
				fom_vs_x = fom_vs_xy.squeeze()
			if xarray.size > 1:
				fom_vs_wl[idx] = np.trapz(y = fom_vs_x, x = xarray)
			else:
				fom_vs_wl[idx] = fom_vs_x[0]
		return fom_vs_wl


	@staticmethod
	def boundary_weights_vs_pos(xarray, yarray, zarray, boundary_func):
		#Creates 3D array of 0s and 1s corresponding to coordinate grid and boundary region
		xarray = xarray.flatten()
		yarray = yarray.flatten()
		zarray = zarray.flatten()
		boundary_weight_vs_pos = np.zeros((xarray.size, yarray.size, zarray.size))
		for i, x in enumerate(xarray):
			for j, y in enumerate(yarray):
				for k, z in enumerate(zarray):
					boundary_weight_vs_pos[i, j, k] = boundary_func(x,y,z)

		#Add check to make sure values are all 0 or 1?
		return boundary_weight_vs_pos


	@staticmethod
	def calculate_power(x, y, z, wl, norm, fields):
		''' Power = 0.5*Re(E x H* . n), where n is a unit normal vector'''
		E = fields.getfield(x,y,z,wl)
		H = fields.getHfield(x,y,z,wl)
		S = np.cross(E, np.conj(H))

		return 0.5*np.dot(S.real, norm)

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
		sim.fdtd.setnamed(index_monitor_name, 'frequency points', frequency_points)
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




