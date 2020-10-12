################################################
# Script: metasurface.py

# Description: This script defines a geometry object corresponding to a metasurface
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

class Metasurface(Geometry):
	"""
		Defines a metasurface composed of a grid of pillars in the (x,y) plane with a height h
	
		:param posx:				Array of shape (N,1) defining x-coordinates of each pillar
		:param posy:				Array of shape (N,1) defining y-coordinates of each pillar
		:param radii:           	Array of shape (N,1) defining radii of each pillar
		:param bounds: 				Bounding ranges (min/max pairs) for each optimization parameter
		:param z:					z-position of bottom of metasurface 
		:param h:               	height of metasurface
		:param radial_precision:	Number of points along circumference of each pillar to calculate gradient
		:param height_precision:	Number of points along height of pillar to calculate gradient
		:param eps_in:				Permittivity of the pillars
		:param eps_out: 			Permittivity of the material around the pillars
		:param dx: 					Step size for computing FOM gradient using permittivity perturbations

	"""

	def __init__(self, posx, posy, init_radii, bounds, z, h, eps_in, eps_out, radial_precision = 25, height_precision = 10, dx = 1.0e-10):
		self.posx = posx
		self.posy = posy
		self.radii = init_radii
		self.z = float(z)
		self.h = float(h)
		self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
		self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
		self.radial_precision = int(radial_precision)
		self.height_precision = int(height_precision)

		if self.h <= 0:
			raise UserWarning("pillar height must be positive.")

		self.dx = float(dx)
		if self.dx < 0.0:
			raise UserWarning("step size must be positive.")
		self.gradients = list()

		self.bounds = np.array(bounds)

		if self.bounds.shape[0] != self.radii.size:
			raise UserWarning("there must be one bound for each parameter.")
		elif self.bounds.shape[1] != 2:
			raise UserWarning("there should be a min and max bound for each parameter.")
		for bound in self.bounds:
			if bound[1] - bound[0] <= 0.0:
				raise UserWarning("bound ranges must be positive.")

	def add_geo(self, sim, params, only_update):
		''' Adds the geometry to a Lumerical simulation'''

		groupname = 'Pillars'
		if params is None:
			radii = self.radii
		else:
			radii = params

		#Saves current data to a .mat file so that structure group script can access it
		#Building many objects within a structure group script is MUCH faster than individually
		self.save_to_mat(radii)
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
		#Force structure to update
		sim.fdtd.set('counter', self.counter)

	def update_geometry(self, params, sim = None):
		'''Sets the radii. Allow option of also setting positions in future?'''
		self.radii = params

	def calculate_gradients(self, gradient_fields):
		'''Gradients are calculated by increasing the radius of each pillar'''
		gradients = list()
		for i, r in enumerate(self.radii):
			gradients.append(self.pillar_derivative(self.posx[i], self.posy[i], r, gradient_fields))
		self.gradients.append(np.array(gradients))
		return self.gradients[-1]

	def get_current_params(self):
		return self.radii.copy()


	def save_to_mat(self, radii):
		'''Saves core parameters to .mat file'''
		scipy.io.savemat('params.mat', mdict={'x': self.posx, 'y': self.posy, 'height': self.h, 'radii': radii})

	def create_script(self, sim, groupname = 'Pillars', only_update = False):
		'''Writes structure group script'''
		struct_script = ('deleteall;\n'
			'data = matlabload("params.mat");\n'
			'for(i=1:length(x)) {\n'
			    'addcircle;\n'
    			'set("name", "pillar_"+num2str(i));\n'
    			'set("x", x(i));\n'
    			'set("y", y(i));\n'
    			'set("z min", 0);\n'
    			'set("z max", height);\n'
    			'set("radius", radii(i));\n')

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
		x = self.posx.copy()*1e6
		y = self.posy.copy()*1e6
		r = self.radii.copy()*1e6
		maxbound = np.amax(self.bounds)*1e6
		ax.clear()
		for indx, rad in enumerate(r):
			circle = plt.Circle((x[indx], y[indx]), rad, color = 'black')
			ax.add_artist(circle)
		ax.set_title('Geometry')
		ax.set_ylim(min(y) - maxbound, max(y) + maxbound)
		ax.set_xlim(min(x) - maxbound, max(x) + maxbound)
		ax.set_xlabel('x (um)')
		ax.set_ylabel('y (um)')
		return True


	def pillar_derivative(self, x0, y0, r, gradient_fields):
		'''Calculates derivative for a particular pillar'''
		#x0, y0, r, z, h should be input in units of meters

		#Parameterize surface by theta and z
		thetav = np.linspace(0,2*np.pi, self.radial_precision)
		zv = np.linspace(self.z,self.z + self.h, self.height_precision)

		integrand_fun = gradient_fields.boundary_perturbation_integrand()
		wavelengths = gradient_fields.forward_fields.wl
		eps_in = self.eps_in.get_eps(wavelengths)
		eps_out = self.eps_out.get_eps(wavelengths)

		#Create a list of derivatives, calculated for each wavelength
		derivs = list()
		for idx, wl in enumerate(wavelengths):
			#Integrate across surface of pillar for each wavelength
			integrand_per_wl = np.zeros((thetav.size, zv.size))
			for i, theta in enumerate(thetav):
				x = x0 + r*np.cos(theta)
				y = y0 + r*np.sin(theta)
				normal_vect = np.array([np.cos(theta), np.sin(theta), 0])
				for j, z in enumerate(zv):
					integrand_per_wl[i,j] = integrand_fun(x,y,z,wl,normal_vect,eps_in[idx], eps_out[idx])
			#Perform integral for each wavelength
			integrand_theta = np.trapz(y = r*integrand_per_wl, x = thetav, axis=0)
			derivs.append(np.trapz(y = integrand_theta, x = zv))

		return np.array(derivs).flatten()



class Metasurface2D(Geometry):
	"""
		Defines a 2D metasurface composed of an array of pillars with given height
	
		:param posx:				Array of shape (N,1) defining x-coordinates of each pillar
		:param widths:      		Array of shape (N,1) defining widthsh pillar
		:param bounds: 				Bounding ranges (min/max pairs) for each optimization parameter
		:param z:					z-position of bottom of metasurface 
		:param h:               	height of metasurface
		:param height_precision:	Number of points along height of pillar to calculate gradient
		:param eps_in:				Permittivity of the pillars
		:param eps_out: 			Permittivity of the material around the pillars
		:param dx: 					Step size for computing FOM gradient using permittivity perturbations

	"""

	def __init__(self, posx, init_widths, bounds, y, h, eps_in, eps_out, height_precision = 10, dx = 1.0e-9):
		self.posx = posx
		self.widths = init_widths
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

		self.bounds = np.array(bounds)

		if self.bounds.shape[0] != self.widths.size:
			raise UserWarning("there must be one bound for each parameter.")
		elif self.bounds.shape[1] != 2:
			raise UserWarning("there should be a min and max bound for each parameter.")
		for bound in self.bounds:
			if bound[1] - bound[0] <= 0.0:
				raise UserWarning("bound ranges must be positive.")

	def add_geo(self, sim, params, only_update):
		''' Adds the geometry to a Lumerical simulation'''

		groupname = 'Pillars'
		if params is None:
			widths = self.widths
		else:
			widths = params

		#Saves current data to a .mat file so that structure group script can access it
		#Building many objects within a structure group script is MUCH faster than individually
		self.save_to_mat(widths)
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
		#Force structure to update
		sim.fdtd.set('counter', self.counter)

	def update_geometry(self, params, sim = None):
		'''Sets the widths. Allow option of also setting positions in future?'''
		self.widths = params

	def calculate_gradients(self, gradient_fields):
		'''Gradients are calculated by increasing the radius of each pillar'''
		gradients = list()
		for i, w in enumerate(self.widths):
			gradients.append(self.pillar_derivative(self.posx[i], w, gradient_fields))
		self.gradients.append(np.array(gradients))
		return self.gradients[-1]

	def get_current_params(self):
		return self.widths.copy()

	def save_to_mat(self, widths):
		'''Saves core parameters to .mat file'''
		scipy.io.savemat('params.mat', mdict={'x': self.posx, 'height': self.h, 'widths': widths})

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
		x = self.posx.copy()*1e6
		w = self.widths.copy()*1e6
		maxbound = np.amax(self.bounds)*1e6
		ax.clear()
		for indx, rad in enumerate(w):
			rect = patches.Rectangle((x[indx] - w[indx]/2, 0), w[indx], self.h*1e6, facecolor='black')
			ax.add_patch(rect)
		ax.set_title('Geometry')
		ax.set_ylim(0, self.h*1e6)
		ax.set_xlim(min(x) - maxbound, max(x) + maxbound)
		ax.set_xlabel('x (um)')
		ax.set_ylabel('y (um)')
		return True

	def pillar_derivative(self, x0, w, gradient_fields):
		'''Calculates derivative for a particular pillar'''

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