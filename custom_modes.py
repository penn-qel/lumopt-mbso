import sys
import numpy as np
import scipy as sp

def AiryDisk(depth, radius, index, x0, y0, z0, norm = np.array([0, 0, -1]), pol_norm = np.array([1, 0, 0])):
	#Depth = focal depth of lens
	#radius = radius of lens
	#index = index of refraction of medium
	#norm: normal vector of propagation direction
	#pol_norm: Polarization direction of electric field
	if norm.size != 3:
		raise UserWarning('Normal vector should be a 3x1 vector')
	if np.dot(norm, pol_norm) != 0:
		raise UserWarning('Polarization normal should be orthogonal to propagation normal')

	norm = norm / np.linalg.norm(norm)
	pol_norm = pol_norm / np.linalg.norm(pol_norm)
	Hnorm = np.cross(norm, pol_norm)
	Z = np.sqrt(sp.constants.mu_0/(np.power(index, 2)*sp.constants.epsilon_0))

	#h = 2J1(v)/v
	#v = 2*pi/lambda *(radius / depth) * r
	def Emodefun(x,y,z,wl):
		a = np.array([x-x0, y-y0, z-z0])
		r = np.linalg.norm(a - norm*np.dot(a,norm))
		v = 2*np.pi/(wl/index)*(radius/depth)*r
		value = 2*sp.special.j1(v)/v if v!= 0 else 1
		return value*pol_norm

	def Hmodefun(x,y,z,wl):
		a = np.array([x-x0, y-y0, z-z0])
		r = np.linalg.norm(a - norm*np.dot(a,norm))
		v = 2*np.pi/(wl/index)*(radius/depth)*r
		value = 2*sp.special.j1(v)/v if v!= 0 else 1
		return value*Hnorm/Z

	return Emodefun, Hmodefun


def PlaneWave(index, norm = np.array([0, 0, 1]), pol_norm = np.array([1, 0, 0])):
	if norm.size != 3:
		raise UserWarning('Normal vector should be a 3x1 vector')
	if np.dot(norm, pol_norm) != 0:
		raise UserWarning('Polarization normal should be orthogonal to propagation normal')

	norm = norm / np.linalg.norm(norm)
	pol_norm = pol_norm / np.linalg.norm(pol_norm)
	Hnorm = np.cross(norm, pol_norm)
	Z = np.sqrt(sp.constants.mu_0/(np.power(index, 2)*sp.constants.epsilon_0))

	def Emodefun(x, y, z, wl):
		return pol_norm

	def Hmodefun(x, y,z,wl):
		return Hnorm/Z

	return Emodefun, Hmodefun

