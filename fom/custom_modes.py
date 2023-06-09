################################################
# Script: custom_modes.py

# Description: This script defines a few helper functions to generate custom mode profiles
# Author: Amelia Klein
###############################################

import sys
import numpy as np
import scipy as sp
import lumapi
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from lumopt.utilities.scipy_wrappers import wrapped_GridInterpolator
from utils.interpolate_fields import interpolate_Efield, interpolate_Hfield
from lumopt.utilities.fields import Fields
from utils.ffthelpers import propagate_fields

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
    #Takes in points as (N,) array
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

def AiryDiskVectorized(depth, radius, index, x0, y0, z0, norm = np.array([0, 0, -1]), pol_norm = np.array([1, 0, 0])):
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
    #Takes in points as (N,) array
    def Emodefun(x,y,z,wl):
        a = np.column_stack((x - x0, y - y0, z - z0))
        r = np.sqrt(np.sum(a*a, axis = -1))
        v = 2*np.pi/(wl/index)*(radius/depth)*r
        out = np.ones(x.size)
        out = np.divide(2*sp.special.j1(v), v, out=out, where=(v!=0))
        return np.outer(out, pol_norm)

    def Hmodefun(x,y,z,wl):
        a = np.column_stack((x - x0, y - y0, z - z0))
        r = np.sqrt(np.sum(a*a, axis = -1))
        v = 2*np.pi/(wl/index)*(radius/depth)*r
        out = np.ones(x.size)
        out = np.divide(2*sp.special.j1(v), v, out=out, where=(v!=0))
        return np.outer(out, Hnorm/Z)

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
        return np.outer(np.ones(x.size), pol_norm)

    def Hmodefun(x, y,z,wl):
        return np.outer(np.ones(x.size), Hnorm/Z)

    return Emodefun, Hmodefun

def InterpolateMonitor(filename, monitorname = 'fom'):
    '''Returns interpolation functions on monitor data in a Lumerical sim file'''

    fdtd = lumapi.FDTD(filename = filename, hide = False)

    #Get fields
    fields = get_fields(fdtd,
                        monitor_name = monitorname,
                        field_result_name = 'fields',
                        get_eps = False,
                        get_D = False,
                        get_H = True,
                        nointerpolation = False)

    fdtd.close()
    
    def Em(x, y, z, wl):
        return interpolate_Efield(x, y, z, wl, fields)

    def Hm(x,y,z,wl):
        return interpolate_Hfield(x, y, z, wl, fields)

    return Em, Hm

def AiryDiskBackpropagated(meas_depth, wavelengths, focal_depth, radius, index, x0, y0, z0, norm = np.array([0, 0, -1]), pol_norm = np.array([1, 0, 0]), grid_res = 15e-9):
    '''Returns backpropagated version of Airy Disk. Separate from callable function to conserve memory'''

    #Create functions to calculate Airy disk
    Eairy, Hairy = AiryDiskVectorized(focal_depth, radius, index, x0, y0, z0, norm = np.array([0, 0, -1]), pol_norm = np.array([1, 0, 0]))

    #Generate coordinate grids for numerical evaluation. Assumes a rectangular size that spans the full radius
    x = np.linspace(-radius, radius, int(2*radius/grid_res) + 1)
    z = np.array([-1*focal_depth])
    xv, yv, zv, wlv = np.meshgrid(x, x, z, wavelengths, indexing = 'ij')

    print("Calculating Airy fields")
    E = Eairy(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((x.size, x.size, 1, len(wavelengths), 3))
    H = Hairy(xv.flatten(), yv.flatten(), zv.flatten(), wlv.flatten()).reshape((x.size, x.size, 1, len(wavelengths), 3)) 

    #Construct fields object of result
    fields = Fields(x, x, z, wavelengths.asarray(), E, None, index**2* np.ones(E.shape, dtype = np.cfloat), H)

    print("Backpropagating fields")
    #Backpropagate E and H. Replace in fields object
    fields.E, fields.H = propagate_fields(fields, (meas_depth - focal_depth))
    fields.z = np.array([-1*meas_depth])

    print("Creating wrapped interpolator")

    return fields

def Interpolate_AiryDiskBackpropagated(meas_depth, wavelengths, focal_depth, radius, index, x0, y0, z0, norm = np.array([0, 0, -1]), pol_norm = np.array([1, 0, 0]), grid_res = 15e-9):
    '''Returns callable functions based on interpolating the backpropagated fields'''

    #Calculates final field object
    fields = AiryDiskBackpropagated(meas_depth, wavelengths, focal_depth, radius, index, x0, y0, z0, norm, pol_norm, grid_res)

    #Creates callables based on interpolation function
    def Em(x, y, z, wl):
        return interpolate_Efield(x,y,z,wl, fields)

    def Hm(x,y,z,wl):
        return interpolate_Hfield(x,y,z,wl, fields)

    return Em, Hm
