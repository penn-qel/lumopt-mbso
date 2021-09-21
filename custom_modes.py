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
        return pol_norm

    def Hmodefun(x, y,z,wl):
        return Hnorm/Z

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
    return fields.getfield, fields.getHfield

def InterpolateMonitor2(filename, monitorname = 'fom'):
    '''Returns interpolation functions on monitor data in a Lumerical sim file'''

    fdtd = lumapi.FDTD(filename = filename, hide = False)

    E = fdtd.getresult(monitorname, 'E')
    H = fdtd.getresult(monitorname, 'H')
    x = np.array([E['x']]).flatten()
    y = np.array([E['y']]).flatten()
    z = np.array([E['z']]).flatten()
    Efield = E['E']
    Hfield = H['H']
    wl = sp.constants.c / (np.array(E['f']).flatten())

    Em = wrapped_GridInterpolator((x,y,z,wl), Efield)
    print(Em(0,0,-0.5e-6,700e-9))