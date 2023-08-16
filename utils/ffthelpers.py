################################################
# Script: ffthelpers.py

# Description: This script defines a few helper functions and wrappers for ffts and propagators
# Author: Amelia Klein
###############################################

import numpy as np
import scipy.constants
import math
try:
    import scipy.fft as fft
except ModuleNotFoundError:
    import scipy.fftpack as fft

def create_NA_boundary(NA):
    '''Returns function corresponding to filter for a boundary of a given NA'''
    
    #Takes in as inputs normalized kx and ky. Returns 1 if within spot and 0 otherwise
    def boundary(kx, ky):
        return np.square(kx) + np.square(ky) < NA**2

    return boundary

def pad_field(A, x, y, wl):
    '''Pads field so that it is still centered but has a good number of points for efficient computation and high resolution'''

    #Calculates size necessary
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = fft.next_fast_len(max(math.ceil(np.max(wl)*25/dx), x.size))
    ny = fft.next_fast_len(max(math.ceil(np.max(wl)*25/dy), y.size))

    #Allocates new array
    origshape = A.shape
    Anew = np.zeros((nx, ny, origshape[2], origshape[3], origshape[4]), dtype = np.cdouble)

    #Calculates size of shift in each dimension
    bufx = int((nx-x.size)/2)
    bufy = int((ny-y.size)/2)

    #Copies in array to be centered
    Anew[bufx:bufx+x.size,bufy:bufy+y.size,:,:,:] = A

    return Anew, bufx, bufy


def filterfields(fields, kboundary_func, Ek, Hk, kx, ky):
    '''Filters fourier space and returns'''

    #Calculates weights
    weights = kboundary_func(kx.flatten(), ky.flatten()).reshape(Ek.shape[0], Ek.shape[1], 1, fields.wl.size, 1)

    return Ek*weights, Hk*weights

def ifftcrop(fields, Ek, Hk, bufx, bufy):
    '''Performs inverse Fourier transform and crops to original spatial size'''

    Einverse = fft.ifftn(Ek, axes = (0,1))
    Hinverse = fft.ifftn(Hk, axes = (0,1))

    return Einverse[bufx:bufx+fields.x.size,bufy:bufy+fields.y.size,:,:,:], Hinverse[bufx:bufx+fields.x.size,bufy:bufy+fields.y.size,:,:,:]

def filterkspace(fields, kboundary_func, Ek, Hk, kx, ky, bufx, bufy):
    '''Filters fourier space and inverts back to spatial domain'''

    Efilt, Hfilt = filterfields(fields, kboundary_func, Ek, Hk, kx, ky)

    return ifftcrop(fields, Efilt, Hfilt, bufx, bufy)

def kpropagate(Ek, Hk, kx, ky, wl, z, eps):
    '''Propagates fields by distance z in material eps'''

    #Calculate kz/k0
    kz = (np.sqrt(eps-kx**2-ky**2)).reshape(Ek.shape[0], Ek.shape[1], 1, wl.size, 1)

    #Calculate free space k0
    k0 = (2*np.pi/wl).reshape(1,1,1, wl.size, 1)

    #Total accumulated phase
    phase = np.exp(1j*kz*k0*z)

    return Ek*phase, Hk*phase

def fft2D(A, x, y):
    '''Calculates fft over first two axes of our 5D field arrays. Returns field and frequencies'''

    #Assume constant grid
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = A.shape[0]
    ny = A.shape[1]

    #Performs fft
    Ak = fft.fftn(A, axes=(0,1))

    #Gets spatial frequencies
    vx = fft.fftfreq(nx, dx)
    vy = fft.fftfreq(ny, dy)

    return Ak, vx, vy 

def getkfields2(A, x, y, wl):
    '''Transforms field into k-space from field and coordinates'''

    #Get zero-buffered fields
    E, bufx, bufy = pad_field(A, x, y, wl)

    #Perform fft
    Ek, vx, vy = fft2D(E, x, y)

    #Gets grid of relevant points. Normalizes spatial frequencies by multiplying by wavelength and gets boundary weights
    vxv, vyv, wlv = np.meshgrid(vx, vy, wl, indexing='ij')
    kx = vxv * wlv
    ky = vyv * wlv

    return Ek, kx, ky, bufx, bufy

def getkfields(fields):
    '''Transforms fields into k-space from fields object'''

    assert(fields.z.size == 1)
    Ek, kx, ky, bufx, bufy = getkfields2(fields.E, fields.x, fields.y, fields.wl)
    Hk, kx, ky, bufx, bufy = getkfields2(fields.H, fields.x, fields.y, fields.wl)

    return Ek, Hk, kx, ky, bufx, bufy

def propagate_fields(fields, z):
    '''Propagate fields a distance z (can be negative) and return'''

    #Transform into Fourier space
    Ek, Hk, kx, ky, bufx, bufy = getkfields(fields)

    #Filter fields across NA of 1 to get rid of impossible frequencies
    kboundary_func = create_NA_boundary(np.sqrt(np.max(fields.eps)))
    Efilt, Hfilt = filterfields(fields, kboundary_func, Ek, Hk, kx, ky)

    #Propagate in k space
    Eprop, Hprop = kpropagate(Efilt, Hfilt, kx, ky, fields.wl, z, fields.eps[0,0,0,:,0].reshape(1,1,fields.wl.size))

    #Perform inverse transform
    return ifftcrop(fields, Eprop, Hprop, bufx, bufy)
    