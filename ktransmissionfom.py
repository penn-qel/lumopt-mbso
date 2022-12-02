###############################################################
# Class: KTransmissionFom

# Description: This class defines a FOM object based on power transmission through some region in spatial k-space, such as fitting within a certain fiber NA
# Author: Amelia Klein
###############################################################

from transmissionfom import TransmissionFom
import numpy as np
import scipy.constants
try:
    import scipy.fft as fft
except ModuleNotFoundError:
    import scipy.fftpack as fft
import math
from lumopt.figures_of_merit.modematch import ModeMatch

class KTransmissionFom(TransmissionFom):
    """Calculates FOM by integrating Poynting vector over a region in spatial k-space determined by boundary function

    Parameters
    -------------------
    :param monitor_name:        name of monitor that records FOM field
    :param direction:           direction of propagation ('Forward' or 'Backward') of source
    :param multi_freq_src:      bool flag to enable/disable multi-frequency source calculation
    :param target_T_Fwd:        function describing target T_forward vs wavelength
    :param target_T_fwd_weights:function weighting different wavelengths for FOM
    :param boundary_func:       function defining boundary function for determining space to integrate over
    :param NA:                  Numerical aperture of target FOM. Used to auto-create a boundary function for this case
    :param norm_p:              exponent of p-norm used to generate FOM
    :param target_fom:          target value for FOM for printing/plotting distance of current design
    """

    def __init__(self, monitor_name, direction = 'Forward', boundary_func = None, NA = 1, 
                multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size),
                target_T_fwd_weights = lambda wl: np.ones(wl.size), 
                source_precision = 10e-9, norm_p = 1, target_fom = 0):

        super().__init__(monitor_name, direction = direction, multi_freq_src = multi_freq_src, target_T_fwd = target_T_fwd,
                        target_T_fwd_weights = target_T_fwd_weights, norm_p = norm_p, target_fom = target_fom)

        self.NA = float(NA)

        self.kboundary_func = boundary_func
        if self.kboundary_func is None:
            print("Creating boundary function for NA of {}".format(self.NA))
            self.kboundary_func = KTransmissionFom.create_NA_boundary(self.NA)

        self.analysis_mode = False
        self.fields_saved = False


    def get_fom_fields(self, sim):
        '''Returns fields after applying appropriate filter in Fourier domain'''
        
        if not self.fields_saved:
            #Get fields from simulation monitor
            monitor_fields = super().get_fom_fields(sim)
            
            #Perform Fourier transform
            Ek, Hk, kx, ky, bufx, bufy = KTransmissionFom.getkfields(monitor_fields)
            
            #Save fields for future evaluations
            if self.analysis_mode:
                self.fields, self.Ek, self.Hk, self.kx, self.ky, self.bufx, self.bufy = monitor_fields, Ek, Hk, kx, ky, bufx, bufy
                self.fields_saved = True

            return KTransmissionFom.filterkspace(monitor_fields, self.kboundary_func, Ek, Hk, kx, ky, bufx, bufy)

        #Uses saved fields instead to skip copying from lumerical and performing forward fourier transform
        return KTransmissionFom.filterkspace(self.fields, self.kboundary_func, self.Ek, self.Hk, self.kx, self.ky, self.bufx, self.bufy)

    def enter_analysis(self):
        '''Sets flag to save computation time when repeatedly calculating ift in post-analysis'''
        self.analysis_mode = True
        
    @staticmethod
    def getkfields(fields):
        '''Transforms fields into k-space'''

        assert(fields.z.size == 1)
        #Get zero-buffered fields
        E, bufx, bufy = KTransmissionFom.pad_field(fields.E, fields.x, fields.y, fields.wl)
        H, bufy, bufy = KTransmissionFom.pad_field(fields.H, fields.x, fields.y, fields.wl)
        
        #Perform fft
        Ek, vx, vy = KTransmissionFom.fft2D(E, fields.x, fields.y)
        Hk, vx, vy = KTransmissionFom.fft2D(H, fields.x, fields.y)

        #Gets grid of relevant points. Normalizes spatial frequencies by multiplying by wavelength and gets boundary weights
        vxv, vyv, wlv = np.meshgrid(vx, vy, fields.wl, indexing = 'ij')
        kx = vxv * wlv
        ky = vyv * wlv

        return Ek, Hk, kx, ky, bufx, bufy

    @staticmethod
    def filterkspace(fields, kboundary_func, Ek, Hk, kx, ky, bufx, bufy):
        '''Filters fourier space and inverts back to spatial domain'''

        #Calculates weights
        weights = kboundary_func(kx.flatten(), ky.flatten()).reshape(Ek.shape[0], Ek.shape[1], 1, fields.wl.size, 1)

        #Perform inverse fft
        Einverse = fft.ifftn(Ek*weights, axes=(0,1))
        Hinverse = fft.ifftn(Hk*weights, axes=(0,1))

        #Crop to original field data
        fields.E = Einverse[bufx:bufx+fields.x.size,bufy:bufy+fields.y.size,:,:,:]
        fields.H = Hinverse[bufx:bufx+fields.x.size,bufy:bufy+fields.y.size,:,:,:]

        return fields
        
    @staticmethod
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

    @staticmethod
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

        
    @staticmethod
    def create_NA_boundary(NA):
        
        #Takes in as inputs normalized kx and ky. Returns 1 if within spot and 0 otherwise
        def boundary(kx, ky):
            return np.square(kx) + np.square(ky) < NA**2

        return boundary