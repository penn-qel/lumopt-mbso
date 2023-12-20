'''Takes in command line a path to simulation file and path for final dataset. Performs simulation on a set of (theta,phi) dipole orientations
   and records KTransmissionFom value at different NA points for each. Saves result matrix into output file

   USAGE: python fom_vs_dipole_orientation.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from fom.ktransmissionfom import KTransmissionFom
from utils.chebyshev import chebyshev
from utils.spectrum_weights import get_spectrum_weights

def transmission_vs_NA(sim, wavelengths, weights):
   '''Creates array of transmission vs NA for particular simulation'''
    
   #Create FOM object
   fom = KTransmissionFom('fom', NA=1, target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom.wavelengths = wavelengths.asarray()

   NArange = np.linspace(0,1,num=101)
   T = np.zeros(NArange.shape)

   fom.enter_analysis()

   T02=np.zeros(fom.wavelengths.size)
   T1 = np.zeros(fom.wavelengths.size)

   def create_NA_boundary(NA):
      def boundary(kx, ky):
         return np.square(kx) + np.square(ky) < NA**2
      return boundary

   for i, NA in enumerate(NArange):
      fom.set_kboundary_func(create_NA_boundary(NA))
      T[i] = fom.get_fom(sim)
      if NA == 0.2:
         T02 = fom.T_fwd_vs_wavelength
      if NA == 1:
         T1 = fom.T_fwd_vs_wavelength

   return T, T02, T1

def get_theta_phi_111(t):
   '''Get theta and phi in degrees for paramter t'''

   v = np.array([np.cos(t), np.sin(t), 0*t])
   theta = np.arccos(v[2])
   phi = np.arctan2(v[1],v[0])

   return np.degrees(theta), np.degrees(phi)

def get_theta_phi_100(t, n=0):
   '''Gets theta and phi in degrees for particular parameter with NV axis at theta=54.7. n = (0,1,2,3) determines
   which possible NV axis orientation this is. t can be scalar or np-array of values'''
   
   #Coordinate (x,y,z) array in <111> frame
   v = np.array([np.cos(t), np.sin(t), 0*t])

   #Rotation matrix around x axis
   thetanv = np.radians(54.7)
   R1 = np.array([[1,0,0],[0,np.cos(thetanv),-np.sin(thetanv)],[0,np.sin(thetanv),np.cos(thetanv)]])

   #Rotation matrix around new z axis
   alpha = np.radians(90*n)
   R2 = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

   #Total rotation matrix
   R = np.matmul(R2,R1)

   #Rotated vector into our coordinated frame.
   vrot = np.matmul(R, v)

   #Theta = arccos(z), phi = arctan(y/x)
   theta = np.arccos(vrot[2])
   phi = np.arctan2(vrot[1],vrot[0])

   return np.degrees(theta), np.degrees(phi)

def iterate_simulations(sim, theta, phi, wavelengths, weights):
   '''Iterate over all orientations and calculate T for each'''

   Tmatrix = np.array([])
   T02matrix = np.array([])
   T1matrix = np.array([])
   for i in range(theta.size):
      #Select dipole object
      sim.fdtd.select('source')
      #set angles
      sim.fdtd.set("theta", theta[i])
      sim.fdtd.set("phi", phi[i])

      #Run simulation
      print("Running simulation for dipole theta = {:.1f}, phi = {:.1f}".format(theta[i], phi[i]))
      sim.fdtd.run()

      #Calculate trans vs NA
      T_vs_NA, T02_vs_wl, T1_vs_wl = transmission_vs_NA(sim, wavelengths, weights)

      #Stack results into matrix
      Tmatrix = np.vstack((Tmatrix, T_vs_NA)) if Tmatrix.size else T_vs_NA
      T02matrix = np.vstack((T02matrix, T02_vs_wl)) if T02matrix.size else T02_vs_wl
      T1matrix = np.vstack((T1matrix, T1_vs_wl)) if T1matrix.size else T1_vs_wl

      #Clear saved data from simulation file
      sim.fdtd.switchtolayout()

   return Tmatrix, T02matrix, T1matrix

def main(args):
   sourcepath = args[0]
   destpath = args[1]

   #Get wavelengths and weights
   wavelengths = Wavelengths(chebyshev(600e-9,800e-9,31))
   spectrumdata = scipy.io.loadmat('../datasets/NVspectrum')
   weights = get_spectrum_weights(wavelengths, spectrumdata['wl'].flatten()*1e-9, spectrumdata['spectrum'].flatten())


   #Open simulation
   sim = Simulation('./', use_var_fdtd=False, hide_fdtd_cad = True)
   sim.load(sourcepath)
   sim.fdtd.switchtolayout()

   #Parameter to generate dipole configurations. num of points should be multiple of 2
   t = np.linspace(0,np.pi,8,endpoint = False)

   #Get dipole orientations for each possible NV axis
   theta0, phi0 = get_theta_phi_100(t, 0)
   theta1, phi1 = get_theta_phi_100(t, 1)
   theta2, phi2 = get_theta_phi_100(t, 2)
   theta3, phi3 = get_theta_phi_100(t, 3)
   theta = np.concatenate((theta0,theta1,theta2,theta3))
   phi = np.concatenate((phi0,phi1,phi2,phi3))

   #Perform simulations and save results
   Tmatrix, T02matrix, T1matrix = iterate_simulations(sim, theta, phi, wavelengths, weights)
   np.savez(destpath + '100', theta=theta, phi=phi, Tmatrix = Tmatrix, T02matrix=T02matrix, T1matrix=T1matrix, wavelengths = wavelengths)

   #Repeat for 111 orientation
   theta, phi = get_theta_phi_111(t)
   Tmatrix, T02matrix, T1matrix = iterate_simulations(sim, theta, phi, wavelengths, weights)
   np.savez(destpath + '111', theta=theta, phi=phi, Tmatrix = Tmatrix, T02matrix=T02matrix, T1matrix=T1matrix, wavelengths = wavelengths)

   sim.remove_data_and_save()

if __name__ == "__main__":
   main(sys.argv[1:])
