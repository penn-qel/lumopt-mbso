'''Takes in command line a path to simulation file and path for final dataset. Performs simulation on a set of (x,y) dipole positions
   and records KTransmissionFom value at NA=0.2 at each. Saves result matrix into output file

   USAGE: python fom_vs_dipole_xy.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from fom.ktransmissionfom import KTransmissionFom
from utils.chebyshev import chebyshev
from utils.spectrum_weights import get_spectrum_weights


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

   #Two dipole orientations
   theta1 = 90
   theta2 = 35.3
   phi1 = 0
   phi2 = 90

   #x values. Total simulations will be 2*num_points^2
   x = np.linspace(-0.25e-6, 0.25e-6, 21)
   y = np.copy(x)
   xv, yv = np.meshgrid(x, y)
   FOM1 = np.zeros(xv.shape)
   FOM2 = np.zeros(xv.shape)

   fom = KTransmissionFom('fom', NA=0.2, target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom.wavelengths = wavelengths.asarray()

   #Iterate through positions
   for i in range(xv.shape[0]):
      for j in range(xv.shape[1]):
         #Place dipole
         sim.fdtd.select("source")
         sim.fdtd.set("x", xv[i,j])
         sim.fdtd.set("y", yv[i,j])

         #Do first orientation
         sim.fdtd.set("theta", theta1)
         sim.fdtd.set("phi", phi1)
         
         print("Running simulation 1 for dipole x= {:.2e}, y = {:.2e}".format(xv[i,j], yv[i,j]))
         sim.fdtd.run()
         FOM1[i,j] = fom.get_fom(sim)
         sim.fdtd.switchtolayout()

         #Second orientation
         sim.fdtd.select("source")
         sim.fdtd.set("theta", theta2)
         sim.fdtd.set("phi", phi2)

         print("Running simulation 2 for dipole x= {:.2e}, y = {:.2e}".format(xv[i,j], yv[i,j]))
         sim.fdtd.run()
         FOM2[i,j] = fom.get_fom(sim)
         sim.fdtd.switchtolayout()

   #Return dipole to origin
   sim.fdtd.select("source")
   sim.fdtd.set("x", 0)
   sim.fdtd.set("y", 0)

   #Save data to file
   np.savez(destpath, theta1=theta1, phi1=phi1, theta2 = theta2, phi2 = phi2, X = xv, Y = yv, fom1 = FOM1, fom2 = FOM2)
   sim.remove_data_and_save()

if __name__ == "__main__":
   main(sys.argv[1:])