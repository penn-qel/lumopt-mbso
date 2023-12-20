'''Takes in command line a path to simulation file and path for final dataset. Performs simulation on a set of (x,y) dipole positions
   and records KTransmissionFom value at NA=0.2 at each. Saves result matrix into output file

   USAGE: python fom_vs_dipole_z.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from fom.ktransmissionfom import KTransmissionFom
from fom.transmissionfom import TransmissionFom
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

   #Get starting z value
   z_init = sim.fdtd.getnamed("source", "z")

   #z values (start, stop, step)
   z = np.arange(-2e-6, 0, 0.05e-6)
   FOM1 = np.zeros(z.shape)
   FOM2 = np.zeros(z.shape)

   #fom = TransmissionFom('fom', target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom = KTransmissionFom('fom', NA=0.2, target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom.wavelengths = wavelengths.asarray()

   #Iterate through positions
   for i, zval in enumerate(z):
      #Adjust FDTD boundary
      sim.fdtd.select('FDTD')
      sim.fdtd.set("z min", min(zval-0.5e-6,-1.5e-6))

      #Place dipole
      sim.fdtd.select("source")
      sim.fdtd.set("z", zval)

      #Do first orientation
      sim.fdtd.set("theta", theta1)
      sim.fdtd.set("phi", phi1)
      
      print("Running simulation 1 for dipole z = {:.2e}".format(zval))
      sim.fdtd.run()
      FOM1[i] = fom.get_fom(sim)
      sim.fdtd.switchtolayout()

      #Second orientation
      sim.fdtd.select("source")
      sim.fdtd.set("theta", theta2)
      sim.fdtd.set("phi", phi2)

      print("Running simulation 2 for dipole z = {:.2e}".format(zval))
      sim.fdtd.run()
      FOM2[i] = fom.get_fom(sim)
      sim.fdtd.switchtolayout()

   #Return dipole to origin
   sim.fdtd.select("source")
   sim.fdtd.set("z", z_init)

   #Save data to file
   np.savez(destpath, theta1=theta1, phi1=phi1, theta2 = theta2, phi2 = phi2, z = z, fom1 = FOM1, fom2 = FOM2)
   sim.remove_data_and_save()

if __name__ == "__main__":
   main(sys.argv[1:])