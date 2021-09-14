'''2D optimization. 600-800 nm, multi-freq source'''

import numpy as np
import scipy.constants

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.optimization import Optimization
from lumopt.utilities.fields import Fields
from lumopt.utilities.gradients import GradientFields

from moving_metasurface import MovingMetasurface2D
from moving_metasurface_old import MovingMetasurfaceOld2D


eps_0 = scipy.constants.epsilon_0

height = 0.25e-6
scaling_factor = 1e7

pitch = 0.25
xmax = 3
xmin = -3
N = 5

x = np.linspace(xmin, xmax, N)*1e-6

widths = 100e-9*np.ones(len(x))

surface = MovingMetasurface2D( posx = x.flatten(), 
                        init_widths = widths.flatten(),
                        min_feature_size = 25e-9,
                        y = 0,
                        h = height,
                        eps_in = np.power(2.4, 2),
                        eps_out = np.power(1,2),
                        height_precision = 4,
                        scaling_factor = scaling_factor,
                        simulation_span = 30.2e-6)

surface_old = MovingMetasurfaceOld2D( posx = x.flatten(), 
                        init_widths = widths.flatten(),
                        min_feature_size = 25e-9,
                        y = 0,
                        h = height,
                        eps_in = np.power(2.4, 2),
                        eps_out = np.power(1,2),
                        height_precision = 4,
                        scaling_factor = scaling_factor,
                        simulation_span = 30.2e-6)

z = 0 

#Ef = (x + y + z + wl)*[1,2,3]
x = np.linspace(-4e-6, 4e-6, 20)
y = np.linspace(0, 0.3, 12)
z = np.array([0])
wl = np.arange(5)

xv, yv, zv, wlv = np.meshgrid(x, y, z, wl, indexing = 'ij')

Ef = np.zeros((x.size, y.size, 1, wl.size, 3))
fun = xv + yv + zv + wlv
Ef[:,:,:,:,0] = fun
Ef[:,:,:,:,1] = 2*fun
Ef[:,:,:,:,2] = 3*fun
Df = Ef*eps_0

#Ea = (2*x - y + z +wl)*[3, 1, 2]
Ea = np.zeros((x.size, y.size, 1, wl.size, 3))
fun = 2*xv - yv + zv + wlv
Ea[:,:,:,:,0] = 3*fun
Ea[:,:,:,:,1] = fun
Ea[:,:,:,:,2] = 2*fun
Da = Ea*eps_0

forward_fields = Fields(x, y, z, wl, Ef, Df, None, None)
adjoint_fields = Fields(x, y, z, wl, Ea, Da, None, None)
gradient_fields = GradientFields(forward_fields, adjoint_fields)

#Comparing interpolation ability:
xtest = np.linspace(-3e-6, 3e-6, 4)
ytest = np.linspace(0, 0.25, 3)
xvtest, yvtest = np.meshgrid(xtest, ytest, indexing='ij')
Einterp, Dinterp = MovingMetasurface2D.interpolate_fields(xvtest, yvtest, 0, forward_fields)
#print('My interpolated E:')
#print(Einterp)

xvtest1, yvtest1, zvtest1, wlvtest1 = np.meshgrid(xtest, ytest, z, wl, indexing='ij')
fun = xvtest1 + yvtest1 + zvtest1 + wlvtest1
Eexact = np.zeros((xtest.size, ytest.size, 1, wl.size, 3))
Eexact[:,:,:,:,0] = fun
Eexact[:,:,:,:,1] = 2*fun
Eexact[:,:,:,:,2] = 3*fun
#print('Exact E:')
Eexact_reshaped = np.reshape(Eexact,(xtest.size, ytest.size, wl.size, 3))
#print(Eexact_reshaped)

#print('% Difference:')
#print((Einterp-Eexact_reshaped)/Eexact_reshaped*100)
print('Max % difference:')
print(np.amax((Einterp-Eexact_reshaped)/Eexact_reshaped*100))

Einterp_old = np.zeros((xtest.size, ytest.size, 1, wl.size, 3))
for i,xval in np.ndenumerate(xtest):
    for j,yval in np.ndenumerate(ytest):
        for k, wlval in np.ndenumerate(wl):
            Einterp_old[i,j,0,k,:] = forward_fields.getfield(xval, yval, 0, wlval)

Einterp_old_reshaped = np.reshape(Einterp_old, (xtest.size, ytest.size, wl.size, 3))
print('max % Difference with old interp:')
print(np.amax((Einterp-Einterp_old_reshaped)/Einterp*100))

'''grad_new = surface.calculate_gradients(gradient_fields)
print('Gradient with new calculation:')
print(grad_new)
print('Gradient with old calculation:')
grad_old = surface_old.calculate_gradients(gradient_fields)
print(grad_old)

h1_n, h2_n = np.split(grad_new, 2)
h1_o, h2_o = np.split(grad_old, 2)

print('% Difference of halves:')
print((h1_n-h2_o)/h2_o*100)
print('Other halves:')
print((h2_n - h1_o)/h1_o*100)

print(grad_new - np.concatenate((h1_n, h2_n)))'''

posnew, widthnew = surface.calculate_gradients(gradient_fields)
posold, widthold = surface_old.calculate_gradients(gradient_fields)

print('New pos derivs:')
print(posnew)
print('Old pos derivs:')
print(posold)
print('New width derivs:')
print(widthnew)
print('Old width derivs:')
print(widthold)

