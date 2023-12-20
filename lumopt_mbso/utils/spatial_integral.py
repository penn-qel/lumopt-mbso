import numpy as np

#Integrate over spatial coordinates, skipping single-valued dimension
def spatial_integral(integrand, xarray, yarray, zarray):
    if zarray.size > 1:
        fom_vs_xy = np.trapz(y = integrand, x=zarray, axis = 2)
    else:
        fom_vs_xy = integrand.squeeze()
    if yarray.size > 1:
        fom_vs_x = np.trapz(y=fom_vs_xy, x = yarray, axis = 1)
    else:
        fom_vs_x = fom_vs_xy.squeeze()
    if xarray.size > 1:
        fom_vs_wl = np.trapz(y = fom_vs_x, x = xarray, axis = 0)
    else:
        fom_vs_wl = fom_vs_x.squeeze()
    return fom_vs_wl