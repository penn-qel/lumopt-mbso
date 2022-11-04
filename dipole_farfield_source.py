import numpy as np
import lumapi
import scipy.constants
import time

def create_source(xsize, ysize, theta, phi, wavelengths, depth, n=2.4, z=0, grid = 40e-9, create_file = True):
    """
    Creates a source in Lumerical format corresponding to a dipole propagated in +z direction through a uniform material
       
    :param xsize:       x-span of plane to calculate fields on
    :param ysize:       y-span of plane to calculate fields on
    :param theta:       Orientation angle from z-axis of dipole source
    :param phi:         Orientation angle in xy plane of dipole source
    :wavelengths:       Array of wavelengths to calculate over
    :depth:             Distance below plane dipole is located
    :n:                 Refractive index of material. Default 2.4
    :z:                 z-position of plane. Default 0
    :grid:              Spatial sampling size for discretizing calculation.
    :create_file:       Boolean flag to create a file "sourcefile.mat" holding the field data. Default true

    """
    x = np.linspace(-xsize/2, xsize/2, xsize/grid)
    y = np.linspace(-ysize/2, ysize/2, ysize/grid)
    xv, yv, zv, wlv = np.meshgrid(x, y, np.array([z+depth]), wavelengths, indexing = 'ij')

    if theta < 0 or theta > 90:
        raise UserWarning('theta should be between 0 and 90 degrees')
    theta = np.radians(theta)
    phi = np.radians(phi)

    # I*l, determines radiation power
    const = 1e-18

    Ix = const*np.sin(theta)*np.cos(phi)
    Iy = const*np.sin(theta)*np.sin(phi)
    Iz = const*np.cos(theta)

    #Normal vector of source, chosen to be along z-direction of new coordinates
    nz2 = np.array([Ix, Iy, Iz])/np.linalg.norm(np.array([Ix, Iy, Iz]))
    if theta == 0:
        nx2 = np.array([1,0,0])
        ny2 = np.array([0,1,0])
    else:
        nx2 = np.cross(nz2, np.array([0,0,1]))/np.linalg.norm(np.cross(nz2, np.array([0,0,1])))
        ny2 = np.cross(nz2, nx2)

    #Stack points so last axis tells (x,y,z) component
    points = np.stack((xv,yv,zv), axis=-1)

    #Get components in new coordinates
    xv2 = np.dot(points, nx2)
    yv2 = np.dot(points, ny2)
    zv2 = np.dot(points, nz2)

    #Distance from source to points on plane
    rv = np.sqrt(np.square(xv2) + np.square(yv2) + np.square(zv2))
    thetav = np.arccos(zv2/rv)
    phiv = np.arctan2(yv2, xv2)

    beta = 2*np.pi/(wlv/n)
    omega = beta*(3e8/n)

    exponent = np.exp(-1j*beta*rv)/rv
    jbr = 1.0/(1j*beta*rv)
    nu = (scipy.constants.mu_0*scipy.constants.c)/n

    #Calculate fields from Balanis pg 282
    Er = const*nu*np.cos(thetav)/(2*np.pi*np.power(rv, 2))*(1+jbr)*exponent
    Etheta = 1j*const*nu*beta*np.sin(thetav)/(4*np.pi*rv)*(1+jbr-1/(np.power(beta*rv, 2)))*exponent
    Hphi = 1j*const*beta*np.sin(thetav)/(4*np.pi*rv)*(1+jbr)*exponent

    #Convert to shifted rectangular coordinates

    Ex2 = np.sin(thetav)*np.cos(phiv)*Er + np.cos(thetav)*np.cos(phiv)*Etheta
    Ey2 = np.sin(thetav)*np.sin(phiv)*Er + np.cos(thetav)*np.sin(phiv)*Etheta
    Ez2 = np.cos(thetav)*Er - np.sin(thetav)*Etheta
    E2 = np.stack((Ex2, Ey2, Ez2), axis=-1)

    Hx2 = -np.sin(phiv)*Hphi
    Hy2 = np.cos(phiv)*Hphi
    Hz2 = np.zeros(Hx2.shape)
    H2 = np.stack((Hx2, Hy2, Hz2), axis=-1)

    #Convert to real rectangular coordinates
    xhat = np.array([nx2[0], ny2[0], nz2[0]])
    yhat = np.array([nx2[1], ny2[1], nz2[1]])
    zhat = np.array([nx2[2], ny2[2], nz2[2]])
    Ex = np.dot(E2, xhat)
    Ey = np.dot(E2, yhat)
    Ez = np.dot(E2, zhat)

    Hx = np.dot(H2, xhat)
    Hy = np.dot(H2, yhat)
    Hz = np.dot(H2, zhat)

    E = np.stack((Ex, Ey, Ez), axis=-1)
    H = np.stack((Hx, Hy, Hz), axis=-1)

    fields = {'E' : E, 'H' : H, 'x': x, 'y': y, 'z':z, 'wl':wavelengths}

    if create_file:
        fdtd = lumapi.FDTD(hide=True)
        lumapi.putMatrix(fdtd.handle, 'x', x)
        lumapi.putMatrix(fdtd.handle, 'y', y)
        lumapi.putMatrix(fdtd.handle, 'z', z*np.ones(1))
        lumapi.putMatrix(fdtd.handle, 'f', np.divide(scipy.constants.speed_of_light,wavelengths))
        lumapi.putMatrix(fdtd.handle, 'Ex', Ex)
        lumapi.putMatrix(fdtd.handle, 'Ey', Ey)
        lumapi.putMatrix(fdtd.handle, 'Ez', Ez)
        lumapi.putMatrix(fdtd.handle, 'Hx', Hx)
        lumapi.putMatrix(fdtd.handle, 'Hy', Hy)
        lumapi.putMatrix(fdtd.handle, 'Hz', Hz)

        fdtd.eval("EM = rectilineardataset('EM fields', x, y, z);")
        fdtd.eval("EM.addparameter('lambda', c/f, 'f', f);")
        fdtd.eval("EM.addattribute('E', Ex, Ey, Ez);")
        fdtd.eval("EM.addattribute('H', Hx, Hy, Hz);")
        fdtd.eval("matlabsave('sourcefile.mat', EM);")
        fdtd.close()

    return fields
