def interpolate_fields(x, y, z, fields):
    #takes in a fields structure, and (N,) arrays of x,y,z points to interpolate on. Returns E and D

    xi = np.searchsorted(fields.x, x).reshape(x.size, 1, 1)
    yi = np.searchsorted(fields.y, y).reshape(x.size, 1, 1)
    zi = np.searchsorted(fields.z, z).reshape(x.size, 1, 1)
    x = x.reshape(x.size, 1, 1)
    y = y.reshape(x.size, 1, 1)
    z = z.reshape(x.size, 1, 1)

    #Follows Wikipedia algorithm for trilinear interpolation
    xd = (x - fields.x[xi-1])/(fields.x[xi] - fields.x[xi-1])
    yd = (y - fields.y[yi-1])/(fields.y[yi] - fields.y[yi-1])
    zd = (z - fields.z[zi-1])/(fields.z[zi] - fields.z[zi-1])

    E00 = fields.E[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.E[xi, yi-1, zi-1,:,:].squeeze()*xd
    E01 = fields.E[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi-1, zi,:,:].squeeze()*xd
    E10 = fields.E[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi-1,:,:].squeeze()*xd
    E11 = fields.E[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi,:,:].squeeze()*xd

    E0 = E00*(1-yd) + E10*yd
    E1 = E01*(1-yd) + E11*yd
    E = E0*(1-zd) + E1*zd

    D00 = fields.D[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.D[xi, yi-1, zi-1,:,:].squeeze()*xd
    D01 = fields.D[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.D[xi, yi-1, zi,:,:].squeeze()*xd
    D10 = fields.D[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.D[xi, yi, zi-1,:,:].squeeze()*xd
    D11 = fields.D[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.D[xi, yi, zi,:,:].squeeze()*xd

    D0 = D00*(1-yd) + D10*yd
    D1 = D01*(1-yd) + D11*yd
    D = D0*(1-zd) + D1*zd

    return E, D

def interpolate_Efield(x, y, z, fields):
    #takes in a fields structure, and (N,) arrays of x,y,z points to interpolate on. Returns E only

    xi = np.searchsorted(fields.x, x).reshape(x.size, 1, 1)
    yi = np.searchsorted(fields.y, y).reshape(x.size, 1, 1)
    zi = np.searchsorted(fields.z, z).reshape(x.size, 1, 1)
    x = x.reshape(x.size, 1, 1)
    y = y.reshape(x.size, 1, 1)
    z = z.reshape(x.size, 1, 1)

    #Follows Wikipedia algorithm for trilinear interpolation
    xd = (x - fields.x[xi-1])/(fields.x[xi] - fields.x[xi-1])
    yd = (y - fields.y[yi-1])/(fields.y[yi] - fields.y[yi-1])
    zd = (z - fields.z[zi-1])/(fields.z[zi] - fields.z[zi-1])

    E00 = fields.E[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.E[xi, yi-1, zi-1,:,:].squeeze()*xd
    E01 = fields.E[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi-1, zi,:,:].squeeze()*xd
    E10 = fields.E[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi-1,:,:].squeeze()*xd
    E11 = fields.E[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.E[xi, yi, zi,:,:].squeeze()*xd

    E0 = E00*(1-yd) + E10*yd
    E1 = E01*(1-yd) + E11*yd
    E = E0*(1-zd) + E1*zd

    return E

def interpolate_Hfield(x, y, z, fields):
    #takes in a fields structure, and (N,) arrays of x,y,z points to interpolate on. Returns H only

    xi = np.searchsorted(fields.x, x).reshape(x.size, 1, 1)
    yi = np.searchsorted(fields.y, y).reshape(x.size, 1, 1)
    zi = np.searchsorted(fields.z, z).reshape(x.size, 1, 1)
    x = x.reshape(x.size, 1, 1)
    y = y.reshape(x.size, 1, 1)
    z = z.reshape(x.size, 1, 1)

    #Follows Wikipedia algorithm for trilinear interpolation
    xd = (x - fields.x[xi-1])/(fields.x[xi] - fields.x[xi-1])
    yd = (y - fields.y[yi-1])/(fields.y[yi] - fields.y[yi-1])
    zd = (z - fields.z[zi-1])/(fields.z[zi] - fields.z[zi-1])

    H00 = fields.H[xi-1, yi-1, zi-1,:,:].squeeze()*(1 - xd) + fields.H[xi, yi-1, zi-1,:,:].squeeze()*xd
    H01 = fields.H[xi-1, yi-1, zi,:,:].squeeze()*(1-xd) + fields.H[xi, yi-1, zi,:,:].squeeze()*xd
    H10 = fields.H[xi-1, yi, zi-1,:,:].squeeze()*(1-xd) + fields.H[xi, yi, zi-1,:,:].squeeze()*xd
    H11 = fields.H[xi-1, yi, zi,:,:].squeeze()*(1-xd) + fields.H[xi, yi, zi,:,:].squeeze()*xd

    H0 = H00*(1-yd) + H10*yd
    H1 = H01*(1-yd) + H11*yd
    H = H0*(1-zd) + H1*zd
    
    return E