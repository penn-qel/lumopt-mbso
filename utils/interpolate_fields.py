import numpy as np

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

    E00 = fields.E[xi-1, yi-1, zi-1,:,:].squeeze(axis=(1,2))*(1 - xd) + fields.E[xi, yi-1, zi-1,:,:].squeeze(axis=(1,2))*xd
    E01 = fields.E[xi-1, yi-1, zi,:,:].squeeze(axis=(1,2))*(1-xd) + fields.E[xi, yi-1, zi,:,:].squeeze(axis=(1,2))*xd
    E10 = fields.E[xi-1, yi, zi-1,:,:].squeeze(axis=(1,2))*(1-xd) + fields.E[xi, yi, zi-1,:,:].squeeze(axis=(1,2))*xd
    E11 = fields.E[xi-1, yi, zi,:,:].squeeze(axis=(1,2))*(1-xd) + fields.E[xi, yi, zi,:,:].squeeze(axis=(1,2))*xd

    E0 = E00*(1-yd) + E10*yd
    E1 = E01*(1-yd) + E11*yd
    E = E0*(1-zd) + E1*zd

    D00 = fields.D[xi-1, yi-1, zi-1,:,:].squeeze(axis=(1,2))*(1 - xd) + fields.D[xi, yi-1, zi-1,:,:].squeeze(axis=(1,2))*xd
    D01 = fields.D[xi-1, yi-1, zi,:,:].squeeze(axis=(1,2))*(1-xd) + fields.D[xi, yi-1, zi,:,:].squeeze(axis=(1,2))*xd
    D10 = fields.D[xi-1, yi, zi-1,:,:].squeeze(axis=(1,2))*(1-xd) + fields.D[xi, yi, zi-1,:,:].squeeze(axis=(1,2))*xd
    D11 = fields.D[xi-1, yi, zi,:,:].squeeze(axis=(1,2))*(1-xd) + fields.D[xi, yi, zi,:,:].squeeze(axis=(1,2))*xd

    D0 = D00*(1-yd) + D10*yd
    D1 = D01*(1-yd) + D11*yd
    D = D0*(1-zd) + D1*zd

    return E, D

def interpolate_Efield(x, y, z, wl, fields):
    #takes in a fields structure, and (N,) arrays of x,y,z,wl points to interpolate on. Returns E only

    xi = np.searchsorted(fields.x, x).reshape(x.size, 1)
    yi = np.searchsorted(fields.y, y).reshape(x.size, 1)
    wli = np.searchsorted(fields.wl, wl).reshape(x.size, 1)
    x1 = (fields.x[xi-1])
    x2 = (fields.x[xi])
    y1 = (fields.y[yi-1])
    y2 = (fields.y[yi])
    x = x.reshape(x.size, 1)
    y = y.reshape(x.size, 1)

    #Calculates rectangle areas for bilinear interpolation (Wikipedia)
    denom = (x2-x1)*(y2-y1)
    w11 = ((x2-x)*(y2-y)/denom)
    w12 = ((x2-x)*(y-y1)/denom)
    w21 = ((x-x1)*(y2-y)/denom)
    w22 = ((x-x1)*(y-y1)/denom)
    
    return fields.E[xi-1,yi-1,0,wli,:].squeeze()*w11 + fields.E[xi-1,yi,0,wli,:].squeeze()*w12 + fields.E[xi,yi-1,0,wli,:].squeeze()*w21 + fields.E[xi,yi,0,wli,:].squeeze()*w22

def interpolate_Hfield(x, y, z, wl, fields):
    #takes in a fields structure, and (N,) arrays of x,y,z,wl points to interpolate on. Returns E only

    xi = np.searchsorted(fields.x, x).reshape(x.size, 1)
    yi = np.searchsorted(fields.y, y).reshape(x.size, 1)
    wli = np.searchsorted(fields.wl, wl).reshape(x.size, 1)
    x1 = (fields.x[xi-1])
    x2 = (fields.x[xi])
    y1 = (fields.y[yi-1])
    y2 = (fields.y[yi])
    x = x.reshape(x.size, 1)
    y = y.reshape(x.size, 1)

    #Calculates rectangle areas for bilinear interpolation (Wikipedia)
    denom = (x2-x1)*(y2-y1)
    w11 = ((x2-x)*(y2-y)/denom)
    w12 = ((x2-x)*(y-y1)/denom)
    w21 = ((x-x1)*(y2-y)/denom)
    w22 = ((x-x1)*(y-y1)/denom)
    
    return fields.H[xi-1,yi-1,0,wli,:].squeeze()*w11 + fields.H[xi-1,yi,0,wli,:].squeeze()*w12 + fields.H[xi,yi-1,0,wli,:].squeeze()*w21 + fields.H[xi,yi,0,wli,:].squeeze()*w22

def interpolate_fields2D(x, y, z, fields):
    #fields.x, fields.y, fields.z, fields.E, fields.D, fields.wl are relevant terms
    #E a 5D matrix in form x:y:z:wl:vector where vector = 0,1,2 for x,y,z components

    
    #Finds meshgrid indices of upper bound x,y,z locations in array
    nx = x.shape[0]
    ny = x.shape[1]

    xi, yi = np.meshgrid(np.searchsorted(fields.x, x[:,0]), np.searchsorted(fields.y, y[0,:]), indexing = 'ij')
    x1 = (fields.x[xi-1]).flatten()
    x2 = (fields.x[xi]).flatten()
    y1 = (fields.y[yi-1]).flatten()
    y2 = (fields.y[yi]).flatten()
    xi = xi.flatten()
    yi = yi.flatten()
    x = x.flatten()
    y = y.flatten()

    #Calculates rectangle areas for bilinear interpolation (Wikipedia)
    denom = (x2-x1)*(y2-y1)
    w11 = ((x2-x)*(y2-y)/denom).reshape(x.size, 1, 1)
    w12 = ((x2-x)*(y-y1)/denom).reshape(x.size, 1, 1)
    w21 = ((x-x1)*(y2-y)/denom).reshape(x.size, 1, 1)
    w22 = ((x-x1)*(y-y1)/denom).reshape(x.size, 1, 1)
    
    E = fields.E[xi-1,yi-1,z,:,:]*w11 + fields.E[xi-1,yi,z,:,:]*w12 + fields.E[xi,yi-1,z,:,:]*w21 + fields.E[xi,yi,z,:,:]*w22
    D = fields.D[xi-1,yi-1,z,:,:]*w11 + fields.D[xi-1,yi,z,:,:]*w12 + fields.D[xi,yi-1,z,:,:]*w21 + fields.D[xi,yi,z,:,:]*w22
    return E.reshape(nx, ny, fields.wl.size, 3), D.reshape(nx, ny, fields.wl.size, 3)