import numpy as np

def pillar_constraints(geo):
    '''Builds constraint objects given geometry'''
    '''Creates upper bound constraint on all pairs of pillars by treating as
    circles with radii equal to major axis radius'''

    def constraint(params):
        '''Returns (N(N-1)/2,) array of constraints'''
        offset_x, offset_y, rx, ry, phi = geo.get_from_params(params)
        x = offset_x + geo.init_x
        y = offset_y + geo.init_y
        rad = np.maximum(rx, ry)
        bound = geo.min_feature_size
        N = x.size
        
        if not geo.limit_nearest_neighbor_cons: 
            cons = np.zeros(N*(N-1)//2)
            counter = 0

            for i in range(N):
                for j in range(i+1, N):
                    cons[counter] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) - rad[i] - rad[j] - bound
                    counter +=1
            
            if np.min(cons) < 0:
                print('Warning: Constraints violated')
            return cons

        #If limited to nearest neighbors, constraint each pillar (i,j) with its (i+1) and (j+1) counterparts
        #Total of (Nx-1)*(Ny-1) constraints. For a square, Nx,Ny = sqrt(N)
        print("Warning: nearest neighbor constraints not fully tested")
        Nx = geo.grid_shape[0]
        Ny = geo.grid_shape[1]
        x = x.reshape(Nx, Ny)
        y = y.reshape(Nx, Ny)
        rad = rad.reshape(Nx, Ny)
        cons = np.zeros((Nx*Ny-Nx-Ny+1))
        counter = 0
        for i in np.arange(Nx-1):
            for j in np.arange(Ny-1):
                for side in np.arange(2):
                    #side=0 corresponds to pillar to right, side=1 pillar above
                    for side in np.arange(2):
                        if side == 0:
                            i2 = i+1
                            j2 = j
                            if i == Nx-1:
                                continue
                        else:
                            i2 = i
                            j2 = j+1
                            if j == Ny-1:
                                continue

                        cons[counter] = np.sqrt((x[i,j] - x[i2,j2])**2 + (y[i,j] - y[i2,j2])**2) - rad[i,j] - rad[i2,j2] - bound
                        counter += 1
        
        return cons

    return {'type': 'ineq', 'fun': constraint} 