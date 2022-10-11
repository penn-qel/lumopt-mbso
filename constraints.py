import numpy as np
import itertools
from nearest_neighbor_iterator import nearest_neighbor_iterator

def pillar_constraints(geo):
    '''Builds constraint objects given geometry'''
    '''Creates upper bound constraint on all pairs of pillars by treating as
    circles with radii equal to major axis radius'''

    num_pillars = geo.offset_x.size
    
    def get_pair_iterator():
        #Creates iterator that generates all possible combinations of pillars. Optionally limited to nearest neighbors.
        if geo.limit_nearest_neighbor_cons:
            nx = geo.grid_shape[0]
            ny = geo.grid_shape[1]
            return nearest_neighbor_iterator(np.arange(num_pillars), nx, ny)
        else:
             return itertools.combinations(np.arange(num_pillars), 2)

    def constraint(params):
        '''Returns array of constraints according to pairs of elements'''
        offset_x, offset_y, rx, ry, phi = geo.get_from_params(params)
        x = offset_x + geo.init_x
        y = offset_y + geo.init_y
        rad = (rx + ry) / 2
        bound = geo.min_feature_size
        cons = []
        for pair in get_pair_iterator():
            i, j = pair[0], pair[1]
            cons.append((np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) - rad[i] - rad[j] - bound))
        
        cons = np.array(cons)
        if np.min(cons) < 0:
            print('Warning: Constraints violated')
        return cons

    def jacobian(params):
        offset_x, offset_y, rx, ry, phi = geo.get_from_params(params)
        x = offset_x + geo.init_x
        y = offset_y + geo.init_y
        N = x.size
        rad = (rx + ry) / 2
        jac = []
        for pair in get_pair_iterator():
            i, j = pair[0], pair[1]
            dx, dy, drx, dry, dphi = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            denom = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

            dx[i] = (x[i] - x[j])/denom
            dx[j] = (x[j] - x[i])/denom

            dy[i] = (y[i] - y[j])/denom
            dy[j] = (y[j] - y[i])/denom

            drx[i] = -0.5
            drx[j] = -0.5
            dry[i] = -0.5
            dry[j] = -0.5

            jac.append(np.concatenate((dx, dy, drx, dry, dphi)))

        return np.stack(jac)

    return {'type': 'ineq', 'fun': constraint, 'jac': jacobian} 