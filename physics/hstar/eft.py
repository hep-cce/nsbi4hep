import numpy as np
from numpy.polynomial import polynomial as P

from ..simulation import mcfm

class Modifier():

  def __init__(self, baseline, events, c6_values = [-20, -10, 0, 10, 20], ct_values = [-1,0,1], cg_values = [-0.01, 0.0, 0.01]):
    self.baseline = baseline
    self.events = events

    self.c6_values = np.array(c6_values)
    self.ct_values = np.array(ct_values)
    self.cg_values = np.array(cg_values)
    self.c6_degree = len(c6_values) - 1
    self.ct_degree = len(ct_values) - 1
    self.cg_degree = len(cg_values) - 1

    X, Y, Z = np.meshgrid(self.c6_values, self.ct_values, self.cg_values, indexing='ij')  # Shape: (5, 3, 3) 
    V = P.polyvander3d(X, Y, Z, [self.c6_degree, self.ct_degree, self.cg_degree])

    msq_sm  = self.events.components[mcfm.mcfm_component_sm[self.baseline]].to_numpy()
    xyz_bsm = []
    for i, c6_val in enumerate(self.c6_values):
      for j, ct_val in enumerate(self.ct_values):
        for k, cg_val in enumerate(self.cg_values):
          xyz_bsm.append((c6_val, ct_val, cg_val))
    xyz_npts = len(xyz_bsm)
    msq_bsm = self.events.components[[mcfm.mcfm_component_bsm[self.baseline][xyz] for xyz in xyz_bsm]].to_numpy()

    bsm_values = msq_bsm / msq_sm[:, np.newaxis]

    V = V.reshape(xyz_npts, xyz_npts)
    bsm_values = bsm_values.reshape(-1, xyz_npts).T
    coefficients = np.linalg.solve(V, bsm_values)

    self.coefficients = coefficients.T.reshape(-1, self.c6_degree+1, self.ct_degree+1, self.cg_degree+1)

    # filter out non-physical coefficients
    # self.coefficients[:, 3, 2:, :] = 0
    # self.coefficients[:, 3, :, 2:] = 0
    # self.coefficients[:, 4, 1:, :] = 0
    # self.coefficients[:, 4, :, 1:] = 0

  def modify(self, c6 = None, ct = None, cg = None):
    c6_powers = np.stack([np.power(c6,i) for i in range(self.c6_degree+1)], axis=0)   # (5, Nx)
    ct_powers = np.stack([np.power(ct,j) for j in range(self.ct_degree+1)], axis=0)   # (3, Ny)
    cg_powers = np.stack([np.power(cg,k) for k in range(self.cg_degree+1)], axis=0)   # (3, Nz)
    msq_bsm_over_sm = np.einsum('nijk,ix,jy,kz->nxyz', self.coefficients, c6_powers, ct_powers, cg_powers)

    w_bsm = msq_bsm_over_sm * self.events.weights.to_numpy()[:, np.newaxis, np.newaxis, np.newaxis]
    p_bsm = w_bsm / np.sum(w_bsm, axis=0, keepdims=True)

    return w_bsm, p_bsm