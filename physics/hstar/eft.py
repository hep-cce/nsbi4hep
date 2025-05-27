import torch
import numpy as np
from numpy.polynomial import polynomial as P

from enum import Enum

from ..simulation import mcfm, msq

def msq_eft_over_sm(eft_coefficients, c6=None, ct=None, cg=None):
    # Move eft_coefficients to tensor on device
    eft_coefficients = torch.tensor(eft_coefficients)

    c6_degree = eft_coefficients.shape[1] - 1
    ct_degree = eft_coefficients.shape[2] - 1
    cg_degree = eft_coefficients.shape[3] - 1

    # If coefficients are None, use 0 scalar, else convert to tensor on device
    c6_val = torch.tensor(c6 if c6 is not None else [0.0], dtype=eft_coefficients.dtype)
    ct_val = torch.tensor(ct if ct is not None else [0.0], dtype=eft_coefficients.dtype)
    cg_val = torch.tensor(cg if cg is not None else [0.0], dtype=eft_coefficients.dtype)

    # Compute powers, shape (degree+1, len(cX_val)) if cX_val is a list, else scalar
    c6_powers = torch.stack([c6_val.pow(i) for i in range(c6_degree + 1)], dim=0)
    ct_powers = torch.stack([ct_val.pow(j) for j in range(ct_degree + 1)], dim=0)
    cg_powers = torch.stack([cg_val.pow(k) for k in range(cg_degree + 1)], dim=0)

    # Einsum - same subscript notation as numpy but for torch.einsum
    msq_eft_over_sm = torch.einsum('nijk,ix,jy,kz->nxyz', eft_coefficients, c6_powers, ct_powers, cg_powers)

    # Determine slicing: if None, select only first element, else slice all
    c6_slice = slice(None) if c6 is not None else 0
    ct_slice = slice(None) if ct is not None else 0
    cg_slice = slice(None) if cg is not None else 0

    return msq_eft_over_sm[:, c6_slice, ct_slice, cg_slice].numpy()

class Modifier():

  def __init__(self, *, events= None, baseline = msq.Component.SBI, c6_points = [-20,-10,0,10,20], ct_values = [-1,0,1], cg_values = [-1,0,1]):
    self.baseline = baseline
    self.events = events

    self.c6_points = np.array(c6_points)
    self.ct_values = np.array(ct_values)
    self.cg_values = np.array(cg_values)
    self.c6_degree = len(c6_points) - 1
    self.ct_degree = len(ct_values) - 1
    self.cg_degree = len(cg_values) - 1

    X, Y, Z = np.meshgrid(self.c6_points, self.ct_values, self.cg_values, indexing='ij')  # Shape: (5, 3, 3) 
    V = P.polyvander3d(X, Y, Z, [self.c6_degree, self.ct_degree, self.cg_degree])

    msq_sm  = self.events.components[mcfm.csv_component_sm[self.baseline]].to_numpy()
    xyz_bsm = []
    for i, c6_val in enumerate(self.c6_points):
      for j, ct_val in enumerate(self.ct_values):
        for k, cg_val in enumerate(self.cg_values):
          xyz_bsm.append((c6_val, ct_val, cg_val))
    xyz_npts = len(xyz_bsm)
    msq_bsm = self.events.components[[mcfm.csv_component_bsm[self.baseline][xyz] for xyz in xyz_bsm]].to_numpy()

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
    
    # Create a tuple with `count_not_none` times np.newaxis
    newaxes = (np.newaxis,) * sum(x is not None for x in (c6, ct, cg))
    w_eft = msq_eft_over_sm(self.coefficients, c6, ct, cg) * self.events.weights.to_numpy()[(slice(None), )+newaxes]
    p_eft = w_eft / np.sum(w_eft, axis=0, keepdims=True)

    return w_eft, p_eft