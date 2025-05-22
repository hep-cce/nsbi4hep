import numpy as np
from numpy.polynomial.polynomial import polyval

from ..simulation import mcfm

class Modifier():

  def __init__(self, baseline, events, c6_values = [-10, -5, 0, 5, 10]):
    self.baseline = baseline
    self.events = events
    self.c6_values = np.array(c6_values)

    self.c6_components = [
      mcfm.csv_component_c6[self.baseline][c6_value] for c6_value in c6_values
    ]

    # Matrix of shape (N_events, N_c6_values)
    msq_sm = self.events.components[mcfm.csv_component_sm[self.baseline]].to_numpy()
    msq_c6 = np.array([
      self.events.components[c6_component].to_numpy()
      for c6_component in self.c6_components
    ]).T  # shape: (N_events, N_c6_values)

    # Create the Vandermonde matrix (shape: N_c6_values x N_c6_values)
    V = np.vander(self.c6_values, len(self.c6_values), increasing=True)

    # Normalize msq_c6 by msq_sm (broadcast over axis 1)
    rhs = np.divide(msq_c6, msq_sm[:, np.newaxis], out=np.ones_like(msq_c6), where=msq_sm[:, np.newaxis]!=0)

    # Solve batched linear systems using np.linalg.solve with broadcasting
    # Broadcasting the Vandermonde matrix to all events
    self.coefficients = np.linalg.solve(V[np.newaxis, :, :], rhs[:, :, np.newaxis]).squeeze(-1)
    
   
  # def modify(self, c6):

  #   if np.isscalar(c6):
  #     c6 = np.array([c6])

  #   # Evaluate the polynomial at c6 for each row
  #   wt_c6 = self.events.weights.to_numpy()[:,np.newaxis] * np.apply_along_axis(lambda x: np.polyval(x, c6), 1, self.coefficients[:, ::-1]) 

  #   # Normalize over events for each c6 (axis=0: sum over events)
  #   prob_c6 = wt_c6 / np.sum(wt_c6, axis=0, keepdims=True)

  #   return wt_c6, prob_c6

  def modify(self, c6):
      if np.isscalar(c6):
          c6 = np.array([c6])
      else:
          c6 = np.asarray(c6)

      # Vandermonde matrix: shape (N_c6, degree+1)
      V = np.vander(c6, N=self.coefficients.shape[1], increasing=True)

      # Evaluate all polynomials: (N_events, degree+1) dot (degree+1, N_c6) => (N_events, N_c6)
      rwt_c6 = self.coefficients @ V.T

      # Evaluate poly for each event (rows) at all c6 (columns)
      wt_c6 = self.events.weights.to_numpy()[:, np.newaxis] * rwt_c6

      # Normalize over events for each c6 (axis=0: sum over events)
      prob_c6 = wt_c6 / np.sum(wt_c6, axis=0, keepdims=True)

      return wt_c6, prob_c6