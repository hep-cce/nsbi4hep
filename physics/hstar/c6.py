import numpy as np

from ..simulation import msq
from ..simulation import mcfm

class Modifier():

  def __init__(self, baseline, sample, c6_values = [-5,-1,0,1,5]):
    self.baseline = baseline
    self.sample = sample

    self.c6_values = np.array(c6_values)
    self.c6_components = [mcfm.component_c6[self.baseline][c6_value] for c6_value in c6_values]

    # solve the polynomial coefficients
    msq_sm = self.sample.components[mcfm.component_sm[self.baseline]].to_numpy()
    msq_c6 = np.array([self.sample.components[c6_components].to_numpy() for c6_components in self.c6_components]).T
    self.coefficients = np.apply_along_axis(lambda x: np.linalg.solve(np.vander(self.c6_values, len(self.c6_values), increasing=True), x), 1, msq_c6 / msq_sm[:, np.newaxis])

  def modify(self, c6):

    if np.isscalar(c6):
      c6 = np.array([c6])
    

    # Evaluate the polynomial at c6 for each row
    wt_c6 = self.sample.weights.to_numpy()[:,np.newaxis] * np.apply_along_axis(lambda x: np.polyval(x, c6), 1, self.coefficients[:, ::-1])
    prob_c6 = wt_c6 / np.sum(wt_c6, axis=0)

    return (wt_c6, prob_c6)

      # cH_modification = -1.0 * events.weights[:,np.newaxis] * np.array(cH)

      # c6_modification = c6_modification[:, np.newaxis, :]

      # cH_modification = cH_modification[:, :, np.newaxis]

      # tot_modification = c6_modification + cH_modification 