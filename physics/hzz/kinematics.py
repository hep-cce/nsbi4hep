import numpy as np

class MomentumFilter():
    def __init__(self, lepton_index, momenta_min: tuple=(-np.inf,-np.inf,-np.inf,-np.inf), momenta_max: tuple=(np.inf,np.inf,np.inf,np.inf)):
        self.lepton_index = lepton_index
        self.momenta_min = momenta_min
        self.momenta_max = momenta_max

    def filter(self, kinematics, components, weights, probabilities):
        lepton_name = ['p3','p4','p5','p6'][self.lepton_index]
        px = kinematics[f'{lepton_name}_px']
        py = kinematics[f'{lepton_name}_py']
        pz = kinematics[f'{lepton_name}_pz']
        pE = kinematics[f'{lepton_name}_E']

        cond1 = np.where((px >= self.momenta_min[0]) & (px <= self.momenta_max[0]))
        cond2 = np.where((py >= self.momenta_min[1]) & (py <= self.momenta_max[1]))
        cond3 = np.where((pz >= self.momenta_min[2]) & (pz <= self.momenta_max[2]))
        cond4 = np.where((pE >= self.momenta_min[3]) & (pE <= self.momenta_max[3]))

        print(len(cond1), len(cond2), len(cond3), len(cond4))

        indices = np.intersect1d(np.intersect1d(np.intersect1d(cond1, cond2), cond3), cond4)

        return indices, indices