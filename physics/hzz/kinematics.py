import numpy as np
import vector
from vector import MomentumObject4D

import pandas as pd


class AngularVariables():
    def __init__(self):
        """
        Calculator class that calculates the kinematics needed for constructing datasets.
        Angles cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙 used in this class are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        """
        self.variable_functions = {'cth_star': self.calc_cth_star, 'cth_1': self.calc_cth_1, 'cth_2': self.calc_cth_2, 'phi_1': self.calc_phi_1, 
                                   'phi': self.calc_phi, 'mZ1': self.calc_mZ1, 'mZ2': self.calc_mZ2}
    
    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})
        
        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](l1, l2, l3, l4)

        return results
    
    def calc_cth_star(self, *leptons: MomentumObject4D):
        Z1 = leptons[0] + leptons[1]
        H = Z1 + leptons[2] + leptons[3]
        return pd.Series(Z1.boost(-H).to_3D().unit().z)

    def calc_cth_1(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z2_in_Z1 = Z2_h.boost(-Z1_h).to_3D()
        l1 = leptons[0].boost(-Z1_h)
        return pd.Series(-z2_in_Z1.dot(l1.to_3D())/np.abs(z2_in_Z1.mag*l1.to_3D().mag))

    def calc_cth_2(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z1_in_Z2 = Z1_h.boost(-Z2_h).to_3D()
        l3 = leptons[2].boost(-Z2_h)
        return pd.Series(-z1_in_Z2.dot(l3.to_3D())/np.abs(z1_in_Z2.mag*l3.to_3D().mag))

    def calc_phi_1(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        z1 = Z1_h.to_3D().unit()

        l1_h = leptons[0].boost(-H).to_3D()
        l2_h = leptons[1].boost(-H).to_3D()

        nz = vector.array({'x': np.zeros(Z1_h.shape[0]), 'y': np.zeros(Z1_h.shape[0]), 'z': np.ones(Z1_h.shape[0])})

        n12 = l1_h.cross(l2_h).unit() # Normal vector of the plane in which the Z1 decay takes place
        nscp = nz.cross(z1).unit() # Normal vector of the plane in which the H -> Z1, Z2 takes place
            
        return pd.Series(z1.dot(n12.cross(nscp))/np.abs(z1.dot(n12.cross(nscp)))*np.arccos(n12.dot(nscp)))
    
    def calc_phi(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        z1 = Z1.boost(-H).to_3D().unit()

        l1_h = leptons[0].boost(-H).to_3D()
        l2_h = leptons[1].boost(-H).to_3D()
        l3_h = leptons[2].boost(-H).to_3D()
        l4_h = leptons[3].boost(-H).to_3D()

        n12 = l1_h.cross(l2_h).unit() # Normal vector of the plane in which the Z1 decay takes place
        n34 = l3_h.cross(l4_h).unit() # Normal vector of the plane in which the Z2 decay takes place

        return pd.Series(z1.dot(n12.cross(n34))/np.abs(z1.dot(n12.cross(n34)))*np.arccos(-n12.dot(n34)))

    def calc_mZ1(self, *leptons: MomentumObject4D):
        return pd.Series((leptons[0]+leptons[1]).mass)

    def calc_mZ2(self, *leptons: MomentumObject4D):
        return pd.Series((leptons[2]+leptons[3]).mass)
    

class FourLeptonSystem():
    def __init__(self):
        """
        Calculator class that calculates the kinematics needed for constructing datasets.
        Angles cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙 used in this class are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        """
        self.variable_functions = {'4l_mass': self.calc_m4l, '4l_rapidity': self.calc_y4l, '4l_pT': self.calc_pT}

    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})
        
        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](l1, l2, l3, l4)

        return results

    def calc_m4l(self, *leptons: MomentumObject4D):
        return pd.Series((leptons[0]+leptons[1]+leptons[2]+leptons[3]).mass)

    def calc_y4l(self, *leptons: MomentumObject4D):
        return pd.Series((leptons[0]+leptons[1]+leptons[2]+leptons[3]).rapidity)
    
    def calc_pT(self, *leptons: MomentumObject4D):
        return pd.Series((leptons[0]+leptons[1]+leptons[2]+leptons[3]).pT)


class M4l():
    def __init__(self, m4l_min=None, m4l_max=None):
        self.m4l_min = m4l_min
        self.m4l_max = m4l_max

    def __call__(self, kinematics, components, weights, probabilities):
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})#negative l1
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})#positive l1
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})#negative l2
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})#positive l2

        m4l = (l1+l2+l3+l4).mass

        if self.m4l_min is not None:
            cond1 = np.where(m4l>=self.m4l_min)
        else:
            cond1 = np.arange(m4l.shape[0])

        if self.m4l_max is not None:
            cond2 = np.where(m4l<=self.m4l_max)
        else:
            cond2 = np.arange(m4l.shape[0])

        indices = np.intersect1d(cond1, cond2)

        return indices, None
    
class LeptonMomentum():
    def __init__(self, lepton_index, momenta_min: tuple=(-np.inf,-np.inf,-np.inf,-np.inf), momenta_max: tuple=(np.inf,np.inf,np.inf,np.inf)):
        self.lepton_index = lepton_index
        self.momenta_min = momenta_min
        self.momenta_max = momenta_max

    def __call__(self, kinematics, components, weights, probabilities):
        lepton_name = ['l1','l2','l3','l4'][self.lepton_index]
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