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
        return Z1.boost(-H).to_3D().unit().z

    def calc_cth_1(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z2_in_Z1 = Z2_h.boost(-Z1_h).to_3D()
        l1 = leptons[0].boost(-Z1_h)
        return -z2_in_Z1.dot(l1.to_3D())/np.abs(z2_in_Z1.mag*l1.to_3D().mag)

    def calc_cth_2(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z1_in_Z2 = Z1_h.boost(-Z2_h).to_3D()
        l3 = leptons[2].boost(-Z2_h)
        return -z1_in_Z2.dot(l3.to_3D())/np.abs(z1_in_Z2.mag*l3.to_3D().mag)

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
            
        return z1.dot(n12.cross(nscp))/np.abs(z1.dot(n12.cross(nscp)))*np.arccos(n12.dot(nscp))
    
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

        return z1.dot(n12.cross(n34))/np.abs(z1.dot(n12.cross(n34)))*np.arccos(-n12.dot(n34))

    def calc_mZ1(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]).mass

    def calc_mZ2(self, *leptons: MomentumObject4D):
        return (leptons[2]+leptons[3]).mass
    

class LeptonMomenta():
    def __call__(self, kinematics):
        if 'l1_px' in kinematics:
            l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})
            l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})
            l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})
            l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})
        else:
            l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']})
            l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']})
            l3 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']})
            l4 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']})

        pt = np.array([l1.pt, l2.pt, l3.pt, l4.pt]).T
        indices = np.argsort(pt, axis=1)[:,::-1]

        leptons = np.array([l1,l2,l3,l4]).T
        leptons_sorted = vector.array(np.take_along_axis(leptons, indices, axis=1), dtype=[("px", np.float32), ("py", np.float32), ("pz", np.float32), ("E", np.float32)])

        return {'l1_pt': leptons_sorted[:,0].pt, 'l1_eta': leptons_sorted[:,0].eta, 'l1_phi': leptons_sorted[:,0].phi, 'l1_energy': leptons_sorted[:,0].energy,
                'l2_pt': leptons_sorted[:,1].pt, 'l2_eta': leptons_sorted[:,1].eta, 'l2_phi': leptons_sorted[:,1].phi, 'l2_energy': leptons_sorted[:,1].energy,
                'l3_pt': leptons_sorted[:,2].pt, 'l3_eta': leptons_sorted[:,2].eta, 'l3_phi': leptons_sorted[:,2].phi, 'l3_energy': leptons_sorted[:,2].energy,
                'l4_pt': leptons_sorted[:,3].pt, 'l4_eta': leptons_sorted[:,3].eta, 'l4_phi': leptons_sorted[:,3].phi, 'l4_energy': leptons_sorted[:,3].energy}


class FourLeptonSystem():
    def __init__(self):
        """
        Calculator class that calculates the kinematics needed for constructing datasets.
        Angles cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙 used in this class are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        """
        self.variable_functions = {'4l_mass': self.calc_m4l, '4l_rapidity': self.calc_y4l, '4l_pT': self.calc_pT, '4l_energy': self.calc_E}

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
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).mass

    def calc_y4l(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).rapidity
    
    def calc_pT(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).pt
    
    def calc_E(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).E

class MandelstamVariables():
    def __init__(self):
        """
        Calculator class that calculates the Mandelstam variables t,u and s=m4l
        """
        self.variable_functions = {'mandelstam_s': self.calc_s, 'mandelstam_t': self.calc_t, 'mandelstam_u': self.calc_u}

    def __call__(self, kinematics):
        g1 = vector.array({'px': kinematics['p1_px'], 'py': kinematics['p1_py'], 'pz': kinematics['p1_pz'], 'E': kinematics['p1_E']})
        g2 = vector.array({'px': kinematics['p2_px'], 'py': kinematics['p2_py'], 'pz': kinematics['p2_pz'], 'E': kinematics['p2_E']})
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})
        
        Z1 = l1 + l2
        Z2 = l3 + l4

        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](g1, g2, Z1, Z2)

        return results

    def calc_s(self, *particles: MomentumObject4D):
        return (particles[2]+particles[3]).mass2

    def calc_t(self, *particles: MomentumObject4D):
        return (particles[0]-particles[2]).mass2

    def calc_u(self, *particles: MomentumObject4D):
        return (particles[0]-particles[3]).mass2

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

        return indices
    
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