import vector
import numpy as np

import pandas as pd

class ZPairCandidate:
    def __init__(self, algorithm: str = 'leastsquare'):
        if algorithm not in ['leastsquare', 'closest', 'truth']:
            raise ValueError('algorithm has to be one of ["leastsquare", "closest", "truth"]')

        self.algorithm = algorithm

        self.Z_mass = 91.18
    
    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']})#negative l1
        l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']})#positive l1
        l3 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']})#negative l2
        l4 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']})#positive l2

        if self.algorithm == 'leastsquare':
            return self.find_Z_lsq(l1, l2, l3, l4)
        elif self.algorithm == 'closest':
            return self.find_Z_closest(l1, l2, l3, l4)
        elif self.algorithm == 'truth':
            return {'l1_px': kinematics['p3_px'], 'l1_py': kinematics['p3_py'], 'l1_pz': kinematics['p3_pz'], 'l1_E': kinematics['p3_E'],
                    'l2_px': kinematics['p4_px'], 'l2_py': kinematics['p4_py'], 'l2_pz': kinematics['p4_pz'], 'l2_E': kinematics['p4_E'],
                    'l3_px': kinematics['p5_px'], 'l3_py': kinematics['p5_py'], 'l3_pz': kinematics['p5_pz'], 'l3_E': kinematics['p5_E'],
                    'l4_px': kinematics['p6_px'], 'l4_py': kinematics['p6_py'], 'l4_pz': kinematics['p6_pz'], 'l4_E': kinematics['p6_E'],
                    'Z1_mass': pd.Series((l1+l2).mass), 'Z2_mass': pd.Series((l1+l2).mass)}

    def find_Z_lsq(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',np.float64),('py',np.float64),('pz',np.float64),('E',np.float64)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',np.float64),('py',np.float64),('pz',np.float64),('E',np.float64)])

        # Squared minimization to determine the closest pair
        sq = np.array([(pair[0].mass - self.Z_mass)**2 + (pair[1].mass - self.Z_mass)**2 for pair in pairs]).T
        closest_pair_indices = np.argmin(sq, axis=1)
        closest_pair = pairs.transpose(2,0,1)[np.arange(len(closest_pair_indices)), closest_pair_indices].T

        # Determine the Z boson with the higher pT
        # That one will be Z1, the other one Z2
        pT_max_ind = np.argmax(closest_pair.pt,axis=0) # Z1
        pT_min_ind = np.argmin(closest_pair.pt,axis=0) # Z2

        # Determine the order manually if both Z bosons have the same pT
        cond=(pT_max_ind==pT_min_ind)

        pT_max_ind[cond] = 0
        pT_min_ind[cond] = 1

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_max_ind)), pT_max_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_min_ind)), pT_min_ind].T

        return {'l1_px': pd.Series(l1_1.px), 'l1_py': pd.Series(l1_1.py), 'l1_pz': pd.Series(l1_1.pz), 'l1_E': pd.Series(l1_1.E),
                'l2_px': pd.Series(l2_1.px), 'l2_py': pd.Series(l2_1.py), 'l2_pz': pd.Series(l2_1.pz), 'l2_E': pd.Series(l2_1.E),
                'l3_px': pd.Series(l1_2.px), 'l3_py': pd.Series(l1_2.py), 'l3_pz': pd.Series(l1_2.pz), 'l3_E': pd.Series(l1_2.E),
                'l4_px': pd.Series(l2_2.px), 'l4_py': pd.Series(l2_2.py), 'l4_pz': pd.Series(l2_2.pz), 'l4_E': pd.Series(l2_2.E),
                'Z1_mass': pd.Series((l1_1 + l2_1).mass), 'Z2_mass': pd.Series((l1_2 + l2_2).mass)}
    
    def find_Z_closest(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Just choose the Z boson pair which contains the Z boson closest to the true rest mass
        pairs_diffs = ((pairs.mass - np.ones(pairs.shape)*self.Z_mass)**2).transpose(2,0,1).reshape(pairs.shape[2],4)
        min_ind = np.floor(np.argmin(pairs_diffs, axis=1)/2.0).astype(int)
        closest_Z_pair = pairs.transpose(2,0,1)[np.arange(len(min_ind)),min_ind].T

        closest_Z_min_ind = np.argmin((closest_Z_pair.mass-self.Z_mass)**2, axis=0)
        closest_Z_max_ind = np.argmax((closest_Z_pair.mass-self.Z_mass)**2, axis=0)

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_min_ind)), closest_Z_min_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_max_ind)), closest_Z_max_ind].T

        return {'l1_px': pd.Series(l1_1.px), 'l1_py': pd.Series(l1_1.py), 'l1_pz': pd.Series(l1_1.pz), 'l1_E': pd.Series(l1_1.E),
                'l2_px': pd.Series(l2_1.px), 'l2_py': pd.Series(l2_1.py), 'l2_pz': pd.Series(l2_1.pz), 'l2_E': pd.Series(l2_1.E),
                'l3_px': pd.Series(l1_2.px), 'l3_py': pd.Series(l1_2.py), 'l3_pz': pd.Series(l1_2.pz), 'l3_E': pd.Series(l1_2.E),
                'l4_px': pd.Series(l2_2.px), 'l4_py': pd.Series(l2_2.py), 'l4_pz': pd.Series(l2_2.pz), 'l4_E': pd.Series(l2_2.E),
                'Z1_mass': pd.Series((l1_1 + l2_1).mass), 'Z2_mass': pd.Series((l1_2 + l2_2).mass)}
    

class ZPairMassWindow():
    def __init__(self, z1: tuple[int, int] = None, z2: tuple[int, int] = None):
        self.z1 = z1
        self.z2 = z2

    def __call__(self, kinematics, components, weights, probabilities) -> np.array:
        #Outgoing leptons
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})#negative l1
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})#positive l1
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})#negative l2
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})#positive l2

        Z1 = l1 + l2
        Z2 = l3 + l4

        if self.z1 is not None:
            cond1 = np.where((Z1.mass>=self.z1[0])&(Z1.mass<=self.z1[1]))
        else:
            cond1 = np.arange(Z1.mass.shape[0])

        if self.z2 is not None:
            cond2 = np.where((Z2.mass>=self.z2[0])&(Z2.mass<=self.z2[1]))
        else:
            cond2 = np.arange(Z2.mass.shape[0])

        # Get only indices where cond1 and cond2 apply
        indices = np.intersect1d(cond1,cond2)

        return indices