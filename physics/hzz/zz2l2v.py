import numpy as np
import vector

import pandas as pd

class ZZ2L2V():
    def __init__(self):
        """
        """

    def __call__(self, kinematics):
        from ..constants import mZ

        l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']})
        l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']})
        v1 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']})
        v2 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']})

        pt = np.array([l1.pt, l2.pt]).T
        indices = np.argsort(pt, axis=1)[:,::-1]
        leptons = np.array([l1,l2]).T
        leptons_sorted = vector.array(np.take_along_axis(leptons, indices, axis=1), dtype=[("px", np.float32), ("py", np.float32), ("pz", np.float32), ("E", np.float32)])
        
        results = {'l1_pt': leptons_sorted[:,0].pt, 'l1_eta': leptons_sorted[:,0].eta, 'l1_phi': leptons_sorted[:,0].phi, 'l1_energy': leptons_sorted[:,0].energy,
                   'l2_pt': leptons_sorted[:,1].pt, 'l2_eta': leptons_sorted[:,1].eta, 'l2_phi': leptons_sorted[:,1].phi, 'l2_energy': leptons_sorted[:,1].energy}

        ll = l1+l2
        met = (v1+v2).to_2D()

        results['ll_pt']   = ll.pt
        results['ll_eta']  = ll.eta
        results['ll_phi']  = ll.phi
        results['ll_mass'] = ll.mass

        results['met']     = met.pt
        results['met_phi'] = met.phi

        results['zz_mt'] = np.sqrt( (np.sqrt(mZ**2 + ll.pt2) + np.sqrt(mZ**2 + met.pt2))**2 - (ll.to_2D() + met).pt2 )

        results['ll_dr'] = l1.deltaR(l2)
        results['llmet_dphi'] = ll.to_2D().deltaphi(met)

        return results

class ZMassWindow():
    def __init__(self, min = 75, max = 105):
        self.min = min
        self.max = max

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        Z_mass = kinematics['ll_mass']
        indices, = np.where((Z_mass>=self.min)&(Z_mass<=max))
        return indices

class MinMETCut():
    def __init__(self, met_min = 100):
        self.met_min = met_min

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        met = kinematics['met']
        indices, = np.where( met > self.met_min )
        return indices

class MinDPhillMETCut():
    def __init__(self, dphi_min = 2.5):
        self.dphi_min = dphi_min

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        dphi = kinematics['llmet_dphi']
        indices, = np.where( np.abs(dphi) > self.dphi_min )
        return indices

class MaxDRllCut():
    def __init__(self, dr_max = 1.8):
        self.dr_max = dr_max

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        dr = kinematics['ll_dr']
        indices, = np.where(dr < self.dr_max)
        return indices

def analyze(events):

    events_analyzed = events.calculate(ZZ2L2V())
    print('Inclusive | ', events_analyzed.weights.sum())

    met_max = 100
    events_analyzed = events_analyzed.filter(MinMETCut(met_max))
    print(f'MET > {met_max} GeV | ', events_analyzed.weights.sum())

    dphillmet_min = 2.5
    events_analyzed = events_analyzed.filter(MinDPhillMETCut(dphillmet_min))
    print(f'DPhillMET > {dphillmet_min} | ', events_analyzed.weights.sum())

    drll_max = 1.8
    events_analyzed = events_analyzed.filter(MaxDRllCut(drll_max))
    print(f'DRll < {drll_max} | ', events_analyzed.weights.sum())

    return events_analyzed