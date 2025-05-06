import numpy as np
import vector

import pandas as pd

class TwoLepTwoNuSystem():
    def __init__(self):
        """
        """
        self.variable_functions = {'2l2v_mt': self.calc_mt, '2l2lv_met': self.calc_met}

    def __call__(self, kinematics):
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
        results['2l2v_mt'] = self.calc_mt(l1, l2, v1, v2)
        results['2l2lv_met'] = self.calc_met(v1, v2)
        results['2l2lv_met_phi'] = self.calc_met_phi(v1, v2)

        return results

    def calc_mt(self, l1, l2, v1, v2):
        """Transverse mass of the (2l + MET) system."""
        return (l1+l2+v1+v2).mt

    def calc_met(self, v1, v2):
        """Missing transverse energy magnitude."""
        return (v1+v2).et

    def calc_met_phi(self, v1, v2):
        """Missing transverse energy magnitude."""
        return (v1+v2).phi

def analyze(events):
    # angular_vars = AngularVariables()
    lepton_momenta = TwoLepTwoNuSystem()
    return events.calculate(lepton_momenta)