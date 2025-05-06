import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .msq import Component

mcfm_kinematics = [
  'p1_px','p1_py','p1_pz','p1_E',
  'p2_px','p2_py','p2_pz','p2_E',
  'p3_px','p3_py','p3_pz','p3_E',
  'p4_px','p4_py','p4_pz','p4_E',
  'p5_px','p5_py','p5_pz','p5_E',
  'p6_px','p6_py','p6_pz','p6_E',
]

mcfm_weight = 'wt'

mcfm_components = [
  'msq_sbi_sm', 'msq_sig_sm', 'msq_bkg_sm', 'msq_int_sm',
  'msq_sig_bsm_1', 'msq_int_bsm_1', 'msq_sbi_bsm_1',
  'msq_sig_bsm_2', 'msq_int_bsm_2', 'msq_sbi_bsm_2',
  'msq_sig_bsm_3', 'msq_int_bsm_3', 'msq_sbi_bsm_3',
  'msq_sig_bsm_4', 'msq_int_bsm_4', 'msq_sbi_bsm_4',
  'msq_sig_bsm_5', 'msq_int_bsm_5', 'msq_sbi_bsm_5',
  'msq_sig_bsm_6', 'msq_int_bsm_6', 'msq_sbi_bsm_6',
  'msq_sig_bsm_7', 'msq_int_bsm_7', 'msq_sbi_bsm_7',
  'msq_sig_bsm_8', 'msq_int_bsm_8', 'msq_sbi_bsm_8',
  'msq_sig_bsm_9', 'msq_int_bsm_9', 'msq_sbi_bsm_9',
  'msq_sig_bsm_10', 'msq_int_bsm_10', 'msq_sbi_bsm_10',
  'msq_sig_bsm_11', 'msq_int_bsm_11', 'msq_sbi_bsm_11',
  'msq_sig_bsm_12', 'msq_int_bsm_12', 'msq_sbi_bsm_12',
  'msq_sig_bsm_13', 'msq_int_bsm_13', 'msq_sbi_bsm_13',
  'msq_sig_bsm_14', 'msq_int_bsm_14', 'msq_sbi_bsm_14',
  'msq_sig_bsm_15', 'msq_int_bsm_15', 'msq_sbi_bsm_15',
  'msq_sig_bsm_16', 'msq_int_bsm_16', 'msq_sbi_bsm_16',
  'msq_sig_bsm_17', 'msq_int_bsm_17', 'msq_sbi_bsm_17',
  'msq_sig_bsm_18', 'msq_int_bsm_18', 'msq_sbi_bsm_18',
  'msq_sig_bsm_19', 'msq_int_bsm_19', 'msq_sbi_bsm_19',
  'msq_sig_bsm_20', 'msq_int_bsm_20', 'msq_sbi_bsm_20',
  'msq_sig_bsm_21', 'msq_int_bsm_21', 'msq_sbi_bsm_21'
]

mcfm_component_sm = {
  Component.SBI: 'msq_sbi_sm',
  Component.SIG: 'msq_sig_sm',
  Component.BKG: 'msq_bkg_sm',
  Component.INT: 'msq_int_sm'
}

mcfm_component_c6 = {
  Component.SBI: {
    -10.0: 'msq_sbi_bsm_1',
    -9.0: 'msq_sbi_bsm_2',
    -8.0: 'msq_sbi_bsm_3',
    -7.0: 'msq_sbi_bsm_4',
    -6.0: 'msq_sbi_bsm_5',
    -5.0: 'msq_sbi_bsm_6',
    -4.0: 'msq_sbi_bsm_7',
    -3.0: 'msq_sbi_bsm_8',
    -2.0: 'msq_sbi_bsm_9',
    -1.0: 'msq_sbi_bsm_10',
    0.0: 'msq_sbi_bsm_11',
    1.0: 'msq_sbi_bsm_12',
    2.0: 'msq_sbi_bsm_13',
    3.0: 'msq_sbi_bsm_14',
    4.0: 'msq_sbi_bsm_15',
    5.0: 'msq_sbi_bsm_16',
    6.0: 'msq_sbi_bsm_17',
    7.0: 'msq_sbi_bsm_18',
    8.0: 'msq_sbi_bsm_19',
    9.0: 'msq_sbi_bsm_20',
    10.0: 'msq_sbi_bsm_21',
    },
  Component.INT: {
    -10.0: 'msq_int_bsm_1',
    -9.0: 'msq_int_bsm_2',
    -8.0: 'msq_int_bsm_3',
    -7.0: 'msq_int_bsm_4',
    -6.0: 'msq_int_bsm_5',
    -5.0: 'msq_int_bsm_6',
    -4.0: 'msq_int_bsm_7',
    -3.0: 'msq_int_bsm_8',
    -2.0: 'msq_int_bsm_9',
    -1.0: 'msq_int_bsm_10',
    0.0: 'msq_int_bsm_11',
    1.0: 'msq_int_bsm_12',
    2.0: 'msq_int_bsm_13',
    3.0: 'msq_int_bsm_14',
    4.0: 'msq_int_bsm_15',
    5.0: 'msq_int_bsm_16',
    6.0: 'msq_int_bsm_17',
    7.0: 'msq_int_bsm_18',
    8.0: 'msq_int_bsm_19',
    9.0: 'msq_int_bsm_20',
    10.0: 'msq_int_bsm_21',
    },
  Component.SIG: {
    -10.0: 'msq_sig_bsm_1',
    -9.0: 'msq_sig_bsm_2',
    -8.0: 'msq_sig_bsm_3',
    -7.0: 'msq_sig_bsm_4',
    -6.0: 'msq_sig_bsm_5',
    -5.0: 'msq_sig_bsm_6',
    -4.0: 'msq_sig_bsm_7',
    -3.0: 'msq_sig_bsm_8',
    -2.0: 'msq_sig_bsm_9',
    -1.0: 'msq_sig_bsm_10',
    0.0: 'msq_sig_bsm_11',
    1.0: 'msq_sig_bsm_12',
    2.0: 'msq_sig_bsm_13',
    3.0: 'msq_sig_bsm_14',
    4.0: 'msq_sig_bsm_15',
    5.0: 'msq_sig_bsm_16',
    6.0: 'msq_sig_bsm_17',
    7.0: 'msq_sig_bsm_18',
    8.0: 'msq_sig_bsm_19',
    9.0: 'msq_sig_bsm_20',
    10.0: 'msq_sig_bsm_21',
    },
  Component.BKG: {
    -10.0: 'msq_bkg_sm',
    -9.0: 'msq_bkg_sm',
    -8.0: 'msq_bkg_sm',
    -7.0: 'msq_bkg_sm',
    -6.0: 'msq_bkg_sm',
    -5.0: 'msq_bkg_sm',
    -4.0: 'msq_bkg_sm',
    -3.0: 'msq_bkg_sm',
    -2.0: 'msq_bkg_sm',
    -1.0: 'msq_bkg_sm',
    0.0: 'msq_bkg_sm',
    1.0: 'msq_bkg_sm',
    2.0: 'msq_bkg_sm',
    3.0: 'msq_bkg_sm',
    4.0: 'msq_bkg_sm',
    5.0: 'msq_bkg_sm',
    6.0: 'msq_bkg_sm',
    7.0: 'msq_bkg_sm',
    8.0: 'msq_bkg_sm',
    9.0: 'msq_bkg_sm',
    10.0: 'msq_bkg_sm',
    },
}

def from_csv(cross_section=1.0, *, file_path, n_rows=None):
  df = pd.read_csv(file_path, nrows=n_rows, float_precision='round_trip')
  kinematics = df[mcfm_kinematics]
  components = df[mcfm_components]
  weights = df[mcfm_weight]
  # print(weights.sum())
  weights *= cross_section / weights.sum() 
  return Process(kinematics, components, weights)

class Process():

  def __init__(self, kinematics=None, components=None, weights=None):
    self.kinematics = kinematics
    self.components = components
    # HACK: to avoid negative weights
    # only e.g. 2/2M events have infinitesimally-small negative weights due to numerical precision
    # if weights.sum() > 0.0:
    #   weights[weights < 0] = 0.0
    self.weights = weights
    self.probabilities = weights / weights.sum()

  def calculate(self, calculator):
    new_kinematics = self.kinematics.copy()

    new_columns = calculator(new_kinematics)
    for column_name, column_series in new_columns.items():
      # IMPORTANT: to_numpy() ignores pandas indexing, since DataFrame and Series might mis-match
      new_kinematics.loc[:, column_name] = column_series

    return Process(
      kinematics=new_kinematics.reset_index(drop=True),
      components=self.components.reset_index(drop=True),
      weights=self.weights.reset_index(drop=True)
    )

  def filter(self, filter):
    accepted_indices = filter(self.kinematics, self.components, self.weights, self.probabilities)
    
    return Process(
      self.kinematics.iloc[accepted_indices].reset_index(drop=True),
      self.components.iloc[accepted_indices].reset_index(drop=True),
      self.weights.iloc[accepted_indices].reset_index(drop=True)
    )

  def shuffle(self, random_state=None):
    shuffled_kinematics, shuffled_components, shuffled_weights = shuffle(self.kinematics, self.components, self.weights, random_state=random_state)
    return Process(
      shuffled_kinematics.reset_index(drop=True), 
      shuffled_components.reset_index(drop=True), 
      shuffled_weights.reset_index(drop=True)
    )
  
  def split(self, train_size=1, val_size=1, test_size=None):

    if test_size is not None:
      total_size = train_size + val_size + test_size
      train_size /= total_size
      val_size /= total_size
      test_size /= total_size
        
      kinematics_train, kinematics_val_test, components_train, components_val_test, weights_train, weights_val_test = train_test_split(self.kinematics, self.components, self.weights, train_size=train_size, test_size=test_size+val_size, shuffle=False)
      kinematics_val, kinematics_test, components_val, components_test, weights_val, weights_test = train_test_split(kinematics_val_test, components_val_test, weights_val_test, train_size=val_size, test_size=test_size, shuffle=False)

      # the weights now must be scaled up so the sum of weights remains the cross-section
      weights_train /= train_size
      weights_val /= val_size
      weights_test /= test_size

      return Process(
        kinematics_train.reset_index(drop=True), components_train.reset_index(drop=True), weights_train.reset_index(drop=True)
      ), Process(
        kinematics_val.reset_index(drop=True), components_val.reset_index(drop=True), weights_val.reset_index(drop=True)
      ), Process(
        kinematics_test.reset_index(drop=True), components_test.reset_index(drop=True), weights_test.reset_index(drop=True)
      )

    else:
      total_size = train_size + val_size
      train_size /= total_size
      val_size /= total_size

      kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(self.kinematics, self.components, self.weights, test_size=val_size, train_size=train_size, shuffle=False)
      
      # the weights now must be scaled up so the sum of weights remains the cross-section
      weights_train /= train_size
      weights_val /= val_size
      
      return Process(
        kinematics_train.reset_index(drop=True), components_train.reset_index(drop=True), weights_train.reset_index(drop=True)
      ), Process(
        kinematics_val.reset_index(drop=True), components_val.reset_index(drop=True), weights_val.reset_index(drop=True)
      )

  def sample(self, n, random_state=None):
    sampled_events_indices = self.weights.sample(n=n, replace=False, weights=None, random_state=random_state).index

    # the weights now must be scaled up so the sum of weights remains the cross-section
    sampled_weights = self.weights.loc[sampled_events_indices].reset_index(drop=True)
    sampled_weights *= self.weights.sum() / sampled_weights.sum()

    return Process(
      self.kinematics.loc[sampled_events_indices].reset_index(drop=True),
      self.components.loc[sampled_events_indices].reset_index(drop=True),
      sampled_weights
    )

  def unweight(self, n, random_state=None):
    unweighted_events_indices = self.weights.sample(n=n, replace=True, weights=self.weights, random_state=random_state).index

    return Process(
      self.kinematics.loc[unweighted_events_indices].reset_index(drop=True),
      self.components.loc[unweighted_events_indices].reset_index(drop=True),
      pd.Series(np.ones_like(unweighted_events_indices) * self.weights.sum() / n).reset_index(drop=True)
    )

  def reweight(self, denominator, numerator):
    reweights = self.weights * self.components[mcfm_component_sm[numerator]] / self.components[mcfm_component_sm[denominator]]
    return Process(
      self.kinematics.reset_index(drop=True), 
      self.components.reset_index(drop=True), 
      reweights.reset_index(drop=True)
    )
  
  def __getitem__(self, item):
    return Process(
      self.kinematics.iloc[item],
      self.components.iloc[item],
      self.weights.iloc[item]
    )