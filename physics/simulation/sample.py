import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from . import mcfm

def from_csv(cross_section, file_path, n_rows=None):
  df = pd.read_csv(file_path, nrows=n_rows)
  kinematics = df[mcfm.kinematics]
  components = df[mcfm.components]
  weights = df[mcfm.weight]
  weights *= cross_section / weights.sum() 
  return Process(kinematics, components, weights)

class Process():

  def __init__(self, kinematics=None, components=None, weights=None):
    self.kinematics = kinematics
    self.components = components
    self.weights = weights
    self.probabilities = weights / weights.sum()

  def calculate(self, calculator):
    new_kinematics = self.kinematics.copy()

    new_columns = calculator(new_kinematics)
    for column_name, column_series in new_columns.items():
      # IMPORTANT: to_numpy() ignores pandas indexing, since DataFrame and Series might mis-match
      new_kinematics.loc[:, column_name] = column_series.to_numpy()

    return Process(
      kinematics=new_kinematics,
      components=self.components.copy(),
      weights=self.weights.copy()
    )

  def filter(self, filter):
    accepted_indices = filter(self.kinematics, self.components, self.weights, self.probabilities)
    
    return Process(
      self.kinematics.iloc[accepted_indices].copy(),
      self.components.iloc[accepted_indices].copy(),
      self.weights.iloc[accepted_indices].copy()
    )

  def shuffle(self, random_state=None):
    shuffled_kinematics, shuffled_components, shuffled_weights = shuffle(self.kinematics, self.components, self.weights, random_state=random_state)
    return Process(shuffled_kinematics, shuffled_components, shuffled_weights)
  
  def split(self, train_size=1, val_size=1, test_size=None):

    if test_size is not None:
      total = train_size + val_size + test_size
      train_size /= total
      val_size /= total
      test_size /= total
        
      split_1_kin, kinematics_test, split_1_comp, components_test, split_1_wt, weights_test = train_test_split(self.kinematics, self.components, self.weights, test_size=test_size, train_size=train_size+val_size, shuffle=False)
      kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(split_1_kin, split_1_comp, split_1_wt, test_size=val_size, train_size=train_size, shuffle=False)

      return Process(
        kinematics_train, components_train, weights_train
      ), Process(
        kinematics_val, components_val, weights_val
      ), Process(
        kinematics_test, components_test, weights_test
      )

    else:
      total = train_size + val_size
      train_size /= total
      val_size /= total

      kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(self.kinematics, self.components, self.weights, test_size=val_size, train_size=train_size, shuffle=False)
      return Process(
        kinematics_train, components_train, weights_train
      ), Process(
        kinematics_val, components_val, weights_val
      )

  def unweight(self, n, random_state=None):
    unweighted_events_indices = self.weights.sample(n=n, replace=True, weights=self.weights, random_state=random_state).index

    return Process(
      self.kinematics.loc[unweighted_events_indices],
      self.components.loc[unweighted_events_indices],
      pd.Series(np.ones_like(unweighted_events_indices))
    )