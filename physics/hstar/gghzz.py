import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from ..simulation import mcfm

class Events():
  def __init__(self):
    self.kinematics = None
    self.components = None
    self.weights = None
    self.probabilities = None

  def calculate(self, calculator):
    events = Events()
    events.kinematics = self.kinematics
    events.components = self.components
    events.weights = self.weights
    events.probabilities = self.probabilities

    new_columns = calculator(events.kinematics)

    for column_name, column_series in new_columns.items():
      events.kinematics[column_name] = column_series

    return events

  def filter(self, filter):
    indices = filter(self.kinematics, self.components, self.weights, self.probabilities)

    events = Events()

    events.kinematics = self.kinematics.take(indices)
    events.components = self.components.take(indices)
    events.weights = self.weights.take(indices)
    events.probabilities = events.weights/events.weights.sum()

    return events

  def shuffle(self, random_state=None):
    events = Events()

    events.kinematics, events.components, events.weights = shuffle(self.kinematics, self.components, self.weights, random_state=random_state)
    events.probabilities = events.weights/events.weights.sum()

    return events
  
  def split(self, training=1, validation=1, testing=None):
    if testing is not None:
      total = training + validation + testing
      training /= total
      validation /= total
      testing /= total
        
      split_1_kin, kinematics_test, split_1_comp, components_test, split_1_wt, weights_test = train_test_split(self.kinematics, self.components, self.weights, test_size=testing, train_size=training+validation, shuffle=False)
      kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(split_1_kin, split_1_comp, split_1_wt, test_size=validation, train_size=training, shuffle=False)

      events_test = Events()
      events_test.kinematics = kinematics_test
      events_test.components = components_test
      events_test.weights = weights_test
      events_train.probabilities = weights_test/weights_test.sum()
    else:
      total = training + validation
      training /= total
      validation /= total

      kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(self.kinematics, self.components, self.weights, test_size=validation, train_size=training, shuffle=False)

    events_train, events_val = Events(), Events()
    events_train.kinematics, events_val.kinematics = kinematics_train, kinematics_val
    events_train.components, events_val.components = components_train, components_val
    events_train.weights, events_val.weights = weights_train, weights_val
    events_train.probabilities, events_val.probabilities = weights_train/weights_train.sum(), weights_val/weights_val.sum()

    if testing is not None:
      return events_train, events_val, events_test
    else:
      return events_train, events_val

  def __getitem__(self, item):
    events = Events()
    
    events.kinematics = self.kinematics[item]
    events.components = self.components[item]
    events.weights = self.weights[item]
    events.probabilities = events.weights/events.weights.sum()

    return events

class Process():

  def __init__(self, baseline, *channels):
    self.baseline = baseline
    self.events = Events()

    kinematics_per_channel = []
    components_per_channel = []
    weights_per_channel = []

    for sample_from_channel in channels:
      xsec = sample_from_channel[0]
      
      if not isinstance(sample_from_channel[1],pd.DataFrame):
        filepath = sample_from_channel[1]
        nrows = None if len(sample_from_channel) < 3 else sample_from_channel[2]
        df = pd.read_csv(filepath, nrows=nrows)
      else:
        df = sample_from_channel[1]
      kinematics_per_channel.append(df[mcfm.kinematics])
      components_per_channel.append(df[mcfm.components])
      weights = df[mcfm.weight]
      # normalize
      weights *= xsec / weights.sum() 
      weights_per_channel.append(weights)

    self.events.kinematics = pd.concat(kinematics_per_channel)
    self.events.components = pd.concat(components_per_channel)
    self.events.weights = pd.concat(weights_per_channel)
    self.events.probabilities = self.events.weights/self.events.weights.sum()

  def __getitem__(self, component):
    events = Events()
    events.kinematics = self.events.kinematics
    events.components = self.events.components
    events.weights = self.events.weights * events.components[mcfm.component_sm[component]] / events.components[mcfm.component_sm[self.baseline]]
    events.probabilities = events.weights / events.weights.sum()
    return events