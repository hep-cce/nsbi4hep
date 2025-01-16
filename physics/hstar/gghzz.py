import pandas as pd

from sklearn.model_selection import train_test_split

from ..simulation import mcfm

class Events():
  def __init__(self):
    self.kinematics = None
    self.components = None
    self.weights = None
    self.probabilities = None

  def filter(self, obj_instance):
    indices, output = obj_instance.filter(self.kinematics, self.components, self.weights, self.probabilities)

    self.kinematics = self.kinematics.take(indices)
    self.components = self.components.take(indices)
    self.weights = self.weights.take(indices)
    self.probabilities = self.weights/self.weights.sum()

    return output

  def shuffle(self, random_state=None):
    events = Events()

    events.kinematics = self.kinematics.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.components = self.components.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.weights = self.weights.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.probabilities = events.weights/events.weights.sum()

    return events
  
  def split(self, train=0.5, validation=0.5, test=None):
    if test is not None:
      if train + validation + test <= 1.0:
        split_1_kin, kinematics_test, split_1_comp, components_test, split_1_wt, weights_test = train_test_split(self.kinematics, self.components, self.weights, test_size=test, train_size=train+validation, shuffle=False)
        kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(split_1_kin, split_1_comp, split_1_wt, test_size=validation, train_size=train, shuffle=False)

        events_test = Events()
        events_test.kinematics = kinematics_test
        events_test.components = components_test
        events_test.weights = weights_test
        events_train.probabilities = weights_test/weights_test.sum()
      else:
        raise ValueError('Train, validation and test fractions must add up to 1.0')
    else:
      if train + validation <= 1.0:
        kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(self.kinematics, self.components, self.weights, test_size=validation, train_size=train, shuffle=False)
      else:
        raise ValueError('Train and validation fractions have must add up to 1.0')

    events_train, events_val = Events(), Events()
    events_train.kinematics, events_val.kinematics = kinematics_train, kinematics_val
    events_train.components, events_val.components = components_train, components_val
    events_train.weights, events_val.weights = weights_train, weights_val
    events_train.probabilities, events_val.probabilities = weights_train/weights_train.sum(), weights_val/weights_val.sum()

    if test is not None:
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