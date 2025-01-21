import pandas as pd

from ..simulation import mcfm

class Events():
  def __init__(self):
    self.kinematics = None
    self.weights = None
    self.probabilities = None

class Process():

  def __init__(self, *channels):
    self.events = Events()

    kinematics_per_channel = []
    weights_per_channel = []
    for sample_from_channel in channels:
      xsec = sample_from_channel[0]
      filepath = sample_from_channel[1]
      nrows = None if len(sample_from_channel) < 3 else sample_from_channel[2]
      df = pd.read_csv(filepath, nrows=nrows)
      kinematics_per_channel.append(df[mcfm.kinematics])
      weights = df[mcfm.weight]
      # normalize
      weights *= xsec / weights.sum() 
      weights_per_channel.append(weights)

    self.events.kinematics = pd.concat(kinematics_per_channel)
    self.events.weights = pd.concat(weights_per_channel)
    self.events.probabilities = self.events.weights/self.events.weights.sum()