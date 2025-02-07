#!/usr/bin/env python3

import torch

import os
import pickle

from physics.simulation import sample, msq
from physics.hzz import zpair, zz4l

import numpy as np
import matplotlib.pyplot as plt

from models import CARL, ALICE
MODEL_MODULE = {
  'carl' : CARL,
  'alice' : ALICE
}

from datasets import AliceDataModule, AliceDataset
DATA_MODULE = {
  'alice' : AliceDataset
}

SAMPLE_FILEPATH = {
    'sig': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sig/events.csv',
    'bkg': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_bkg/events.csv',
    'int': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_int/events.csv',
    'sbi': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sbi/events.csv'
}

COMPONENT = {
  'sig': msq.Component.SIG,
  'bkg': msq.Component.BKG,
  'int': msq.Component.INT,
  'sbi': msq.Component.SBI
}

def load_model(model_name, checkpoint_file, hparams):
  return MODEL_MODULE[model_name].load_from_checkpoint(checkpoint_file, **hparams)

def get_true_targets(signal_sample, signal_process, background_process):
  background_sample = signal_sample.reweight(numerator = background_process, denominator = signal_process)
  return signal_sample.probabilities/background_sample.probabilities

def get_scaled_inputs(sample, model, scaler_file):
  # X = sample.kinematics[model.input_features].to_numpy()
  feature_names = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']
  X = sample.kinematics[feature_names].to_numpy()
  scaler = pickle.load(scaler_file, 'rb')
  X = scaler.transform(X)
  return X

def main(args):

  h4lp4 = zz4l.FourLeptonSystem()
  h4lcp = zz4l.AngularVariables()
  zcands = zpair.ZPairCandidate(algorithm='leastsquare')
  zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
  sig = events.from_csv(0.2, args., n_rows = 10000).calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)

  from physics.hzz import zpair, zz4l
  h4lp4 = zz4l.FourLeptonSystem()
  h4lcp = zz4l.AngularVariables()
  zcands = zpair.ZPairCandidate(algorithm='leastsquare')
  zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
  sig = sig.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)
  bkg = sig.reweight(msq.Component.SIG, msq.Component.BKG)
  # bkg = sig.reweight(msq.Component.SBI, msq.Component.BKG)
  r_true = sig.probabilities/bkg.probabilities

  observables = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']
  X_test = sig.kinematics[observables].to_numpy()

  scaler = pickle.load(open(os.path.join(JOB_DIR, 'scaler.pkl'), 'rb'))
  X_test = scaler.transform(X_test)

  s_hat = model(torch.tensor(X_test,dtype=torch.float32))
  r_hat = (s_hat / (1 - s_hat)).detach().numpy().reshape(-1)

  # r_bins = np.arange(0.0,21,1)

  r_min = 0.95
  r_max = 1.05
  n_bins = 30


  # r_bins = np.logspace(np.log(r_min),np.log(r_max),n_bins+1,base=np.e)
  # r_centers = (r_bins[:-1] + r_bins[1:]) / 2

  logr_min = -3
  logr_max = 3.0
  n_bins = 25

  logr_bins = np.linspace(logr_min,logr_max,n_bins+1)

  r_true_vals = np.zeros(len(logr_bins)-1)
  logr_true_vals = np.zeros(len(logr_bins)-1)
  logr_hat_vals = np.zeros(len(logr_bins)-1)
  logr_hat_errs = np.zeros(len(logr_bins)-1)

  for i_bin in range(len(logr_bins)-1):
    bin_mask = np.where(np.logical_and(np.log(r_true.to_numpy()) >= logr_bins[i_bin], np.log(r_true.to_numpy())<logr_bins[i_bin+1]))
    r_true_bin = r_true.to_numpy()[bin_mask]
    r_hat_bin = r_hat[bin_mask]
    response_bin = r_hat_bin
    r_true_vals[i_bin] = np.mean(r_true_bin)
    logr_true_vals[i_bin] = np.mean(np.log(r_true_bin))
    logr_hat_vals[i_bin] = np.mean(np.log(r_hat_bin))
    logr_hat_errs[i_bin] = np.std(np.log(r_hat_bin))

  print(logr_bins)
  print(logr_true_vals)
  print(logr_hat_vals)

  plt.cla()
  plt.figure(figsize=(5,5))
  plt.errorbar(x=logr_true_vals, y=logr_hat_vals, yerr=logr_hat_errs, fmt='o')
  plt.xlim(logr_bins[0], logr_bins[-1])
  plt.ylim(logr_bins[0], logr_bins[-1])
  plt.plot([-100,100], [-100,100], '--')
  plt.xlabel('True probability ratio, $\\log(r = p_{gg\\to h^{\\ast}\\to 4\\ell} / p_{gg\\to 4\\ell})$')
  plt.ylabel('Estimated probability ratio, $\\log(\\hat{r} = \\hat{s} / (1-\\hat{s}))$')
  plt.tight_layout()
  plt.savefig('calibration.png')

  plt.cla()
  plt.figure(figsize=(5,5))
  plt.hist2d(r_true, r_hat, bins=[n_bins*2,n_bins*2], range=[[r_min,r_max],[r_min,r_max]], cmap='viridis')
  plt.plot([0, 20], [0, 20], '--')
  plt.xlabel('True probability ratio, $r = p_{gg\\to h^{\\ast}\\to 4\\ell} / p_{gg\\to 4\\ell}$')
  plt.ylabel('Estimated probability ratio, $\\hat{r} = \\hat{s} / (1-\\hat{s})$')
  plt.tight_layout()
  plt.savefig('migration.png')

  plt.cla()
  plt.hist(r_hat/r_true, bins=np.arange(0,5.1,0.1))
  plt.xlabel('Probability ratio response, $\\hat{r} / r$')
  plt.ylabel('Number of events')
  plt.tight_layout()
  plt.savefig('response.png')

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Plot the calibration curve of a likelihood approximator.')
  parser.add_argument('--model', type=str, required=True, help='Model name')
  parser.add_argument('--model-file', type=str, required=True, help='Path to the model file.')
  parser.add_argument('--hparams-file', type=str, required=True, help='Path to the model file.')
  parser.add_argument('--scaler-file', type=str, required=True, help='Path to the checkpoint file.')
  parser.add_argument('--sample-file', type=str, help='Sample file')
  parser.add_argument('--signal-process', type=str, default='sbi', help='Name of the signal process.')
  parser.add_argument('--background-process', type=str, default='bkg', help='Name of the background process.')
  
  args = parser.parse_args()
  main(args)