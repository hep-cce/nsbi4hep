from physics.simulation import msq
from physics.hstar import gghzz, c6
from physics.hzz import kinematics, zpair

import os
import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


def get_components(config):
    component_flag = np.array(config['flags'])[np.where([ (flag in ['sig', 'int', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi', 'sbi-vs-sig', 'int-vs-sig', 'bkg-vs-sig', 'sig-vs-bkg']) for flag in config['flags'] ])]
    component_flag = component_flag[0] if component_flag.shape[0] != 0 else 'sbi'
    component_1, component_2 = component_flag.split('-')[0], component_flag.split('-')[-1]
    
    comp_dict = {'sig': msq.Component.SIG,
                 'int': msq.Component.INT,
                 'bkg': msq.Component.BKG,
                 'sbi': msq.Component.SBI}

    return (comp_dict[component_1], comp_dict[component_2])

def build_dataset(x_arr, signal_probabilities, background_probabilities, weights=None):
    data = []

    signal_probabilities = tf.cast(signal_probabilities, tf.float32)
    background_probabilities = tf.cast(background_probabilities, tf.float32)

    ratios = signal_probabilities[:,tf.newaxis]/background_probabilities[:,tf.newaxis]

    inputs = x_arr
    targets = ratios/(1+ratios)

    print(inputs.shape)
    print(targets.shape)

    if weights is None:
        data = tf.concat([inputs, targets], axis=1)

        return data
    else:
        sample_weights = tf.cast(weights, tf.float32)[:,tf.newaxis]

        data = tf.concat([inputs, targets, sample_weights], axis=1)

        return data

def load_sample(config, component):
    if config['num_events'] is None:
        n_i = None
    else:
        n_i = int(config['num_events']*1.2)

    def match_comp(config, component, n_i):
        match component:
            case msq.Component.SIG:
                sample = gghzz.Process(msq.Component.SIG, (0.1, os.path.join(config['sample_dir'], 'ggZZ2e2m_sig.csv'), n_i))
            case msq.Component.SBI:
                sample = gghzz.Process(msq.Component.SBI, (1.5, os.path.join(config['sample_dir'], 'ggZZ2e2m_sbi.csv'), n_i))
            case msq.Component.BKG:
                sample = gghzz.Process(msq.Component.BKG, (1.6, os.path.join(config['sample_dir'], 'ggZZ2e2m_bkg.csv'), n_i))
            case msq.Component.INT:
                sample = gghzz.Process(msq.Component.INT, (-0.2, os.path.join(config['sample_dir'], 'ggZZ2e2m_int.csv'), n_i))
        return sample

    return match_comp(config, component, n_i)


def build(config, seed, strategy=None):
    component_1, component_2 = get_components(config)

    # Sample will be loaded as component_2 (denominator in pdf ratio)
    # Later reweighted as component_1 to get the numerator probabilities
    sample = load_sample(config, component_2)

    bkg_null_filter = msq.MSQFilter('msq_bkg_sm', value=0.0)
    bkg_nan_filter = msq.MSQFilter('msq_bkg_sm', value=np.nan)

    z_candidate = zpair.ZPairCandidate(algorithm='leastsquare')
    z_masses = zpair.ZMasses(bounds1 = (70,115), bounds2 = (70,115))

    angles = kinematics.AngularVariables()
    four_lepton = kinematics.FourLeptonSystem()

    events_training, events_validation = sample.events.filter(bkg_null_filter).filter(bkg_nan_filter).calculate(z_candidate).filter(z_masses).calculate(angles).calculate(four_lepton)[:int(config['num_events'])].shuffle(random_state=seed).split(training=0.5, validation=0.5)
    sig_prob_training, sig_prob_validation = [evt.probabilities for evt in sample[component_1].filter(bkg_null_filter).filter(bkg_nan_filter).calculate(z_candidate).filter(z_masses).calculate(angles).calculate(four_lepton)[:int(config['num_events'])].shuffle(random_state=seed).split(training=0.5, validation=0.5)]

    print(f'Initial base size of {["SIG", "INT", "BKG", "SBI"][component_2.value-1]}(SM) set to {int(sample.events.kinematics.shape[0])}.')
    print(f'Dataset size after splitting: training={events_training.kinematics.shape[0]}, validation={events_validation.kinematics.shape[0]}')

    kin_variables = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']

    kinematics_training = events_training.kinematics[kin_variables].to_numpy()
    kinematics_validation = events_validation.kinematics[kin_variables].to_numpy()

    train_data = build_dataset(x_arr = kinematics_training,
                               signal_probabilities = sig_prob_training,
                               background_probabilities = events_training.probabilities)
    
    val_data = build_dataset(x_arr = kinematics_validation,
                             signal_probabilities = sig_prob_validation,
                             background_probabilities = events_validation.probabilities)
    
    # The following will scale only kinematics for nonprm and kinematics + c6 for prm
    train_scaler = MinMaxScaler()
    train_data = tf.concat([train_scaler.fit_transform(train_data[:,:-1]), train_data[:,-1][:, tf.newaxis]], axis=1)
    train_data = tf.random.shuffle(train_data, seed=seed)

    val_data = tf.concat([train_scaler.transform(val_data[:,:-1]), val_data[:,-1][:, tf.newaxis]], axis=1)
    val_data = tf.random.shuffle(val_data, seed=seed)

    scaler_config = {'scaler.scale_': train_scaler.scale_.tolist(), 'scaler.min_': train_scaler.min_.tolist()}
    with open('scaler.json', 'w') as scaler_file:
        scaler_file.write(json.dumps(scaler_config, indent=4))

    # Build tf Dataset objects and batch data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:,:-1], train_data[:,-1][:,tf.newaxis]))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:,:-1], train_data[:,-1][:,tf.newaxis]))

    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            train_dataset = train_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
            val_dataset = val_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
    else:
        train_dataset = train_dataset.batch(config['batch_size'])
        val_dataset = val_dataset.batch(config['batch_size'])

    return (train_dataset, val_dataset)