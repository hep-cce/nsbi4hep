from physics.simulation import msq
from physics.hstar import gghzz, c6
from physics.hzz import kinematics, zpair

import os
import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


def get_component(config):
    component_flag = np.array(config['flags'])[np.where([ (flag in ['sig', 'int', 'sbi', 'bkg']) for flag in config['flags'] ])]
    component = component_flag[0] if component_flag.shape[0] != 0 else 'sbi'
    
    comp_dict = {'sig': msq.Component.SIG,
                 'int': msq.Component.INT,
                 'bkg': msq.Component.BKG,
                 'sbi': msq.Component.SBI}

    return comp_dict[component]

def build_dataset(x_arr, target, weights=None):
    inputs = tf.cast(x_arr, tf.float32)
    targets = tf.cast(tf.convert_to_tensor(target)[:, tf.newaxis], tf.float32)

    if weights is None:
        data = tf.concat([inputs, targets], axis=1)

        return data
    else:
        weights = tf.cast(tf.convert_to_tensor(weights)[:, tf.newaxis], tf.float32)

        data = tf.concat([inputs, targets, weights], axis=1)

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

def build(config, seed):
    component = get_component(config)

    sample = load_sample(config, component)

    int_null_filter = msq.MSQFilter('msq_int_sm', value=0.0)
    int_nan_filter = msq.MSQFilter('msq_int_sm', value=np.nan)

    z_candidate = zpair.ZPairCandidate(algorithm='leastsquare')
    z_masses = zpair.ZMasses(bounds1 = (70,115), bounds2 = (70,115))

    angles = kinematics.AngularVariables()
    four_lepton = kinematics.FourLeptonSystem()

    events_training, events_validation = sample.events.filter(int_null_filter).filter(int_nan_filter).calculate(z_candidate).filter(z_masses).calculate(angles).calculate(four_lepton)[:int(config['num_events'])].shuffle(random_state=seed).split(training=0.5, validation=0.5)

    kin_vars = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']

    kinematics_training = events_training.kinematics[kin_vars].to_numpy()
    kinematics_validation = events_validation.kinematics[kin_vars].to_numpy()

    print(f'Building dataset for {["SIG", "INT", "BKG", "SBI"][component.value-1]}.')
    print(f'Sampling {int(sample.events.kinematics.shape[0])} events from {["SIG", "INT", "BKG", "SBI"][component.value-1]}.')
    print(f'Dataset size after filtering and splitting: training={events_training.kinematics.shape[0]}, validation={events_validation.kinematics.shape[0]}')

    c6_mod_training = c6.Modifier(baseline = component, events=events_training, c6_values = [-5,-1,0,1,5]) if component != msq.Component.INT else c6.Modifier(baseline = component, sample=sample, c6_values = [-5,0,5])
    coeff_training = c6_mod_training.coefficients[:, config['coeff']]

    c6_mod_validation = c6.Modifier(baseline = component, events=events_validation, c6_values = [-5,-1,0,1,5]) if component != msq.Component.INT else c6.Modifier(baseline = component, sample=sample, c6_values = [-5,0,5])
    coeff_validation = c6_mod_validation.coefficients[:, config['coeff']]

    train_data = build_dataset(x_arr = kinematics_training,
                               target = coeff_training)
    
    val_data = build_dataset(x_arr = kinematics_validation,
                             target = coeff_validation)
    
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
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:,:-1], val_data[:,-1][:,tf.newaxis]))

    train_dataset = train_dataset.batch(config['batch_size'])
    val_dataset = val_dataset.batch(config['batch_size'])

    return (train_dataset, val_dataset)