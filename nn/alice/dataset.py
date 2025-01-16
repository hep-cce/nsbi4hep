from physics.simulation import msq
from physics.hstar import gghzz, c6
from physics.hzz import angles, zpair

import os
import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


def get_components(config):
    component_flag = np.array(config['flags'])[np.where([ (flag in ['sig', 'int', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi', 'sbi-vs-sig', 'int-vs-sig', 'bkg-vs-sig']) for flag in config['flags'] ])]
    component_flag = component_flag[0] if component_flag.shape[0] != 0 else 'sbi'
    component_1, component_2 = component_flag.split('-')[0], component_flag.split('-')[-1]
    
    comp_dict = {'sig': msq.Component.SIG,
                 'int': msq.Component.INT,
                 'bkg': msq.Component.BKG,
                 'sbi': msq.Component.SBI}

    return (comp_dict[component_1], comp_dict[component_2])

def build_dataset(x_arr, param_values, signal_probabilities, background_probabilities, weights):
    data = []

    signal_probabilities = tf.convert_to_tensor(signal_probabilities)
    background_probabilities = tf.convert_to_tensor(background_probabilities)

    non_prm=False

    if np.isscalar(param_values):
        param_values = [param_values]
        non_prm=True
    elif len(param_values) == 1:
        non_prm=True

    for i in range(len(param_values)):
        param = param_values[i]
        # x | param | y | weight
        ratios = tf.cast(signal_probabilities[:,i][:,tf.newaxis]/background_probabilities[:,tf.newaxis], tf.float32)

        if not non_prm:
            inputs = tf.concat([x_arr, tf.ones(x_arr.shape[0])[:,tf.newaxis]*param], axis=1)
            targets = ratios/(1+ratios)
            sample_weights = tf.cast(weights, tf.float32)[:,tf.newaxis]

            data.append(tf.concat([inputs, targets, sample_weights], axis=1))
        else:
            inputs = x_arr
            targets = ratios/(1+ratios)
            sample_weights = tf.cast(weights, tf.float32)[:,tf.newaxis]

            data.append(tf.concat([inputs, targets, sample_weights], axis=1))

    data = tf.reshape(tf.convert_to_tensor(data), (tf.convert_to_tensor(data).shape[0]*tf.convert_to_tensor(data).shape[1], tf.convert_to_tensor(data).shape[2]))

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

def load_kinematics(sample, bounds1=(70,115), bounds2=(70,115), algorithm='leastsquare'):
    z_chooser = zpair.ZPairChooser(bounds1=bounds1, bounds2=bounds2, algorithm=algorithm)

    # calculate_2 uses y4l as additional kinematic
    return angles.calculate_2(*sample.events.filter(z_chooser))

def build(config, seed, strategy=None):
    component_1, component_2 = get_components(config)

    sample_2 = load_sample(config, component_2)

    set_size_2 = sample_2.events.kinematics.shape[0]

    sample_2.events.filter(msq.MSQFilter('msq_bkg_sm', value=0.0))
    sample_2.events.filter(msq.MSQFilter('msq_bkg_sm', value=np.nan))

    kin_vars_2 = load_kinematics(sample_2)

    sample_2.events = sample_2.events[:int(config['num_events'])]
    kin_vars_2 = kin_vars_2[:int(config['num_events'])]

    true_size_2 = kin_vars_2.shape[0]

    print(f'Initial base size of {["SIG", "INT", "BKG", "SBI"][component_2.value-1]}(SM) set to {int(set_size_2)}. Train and validation data will be {int(true_size_2/2)} each after Z mass cuts.')
    print(f'Total dataset size after filters (per train, val): {int(true_size_2/2)}')

    if component_1 != msq.Component.BKG:
        c6_mod = c6.Modifier(baseline = component_1, sample=sample_2, c6_values = [-5,-1,0,1,5])
        sig_wt, sig_prob = c6_mod.modify(c6=config['c6_values'])
    else:
        sig_wt, sig_prob = np.array(sample_2.events.weights)[:,np.newaxis], np.array(sample_2.events.probabilities)[:,np.newaxis]
    
    if component_1 == msq.Component.INT: #TODO: Fix this somehow
        c6_mod = c6.Modifier(baseline = component_1, sample=sample_2, c6_values = [-5,0,5])
        sig_wt, sig_prob = c6_mod.modify(c6=config['c6_values'])


    # Renormalize probabilities after splitting the sample into training and validation
    sig_prob_train = sig_wt[:int(true_size_2/2)]/np.sum(sig_wt[:int(true_size_2/2)])
    bkg_prob_train = np.array(sample_2.events.weights)[:int(true_size_2/2)]/np.sum(np.array(sample_2.events.weights)[:int(true_size_2/2)])

    sig_prob_val = sig_wt[int(true_size_2/2):]/np.sum(sig_wt[int(true_size_2/2):])
    bkg_prob_val = np.array(sample_2.events.weights)[int(true_size_2/2):]/np.sum(np.array(sample_2.events.weights)[int(true_size_2/2):])


    train_data = build_dataset(x_arr = kin_vars_2[:int(true_size_2/2)], 
                               param_values = config['c6_values'],
                               signal_probabilities = sig_prob_train,
                               background_probabilities = bkg_prob_train,
                               weights = np.array(sample_2.events.weights)[:int(true_size_2/2)])
    
    val_data = build_dataset(x_arr = kin_vars_2[int(true_size_2/2):],
                             param_values = config['c6_values'],
                             signal_probabilities = sig_prob_val,
                             background_probabilities = bkg_prob_val,
                             weights = np.array(sample_2.events.weights)[int(true_size_2/2):])
    
    # The following will scale only kinematics for nonprm and kinematics + c6 for prm
    train_scaler = MinMaxScaler()
    train_data = tf.concat([train_scaler.fit_transform(train_data[:,:-2]), train_data[:,-2:]], axis=1)
    train_data = tf.random.shuffle(train_data, seed=seed)

    val_data = tf.concat([train_scaler.transform(val_data[:,:-2]), val_data[:,-2:]], axis=1)
    val_data = tf.random.shuffle(val_data, seed=seed)

    scaler_config = {'scaler.scale_': train_scaler.scale_.tolist(), 'scaler.min_': train_scaler.min_.tolist()}
    with open('scaler.json', 'w') as scaler_file:
        scaler_file.write(json.dumps(scaler_config, indent=4))

    # Build tf Dataset objects and batch data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:,:-2], train_data[:,-2][:,tf.newaxis], train_data[:,-1][:,tf.newaxis]))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:,:-2], train_data[:,-2][:,tf.newaxis], val_data[:,-1][:,tf.newaxis]))

    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            train_dataset = train_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
            val_dataset = val_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
    else:
        train_dataset = train_dataset.batch(config['batch_size'])
        val_dataset = val_dataset.batch(config['batch_size'])

    return (train_dataset, val_dataset)