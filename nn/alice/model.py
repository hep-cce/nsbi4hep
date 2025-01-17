import tensorflow as tf
from tensorflow import keras

import os
import numpy as np

@keras.utils.register_keras_serializable()
def ALICE_loss(y_true, y_pred):
    return - (1-y_true) * tf.math.log(1 - y_pred) - y_true * tf.math.log(y_pred)

@keras.utils.register_keras_serializable()
def swish_activation(x, b=1):
    return x*keras.backend.sigmoid(b*x)

keras.utils.get_custom_objects().update({'swish_activation': keras.layers.Activation(swish_activation)})

@keras.utils.register_keras_serializable()
class ALICE_reg(keras.Model):
    def __init__(self, num_layers=10, num_nodes=2000, input_dim=9, **kwargs):
        super(ALICE_reg, self).__init__(**kwargs)
        
        swish = keras.layers.Activation(swish_activation, name='Swish')

        if np.isscalar(num_nodes) or len(num_nodes) == 1:
            num_nodes = np.array(num_nodes).item() * np.ones(num_layers)
        elif len(np.array(num_nodes).shape) != 0 and np.array(num_nodes).shape[0] == num_layers:
            num_nodes = np.array(num_nodes)
        else:
            raise ValueError('num_nodes has to be single valued or has to fulfill len(num_nodes)=num_layers') 
        
          
        self.custom_layers = [keras.layers.Dense(num_nodes[0], activation=swish, input_dim=input_dim, kernel_initializer='he_normal')]

        for i in range(1,num_layers):
            self.custom_layers.append(keras.layers.Dense(num_nodes[i], activation=swish, kernel_initializer='he_normal'))

        self.custom_layers.append(keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal'))

    def call(self, inputs):
        x = self.custom_layers[0](inputs)

        for i in range(1,len(self.custom_layers)):
            x = self.custom_layers[i](x)

        return x
    
def build(config, strategy=None):
    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            model = ALICE_reg(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=9)

            optimizer = keras.optimizers.Nadam(
                learning_rate=config['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            model.compile(optimizer=optimizer, loss=ALICE_loss, weighted_metrics=[])
    else:
        model = ALICE_reg(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=9)

        optimizer = keras.optimizers.Nadam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(optimizer=optimizer, loss=ALICE_loss, weighted_metrics=[])
    
    return model

def train(model, config, train_dataset, val_dataset, strategy=None):
    # Setup keras callbacks
    checkpoint_filepath = os.path.join('.', 'checkpoint.model.tf')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, start_from_epoch=30)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join('.', 'logs'))

    callbacks = [model_checkpoint_callback, tensorboard_callback]

    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            history_callback = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=config['epochs'], verbose=2)
    else:
        history_callback = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=config['epochs'], verbose=2)
        
    return history_callback

def save(model, history_callback):
    model.save(os.path.join('.', 'final.model.tf'), save_format='tf')

    with open(os.path.join('.', 'history.txt'), 'w') as hist_file:
        hist_file.write(str(history_callback.history['loss']))
        hist_file.write(str(history_callback.history['val_loss']))

def load(model_path):
    return keras.models.load_model(model_path, custom_objects={'ALICE_reg': ALICE_reg, 'swish_activation': swish_activation, 'ALICE_loss': ALICE_loss})