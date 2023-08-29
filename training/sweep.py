""" Sweeps, experiment tracking.

    Prerequisite: Weights & Biases platform
    References: 
    - https://wandb.ai/site/sweeps
    - https://docs.wandb.ai/tutorials/sweeps

MIT License

Copyright (c) 2023 - J.R.Verbiest

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# Parameters
config_file = '/sweep_config.yaml'
sweep_file  = '/sweep.yaml'        

# -----------------------------------------------------------------------------------------------------------

import os
import yaml
from yaml.loader import SafeLoader

import numpy as np
import random

from numba import cuda

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape, MaxPooling1D, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping

import wandb
from wandb.keras import WandbCallback

# logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# config (yaml) file
with open(os.getcwd()+config_file, "r") as yaml_file:
    project_config = yaml.load(yaml_file, Loader=SafeLoader)
yaml_file.close()

## sweep (yaml) file
with open(os.getcwd()+sweep_file,"r") as wandb_yaml_file:
    sweep_config = yaml.load(wandb_yaml_file, Loader=SafeLoader)
wandb_yaml_file.close()

import helper

# -----------------------------------------------------------------------------------------------------------

RANDOM_SEED = 3

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------------------------------------
def run():
    # -------------------------------------------------------------------------------------------------------
    # Default values for hyper-parameters we're going to sweep over
    # -------------------------------------------------------------------------------------------------------   
    config_default = {
        'epochs': project_config['default']['epochs'],
        'batch_size': project_config['default']['batch_size'],
        'lr': project_config['default']['lr'],       
        'epsilon': project_config['default']['epsilon'],
        'beta_1': project_config['default']['beta_1'],
        'beta_2': project_config['default']['beta_2'],
        'conv_1D_layer_1_filter': project_config['default']['conv_1D_layer_1_filter'],
        'conv_1D_layer_2_filter': project_config['default']['conv_1D_layer_2_filter'],
        'conv_1D_layer_3_filter': project_config['default']['conv_1D_layer_3_filter'],
        'conv_1D_layer_1_kernel_size': project_config['default']['conv_1D_layer_1_kernel_size'],
        'conv_1D_layer_2_kernel_size': project_config['default']['conv_1D_layer_2_kernel_size'],
        'conv_1D_layer_3_kernel_size': project_config['default']['conv_1D_layer_3_kernel_size'],
        'dropout_layer_1': project_config['default']['dropout_layer_1'],
        'dense_layer_1_units': project_config['default']['dense_layer_1_units'],
        'fold': project_config['default']['fold'],
        } 

    wandb.init(config = config_default)
    config = wandb.config

    # -------------------------------------------------------------------------------------------------------
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.experimental.numpy.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------------------------------------   
    X_train, Y_train, _ = helper.create_dataset(project_config, config.fold, "train_set")
    X_val, Y_val, _     = helper.create_dataset(project_config, config.fold, "validation_set")

    train_dataset       = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset       = train_dataset.batch(config.batch_size, drop_remainder=False)

    validation_dataset  = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    validation_dataset  = validation_dataset.batch(config.batch_size, drop_remainder=False)

    # -------------------------------------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------------------------------------
    # Initializer that generates a truncated normal distribution.
    # https://cs231n.github.io/neural-networks-2/

    
    # input_length = X_train[0].shape[0]
    # Sequential API is used
    model = Sequential()

    model.add(Reshape((int(X_train[0].shape[0] / 6), 6), input_shape=(X_train[0].shape[0], )))
    
    # Layer 1
    model.add(Conv1D(
        config.conv_1D_layer_1_filter, 
        kernel_size = config.conv_1D_layer_1_kernel_size,
        kernel_initializer = initializers.TruncatedNormal(mean=0., stddev=0.1, seed=RANDOM_SEED), 
        bias_initializer   = initializers.Constant(0.1), 
        activation = 'relu', 
        padding = 'same'))

    model.add(MaxPooling1D(
        pool_size = 2, 
        strides = 2, 
        padding = 'same'))
   
    # Layer 2
    model.add(Conv1D(
        config.conv_1D_layer_2_filter, 
        kernel_size = config.conv_1D_layer_2_kernel_size,
        kernel_initializer = initializers.TruncatedNormal(mean=0., stddev=0.1, seed=RANDOM_SEED), 
        bias_initializer   = initializers.Constant(0.1),
        activation = 'relu', 
        padding = 'same'))

    model.add(MaxPooling1D(
        pool_size = 2, 
        strides = 2, 
        padding = 'same'))

    # Layer 3
    model.add(Conv1D(config.conv_1D_layer_3_filter, 
        kernel_size = config.conv_1D_layer_3_kernel_size, 
        kernel_initializer = initializers.TruncatedNormal(mean=0., stddev=0.1, seed=RANDOM_SEED), 
        bias_initializer   = initializers.Constant(0.1), 
        activation = 'relu', 
        padding = 'same'))

    model.add(MaxPooling1D(
        pool_size = 2, 
        strides = 2, 
        padding = 'same'))
    
    # Flatten
    model.add(Flatten())
    
    # Dense_Layer 1
    model.add(Dense(
        config.dense_layer_1_units, 
        kernel_initializer = initializers.TruncatedNormal(mean=0., stddev=0.1, seed=RANDOM_SEED), 
        bias_initializer   = initializers.Constant(0.1), 
        activation = 'relu'))
        
    # Dropout Layer
    model.add(Dropout(
        config.dropout_layer_1))

    # Output Layer
    model.add(Dense(
        1, 
        kernel_initializer = initializers.TruncatedNormal(mean=0., stddev=0.1, seed=RANDOM_SEED), 
        bias_initializer   = initializers.Constant(0.1), 
        activation = 'linear', 
        name = 'out'))

    # -------------------------------------------------------------------------------------------------------
    
    # early stopping
    if project_config["training_settings"]["early_stopping"] == True:
        callbacks = [WandbCallback(), EarlyStopping(monitor=project_config["training_settings"]["early_stopping_monitor"], patience=project_config["training_settings"]["early_stopping_patience"])]
    else:
        callbacks = [WandbCallback()]
    
    opt = Adam(learning_rate=config.lr, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon)
    metrics=[tf.keras.metrics.MeanSquaredError()]
    
    model.compile(loss = 'mean_squared_error' , optimizer = opt, metrics = metrics)
    model.fit(train_dataset, epochs=config.epochs, validation_data=validation_dataset, verbose=1, callbacks=callbacks)

    # -------------------------------------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------------------------------------
    y_pred = model.predict(X_val)
    y_prediction = []
    for element in y_pred:
        y_prediction.append(element[0])

    m_val, b_val, MSE_val, bias_val, std_bias_val, MDC_val = helper.outcome_metrics(Y_val, y_prediction)

    # -------------------------------------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------------------------------------
    # References:
    # https://wandb.ai/sayakpaul/tale-of-quantization/reports/A-Tale-of-Model-Quantization-in-TF-Lite--Vmlldzo5MzQwMA
    # https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite
    # converting a tf Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # post-training dynamic range quantization (quantizes the weights of the model to 8-bits of precision)
    # https://www.tensorflow.org/lite/performance/post_training_quantization
    # The post-training dynamic range quantization is used to get a first estimation of the memory usage. 
    # Finally the Edge Impulse ION Tuner is used to estimate the memory for a given device, this by retraining the model in Edge Impulse Studio.
    # In future implementations it is recommended to use the Profiling and Deploy, see [🔗 Edge Impulse Python SDK](https://docs.edgeimpulse.com/docs/tools/overview) 
    # combined with [🔗 Weights & Biases](https://docs.edgeimpulse.com/docs/integrations/weights-and-biases).

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_range_quantization = converter.convert()
    
    # save tflite models
    tflite_file = 'tflite.tflite'
    dynamic_range_quantization_file = 'dr_quant.tflite'
    
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    f.close() 

    with open(dynamic_range_quantization_file, 'wb') as f:
        f.write(dynamic_range_quantization)     
    f.close() 

    # model size                     
    tflite_model_size = round(os.path.getsize(tflite_file) / float(1024*1024), 1)
    dynamic_range_quantization_model_size = round(os.path.getsize(dynamic_range_quantization_file) / float(1024*1024), 1)

    # -------------------------------------------------------------------------------------------------------
    # Logging - Save artifacts
    # -------------------------------------------------------------------------------------------------------

    wandb.log({
        "m_val": m_val,
        "b_val": b_val,
        "MSE_val": MSE_val,
        "bias_val": bias_val,
        "std_bias_val": std_bias_val,
        "MDC_val": MDC_val,
        "tflite_model_size [MB]": tflite_model_size,
        "dynamic_range_quantization_model_size [MB]": dynamic_range_quantization_model_size,
        "fold": config.fold
        })

    if project_config["wandb"]["save_artifacts"] == True:
        for file in project_config["wandb"]["artifacts"]:
            wandb.save(file)
    else: 
        pass

# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    print(f"Random seed set as {RANDOM_SEED}")
    project = project_config['wandb']['project_name']

    sweep_id = wandb.sweep(sweep_config, project=project)

    wandb.agent(sweep_id, function=run)

# -----------------------------------------------------------------------------------------------------------