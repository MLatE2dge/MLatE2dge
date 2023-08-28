"""
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

import json
import os

import requests

import pandas as pd
import numpy as np
import pingouin as pg

from sklearn.metrics import mean_squared_error
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def outcome_metrics(df, gold_standard, new_method, model):
    """
    Parameters
    ----------
    df:
        dataframe, contains gold standard and target values

    gold_standard: str
        Column header name, ground truth (correct) target values. 

    new_method: str
        Column header name, estimated target values.. 

    """

    df['error'] = df[gold_standard]-df[new_method]

    bias = df['error'].mean()
    std_bias = df['error'].std()
    agreement = 1.96
    high = bias + agreement * std_bias
    low = bias - agreement * std_bias
    MDC = 0.5*(high-low)
    m, b = np.polyfit(df[gold_standard], df[new_method], 1)

    print('y = %0.2fx + %0.2f' % (m, b))
    print('MSE: %s' %
          (round(mean_squared_error(df[gold_standard], df[new_method]), 2)))
    print('bias: %s +/- %s cm' % (round(bias, 2), round(std_bias, 2)))
    print("LoA_U: %s" % round(high, 2))
    print("LoA_L: %s" % round(low, 2))
    print("Minimal detectable change (MDC): %s cm" % round(MDC, 2))
    print('Percentage of all samples < MDC: %s%%' %
          round(100*df[df['error'] < MDC].shape[0] / df.shape[0], 2))

    df.plot(kind='scatter', x=gold_standard, y=new_method,
            style='o', color='blue', legend=None)
    plt.plot(np.array(df[gold_standard]), m *
             np.array(df[gold_standard]) + b, 'r-', label='regression line')
    plt.plot([80, 160], [80, 160], 'k--', label='unity slope')
    plt.xlim(80, 160)
    plt.ylim(80, 160)
    plt.legend()
    
    if model == 'tflite.tflite': 
        plt.xlabel('Stride length [cm] - OptoGait')
        plt.ylabel('Stride length [cm] - float32 model')
    elif model == 'dr_quant.tflite':
        plt.xlabel('Stride length [cm] - OptoGait')
        plt.ylabel('Stride length [cm] - dr. quant model')
    elif model == 'ei_int8':
        plt.xlabel('Stride length [cm] - OptoGait')
        plt.ylabel('Stride length [cm] - int8 quantized  model')
    elif model == 'ei_float32':
        plt.xlabel('Stride length [cm] - OptoGait')
        plt.ylabel('Stride length [cm] - float32 model')

    plt.grid(color='gray', linestyle='--', linewidth=0.25)
    plt.savefig('regression-'+model+'.png', bbox_inches='tight')

    pg.plot_blandaltman(df[gold_standard], df[new_method], scatter_kws={
                        'marker': 'o', 'alpha': 0.8, 'color': 'blue'}, figsize=(10, 6))
    plt.ylim(-20, 20)
    plt.xlim(90, 160)
    plt.title('')
    if model == 'tflite.tflite': 
        plt.xlabel('Measurement agreement [cm], (OptoGait + float32 model)/2')
        plt.ylabel('Measurement error [cm], (OptoGait - float32 model)')
    elif model == 'dr_quant.tflite':   
        plt.xlabel('Measurement agreement [cm], (OptoGait + dr. quant model)/2')
        plt.ylabel('Measurement error [cm], (OptoGait - dr. quant model)')
    elif model == 'ei_int8':
        plt.xlabel('Measurement agreement [cm], (OptoGait + int8 quantized model)/2')
        plt.ylabel('Measurement error [cm], (OptoGait - int8 quantized model)')
    elif model == 'ei_float32':
        plt.xlabel('Measurement agreement [cm], (OptoGait + float32 model)/2')
        plt.ylabel('Measurement error [cm], (OptoGait - float32 model)')           

    plt.grid(color='gray', linestyle='--', linewidth=0.25)
    plt.savefig('Bland-Altman-'+model+'.png', bbox_inches='tight')
    plt.show()


def create_dataset(config, fold, dset):

    path_fold = config['path']['fold']
    path_metadata = config['path']['metadata']

    df_fold = pd.read_json(path_fold)
    df_meta = pd.read_json(path_metadata)

    sub_set = df_fold[fold][dset]

    filenames = []
    for file in df_meta["filename"]:
        if file.split("-")[0] in sub_set:
            filenames.append(file)

    tmp = []
    tmp_X = []
    tmp_Y = []

    for filename in filenames:
        json_file = open(os.path.join(config['path']['artifacts'], filename))
        json_data = json.load(json_file)
        data = np.array(json_data["payload"]["values"])
        label = filename.split("-")[-1].split(".json")[0]

        data_reshape = data.reshape(int(data.shape[0]*data.shape[1]))
        tmp.append(data_reshape.tolist())
        tmp_Y.append(float(label))

     # https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    nmb_samples = config['dataset']['signal_length']
    if nmb_samples >= len(max(tmp, key=len)):
        tmp_X = np.array([i + [0]*(nmb_samples-len(i)) for i in tmp])

    tmp_Y = np.array(tmp_Y)

    return tmp_X, tmp_Y, filenames


def evaluate_model(tf_model, X_test_data, Y_test_data):
    interpreter = tf.lite.Interpreter(tf_model)
    interpreter.allocate_tensors()

    # get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # test the model on random input data.
    input_shape = input_details[0]['shape']

    results = []
    for i in range(X_test_data.shape[0]):
        input_data = np.array([X_test_data[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        results.append([output_data[0][0], Y_test_data[i]])

    df = pd.DataFrame(results, columns=['result', 'expected result'])
    return df


def testing_statistical_analyse(config, path, model, fold):
    # -----------------------------------------------------------------------------------------------
    # Download model
    url = "https://api.wandb.ai/files/"+os.path.join(path, model)
    r = requests.get(url, allow_redirects=True)
    open(model, "wb").write(r.content)

    # -----------------------------------------------------------------------------------------------
    X_val, Y_val, filenames = create_dataset(config, fold, "validation_set")
    df = evaluate_model(model, X_val, Y_val)
    df['filename'] = filenames

    gold_standard = 'OptoGait stride length [cm]'
    new_method = 'ML-model stride length [cm]'

    df.rename(columns={'expected result': gold_standard,
              'result': new_method}, inplace=True)
    outcome_metrics(df=df, gold_standard=gold_standard, new_method=new_method, model=model)
    df[[gold_standard, new_method]].corr()


def json_file_to_npy_array(file, nmb_samples):
    # print(file)

    json_file = open(file)
    json_data = json.load(json_file)

    data = np.array(json_data["payload"]["values"])
    label = file.split("-")[-1].split(".json")[0]

    data_reshape = data.reshape(int(data.shape[0]*data.shape[1]))
    tmp = np.array(list(data_reshape))

    X_testing = (np.pad(np.array(list(data_reshape)),
                 (0, nmb_samples-tmp.shape[0]), 'constant'))
    Y_testing = np.array(float(label))

    return X_testing, Y_testing
