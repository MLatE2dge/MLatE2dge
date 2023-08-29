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

import random
import os
import json
import pandas as pd
import numpy as np

import wandb

from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------------------------------------

RANDOM_SEED = 3

# -----------------------------------------------------------------------------------------------------------

def create_dataset(config, fold, dset):
    
    np.random.seed(RANDOM_SEED)
    
    path_fold = config['path']['fold']
    path_metadata = config['path']['metadata']
    
    df_fold = pd.read_json(path_fold)
    df_meta  = pd.read_json(path_metadata)
    
    sub_set = df_fold[fold][dset]

    if dset == 'trainset': 
        wandb.log({'trainset':sub_set})
    elif dset == 'validation_set': 
        wandb.log({'validation_set':sub_set})
    
    filenames = []
    for file in df_meta["filename"]:
        if file.split("-")[0] in sub_set:
            filenames.append(file)

    random.seed(RANDOM_SEED)        
    random.shuffle(filenames)
    
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

    # reference: https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    nmb_samples = config['dataset']['signal_length']
    if nmb_samples >= len(max(tmp, key=len)):
        tmp_X = np.array([i + [0]*(nmb_samples-len(i)) for i in tmp])   
    
    tmp_Y = np.array(tmp_Y)

    return  tmp_X, tmp_Y, filenames
    

def outcome_metrics(y_test, y_pred):
    """
    Parameters:
    -----------
    y_test: float
        ground truth
    y_pred: float
        prediction

    """

    m, b  = np.polyfit(y_test, y_pred, 1) 

    error = y_test-y_pred 

    bias = np.mean(error)
    std_bias = np.std(error, axis=0)
    
    agreement = 1.96
    high = bias + agreement * std_bias
    low  = bias - agreement * std_bias
    MDC  = 0.5*(high-low)

    MSE = round(mean_squared_error(y_test, y_pred),2)
 
    return m, b, MSE, bias, std_bias, MDC

# -----------------------------------------------------------------------------------------------------------   