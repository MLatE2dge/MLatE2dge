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

import time

import pandas as pd
import numpy as np

from scipy import signal
from scipy.signal import butter, filtfilt

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Span


def select_column(df, column):

    return df[column]


def select_rows(df, column, value):
    
    return df[df[column] == value]


def create_list(df):
    
    return list(df)


def normalize_imu_data(df, range_acc, range_gyr):
    
    df_norm = pd.DataFrame()
    
    df_norm["Time"] = df["Time"]
    df_norm["Ax"] = df["Ax"]/range_acc
    df_norm["Ay"] = df["Ay"]/range_acc
    df_norm["Az"] = df["Az"]/range_acc
    df_norm["Gx"] = df["Gx"]/range_gyr
    df_norm["Gy"] = df["Gy"]/range_gyr
    df_norm["Gz"] = df["Gz"]/range_gyr
    
    return df_norm


def resample_imu_data(df, fs, fs_resample):
    
    num_resample = int(fs_resample*len(df['Time'])/fs)
    
    df_resample = pd.DataFrame()
    
    df_resample['Time'] = [x / fs_resample for x in [*range(0, num_resample, 1)]]       
    df_resample['Gx'] = signal.resample(x = df['Gx'], num=num_resample)
    df_resample['Gy'] = signal.resample(x = df['Gy'], num=num_resample)
    df_resample['Gz'] = signal.resample(x = df['Gz'], num=num_resample)
    df_resample['Ax'] = signal.resample(x = df['Ax'], num=num_resample)
    df_resample['Ay'] = signal.resample(x = df['Ay'], num=num_resample)
    df_resample['Az'] = signal.resample(x = df['Az'], num=num_resample)
    
    return df_resample


def butter_lpf_filtfilt(data, cutoff, fs, order=2):
    
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,data)
    
    return y, b, a


def lpf_signal(df,  fs, cutoff, order):
    
    df_imu_lpf = pd.DataFrame()
    
    df_imu_lpf["Time"]     = df["Time"]
    df_imu_lpf["Gx"], _, _ = butter_lpf_filtfilt(df["Gx"], cutoff, fs, order)
    df_imu_lpf["Gy"], _, _ = butter_lpf_filtfilt(df["Gy"], cutoff, fs, order)
    df_imu_lpf["Gz"], _, _ = butter_lpf_filtfilt(df["Gz"], cutoff, fs, order)
    
    df_imu_lpf["Ax"], _, _ = butter_lpf_filtfilt(df["Ax"], cutoff, fs, order)
    df_imu_lpf["Ay"], _, _ = butter_lpf_filtfilt(df["Ay"], cutoff, fs, order)
    df_imu_lpf["Az"], _, _ = butter_lpf_filtfilt(df["Az"], cutoff, fs, order)

    
    return df_imu_lpf


def get_first_ic(df, subject, run):
    first_ic = (df
                .pipe(select_rows, column = 'subject', value = subject)
                .pipe(select_rows, column = 'run', value = run)
                .pipe(select_column, column = 'ic_time')
                .pipe(create_list)
                )[0]
    
    return first_ic


def add_ic(df, first_ic):
    df['IC_n-1 [sec]'] = df['timestamp']-df['timestamp'].iloc[0]+first_ic
    df['IC_n-1 [sec]'] = df['IC_n-1 [sec]'].shift(1)
    df['IC_n [sec]'] = df['IC_n-1 [sec]'].shift(-1)
    df = df.drop(columns=['timestamp'])

    return df


def get_signal(df, signal_duration, first_ic):
    
    return df[df['IC_n [sec]'] <= (signal_duration+first_ic)]


def get_optogait_data(df, first_ic, signal_duration, sensor_location):
    """
    Reference: [TRIPOD—A Treadmill Walking Dataset with IMU, Pressure-Distribution and Photoelectric Data for Gait Analysis](https://www.mdpi.com/2306-5729/6/9/95)
    """
    
    if sensor_location == 'RF':
        sensor_location_LR = 'R'
    else:
        sensor_location_LR = 'L'
        
    optogait_data = (df
                    .pipe(select_column, column = ['L/R', 'Split', 'Stride', 'StrideTime\\Cycle'])
                    .rename(columns = {
                        'Split': 'timestamp',
                        'Stride': 'stride length (OptoGait) [cm]',
                        'StrideTime\\Cycle': 'stride time (OptoGait) [sec]'
                        })
                    .pipe(select_rows, column = 'L/R', value=sensor_location_LR)
                    .reset_index(drop=True)
                    .pipe(add_ic, first_ic)
                    .shift(-1)
                    .pipe(get_signal, signal_duration, first_ic)
                    .drop(columns=['L/R'])
                    )    
    
    return optogait_data


def gait_event_detection(signal, threshold):
    """
    Reference: [A Real-Time Gait Event Detection for Lower Limb Prosthesis Control and Evaluation](https://ieeexplore.ieee.org/document/7776971)
    
    """
    
    W_ = []
    Dn_ = []
    MSW_ = []
    IC_ = []

    MSW = False

    for i in range(len(signal)):

        if i == 0:
            W = signal[i]
            W_.append(W)

        else:
            Dn = signal[i]-W
            W = signal[i]
            if i == 1:
                Dn_1 = Dn
            else:
                if Dn < 0 and Dn_1 > 0:
                    if W > threshold:
                        MSW_.append(i-1)
                        MSW = True
                elif Dn > 0 and Dn_1 < 0:
                    if MSW == True and W < 0: 
                        IC_.append(i-1)
                        MSW = False
                Dn_1 = Dn
            W_.append(W)
            Dn_.append(Dn)
    
    
    return np.array(IC_)


def get_gait_cycle(df, IC_a, IC_b):

    return df[(df["Time"] >= IC_a) & (df["Time"] < IC_b)]


def json_sensordata(device_type, sensors, sensors_values, interval_ms):
  """ JSON wrapper for sensor data
  
  Parameters
  ----------
    device_type: str
        type device
    interval_ms: float
        time interval between sample is ms
    sensors_values: list
        list of values
        
  Returns
  -------
  data: struct
      json template
      
  """

  # Start with all zeros. Hs256 gives 32 bytes and we encode in hex. So, we need 64 characters here.
  empty_signature = ''.join(['0'] * 64)

  # Create JSON wrapper for data
  data = {
      "protected": {
          "ver": "v1",
          "alg": "HS256",
          "iat": time.time()                  
      },
      "signature": empty_signature,
      "payload": {
          "device_type": device_type,          
          "interval_ms": interval_ms,                  
          "sensors": sensors,
          "values": sensors_values
      }
  }

  return data


def drop_false_IC(delta, ICa):
    IC = []
    for i in range(len(ICa)-1):
        if np.abs((ICa[i]-ICa[i+1])) > delta:
            IC.append(ICa[i+1])
    
    return IC


def plot_imu_axis(imu_norm, imu_lpf, firstIC,IC_ref, IC_gs, axis):
    graph = figure(
            title='IMU normalized (red) / IMU filtered (green) - OptoGait = gray / GE = red',
            sizing_mode="stretch_width",
            height=500,
            x_axis_label='time [s]',
            y_axis_label=axis,
            toolbar_location = "below"
        )

    graph.line(imu_norm['Time'], imu_norm[axis.split(' ')[0]], line_color='red',   line_width=2)
    graph.line(imu_lpf['Time'],  imu_lpf[axis.split(' ')[0]],  line_color='green', line_width=2)

    line = Span(location= firstIC, dimension='height', line_color='black', line_dash='solid', line_width=1)
    graph.add_layout(line)

    for ic in IC_gs:
        line = Span(location= ic, dimension='height', line_color='red', line_dash='dashed', line_width=1)
        graph.add_layout(line)

    for ic in IC_ref:
        line = Span(location= ic, dimension='height', line_color='darkslategray', line_dash='dashdot', line_width=1)
        graph.add_layout(line)

    show(graph)
