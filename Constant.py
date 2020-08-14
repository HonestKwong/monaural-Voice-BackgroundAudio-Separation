#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一些常用的默认参数
@author: KuangHaoheng
Sichuan University
Tel:13547937661
"""
#  U_net卷积神经网络
patch_size = 128 # roughly 33 seconds
Sample_rate = 8192
PATH_FFT = "./MIR-1K/UnetFFT_Full"
DEFAULT_WIN_LEN_PARAM = 0.03     #默认每一帧窗口长度长度30ms
window_length = 1024
hop_length = 768
EPOCH = 80000
BATCH = 16
SAMPLING_STRIDE = 10
VOX_ACC_DICT = {'vocals': {'vocals': 1}, 'accompaniment': {'accompaniment': 1}}


#   循环神经网络
RNN_PATH_FFT = "./MIR-1K/RNN_FFT"
num_hidden_units = [1024, 1024, 1024, 1024, 1024]
RNN_window_length = window_length
RNN_num_features = RNN_window_length // 2 + 1
RNN_BATCH = 64
RNN_EPOCH = 80000

#   一维卷积神经网络
CNN_1d_context = True
CNN_1d_output_filter_size = 1
CNN_1d_num_layers = 12
CNN_1d_merge_filter_size = 5
CNN_1d_filter_size = 15
CNN_1d_input_filter_size = 15
CNN_1d_num_channels = 1


Unet_recognition_Path = "./MIR-1K/Unet_recognition_FFT"

