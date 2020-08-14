#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    该文件对MIR-1K数据集进行划分
    @author: KuangHaoheng
    Sichuan University
    Tel:13547937661
"""

import numpy as np
import librosa
#from librosa.core import load
from librosa.util import find_files
import scipy
import os
import Constant
import utils


def U_net_process():
    #MIR1K_path = "./MIR-1K/Wavfile"    #添加MIR1K数据集切割后的路径
    MIR1K_path = "./MIR-1K/广播数据"    #添加MIR1K数据集切割后的路径
    filelist = find_files(MIR1K_path,ext="wav")

    if not os.path.exists("./MIR-1K/Vocal_Train"): #如果不存在文件夹则创建相应存放数据的文件夹
        os.mkdir("./MIR-1K/Vocal_Train")

    if not os.path.exists("./MIR-1K/Music_Train"):
        os.mkdir("./MIR-1K/Music_Train")

    if not os.path.exists("./MIR-1K/MIX"):
        os.mkdir("./MIR-1K/MIX")

    if not os.path.exists("./MIR-1K/UnetFFT"):
        os.mkdir("./MIR-1K/UnetFFT")

    if not os.path.exists("./MIR-1K/UnetFFT_Full"):
        os.mkdir("./MIR-1K/UnetFFT_Full")

    if not os.path.exists("./MIR-1K/MIX_Full"):
        os.mkdir("./MIR-1K/MIX_Full")
    # if not os.path.exists("./MIR-1K/MIX"):
    #     os.mkdir("./MIR-1K/MIX")

    for audiofile in filelist:
        Audio_name = os.path.split(audiofile)[-1]
        Music_name = Audio_name.split('.')[0]
        print("Processing: %s" % Audio_name)

    #论文降采样率为8192
        y, sr = librosa.load(audiofile, sr=Constant.Sample_rate, mono=False)
        music = y[0, :]
        mix = y[0, :]+y[1, :]      #人声与音乐混合，加性混合
        vocal = y[1, :]
        
        # scipy.io.wavfile.write(os.path.join("./MIR-1K/Music_Train", Music_name+"_Music"+".wav"), data=music, rate=sr)
        scipy.io.wavfile.write(os.path.join("./MIR-1K/MIX_Full", Music_name+"_Mix"+".wav"), data=mix, rate=sr)
        # scipy.io.wavfile.write(os.path.join("./MIR-1K/Vocal_Train", Music_name+"_Vocal"+".wav"), data=vocal, rate=sr)

        #utils.SaveSpectrogram(mix, vocal, music, Music_name,original_Sample_rate=sr)


def RNN_Process():
     #MIR1K_path = "./MIR-1K/Wavfile"    #添加MIR1K数据集切割后的路径
    MIR1K_path = "./MIR-1K/Wavfile"    #添加MIR1K数据集切割后的路径
    filelist = find_files(MIR1K_path,ext="wav")


    if not os.path.exists("./MIR-1K/RNN_FFT"):
        os.mkdir("./MIR-1K/RNN_FFT")

    for audiofile in filelist:
        Audio_name = os.path.split(audiofile)[-1]
        Music_name = Audio_name.split('.')[0]
        print("Processing: %s" % Audio_name)

    #论文降采样率为8192
        y, sr = librosa.load(audiofile, sr=Constant.Sample_rate, mono=False)
        music = y[0, :]
        mix = y[0, :]+y[1, :]      #人声与音乐混合，加性混合
        vocal = y[1, :]
        
        # scipy.io.wavfile.write(os.path.join("./MIR-1K/Music_Train", Music_name+"_Music"+".wav"), data=music, rate=sr)
        #scipy.io.wavfile.write(os.path.join("./MIR-1K/MIX_Full", Music_name+"_Mix"+".wav"), data=mix, rate=sr)
        # scipy.io.wavfile.write(os.path.join("./MIR-1K/Vocal_Train", Music_name+"_Vocal"+".wav"), data=vocal, rate=sr)

        utils.RNN_SaveSpectrogram(mix, vocal, music, Music_name,original_Sample_rate=sr) 

def CNN_1D_Process():

    # MIR1K_train_path = "./MIR-1K/MIX_1D/Train"    #添加MIR1K数据集切割后的路径
    # filelist_train = find_files(MIR1K_train_path,ext="wav")  
    MIR1K_valide_path = "./MIR-1K/MIX_1D/Validation"    #添加MIR1K数据集切割后的路径
    filelist_valide = find_files(MIR1K_valide_path,ext="wav")  

    # for audiofile in filelist_train:
    #     Audio_name = os.path.split(audiofile)[-1]
    #     Music_name = Audio_name.split('.')[0]
    #     print("Processing: %s" % Audio_name)
    #     y, sr = librosa.load(audiofile, sr=Constant.Sample_rate, mono=False)
    #     music = y[0, :]
    #     mix = y[0, :]+y[1, :]      #人声与音乐混合，加性混合
    #     vocal = y[1, :]
    #     utils.CNN_1D_SaveAudio(mix, vocal, music, Music_name, "./MIR-1K/MIX_1D_npz/train")

    for audiofile in filelist_valide:
        Audio_name = os.path.split(audiofile)[-1]
        Music_name = Audio_name.split('.')[0]
        print("Processing: %s" % Audio_name)
        y, sr = librosa.load(audiofile, sr=Constant.Sample_rate, mono=False)
        music = y[0, :]
        mix = y[0, :]+y[1, :]      #人声与音乐混合，加性混合
        vocal = y[1, :]        
        utils.CNN_1D_SaveAudio(mix, vocal, music, Music_name, "./MIR-1K/MIX_1D_npz/validation")


if __name__ == "__main__":
    CNN_1D_Process()