#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    判断识别率预处理
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
import utils

def Unet_Generate_data():

    Vocal_pure_filelist = librosa.util.find_files('./MIR-1K/Vocal_pure',ext="wav")
    original_filelist = librosa.util.find_files('./MIR-1K/UndividedWavfile',ext="wav")

    for Vocal_pure_filepath, original_filepath in zip(Vocal_pure_filelist, original_filelist):
        Audio_path = Vocal_pure_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        vox_audio, mix_sr = librosa.load(Vocal_pure_filepath, sr=8192) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=8192)
        vox_length = len(vox_audio)
        if len(vox_audio) > original_audio.shape[1]:
            vox_length = original_audio.shape[1]
        else :
            vox_length = len(vox_audio)
        vox_audio = vox_audio[:vox_length]               #[:163840]
        acc_audio = original_audio[0,:vox_length]

        mix = vox_audio + acc_audio
        vox_path = os.path.join('.','MIR-1K' , 'recognize', 'vox', Music_name+'.wav')  
        mix_path = os.path.join('.','MIR-1K' , 'recognize', 'mix', Music_name+'_mix.wav')     
        librosa.output.write_wav(vox_path, vox_audio, sr=8192,norm=False)
        librosa.output.write_wav(mix_path, mix, sr=8192,norm=False)

        utils.SaveSpectrogram(mix, vox_audio, acc_audio, Music_name,original_Sample_rate=8192)

def CNN_1D_Generate_data():
    Vocal_pure_filelist = librosa.util.find_files('./MIR-1K/Vocal_pure',ext="wav")
    original_filelist = librosa.util.find_files('./MIR-1K/UndividedWavfile',ext="wav")

    for Vocal_pure_filepath, original_filepath in zip(Vocal_pure_filelist, original_filelist):
        Audio_path = Vocal_pure_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        vox_audio, mix_sr = librosa.load(Vocal_pure_filepath, sr=8192) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=8192)
        vox_length = len(vox_audio)
        if len(vox_audio) > original_audio.shape[1]:
            vox_length = original_audio.shape[1]
        else :
            vox_length = len(vox_audio)
        vox_audio = vox_audio[:vox_length]               #[:163840]
        acc_audio = original_audio[0,:vox_length]

        mix = vox_audio + acc_audio
        vox_path = os.path.join('.','MIR-1K' , 'recognize', 'vox', Music_name+'.wav')  
        mix_path = os.path.join('.','MIR-1K' , 'recognize', 'mix', Music_name+'_mix.wav')     
        librosa.output.write_wav(vox_path, vox_audio, sr=8192,norm=False)
        librosa.output.write_wav(mix_path, mix, sr=8192,norm=False)

        utils.CNN_1D_SaveAudio(mix, vox_audio, acc_audio, Music_name, "./MIR-1K/MIX_1D_recognize_npz")


def RNN_Generate_data():

    Vocal_pure_filelist = librosa.util.find_files('./MIR-1K/Vocal_pure',ext="wav")
    original_filelist = librosa.util.find_files('./MIR-1K/UndividedWavfile',ext="wav")

    for Vocal_pure_filepath, original_filepath in zip(Vocal_pure_filelist, original_filelist):
        Audio_path = Vocal_pure_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        vox_audio, mix_sr = librosa.load(Vocal_pure_filepath, sr=8192) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=8192)
        vox_length = len(vox_audio)
        if len(vox_audio) > original_audio.shape[1]:
            vox_length = original_audio.shape[1]
        else :
            vox_length = len(vox_audio)
        vox_audio = vox_audio[:vox_length]               #[:163840]
        acc_audio = original_audio[0,:vox_length]

        mix = vox_audio + acc_audio
        vox_path = os.path.join('.','MIR-1K' , 'recognize', 'vox', Music_name+'.wav')  
        mix_path = os.path.join('.','MIR-1K' , 'recognize', 'mix', Music_name+'_mix.wav')     
        librosa.output.write_wav(vox_path, vox_audio, sr=8192,norm=False)
        librosa.output.write_wav(mix_path, mix, sr=8192,norm=False)

        utils.RNN_SaveSpectrogram(mix, vox_audio, acc_audio, Music_name,original_Sample_rate=8192)


if __name__ == "__main__":
    # Generate_data()
    # CNN_1D_Generate_data()
    RNN_Generate_data()

