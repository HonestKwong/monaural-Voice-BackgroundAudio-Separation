#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估分离效果：SDR, SIR, SAR etc..
使用museval
@author: KuangHaoheng
Sichuan University
Tel:13547937661
"""

import museval
import librosa
import numpy as np
import utils
from utils import track_like
import Constant
import os




def UnetCNN_evaluate():
    
    mix_filelist = librosa.util.find_files("./outputs/MIX_FULL")
    original_filelist = librosa.util.find_files("./outputs/UndividedWavFile")
    loss_type = "MSE_SIR"
    if loss_type == "L1":
        vox_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/L1/vocals")
        acc_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/L1/accompaniment")
    if loss_type == "MSE":
        vox_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/MSE/vocals")
        acc_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/MSE/accompaniment")
    if loss_type == "L1_SIR":
        vox_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/L1_SIR/vocals")
        acc_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/L1_SIR/accompaniment")        
    if loss_type == "MSE_SIR":
        vox_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/MSE_SIR/vocals")
        acc_filelist = librosa.util.find_files("./outputs/UnetCNN_outputs/MSE_SIR/accompaniment")

    for mix_filepath, vox_filepath, acc_filepath, original_filepath in zip(mix_filelist, vox_filelist, acc_filelist, original_filelist):
        # mix_filepath = mix_filelist[0]
        # vox_filepath = vox_filelist[0]
        # acc_filepath = acc_filelist[0]
        # original_filepath = original_filelist[0]
        Audio_path = mix_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        target_dict=Constant.VOX_ACC_DICT
        mix_audio, mix_sr = librosa.load(mix_filepath, sr=None) 
        vox_audio, vox_sr = librosa.load(vox_filepath, sr=None) 
        acc_audio, acc_sr = librosa.load(acc_filepath, sr=None) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=mix_sr)

        Track_mix = track_like(file_name=Music_name, audio_data=mix_audio, sample_rate= mix_sr, shape=mix_audio.shape)
        Track_original_acc = track_like(file_name=Music_name+"_original_acc", audio_data=original_audio[0], 
                                        sample_rate= original_sr, shape=original_audio[0].shape)
        Track_original_vox = track_like(file_name=Music_name+"_original_vox", audio_data=original_audio[1], 
                                        sample_rate= original_sr, shape=original_audio[1].shape) 
        Track_separated_acc = track_like(file_name=Music_name+"_separated_acc", audio_data=acc_audio, 
                                        sample_rate= acc_sr, shape=acc_audio.shape) 
        Track_separated_vox = track_like(file_name=Music_name+"_separated_vox", audio_data=vox_audio, 
                                        sample_rate= vox_sr, shape=vox_audio.shape)

                                        
                                        
        original_audio = {"vocals":Track_original_vox, "accompaniment":Track_original_acc}
        separated_audio = {"vocals":Track_separated_vox.audio_data.T, "accompaniment":Track_separated_acc.audio_data.T}
        track = utils.track_like_to_musdb_track(Track_mix, original_audio, target_dict)

        if loss_type == "L1":
            if not os.path.exists("./Estimate_result/UnetCNN_evaluate/L1"):
                os.mkdir("./Estimate_result/UnetCNN_evaluate/L1")
            results_dir = "./Estimate_result/UnetCNN_evaluate/L1"
        if loss_type == "MSE":
            if not os.path.exists("./Estimate_result/UnetCNN_evaluate/MSE"):
                os.mkdir("./Estimate_result/UnetCNN_evaluate/MSE")
            results_dir = "./Estimate_result/UnetCNN_evaluate/MSE"
        if loss_type == "L1_SIR":
            if not os.path.exists("./Estimate_result/UnetCNN_evaluate/L1_SIR"):
                os.mkdir("./Estimate_result/UnetCNN_evaluate/L1_SIR")
            results_dir = "./Estimate_result/UnetCNN_evaluate/L1_SIR"
        if loss_type == "MSE_SIR":
            if not os.path.exists("./Estimate_result/UnetCNN_evaluate/MSE_SIR"):
                os.mkdir("./Estimate_result/UnetCNN_evaluate/MSE_SIR")
            results_dir = "./Estimate_result/UnetCNN_evaluate/MSE_SIR"
        track.subset = "train"
        scores = museval.eval_mus_track(track, separated_audio, output_dir=results_dir)
        print ("Estimator:\n",scores)

def CNN_1D_evaluate():
    mix_filelist = librosa.util.find_files("./outputs/MIX_FULL")
    original_filelist = librosa.util.find_files("./outputs/UndividedWavFile")
    loss_type = "MSE_SIR"
    if loss_type == "MSE":
        vox_filelist = librosa.util.find_files("./outputs/CNN_1D_outputs/MSE/vocals")
        acc_filelist = librosa.util.find_files("./outputs/CNN_1D_outputs/MSE/accompaniment")
    if loss_type == "MSE_SIR":
        vox_filelist = librosa.util.find_files("./outputs/CNN_1D_outputs/MSE_SIR/vocals")
        acc_filelist = librosa.util.find_files("./outputs/CNN_1D_outputs/MSE_SIR/accompaniment")

    for mix_filepath, vox_filepath, acc_filepath, original_filepath in zip(mix_filelist, vox_filelist, acc_filelist, original_filelist):
        # mix_filepath = mix_filelist[0]
        # vox_filepath = vox_filelist[0]
        # acc_filepath = acc_filelist[0]
        # original_filepath = original_filelist[0]
        Audio_path = mix_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        target_dict=Constant.VOX_ACC_DICT
        mix_audio, mix_sr = librosa.load(mix_filepath, sr=None) 
        vox_audio, vox_sr = librosa.load(vox_filepath, sr=None) 
        acc_audio, acc_sr = librosa.load(acc_filepath, sr=None) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=mix_sr)

        Track_mix = track_like(file_name=Music_name, audio_data=mix_audio, sample_rate= mix_sr, shape=mix_audio.shape)
        Track_original_acc = track_like(file_name=Music_name+"_original_acc", audio_data=original_audio[0], 
                                        sample_rate= original_sr, shape=original_audio[0].shape)
        Track_original_vox = track_like(file_name=Music_name+"_original_vox", audio_data=original_audio[1], 
                                        sample_rate= original_sr, shape=original_audio[1].shape) 
        Track_separated_acc = track_like(file_name=Music_name+"_separated_acc", audio_data=acc_audio, 
                                        sample_rate= acc_sr, shape=acc_audio.shape) 
        Track_separated_vox = track_like(file_name=Music_name+"_separated_vox", audio_data=vox_audio, 
                                        sample_rate= vox_sr, shape=vox_audio.shape)

                                        
                                        
        original_audio = {"vocals":Track_original_vox, "accompaniment":Track_original_acc}
        separated_audio = {"vocals":Track_separated_vox.audio_data.T, "accompaniment":Track_separated_acc.audio_data.T}
        track = utils.track_like_to_musdb_track(Track_mix, original_audio, target_dict)
        if loss_type == "MSE":
            if not os.path.exists("./Estimate_result/CNN_1D_evaluate/MSE"):
                os.mkdir("./Estimate_result/CNN_1D_evaluate/MSE")
            results_dir = "./Estimate_result/CNN_1D_evaluate/MSE"
        if loss_type == "MSE_SIR":
            if not os.path.exists("./Estimate_result/CNN_1D_evaluate/MSE_SIR"):
                os.mkdir("./Estimate_result/CNN_1D_evaluate/MSE_SIR")
            results_dir = "./Estimate_result/CNN_1D_evaluate/MSE_SIR"
        track.subset = "train"
        scores = museval.eval_mus_track(track, separated_audio, output_dir=results_dir)
        print ("Estimator:\n",scores)
        
def RNN_evaluate(): 
    mix_filelist = librosa.util.find_files("./outputs/MIX_FULL")
    original_filelist = librosa.util.find_files("./outputs/UndividedWavFile")
    loss_type = "MSE"
    if loss_type == "MSE":
        vox_filelist = librosa.util.find_files("./outputs/RNN_outputs/MSE/vocals")
        acc_filelist = librosa.util.find_files("./outputs/RNN_outputs/MSE/accompaniment")
    if loss_type == "MSE_SIR":
        vox_filelist = librosa.util.find_files("./outputs/RNN_outputs/MSE_SIR/vocals")
        acc_filelist = librosa.util.find_files("./outputs/RNN_outputs/MSE_SIR/accompaniment")

    for mix_filepath, vox_filepath, acc_filepath, original_filepath in zip(mix_filelist, vox_filelist, acc_filelist, original_filelist):
        # mix_filepath = mix_filelist[0]
        # vox_filepath = vox_filelist[0]
        # acc_filepath = acc_filelist[0]
        # original_filepath = original_filelist[0]
        Audio_path = mix_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        target_dict=Constant.VOX_ACC_DICT
        mix_audio, mix_sr = librosa.load(mix_filepath, sr=None) 
        vox_audio, vox_sr = librosa.load(vox_filepath, sr=None) 
        acc_audio, acc_sr = librosa.load(acc_filepath, sr=None) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=mix_sr)

        Track_mix = track_like(file_name=Music_name, audio_data=mix_audio, sample_rate= mix_sr, shape=mix_audio.shape)
        Track_original_acc = track_like(file_name=Music_name+"_original_acc", audio_data=original_audio[0], 
                                        sample_rate= original_sr, shape=original_audio[0].shape)
        Track_original_vox = track_like(file_name=Music_name+"_original_vox", audio_data=original_audio[1], 
                                        sample_rate= original_sr, shape=original_audio[1].shape) 
        Track_separated_acc = track_like(file_name=Music_name+"_separated_acc", audio_data=acc_audio, 
                                        sample_rate= acc_sr, shape=acc_audio.shape) 
        Track_separated_vox = track_like(file_name=Music_name+"_separated_vox", audio_data=vox_audio, 
                                        sample_rate= vox_sr, shape=vox_audio.shape)

                                        
                                        
        original_audio = {"vocals":Track_original_vox, "accompaniment":Track_original_acc}
        separated_audio = {"vocals":Track_separated_vox.audio_data.T, "accompaniment":Track_separated_acc.audio_data.T}
        track = utils.track_like_to_musdb_track(Track_mix, original_audio, target_dict)
        if loss_type == "MSE":
            if not os.path.exists("./Estimate_result/RNN_evaluate/MSE"):
                os.mkdir("./Estimate_result/RNN_evaluate/MSE")
            results_dir = "./Estimate_result/RNN_evaluate/MSE"
        if loss_type == "MSE_SIR":
            if not os.path.exists("./Estimate_result/RNN_evaluate/MSE_SIR"):
                os.mkdir("./Estimate_result/RNN_evaluate/MSE_SIR")
            results_dir = "./Estimate_result/RNN_evaluate/MSE_SIR"
        track.subset = "train"
        scores = museval.eval_mus_track(track, separated_audio, output_dir=results_dir)
        print ("Estimator:\n",scores)

def RNN_rectified_outputs_evaluate(): 
    mix_filelist = librosa.util.find_files("./outputs/MIX_FULL")
    original_filelist = librosa.util.find_files("./outputs/UndividedWavFile")
    loss_type = "MSE_SIR"
    if loss_type == "MSE":
        vox_filelist = librosa.util.find_files("./outputs/RNN_rectified_outputs/MSE/vocals")
        acc_filelist = librosa.util.find_files("./outputs/RNN_rectified_outputs/MSE/accompaniment")
    if loss_type == "MSE_SIR":
        vox_filelist = librosa.util.find_files("./outputs/RNN_rectified_outputs/MSE_SIR/vocals")
        acc_filelist = librosa.util.find_files("./outputs/RNN_rectified_outputs/MSE_SIR/accompaniment")

    for mix_filepath, vox_filepath, acc_filepath, original_filepath in zip(mix_filelist, vox_filelist, acc_filelist, original_filelist):
        # mix_filepath = mix_filelist[0]
        # vox_filepath = vox_filelist[0]
        # acc_filepath = acc_filelist[0]
        # original_filepath = original_filelist[0]
        Audio_path = mix_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        target_dict=Constant.VOX_ACC_DICT
        mix_audio, mix_sr = librosa.load(mix_filepath, sr=None) 
        vox_audio, vox_sr = librosa.load(vox_filepath, sr=None) 
        acc_audio, acc_sr = librosa.load(acc_filepath, sr=None) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=mix_sr)

        Track_mix = track_like(file_name=Music_name, audio_data=mix_audio, sample_rate= mix_sr, shape=mix_audio.shape)
        Track_original_acc = track_like(file_name=Music_name+"_original_acc", audio_data=original_audio[0], 
                                        sample_rate= original_sr, shape=original_audio[0].shape)
        Track_original_vox = track_like(file_name=Music_name+"_original_vox", audio_data=original_audio[1], 
                                        sample_rate= original_sr, shape=original_audio[1].shape) 
        Track_separated_acc = track_like(file_name=Music_name+"_separated_acc", audio_data=acc_audio, 
                                        sample_rate= acc_sr, shape=acc_audio.shape) 
        Track_separated_vox = track_like(file_name=Music_name+"_separated_vox", audio_data=vox_audio, 
                                        sample_rate= vox_sr, shape=vox_audio.shape)

                                        
                                        
        original_audio = {"vocals":Track_original_vox, "accompaniment":Track_original_acc}
        separated_audio = {"vocals":Track_separated_vox.audio_data.T, "accompaniment":Track_separated_acc.audio_data.T}
        track = utils.track_like_to_musdb_track(Track_mix, original_audio, target_dict)
        if loss_type == "MSE":
            if not os.path.exists("./Estimate_result/RNN_rectified_evaluate/MSE"):
                os.mkdir("./Estimate_result/RNN_rectified_evaluate/MSE")
            results_dir = "./Estimate_result/RNN_rectified_evaluate/MSE"
        if loss_type == "MSE_SIR":
            if not os.path.exists("./Estimate_result/RNN_rectified_evaluate/MSE_SIR"):
                os.mkdir("./Estimate_result/RNN_rectified_evaluate/MSE_SIR")
            results_dir = "./Estimate_result/RNN_rectified_evaluate/MSE_SIR"
        track.subset = "train"
        scores = museval.eval_mus_track(track, separated_audio, output_dir=results_dir)
        print ("Estimator:\n",scores)

def NMF_outputs_evaluate(): 
    original_filelist = librosa.util.find_files("./outputs/WAV")
    vox_filelist = librosa.util.find_files("./outputs/NMF/vocals")
    acc_filelist = librosa.util.find_files("./outputs/NMF/accompaniment")

    for vox_filepath, acc_filepath, original_filepath in zip(vox_filelist, acc_filelist, original_filelist):
        # mix_filepath = mix_filelist[0]
        # vox_filepath = vox_filelist[0]
        # acc_filepath = acc_filelist[0]
        # original_filepath = original_filelist[0]
        Audio_path = original_filepath
        Audio_name = os.path.split(Audio_path)[-1]
        print ("processing", Audio_name)
        Music_name = Audio_name.split('.')[0]

        target_dict=Constant.VOX_ACC_DICT
        mix_audio, mix_sr = librosa.load(original_filepath, sr=8192)
        vox_audio, vox_sr = librosa.load(vox_filepath, sr=None) 
        acc_audio, acc_sr = librosa.load(acc_filepath, sr=None) 
        original_audio, original_sr = librosa.load(original_filepath, mono=False, sr=vox_sr)
        

        Track_mix = track_like(file_name=Music_name, audio_data=mix_audio, sample_rate= mix_sr, shape=mix_audio.shape)
        Track_original_acc = track_like(file_name=Music_name+"_original_acc", audio_data=original_audio[0], 
                                        sample_rate= original_sr, shape=original_audio[0].shape)
        Track_original_vox = track_like(file_name=Music_name+"_original_vox", audio_data=original_audio[1], 
                                        sample_rate= original_sr, shape=original_audio[1].shape) 
        Track_separated_acc = track_like(file_name=Music_name+"_separated_acc", audio_data=acc_audio, 
                                        sample_rate= acc_sr, shape=acc_audio.shape) 
        Track_separated_vox = track_like(file_name=Music_name+"_separated_vox", audio_data=vox_audio, 
                                        sample_rate= vox_sr, shape=vox_audio.shape)

                                        
                                        
        original_audio = {"vocals":Track_original_vox, "accompaniment":Track_original_acc}
        separated_audio = {"vocals":Track_separated_vox.audio_data.T, "accompaniment":Track_separated_acc.audio_data.T}
        track = utils.track_like_to_musdb_track(Track_mix, original_audio, target_dict)
    
        if not os.path.exists("./Estimate_result/NMF_evaluate"):
            os.mkdir("./Estimate_result/NMF_evaluate")
        results_dir = "./Estimate_result/NMF_evaluate"
        track.subset = "train"
        scores = museval.eval_mus_track(track, separated_audio, output_dir=results_dir)
        print ("Estimator:\n",scores)


if __name__ == "__main__" : 
    # RNN_evaluate()
    # CNN_1D_evaluate()
    # UnetCNN_evaluate()

    utils.draw_sdr("./Estimate_result/NMF_evaluate/train", target="SAR")
    utils.draw_sdr("./Estimate_result/CNN_1D_evaluate/MSE_SIR/train", target="SAR")
    utils.draw_sdr("./Estimate_result/RNN_rectified_evaluate/MSE_SIR/train", target="SAR")
    utils.draw_sdr("./Estimate_result/UnetCNN_evaluate/L1/train", target="SAR")


    # NMF_outputs_evaluate()
    # RNN_rectified_outputs_evaluate()
    # inst_list_CNN_1d = utils.compute_mean_metrics("./Estimate_result/CNN_1D_evaluate/MSE/train", compute_averages=True, metric="SIR")
    # inst_list_RNN = utils.compute_mean_metrics("./Estimate_result/RNN_evaluate/MSE/train", compute_averages=True, metric="SIR")
    
    # inst_list_UnetCNN = utils.compute_mean_metrics("./Estimate_result/UnetCNN_evaluate/MSE/train", compute_averages=True, metric="SIR")
    # inst_list_NMF = utils.compute_mean_metrics("./Estimate_result/NMF_evaluate/train", compute_averages=True, metric="SAR")
    # inst_list_RNN_rectified = utils.compute_mean_metrics("./Estimate_result/RNN_rectified_evaluate/MSE_SIR/train", compute_averages=True, metric="SDR")
    # print("一维卷积",inst_list_CNN_1d)
    # print("循环",inst_list_RNN)
    # print("Unet",inst_list_UnetCNN) 
    # print("循环2",inst_list_RNN_rectified) 
    # print("NMF",inst_list_NMF)  
    # utils.draw_sdr("./Estimate_result/RNN_evaluate/train")