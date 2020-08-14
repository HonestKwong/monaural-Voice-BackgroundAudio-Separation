#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用已经训练好的神经网络模型对语音进行分离，包含DRNN、U_net_CNN以及CNN_1D分离方法
@author: KuangHaoheng
Sichuan University
Tel:13547937661
"""  
import librosa
import os
import numpy as np
import tensorflow as tf
import Constant
import utils
from network import DRNN, U_netCNN, CNN_1D, DRNN_Output_rectified
import tkinter.filedialog

def predict_DRNN():
    input_file_name = tkinter.filedialog.askopenfilename()
    # file_path = "./MIR-1K/Mix_Full"
    # filelist = librosa.util.find_files(file_path,ext="wav")
    Audio_path = input_file_name
    Audio_name = os.path.split(Audio_path)[-1]
    print ("processing", Audio_name)
    Music_name = Audio_name.split('.')[0]

    loss_type = "MSE+SIR"
    if loss_type == "MSE":
        if not os.path.exists("./outputs/RNN_outputs/MSE/accompaniment"):
            os.mkdir("./outputs/RNN_outputs/MSE/accompaniment")
        if not os.path.exists("./outputs/RNN_outputs/MSE/vocals"):
            os.mkdir("./outputs/RNN_outputs/MSE/vocals") 
        tensorboard_directory = './model_cpk/model_RNN/MSE/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_RNN/MSE'
        model_filename = 'DRNN_MSE.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'RNN_outputs','MSE', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'RNN_outputs','MSE', "accompaniment" , Music_name+'_music.wav')
    elif loss_type == "MSE+SIR":
        if not os.path.exists("./outputs/RNN_outputs/MSE_SIR/accompaniment"):
            os.mkdir("./outputs/RNN_outputs/MSE_SIR/accompaniment")
        if not os.path.exists("./outputs/RNN_outputs/MSE_SIR/vocals"):
            os.mkdir("./outputs/RNN_outputs/MSE_SIR/vocals")
        tensorboard_directory = './model_cpk/model_RNN/MSE_SIR/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_RNN/MSE_SIR'
        model_filename = 'DRNN_MSE_SIR.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'RNN_outputs','MSE_SIR', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'RNN_outputs','MSE_SIR', "accompaniment" , Music_name+'_music.wav')

     
    n_fft = 1024
    dropout_rate = 0.95
    num_hidden_units = Constant.num_hidden_units

    model_filepath = os.path.join(model_directory, model_filename)

    model = DRNN(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load_predict(file_dir = model_filepath)
    mix_wav_mag1, mix_wav_mag2, mix_wav_phase, audio_length = utils.LoadAudio(Audio_path)
    X = mix_wav_mag1
    X_width = np.array(X).shape[1]
    X_hight = np.array(X).shape[0]
    # X = tf.reshape(X,[1,X_hight,X_width])  
    X = np.array(X).transpose()
    X = np.array(X).reshape([1,X_width,X_hight])  

    MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred = model.test(x_source_mix = X, dropout_rate = dropout_rate)
    music_mask = MASK_acc.transpose()
    vocal_mask = MASK_vox.transpose()
    Mag_vox = y_voice_src_pred.transpose()
    Mag_acc = y_music_src_pred.transpose()
    
    
    utils.SaveAudio(vocal_path, Mag_vox[:,:,0],mix_wav_phase, audio_length=audio_length)
    utils.SaveAudio(music_path, Mag_acc[:,:,0],mix_wav_phase, audio_length=audio_length)

    # utils.SaveAudio(vocal_path,vocal_mask*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)
    # utils.SaveAudio(music_path,music_mask*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)

def predict_Unet_CNN():
    input_file_name = tkinter.filedialog.askopenfilename()
    # file_path = "./MIR-1K/Mix_Full"
    # filelist = librosa.util.find_files(file_path,ext="wav")
    # Audio_path = filelist[100]
    Audio_path = input_file_name
    Audio_name = os.path.split(Audio_path)[-1]
    print ("processing", Audio_name)
    Music_name = Audio_name.split('.')[0]

    loss_type = "L1"
    if loss_type == "MSE":
        if not os.path.exists("./outputs/UnetCNN_outputs/MSE/accompaniment"):
            os.mkdir("./outputs/UnetCNN_outputs/MSE/accompaniment")
        if not os.path.exists("./outputs/UnetCNN_outputs/MSE/vocals"):
            os.mkdir("./outputs/UnetCNN_outputs/MSE/vocals") 
        tensorboard_directory = './model_cpk/model_CNN/MSE/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN/MSE'
        model_filename = 'UNETCNN_MSE.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','MSE', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','MSE', "accompaniment" , Music_name+'_music.wav')

    elif loss_type == "MSE+SIR":
        if not os.path.exists("./outputs/UnetCNN_outputs/MSE_SIR/accompaniment"):
            os.mkdir("./outputs/UnetCNN_outputs/MSE_SIR/accompaniment")
        if not os.path.exists("./outputs/UnetCNN_outputs/MSE_SIR/vocals"):
            os.mkdir("./outputs/UnetCNN_outputs/MSE_SIR/vocals")
        tensorboard_directory = './model_cpk/model_CNN/MSE_SIR/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN/MSE_SIR'
        model_filename = 'UNETCNN_MSE_SIR.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','MSE_SIR', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','MSE_SIR', "accompaniment" , Music_name+'_music.wav')

    elif loss_type == "L1":
        if not os.path.exists("./outputs/UnetCNN_outputs/L1/accompaniment"):
            os.mkdir("./outputs/UnetCNN_outputs/L1/accompaniment")
        if not os.path.exists("./outputs/UnetCNN_outputs/L1/vocals"):
            os.mkdir("./outputs/UnetCNN_outputs/L1/vocals")
        tensorboard_directory = './model_cpk/model_CNN/L1/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN/L1'
        model_filename = 'UNETCNN_L1.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','L1', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','L1', "accompaniment" , Music_name+'_music.wav')

    elif loss_type == "L1+SIR":
        if not os.path.exists("./outputs/UnetCNN_outputs/L1_SIR/accompaniment"):
            os.mkdir("./outputs/UnetCNN_outputs/L1_SIR/accompaniment")
        if not os.path.exists("./outputs/UnetCNN_outputs/L1_SIR/vocals"):
            os.mkdir("./outputs/UnetCNN_outputs/L1_SIR/vocals")
        tensorboard_directory = './model_cpk/model_CNN/L1_SIR/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN/L1_SIR'
        model_filename = 'UNETCNN_L1_SIR.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','L1_SIR', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'UnetCNN_outputs','L1_SIR', "accompaniment" , Music_name+'_music.wav')        

    model_filepath = os.path.join(model_directory, model_filename)
    dropout_rate = 0.5
    model = U_netCNN(tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load_predict(file_dir = model_filepath)
    mix_wav_mag1, mix_wav_mag2, mix_wav_phase, audio_length = utils.LoadAudio(Audio_path)
    spectrogram_height = mix_wav_mag1.shape[0]
    spectrogram_length = mix_wav_mag1.shape[1]

    num_frame = spectrogram_length // Constant.patch_size
    Result_Mask = []
    for i in range(num_frame+1):
        START = i * Constant.patch_size
        END = START + Constant.patch_size  # 11 seconds
        #
        mix_wav_mag1_process = mix_wav_mag1[:, START:END]
        mix_wav_mag2_process = mix_wav_mag2[:, START:END]
        if i == num_frame:
            mix_wav_mag1_process = mix_wav_mag1[:, (spectrogram_length-Constant.patch_size):spectrogram_length]
            mix_wav_mag2_process = mix_wav_mag2[:, (spectrogram_length-Constant.patch_size):spectrogram_length]
        #mix_wav_phase = mix_wav_phase[:, START:END]
        
        X = mix_wav_mag2_process[1:].reshape(1,512,128,1)
        MASK_vox = model.test(x_source_mix = X, dropout_rate = dropout_rate, istraining = False)

        Bianry_mask = False  #是否使用2进制掩码    
        target_pred_mag = np.vstack((np.zeros((128)), np.squeeze(MASK_vox)))
        if Bianry_mask:
            target_pred_mag[target_pred_mag>0.5] = 1  
            target_pred_mag[target_pred_mag<0.5] = 0        
        if i == 0:
            Result_Mask = target_pred_mag
        elif i == num_frame:
            Result_Mask = np.c_[Result_Mask, target_pred_mag[:,(Constant.patch_size-(spectrogram_length-START)):Constant.patch_size]]
        else:
            Result_Mask = np.c_[Result_Mask, target_pred_mag]   

    
    utils.SaveAudio(vocal_path, Result_Mask*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)
    utils.SaveAudio(music_path, (1-Result_Mask)*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)

    # utils.SaveAudio(vocal_path,vocal_mask*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)
    # utils.SaveAudio(music_path,music_mask*mix_wav_mag1,mix_wav_phase, audio_length=audio_length)

def predict_CNN_1D():
    input_file_name = tkinter.filedialog.askopenfilename()
    # file_path = "./MIR-1K/Mix_Full"
    # filelist = librosa.util.find_files(file_path,ext="wav")
    # Audio_path = filelist[100]
    Audio_path = input_file_name
    Audio_name = os.path.split(Audio_path)[-1]
    Music_name = Audio_name.split('.')[0]
    print ("processing(CNN_1D)", Audio_name)

    loss_type = "MSE+SIR"
    if loss_type == "MSE":
        if not os.path.exists("./outputs/CNN_1D_outputs/MSE/accompaniment"):
            os.mkdir("./outputs/CNN_1D_outputs/MSE/accompaniment")
        if not os.path.exists("./outputs/CNN_1D_outputs/MSE/vocals"):
            os.mkdir("./outputs/CNN_1D_outputs/MSE/vocals") 
        tensorboard_directory = './model_cpk/model_CNN_1D/MSE/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN_1D/MSE'
        model_filename = 'CNN_1D_MSE.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'CNN_1D_outputs','MSE', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'CNN_1D_outputs','MSE', "accompaniment" , Music_name+'_music.wav')
    elif loss_type == "MSE+SIR":
        if not os.path.exists("./outputs/CNN_1D_outputs/MSE_SIR/accompaniment"):
            os.mkdir("./outputs/CNN_1D_outputs/MSE_SIR/accompaniment")
        if not os.path.exists("./outputs/CNN_1D_outputs/MSE_SIR/vocals"):
            os.mkdir("./outputs/CNN_1D_outputs/MSE_SIR/vocals")
        tensorboard_directory = './model_cpk/model_CNN_1D/MSE_SIR/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_CNN_1D/MSE_SIR'
        model_filename = 'CNN_1D_MSE_SIR.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'CNN_1D_outputs','MSE_SIR', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'CNN_1D_outputs','MSE_SIR', "accompaniment" , Music_name+'_music.wav')


    time_start = time.clock()
    audio, sr = librosa.load(Audio_path,sr=Constant.Sample_rate)

    training = False
 
    model_filepath = os.path.join(model_directory, model_filename)
    num_frames = 16384
    disc_input_shape = [1, num_frames, 0]

    sep_input_shape, sep_output_shape = utils.get_padding(np.array(disc_input_shape))

    mix_length = len(audio)

    if mix_length < sep_input_shape[1]:
        extra_pad = sep_input_shape[1] - mix_length
        audio = np.pad(audio, [(0, extra_pad), (0,0)], mode="constant", constant_values=0.0)
    else:
        extra_pad = 0

    # 预分配源预测（与输入混合物形状相同）
    source_time_frames = mix_length

    input_time_frames = sep_input_shape[1]
    output_time_frames = sep_output_shape[1]

    # 在开始和结束时通过时间扩充混合音频，以便神经网络可以在信号的开始和结束时进行预测
    pad_time_frames = int((input_time_frames - output_time_frames)/2)
    mix_audio_padded = np.pad(audio, [pad_time_frames, pad_time_frames], mode="constant", constant_values=0.0)

    model = CNN_1D(sep_input_shape, sep_output_shape, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load_predict(file_dir = model_filepath)
    music_pred_full = []
    voice_pred_full = []
    # 迭代混合音乐的值，获取网络预测的分离音乐
    for source_pos in range(0, source_time_frames, output_time_frames):
        # 如果此输出补丁将超过源频谱图的末尾，设置它以便我们预测输出的结尾，然后停止
        if source_pos + output_time_frames > source_time_frames:
            source_pos = source_time_frames - output_time_frames

        # 通过选择时间间隔将混合音乐送入神经网络模型进行预测
        mix_part = mix_audio_padded[source_pos:source_pos + input_time_frames]
        mix_part = np.expand_dims(mix_part, axis=0)
        mix_part = np.expand_dims(mix_part, axis=2)
        y_voice_src_pred, y_music_src_pred = model.test(x_source_mix = mix_part, training = training)

        # 保存预测值
        music_pred_full[source_pos:source_pos + output_time_frames] = y_music_src_pred[0, :, :]
        voice_pred_full[source_pos:source_pos + output_time_frames] = y_voice_src_pred[0, :, :]

    # 如果之前我们在最后填充混合物，现在从预测中删除这些样本
    if extra_pad > 0:
        music_pred_full = music_pred_full[:-extra_pad,:]
        voice_pred_full = voice_pred_full[:-extra_pad,:]
    
    
    utils.SaveAudio_1D(vocal_path, np.asarray(voice_pred_full))
    utils.SaveAudio_1D(music_path, np.asarray(music_pred_full))
    time_flies = (time.clock() - time_start)        
    print ("CNN_1D分离音乐", Music_name, "耗时：", time_flies,"s")


def predict_DRNN_outputs():
    input_file_name = tkinter.filedialog.askopenfilename()
    # file_path = "./MIR-1K/Mix_Full"
    # filelist = librosa.util.find_files(file_path,ext="wav")
    # Audio_path = filelist[1]
    Audio_path = input_file_name
    Audio_name = os.path.split(Audio_path)[-1]
    print ("processing", Audio_name)
    Music_name = Audio_name.split('.')[0]

    loss_type = "MSE"
    if loss_type == "MSE":
        if not os.path.exists("./outputs/RNN_rectified_outputs/MSE/accompaniment"):
            os.mkdir("./outputs/RNN_rectified_outputs/MSE/accompaniment")
        if not os.path.exists("./outputs/RNN_rectified_outputs/MSE/vocals"):
            os.mkdir("./outputs/RNN_rectified_outputs/MSE/vocals") 
        tensorboard_directory = './model_cpk/model_RNN_outputs/MSE/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_RNN_outputs/MSE'
        model_filename = 'DRNN_MSE_outputs.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'RNN_rectified_outputs','MSE', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'RNN_rectified_outputs','MSE', "accompaniment" , Music_name+'_music.wav')
    elif loss_type == "MSE+SIR":
        if not os.path.exists("./outputs/RNN_rectified_outputs/MSE_SIR/accompaniment"):
            os.mkdir("./outputs/RNN_rectified_outputs/MSE_SIR/accompaniment")
        if not os.path.exists("./outputs/RNN_rectified_outputs/MSE_SIR/vocals"):
            os.mkdir("./outputs/RNN_rectified_outputs/MSE_SIR/vocals")
        tensorboard_directory = './model_cpk/model_RNN_outputs/MSE_SIR/tensorboard'
        clear_tensorboard = False
        model_directory = './model_cpk/model_RNN_outputs/MSE_SIR'
        model_filename = 'DRNN_outputs_MSE_SIR.ckpt'
        vocal_path = os.path.join('.', 'outputs', 'RNN_rectified_outputs','MSE_SIR', "vocals" , Music_name+'_vocal.wav')
        music_path = os.path.join('.', 'outputs', 'RNN_rectified_outputs','MSE_SIR', "accompaniment" , Music_name+'_music.wav')

     
    n_fft = 1024
    dropout_rate = 0.95
    num_hidden_units = Constant.num_hidden_units

    model_filepath = os.path.join(model_directory, model_filename)

    model = DRNN_Output_rectified(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load_predict(file_dir = model_filepath)
    mix_wav_mag1, mix_wav_mag2, mix_wav_phase, audio_length = utils.LoadAudio(Audio_path)
    X = mix_wav_mag1
    X_width = np.array(X).shape[1]
    X_hight = np.array(X).shape[0]
    # X = tf.reshape(X,[1,X_hight,X_width])  
    X = np.array(X).transpose()
    X = np.array(X).reshape([1,X_width,X_hight])  

    MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred = model.test(x_source_mix = X, dropout_rate = dropout_rate)
    music_mask = MASK_acc.transpose()
    vocal_mask = MASK_vox.transpose()
    Mag_vox = y_voice_src_pred.transpose()
    Mag_acc = y_music_src_pred.transpose()
    
    
    utils.SaveAudio(vocal_path, Mag_vox[:,:,0],mix_wav_phase, audio_length=audio_length)
    utils.SaveAudio(music_path, Mag_acc[:,:,0],mix_wav_phase, audio_length=audio_length)

if __name__ == "__main__":
    predict_Unet_CNN()
    # predict_CNN_1D()
    # predict_DRNN()
    # predict_DRNN_outputs()