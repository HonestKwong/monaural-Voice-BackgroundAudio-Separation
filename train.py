#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对数据集进行RNN、UnetCNN以及CNN_1D训练模型
@author: KuangHaoheng
Sichuan University
Tel:13547937661
"""  
import tensorflow as tf
from network import DRNN, U_netCNN, CNN_1D, DRNN_Output_rectified
from librosa.util import find_files
import numpy as np
import utils
import Constant
import os

def train_RNN():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #
    n_fft = 1024
    # 步幅;帧移对应卷积中的stride;
    hop_length = n_fft // 4    
    # 学习率
    learning_rate =0.0001
    # 用于创建rnn节点数
    num_hidden_units = [1024, 1024, 1024, 1024, 1024]
    # batch 长度
    batch_size = 64
    # 获取多少帧数据
    sample_frames = 10
    # 训练迭代次数
    EPOCH = Constant.RNN_EPOCH
    # 随机失活率
    dropout_rate = 0.95
    clear_tensorboard = False

    # loss_type有"MSE","DKL","DKL+SIR", "MSE+SIR",
    loss_type = "MSE+recognize"
    

    log_directory = './loss_pic'

    os.path.join('.', 'model_cpk', 'model_RNN', 'model_RNN'+loss_type)
    if not os.path.exists("./model_cpk/model_RNN"):             #如果不存在文件夹则创建相应存放数据的文件夹
        os.mkdir("./model_cpk/model_RNN")  

    # 模型保存路径
    if loss_type == "L1":
        model_dir = "./model_cpk/model_RNN/L1"
        tensorboard_directory = "./model_cpk/model_RNN/L1/tensorboard"
        model_filename = 'DRNN_L1.ckpt'
        train_log_filename = 'train_loss_DRNN_L1.csv' 

    elif loss_type == "L1+SIR":
        model_dir = "./model_cpk/model_RNN/L1_SIR"
        tensorboard_directory = "./model_cpk/model_RNN/L1_SIR/tensorboard"
        model_filename = 'DRNN_L1_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_L1_SIR.csv' 

    elif loss_type == "MSE":
        model_dir = "./model_cpk/model_RNN/MSE"
        tensorboard_directory = "./model_cpk/model_RNN/MSE/tensorboard"
        model_filename = 'DRCNN_MSE.ckpt'
        train_log_filename = 'train_loss_DRNN_MSE.csv' 

    elif loss_type == "DKL":
        model_dir = "./model_cpk/model_RNN/DKL"
        tensorboard_directory = "./model_cpk/model_RNN/DKL/tensorboard"
        model_filename = 'DRNN_DKL.ckpt'
        train_log_filename = 'train_loss_DRNN_DKL.csv'

    elif loss_type == "MSE+SIR":
        model_dir = "./model_cpk/model_RNN/MSE_SIR"
        tensorboard_directory = "./model_cpk/model_RNN/MSE_SIR/tensorboard"
        model_filename = 'DRNN_MSE_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_MSE_SIR.csv'

    elif loss_type == "DKL+SIR":
        model_dir = "./model_cpk/model_RNN/DKL_SIR"
        tensorboard_directory = "./model_cpk/model_RNN/DKL_SIR/tensorboard"
        model_filename = 'DRNN_DKL_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_DKL_SIR.csv'
    elif loss_type == "MSE+recognize":
        model_dir = "./model_cpk/model_RNN/MSE_recognize"
        tensorboard_directory = "./model_cpk/model_RNN/MSE_recognize/tensorboard"
        model_filename = 'DRCNN_MSE_recognize.ckpt'
        train_log_filename = 'train_loss_DRNN_MSE_recognize.csv' 


    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()    
    if loss_type == "MSE+recognize":
        MIX_list, Vocal_list, Music_list = utils.LoadDataset_RNN(file_Path="./MIR-1K/Unet_recognition_FFT/Train")          # 加载混合语谱图
    else:
        MIX_list, Vocal_list, Music_list = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNN_FFT/Train")          # 加载混合语谱图

    MIX_mag, MIX_phase = utils.Magnitude_phase(MIX_list) #得到幅度相位
    Vocal_mag, Vocal_phase = utils.Magnitude_phase(Vocal_list)      
    Music_mag, Music_phase = utils.Magnitude_phase(Music_list)

    if loss_type == "MSE+recognize":
        MIX_list_Valide, Vocal_list_Valide, Music_list_Valide = utils.LoadDataset_RNN(file_Path="./MIR-1K/Unet_recognition_FFT/Validation") # 加载混合语谱图
    else:
        MIX_list_Valide, Vocal_list_Valide, Music_list_Valide = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNN_FFT/Validation") # 加载混合语谱图
           
    MIX_mag_Valide, MIX_phase_Valide = utils.Magnitude_phase(MIX_list_Valide) #得到幅度相位
    Vocal_mag_Valide, Vocal_phase_Valide = utils.Magnitude_phase(Vocal_list_Valide)      
    Music_mag_Valide, Music_phase_Valide = utils.Magnitude_phase(Music_list_Valide)

    DRNN_model =  DRNN(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units, 
                    tensorboard_directory = tensorboard_directory, clear_tensorboard = False, loss_type = loss_type)
    
    startepo = DRNN_model.load(file_dir = model_dir)
    print('startepo:' + str(startepo))
    print('loss_type: ' + loss_type)
    for e in range(EPOCH) :
        # 随机为训练采样，因为卷积神经网络输入为(512,128)大小的语谱图
        print('当前训练代数：',e+1)
        if loss_type == "MSE+recognize":
            sample_frames = 128
        else:
            sample_frames = 10
        X = list()
        y_accompaniment = list()
        y_vocals = list()
        for mix, music, vocal in zip(MIX_mag, Music_mag, Vocal_mag):
            num_frames = np.array(mix).shape[1]
            start = np.random.randint(num_frames - sample_frames + 1)
            end = start + sample_frames
            X.append(mix[:,start:end])
            y_accompaniment.append(music[:,start:end])
            y_vocals.append(vocal[:,start:end])

        X = np.array(X)
        y_accompaniment = np.array(y_accompaniment)
        y_vocals = np.array(y_vocals)

        X = X.transpose((0, 2, 1))
        y_accompaniment = y_accompaniment.transpose((0, 2, 1))
        y_vocals = y_vocals.transpose((0, 2, 1))

        train_loss = DRNN_model.train(x_source_mix = X, y_source_acc = y_accompaniment, y_source_vox = y_vocals,
                                 learning_rate = learning_rate, dropout_rate = dropout_rate)       
        if e % 10 == 0:
            print('Step: %d Train Loss: %f' %(e, train_loss))


        if e % 200 == 0:
            #这里是测试模型准确率的
            print('==============================================')
            X = []
            y_accompaniment = []
            y_vocals = []
            for mix, music, vocal in zip(MIX_mag_Valide, Music_mag_Valide, Vocal_mag_Valide):
                num_frames = np.array(mix).shape[1]
                start = np.random.randint(num_frames - sample_frames + 1)
                end = start + sample_frames
                X.append(mix[:,start:end])
                y_accompaniment.append(music[:,start:end])
                y_vocals.append(vocal[:,start:end])

            X = np.array(X)
            y_accompaniment = np.array(y_accompaniment)
            y_vocals = np.array(y_vocals)

            X = X.transpose((0, 2, 1))
            y_accompaniment = y_accompaniment.transpose((0, 2, 1))
            y_vocals = y_vocals.transpose((0, 2, 1))

            y_music_src_pred, y_voice_src_pred, validate_loss = DRNN_model.validate(x_source_mix = X,
                    y_source_acc = y_accompaniment, y_source_vox = y_vocals, dropout_rate = dropout_rate)

            print('Step: %d Test Loss: %f' %(e, validate_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(e, train_loss, validate_loss))        
        
        if e % 10 == 0:
            DRNN_model.save(directory = model_dir, filename = model_filename, global_step=e)    
                   
def train_U_net_CNN():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 学习率
    learning_rate = 1e-4
    # batch 长度
    batch_size = 16
    # 训练迭代次数
    EPOCH = Constant.EPOCH
    # 随机失活率
    dropout_rate = 0.5
    clear_tensorboard = False
    # 训练损失函数类型
    loss_type = "L1+recognize"   
    log_directory = './loss_pic'

    os.path.join('.', 'model_cpk', 'model_CNN', 'model_CNN'+loss_type)
    if not os.path.exists("./model_cpk/model_CNN"):             #如果不存在文件夹则创建相应存放数据的文件夹
        os.mkdir("./model_cpk/model_CNN")  

    # 模型保存路径
    if loss_type == "L1":
        model_dir = "./model_cpk/model_CNN/L1"
        tensorboard_directory = "./model_cpk/model_CNN/L1/tensorboard"
        model_filename = 'UNETCNN_L1.ckpt'
        train_log_filename = 'train_loss_UnetCNN_L1.csv' 

    elif loss_type == "L1+SIR":
        model_dir = "./model_cpk/model_CNN/L1_SIR"
        tensorboard_directory = "./model_cpk/model_CNN/L1_SIR/tensorboard"
        model_filename = 'UNETCNN_L1_SIR.ckpt'
        train_log_filename = 'train_loss_UnetCNN_L1_SIR.csv' 

    elif loss_type == "MSE":
        model_dir = "./model_cpk/model_CNN/MSE"
        tensorboard_directory = "./model_cpk/model_CNN/MSE/tensorboard"
        model_filename = 'UNETCNN_MSE.ckpt'
        train_log_filename = 'train_loss_UnetCNN_MSE.csv' 

    elif loss_type == "DKL":
        model_dir = "./model_cpk/model_CNN/DKL"
        tensorboard_directory = "./model_cpk/model_CNN/DKL/tensorboard"
        model_filename = 'UNETCNN_DKL.ckpt'
        train_log_filename = 'train_loss_UnetCNN_DKL.csv'

    elif loss_type == "MSE+SIR":
        model_dir = "./model_cpk/model_CNN/MSE_SIR"
        tensorboard_directory = "./model_cpk/model_CNN/MSE_SIR/tensorboard"
        model_filename = 'UNETCNN_MSE_SIR.ckpt'
        train_log_filename = 'train_loss_UnetCNN_MSE_SIR.csv'

    elif loss_type == "DKL+SIR":
        model_dir = "./model_cpk/model_CNN/DKL_SIR"
        tensorboard_directory = "./model_cpk/model_CNN/DKL_SIR/tensorboard"
        model_filename = 'UNETCNN_DKL_SIR.ckpt'
        train_log_filename = 'train_loss_UnetCNN_DKL_SIR.csv'

    elif loss_type == "L1+recognize":
        model_dir = "./model_cpk/model_CNN/L1_recognize"
        tensorboard_directory = "./model_cpk/model_CNN/L1_recognize/tensorboard"
        model_filename = 'UNETCNN_L1_recognize.ckpt'
        train_log_filename = 'train_loss_UnetCNN_L1_recognize.csv' 

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()    

    if loss_type == "L1+recognize":
        X_list, Y_list = utils.LoadDataset(target="vocal", FFT_Path="./MIR-1K/Unet_recognition_FFT/Train")          # 加载混合语谱图
        X_mag,X_phase = utils.Magnitude_phase(X_list) #得到幅度相位
        Y_mag,_ = utils.Magnitude_phase(Y_list)       #得到

        X_list_Validate, Y_list_Validate = utils.LoadDataset(target="vocal", FFT_Path="./MIR-1K/Unet_recognition_FFT/Validation")          # 加载混合语谱图
        X_mag_Validate,X_phase_Validate = utils.Magnitude_phase(X_list_Validate) #得到幅度相位
        Y_mag_Validate,_ = utils.Magnitude_phase(Y_list_Validate)       #得到
    else:
        X_list, Y_list = utils.LoadDataset(target="vocal", FFT_Path="./MIR-1K/UnetFFT_Full/Train")          # 加载混合语谱图
        X_mag,X_phase = utils.Magnitude_phase(X_list) #得到幅度相位
        Y_mag,_ = utils.Magnitude_phase(Y_list)       #得到

        X_list_Validate, Y_list_Validate = utils.LoadDataset(target="vocal", FFT_Path="./MIR-1K/UnetFFT_Full/Validation")          # 加载混合语谱图
        X_mag_Validate,X_phase_Validate = utils.Magnitude_phase(X_list_Validate) #得到幅度相位
        Y_mag_Validate,_ = utils.Magnitude_phase(Y_list_Validate)       #得到

    UnetCNN_model =  U_netCNN(tensorboard_directory = tensorboard_directory, clear_tensorboard = False, loss_type = loss_type)
    
    startepo = UnetCNN_model.load(file_dir = model_dir)
    print('startepo:' + str(startepo))

    for e in range(EPOCH) :
        # 随机为训练采样，因为卷积神经网络输入为(512,128)大小的语谱图
        print('当前训练代数：',e+1)        
        # X,y = utils.sampling(X_mag,Y_mag)     
        X,y = utils.sampling(X_mag,Y_mag)  
        train_loss = UnetCNN_model.train(x_source_mix = X, y_source_vox = y,
                                 learning_rate = learning_rate, dropout_rate = dropout_rate, istraining = True)       
        if e % 10 == 0:
            print('Step: %d Train Loss: %f' %(e, train_loss))
        if e % 200 == 0:
            #这里是测试模型准确率的
            print('==============================================')

            X,y = utils.sampling(X_mag_Validate,Y_mag_Validate) 
            validate_loss = UnetCNN_model.validate(x_source_mix = X, y_source_vox = y, dropout_rate = dropout_rate, istraining = False)

            print('Step: %d Test Loss: %f' %(e, validate_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(e, train_loss, validate_loss))        
        
        if e % 10 == 0:
            UnetCNN_model.save(directory = model_dir, filename = model_filename, global_step=e)   

def train_CNN_1D():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 学习率
    learning_rate = 1e-4
    # batch 长度
    batch_size = 16
    num_frames = 16384
    # 训练迭代次数
    EPOCH = Constant.EPOCH
    # 随机失活率
    clear_tensorboard = False
    # loss_type有"MSE","DKL","DKL+SIR", "MSE+SIR",
    loss_type = "MSE+SIR+recognize"

    os.path.join('.', 'model_cpk', 'model_CNN_1D', 'model_CNN_1D'+loss_type)
    if not os.path.exists("./model_cpk/model_CNN_1D"):             #如果不存在文件夹则创建相应存放数据的文件夹
        os.mkdir("./model_cpk/model_CNN_1D")  

    # 模型保存路径
    if loss_type == "MSE":
        model_dir = "./model_cpk/model_CNN_1D/MSE"
        tensorboard_directory = "./model_cpk/model_CNN_1D/MSE/tensorboard"
        model_filename = 'CNN_1D_MSE.ckpt'
        train_log_filename = 'train_loss_CNN1D_MSE.csv'
  

    elif loss_type == "DKL":
        model_dir = "./model_cpk/model_CNN_1D/DKL"
        tensorboard_directory = "./model_cpk/model_CNN_1D/DKL/tensorboard"
        model_filename = 'CNN_1D_DKL.ckpt'
        train_log_filename = 'train_loss_CNN1D_DKL.csv'

    elif loss_type == "MSE+SIR":
        model_dir = "./model_cpk/model_CNN_1D/MSE_SIR"
        tensorboard_directory = "./model_cpk/model_CNN_1D/MSE_SIR/tensorboard"
        model_filename = 'CNN_1D_MSE_SIR.ckpt'
        train_log_filename = 'train_loss_CNN1D_MSE_SIR.csv'

    elif loss_type == "DKL+SIR":
        model_dir = "./model_cpk/model_CNN_1D/DKL_SIR"
        tensorboard_directory = "./model_cpk/model_CNN_1D/DKL_SIR/tensorboard"
        model_filename = 'CNN_1D_DKL_SIR.ckpt'
        train_log_filename = 'train_loss_CNN1D_DKL_SIR.csv'

    elif loss_type == "MSE+SIR+recognize":
        model_dir = "./model_cpk/model_CNN_1D/MSE_SIR_recognize"
        tensorboard_directory = "./model_cpk/model_CNN_1D/MSE_SIR_recognize/tensorboard"
        model_filename = 'CNN_1D_MSE_SIR_recognize.ckpt'
        train_log_filename = 'train_loss_CNN1D_MSE_SIR_recognize.csv'

    log_directory = './loss_pic'
    
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()  

    if loss_type == "MSE+SIR+recognize":
        MIR1K_train_path = "./MIR-1K/MIX_1D_recognize_npz/Train"    #添加MIR1K数据集切割后的路径
        MIR1K_valide_path = "./MIR-1K/MIX_1D_recognize_npz/Validation"    #添加MIR1K数据集切割后的路径
    else:
        MIR1K_train_path = "./MIR-1K/MIX_1D_npz/train"    #添加MIR1K数据集切割后的路径
        MIR1K_valide_path = "./MIR-1K/MIX_1D_npz/validation"    #添加MIR1K数据集切割后的路径

    MIX_train_list, ACC_train_list, VOX_train_list = utils.LoadDataset_CNN_1D(filepath=MIR1K_train_path)
    MIX_valide_list, ACC_valide_list, VOX_valide_list = utils.LoadDataset_CNN_1D(filepath=MIR1K_valide_path)

    disc_input_shape = [batch_size, num_frames, 0]
    sep_input_shape, sep_output_shape = utils.get_padding(np.array(disc_input_shape))

    CNN_1d_model =  CNN_1D(input_shape=sep_input_shape, output_shape=sep_output_shape, tensorboard_directory = tensorboard_directory, clear_tensorboard = False, loss_type = loss_type)
    
    startepo = CNN_1d_model.load(file_dir = model_dir)
    print('startepo:' + str(startepo))
    print('训练开始')
    for e in range(EPOCH) :
        # 随机为训练采样
        mix, vox, acc = utils.sampling_CNN_1D(MIX_valide_list, ACC_valide_list, VOX_valide_list, sep_input_shape, sep_output_shape)
        mix, vox, acc = utils.sampling_CNN_1D(MIX_train_list, ACC_train_list, VOX_train_list, sep_input_shape, sep_output_shape)      
        train_loss = CNN_1d_model.train(x_source_mix = mix, y_source_acc = acc, y_source_vox = vox
                                ,learning_rate = learning_rate, training = True)       
        if e % 10 == 0:
            print('当前训练代数：',e+1)  
            print('Step: %d Train Loss: %f' %(e, train_loss))
        if e % 200 == 0:
            #这里是测试模型准确率的
            print('==============================================')

            mix, vox, acc = utils.sampling_CNN_1D(MIX_valide_list, ACC_valide_list, VOX_valide_list, sep_input_shape, sep_output_shape)
            validate_loss = CNN_1d_model.validate(x_source_mix = mix, y_source_acc = acc, y_source_vox = vox, training = False)

            print('Step: %d Test Loss: %f' %(e, validate_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(e, train_loss, validate_loss))        
        
        if e % 10 == 0:
            CNN_1d_model.save(directory = model_dir, filename = model_filename, global_step=e)    


def train_RNN_outputs():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #
    n_fft = 1024
    # 步幅;帧移对应卷积中的stride;
    hop_length = n_fft // 4    
    # 学习率
    learning_rate =0.0001
    # 用于创建rnn节点数
    num_hidden_units = [1024, 1024, 1024, 1024, 1024]
    # batch 长度
    batch_size = 64
    # 获取多少帧数据
    sample_frames = 10
    # 训练迭代次数
    EPOCH = Constant.RNN_EPOCH
    # 随机失活率
    dropout_rate = 0.95
    clear_tensorboard = False

    # loss_type有"MSE","DKL","DKL+SIR", "MSE+SIR",
    loss_type = "MSE+SIR+recognize"
    

    log_directory = './loss_pic'

    os.path.join('.', 'model_cpk', 'model_RNN_outputs', 'model_RNN_outputs'+loss_type)
    if not os.path.exists("./model_cpk/model_RNN_outputs"):             #如果不存在文件夹则创建相应存放数据的文件夹
        os.mkdir("./model_cpk/model_RNN_outputs")  

    # 模型保存路径
    if loss_type == "L1":
        model_dir = "./model_cpk/model_RNN_outputs/L1"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/L1/tensorboard"
        model_filename = 'DRNN_outputs_L1.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_L1.csv' 

    elif loss_type == "L1+SIR":
        model_dir = "./model_cpk/model_RNN_outputs/L1_SIR"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/L1_SIR/tensorboard"
        model_filename = 'DRNN_outputs_L1_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_L1_SIR.csv' 

    elif loss_type == "MSE":
        model_dir = "./model_cpk/model_RNN_outputs/MSE"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/MSE/tensorboard"
        model_filename = 'DRNN_MSE_outputs.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_MSE.csv' 

    elif loss_type == "DKL":
        model_dir = "./model_cpk/model_RNN_outputs/DKL"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/DKL/tensorboard"
        model_filename = 'DRNN_outputs_DKL.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_DKL.csv'

    elif loss_type == "MSE+SIR":
        model_dir = "./model_cpk/model_RNN_outputs/MSE_SIR"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/MSE_SIR/tensorboard"
        model_filename = 'DRNN_outputs_MSE_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_MSE_SIR.csv'

    elif loss_type == "DKL+SIR":
        model_dir = "./model_cpk/model_RNN_outputs/DKL_SIR"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/DKL_SIR/tensorboard"
        model_filename = 'DRNN_outputs_DKL_SIR.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_DKL_SIR.csv'

    elif loss_type == "MSE+SIR+recognize":
        model_dir = "./model_cpk/model_RNN_outputs/MSE_SIR_recognize"
        tensorboard_directory = "./model_cpk/model_RNN_outputs/MSE_SIR_recognize/tensorboard"
        model_filename = 'DRNN_outputs_MSE_SIR_recognize.ckpt'
        train_log_filename = 'train_loss_DRNN_outputs_MSE_SIR_recognize.csv'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()    

    if loss_type == "MSE+SIR+recognize":
        MIX_list, Vocal_list, Music_list = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNNFFT_recognize/Train")          # 加载混合语谱图
    else:    
        MIX_list, Vocal_list, Music_list = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNN_FFT/Train")          # 加载混合语谱图
    MIX_mag, MIX_phase = utils.Magnitude_phase(MIX_list) #得到幅度相位
    Vocal_mag, Vocal_phase = utils.Magnitude_phase(Vocal_list)      
    Music_mag, Music_phase = utils.Magnitude_phase(Music_list)

    if loss_type == "MSE+SIR+recognize":
        MIX_list_Valide, Vocal_list_Valide, Music_list_Valide = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNNFFT_recognize/Validation")          # 加载混合语谱图
    else:
        MIX_list_Valide, Vocal_list_Valide, Music_list_Valide = utils.LoadDataset_RNN(file_Path="./MIR-1K/RNN_FFT/Validation")          # 加载混合语谱图
    MIX_mag_Valide, MIX_phase_Valide = utils.Magnitude_phase(MIX_list_Valide) #得到幅度相位
    Vocal_mag_Valide, Vocal_phase_Valide = utils.Magnitude_phase(Vocal_list_Valide)      
    Music_mag_Valide, Music_phase_Valide = utils.Magnitude_phase(Music_list_Valide)

    DRNN_model =  DRNN_Output_rectified(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units, 
                    tensorboard_directory = tensorboard_directory, clear_tensorboard = False, loss_type = loss_type)
    
    startepo = DRNN_model.load(file_dir = model_dir)
    print('startepo:' + str(startepo))
    print('loss_type: ' + loss_type)
    for e in range(EPOCH) :
        # 随机为训练采样，因为卷积神经网络输入为(512,128)大小的语谱图
        print('当前训练代数：',e+1)
        sample_frames = 10
        X = list()
        y_accompaniment = list()
        y_vocals = list()
        for i in range(8):
            for mix, music, vocal in zip(MIX_mag, Music_mag, Vocal_mag):
                num_frames = np.array(mix).shape[1]
                start = np.random.randint(num_frames - sample_frames + 1)
                end = start + sample_frames
                X.append(mix[:,start:end])
                y_accompaniment.append(music[:,start:end])
                y_vocals.append(vocal[:,start:end])

        X = np.array(X)
        y_accompaniment = np.array(y_accompaniment)
        y_vocals = np.array(y_vocals)

        X = X.transpose((0, 2, 1))
        y_accompaniment = y_accompaniment.transpose((0, 2, 1))
        y_vocals = y_vocals.transpose((0, 2, 1))

        train_loss = DRNN_model.train(x_source_mix = X, y_source_acc = y_accompaniment, y_source_vox = y_vocals,
                                 learning_rate = learning_rate, dropout_rate = dropout_rate)       
        if e % 10 == 0:
            print('Step: %d Train Loss: %f' %(e, train_loss))


        if e % 200 == 0:
            #这里是测试模型准确率的
            print('==============================================')
            X = []
            y_accompaniment = []
            y_vocals = []
            for i in range(8):
                for mix, music, vocal in zip(MIX_mag_Valide, Music_mag_Valide, Vocal_mag_Valide):
                    num_frames = np.array(mix).shape[1]
                    start = np.random.randint(num_frames - sample_frames + 1)
                    end = start + sample_frames
                    X.append(mix[:,start:end])
                    y_accompaniment.append(music[:,start:end])
                    y_vocals.append(vocal[:,start:end])

            X = np.array(X)
            y_accompaniment = np.array(y_accompaniment)
            y_vocals = np.array(y_vocals)

            X = X.transpose((0, 2, 1))
            y_accompaniment = y_accompaniment.transpose((0, 2, 1))
            y_vocals = y_vocals.transpose((0, 2, 1))

            y_music_src_pred, y_voice_src_pred, validate_loss = DRNN_model.validate(x_source_mix = X,
                    y_source_acc = y_accompaniment, y_source_vox = y_vocals, dropout_rate = dropout_rate)

            print('Step: %d Test Loss: %f' %(e, validate_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(e, train_loss, validate_loss))        
        
        if e % 10 == 0:
            DRNN_model.save(directory = model_dir, filename = model_filename, global_step=e)    
  
if __name__ == "__main__":
    # train_U_net_CNN()    
    # train_CNN_1D()
    train_RNN_outputs()
    # train_RNN()