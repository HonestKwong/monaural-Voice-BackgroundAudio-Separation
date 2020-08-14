#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    该文件使用NMF方法对单声道语音进行语音分离
    @author: KuangHaoheng
    Sichuan University
    Tel:13547937661
"""

#%%
#导入所有扩展包
import scipy,sklearn
import librosa
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import Constant
import os

#%%
def SaveAudio_1D(file_path, audio) :    
    librosa.output.write_wav(file_path,audio,sr=Constant.Sample_rate,norm=False)
    print("Save complete!!")
    
#%%
file_path = "./MIR-1K/Wavfile"
#file_path = "./MIR-1K/MIX"
filelist = librosa.util.find_files(file_path,ext="wav")
Audio_path = filelist[10]
Audio_name = os.path.split(Audio_path)[-1]
print ("processing", Audio_name)
Music_name = Audio_name.split('.')[0]


#%%
x, sample_rate = librosa.load(Audio_path, sr=8192)
x_length=len(x)
IPython.display.Audio(x,rate=sample_rate)
#%%
librosa.display.waveplot(x[0:512], sr=8192)
#%%

windows_length = int(2 ** (np.ceil(np.log2(Constant.DEFAULT_WIN_LEN_PARAM * sample_rate))))
hop_length = int(windows_length//2)
# 对时域的语音信号进行短时傅里叶变换
Spectrogram = librosa.stft(x, win_length=windows_length, hop_length=hop_length, window='hamm')
plt.figure()
log_Spectrogram = librosa.amplitude_to_db(np.abs(Spectrogram))
librosa.display.specshow(log_Spectrogram, sr=sample_rate, x_axis='time', y_axis='log')


#%%
X = np.absolute(Spectrogram)
# 进行非负矩阵分解，其中使用的损失函数为frobenius，并将其降到20维
n_components = 16
W, H = librosa.decompose.decompose(X, n_components=n_components, sort=True)
print ("W矩阵的结构大小：", W.shape)
print ("H矩阵的结构大小：",H.shape)

#%%
# 对基矩阵提取其梅尔倒谱系数MFCC来提取人耳蜗感兴趣频率成分的声音
n_mfcc = 24
mfcc_start = 1
mfcc_end = 12 
cluster_templates = librosa.feature.mfcc(S=W, n_mfcc=n_mfcc)[mfcc_start:mfcc_end]
print (cluster_templates.shape)

#%%
n_sources = 2
clusterer = sklearn.cluster.KMeans(n_clusters=n_sources)
clusterer.fit_transform(cluster_templates.T)
labeled_templates = clusterer.labels_
print (labeled_templates.shape)

#%%
uncollated_masks = []
for source_index in range(n_sources):
    source_indices = np.where(labeled_templates == source_index)[0]
    # 用于存放语音分离用的掩码的掩码
    W_mask = np.zeros_like(W)
    H_mask = np.zeros_like(H)
    # 保留由聚类确定的每个源的值
    for idx in source_indices:
        W_mask[:, idx] = W[:, idx]
        H_mask[idx, :] = H[idx, :]
    mask_matrix = W_mask.dot(H_mask)
    music_stft_max = np.maximum(mask_matrix, np.abs(Spectrogram))
    mask_matrix = np.divide(mask_matrix, music_stft_max)  #点除
    mask = np.nan_to_num(mask_matrix)                  #去掉不必要的非数或无限大的数字
    uncollated_masks.append(mask)

# print(uncollated_masks)

collated_masks = [np.dstack([uncollated_masks[s + ch*n_sources]for ch in range(1)])  #整理的掩码
                for s in range(n_sources)]

print(collated_masks)


#%%
result_masks = []
for mask in collated_masks:
    mask = np.round(mask)
    result_masks.append(mask)

sources = []
for masked in result_masks:
    masked_stft = Spectrogram * (np.squeeze(masked, axis=(2,))) 
    source = librosa.istft(masked_stft, win_length=windows_length, hop_length=hop_length, window='hamm',length=x_length)
    sources.append(source) 
vocal_path = os.path.join('.', 'outputs', 'NMF', "vocals" , Music_name+'_vocal.wav')
music_path = os.path.join('.', 'outputs', 'NMF', "accompaniment" , Music_name+'_music.wav')
SaveAudio_1D(vocal_path, np.asarray(sources[0]))
SaveAudio_1D(music_path, np.asarray(sources[1]))

#%%
IPython.display.Audio(sources[0],rate=sample_rate)
#%%

IPython.display.Audio(sources[1],rate=sample_rate)
