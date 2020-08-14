#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存放神经网络模型的文件，包含DRNN类、U_net_CNN类以及CNN_1D类
@author: KuangHaoheng
Sichuan University
Tel:13547937661
"""  

import tensorflow as tf
# from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.keras.layers import concatenate 
import numpy as np
import utils
import os
import shutil
from datetime import datetime

#以下为循环神经网络DRNN类
"""
    属性：num_hidden_units：隐藏单元的数量
        tensorboard_directory:保存tensorboard训练数据地址
        clear_tensorboard：是否清除tensorboard
"""
class DRNN(object):


    def __init__(self, num_features, num_hidden_units=[256,256,256], tensorboard_directory = './RNN_model/tensorboard', clear_tensorboard = True, loss_type = "MSE"):
        self.loss_type = loss_type    
        self.gamma = 0.001  
        #设置特征数量
        self.num_features = num_features
        #循环神经网络有多少层
        self.num_rnn_layer = len(num_hidden_units)
        #有多少个隐藏单元
        self.num_hidden_units = num_hidden_units
        #设置训练步数变量
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

        #设置学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据、人声、背景音乐
        self.x_source_mix = tf.placeholder(tf.float32, shape=[None, None, num_features], name='x_source_mix')
        self.y_source_vox = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_vox')
        self.y_source_acc = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_acc')
        
        #设置随机失活
        self.dropout_rate = tf.placeholder(tf.float32)

        #初始化神经网络得到输出的人声和伴奏
        self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox = self.network_init()
        # 设置损失函数
        self.loss = self.loss_init()
        # 设置优化器
        self.optimizer = self.optimizer_init()
        #创建会话
        self.sess = tf.Session()
        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def Generalized_KL_Divergence(self, y, y_hat):
        return tf.reduce_sum(y * tf.log(y / y_hat))

    #该模型损失函数的初始化
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            
            if self.loss_type == "MSE":
                #求方差
                loss = tf.reduce_mean(
                    tf.square(self.y_source_acc - self.y_predict_acc)
                    + tf.square(self.y_source_vox - self.y_predict_vox), name='loss')
            elif self.loss_type == "DKL":
            # 一般 KL 散度 损失
                loss = tf.add(
                    x = self.Generalized_KL_Divergence(y = self.y_source_acc, y_hat = self.y_predict_acc), 
                    y = self.Generalized_KL_Divergence(y = self.y_source_vox, y_hat = self.y_predict_vox), 
                    name = 'GKL_loss')

            elif self.loss_type == "MSE+SIR":                    
                # 均方差 + SIR 损失
                loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
           
            elif self.loss_type == "DKL+SIR":  
                # 一般 KL 散度 + SIR 损失
                loss = tf.subtract(
                    x = (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_acc) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_vox)), 
                    y = self.gamma * (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_vox) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_acc)), 
                    name = 'GKL_SIR_loss')
            elif self.loss_type == "MSE+recognize":
                #求方差
                loss = tf.reduce_mean(
                    tf.square(self.y_source_acc - self.y_predict_acc)
                    + tf.square(self.y_source_vox - self.y_predict_vox), name='loss')
        return loss
    
    #初始化该模型的优化器，使用Adam优化器
    def optimizer_init(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.g_step)
        return optimizer

    #神经网络初始化
    def network_init(self):
        rnn_layer = []
        #根据num_hidden_units的长度来决定创建几层RNN，每个RNN长度为size
        for size in self.num_hidden_units:
            #使用GRU，同时，加上dropout
            layer_cell = tf.nn.rnn_cell.GRUCell(size)
            layer_cell = tf.contrib.rnn.DropoutWrapper(layer_cell, input_keep_prob=self.dropout_rate)
            rnn_layer.append(layer_cell)

        #创建多层RNN
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_source_mix, dtype = tf.float32)

        #全连接层
        y_dense_music_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,   #有多少个特征有多少个神经元
            activation = tf.nn.relu,     #激活函数为relu
            name = 'y_dense_music_src')

        y_dense_voice_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_dense_voice_src')

        Mask_acc = y_dense_music_src / (y_dense_music_src + y_dense_voice_src + np.finfo(float).eps)
        Mask_vox = y_dense_voice_src / (y_dense_music_src + y_dense_voice_src + np.finfo(float).eps)
        y_pred_acc = Mask_acc * self.x_source_mix
        y_pred_vox = Mask_vox * self.x_source_mix

        return Mask_acc, Mask_vox, y_pred_acc, y_pred_vox

    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        kpt = tf.train.latest_checkpoint(file_dir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            self.saver.restore(self.sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])

        return startepo
    def load_predict(self, file_dir):

        self.saver.restore(self.sess, file_dir)

    #开始训练
    def train(self, x_source_mix, y_source_acc, y_source_vox, learning_rate, dropout_rate):
        #已经训练了多少步
        step = self.sess.run(self.g_step)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox,
                         self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    #验证
    def validate(self, x_source_mix, y_source_acc, y_source_vox, dropout_rate):
        y_music_src_pred, y_voice_src_pred, validate_loss = self.sess.run([self.y_predict_acc, self.y_predict_vox, self.loss],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox, self.dropout_rate: dropout_rate})
        return y_music_src_pred, y_voice_src_pred, validate_loss

    #测试
    def test(self, x_source_mix, dropout_rate):
        MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred = self.sess.run([self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox],
                                         feed_dict = {self.x_source_mix: x_source_mix, self.dropout_rate: dropout_rate})

        return MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            
            # tf.summary.histogram('x_mixed', self.x_mixed)
            # tf.summary.histogram('y_src1', self.y_src1)
            # tf.summary.histogram('y_src2', self.y_src2)
            summary_op = tf.summary.merge_all()

        return summary_op



#以下为卷积神经网络
    """
        属性：num_hidden_units：隐藏单元的数量
            tensorboard_directory:保存tensorboard训练数据地址
            clear_tensorboard：是否清除tensorboard
    """
class U_netCNN(object):

    def __init__(self, tensorboard_directory = './CNN_model/tensorboard', clear_tensorboard = True, loss_type = "L1"):
        self.loss_type = loss_type    
        self.gamma = 0.001         
        
        #设置训练步数变量
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

        #设置学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据、人声
        self.x_source_mix = tf.placeholder(tf.float32, shape=[None, 512, 128, 1], name='x_source_mix')
        self.y_source_vox = tf.placeholder(tf.float32, shape=[None, 512, 128, 1], name='y_source_vox')
        self.y_source_acc = self.x_source_mix - self.y_source_vox
        self.training = tf.placeholder(tf.bool, name='training')

        #设置随机失活
        self.dropout_rate = tf.placeholder(tf.float32)

        #初始化神经网络得到输出的人声和伴奏
        self.MASK_vox, self.y_pred_vox = self.network_init()
        # 设置损失函数
        self.loss = self.loss_init()
        # 设置优化器
        self.optimizer = self.optimizer_init()
        #创建会话
        self.sess = tf.Session()
        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def Generalized_KL_Divergence(self, y, y_hat):
        return tf.reduce_sum(y * tf.log(y / y_hat) - y + y_hat)

    #该模型损失函数的初始化
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            if self.loss_type == "L1":
                #求1范数损失
                loss = tf.reduce_mean(tf.abs(self.y_source_vox - self.y_pred_vox))

            elif self.loss_type == "L1+SIR":
                #求1范数损失
                loss = tf.reduce_mean(tf.abs(self.y_source_vox - self.y_pred_vox) - self.gamma * tf.abs(self.y_source_acc - self.y_pred_vox))
            #求方差
            elif self.loss_type == "MSE":
                loss = tf.reduce_mean(tf.square(self.y_source_vox - self.y_pred_vox), name='loss')

            elif self.loss_type == "DKL":
            # 一般 KL 散度 损失
                loss = self.Generalized_KL_Divergence(y = self.y_source_vox, y_hat = self.y_pred_vox)

            elif self.loss_type == "MSE+SIR":                    
                # 均方差 + SIR 损失
                loss = tf.reduce_mean(tf.square(self.y_source_vox - self.y_pred_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_pred_vox)), name = 'MSE_SIR_loss')
           
            elif self.loss_type == "DKL+SIR":  
                # 一般 KL 散度 + SIR 损失
                loss = tf.subtract(
                    x = self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_pred_vox), 
                    y = self.gamma * (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_pred_vox)), 
                    name = 'GKL_SIR_loss')
            elif self.loss_type == "L1+recognize":
                #求1范数损失
                loss = tf.reduce_mean(tf.abs(self.y_source_vox - self.y_pred_vox))
        return loss
    
    #初始化该模型的优化器，使用Adam优化器
    def optimizer_init(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.g_step)
        return optimizer

    #神经网络初始化
    def network_init(self):    
        # 卷积层1
        conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = self.x_source_mix, filters=16, kernel_size=[5,5], 
                                                        strides=[2,2], padding="same", activation=tf.nn.leaky_relu))
        
        # 卷积层2
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size = [5,5], 
                                                        strides = [2,2], padding="same", activation = tf.nn.leaky_relu))

        # 卷积层3
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size = [5,5], 
                                                        strides = [2,2], padding="same", activation = tf.nn.leaky_relu))

        # 卷积层4
        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv3, filters = 128, kernel_size = [5,5], 
                                                        strides = [2,2], padding="same", activation = tf.nn.leaky_relu))

        # 卷积层5
        conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv4, filters = 256, kernel_size = [5,5], 
                                                        strides = [2,2], padding="same", activation = tf.nn.leaky_relu))

        # 卷积层6
        conv6 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv5, filters = 512, kernel_size = [5,5], 
                                                        strides = [2,2], padding="same", activation = tf.nn.leaky_relu))

        # 逆卷积层1 (随机失活)
        deconv1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = conv6, filters = 256, kernel_size = [5,5], 
                                                                    strides = [2,2], padding="same", activation = tf.nn.relu))
        concate1 = concatenate([deconv1,conv5],3)
        dropout1 = tf.layers.dropout(inputs = concate1, rate = self.dropout_rate, training = self.training)
        
        # 逆卷积层2 (随机失活)
        deconv2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = dropout1, filters = 128, kernel_size = [5,5], 
                                                                    strides = [2,2], padding="same", activation = tf.nn.relu))
        concate2 =  concatenate([deconv2,conv4],3)
        dropout2 = tf.layers.dropout(inputs = concate2, rate = self.dropout_rate, training = self.training)
                                        
        # 逆卷积层3 (随机失活)
        deconv3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = dropout2, filters = 64, kernel_size = [5,5], 
                                                                    strides = [2,2], padding="same", activation = tf.nn.relu))
        concate3 = concatenate([deconv3,conv3],3)
        dropout3 = tf.layers.dropout(inputs = concate3, rate = self.dropout_rate, training = self.training)
                                        
        # 逆卷积层4 (随机失活)
        deconv4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = dropout3, filters = 32, kernel_size = [5,5], 
                                                                    strides = [2,2], padding="same", activation = tf.nn.relu))
        concate4 = concatenate([deconv4,conv2],3)
        # 逆卷积层5 (随机失活)
        deconv5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = concate4, filters = 16, kernel_size = [5,5], 
                                                                    strides = [2,2], padding="same", activation = tf.nn.relu))
        concate5 = concatenate([deconv5,conv1],3)
        # 逆卷积层6 (随机失活)
        deconv6 = tf.layers.conv2d_transpose(inputs = concate5, filters = 1, kernel_size = [5,5],
                                                                    strides = [2,2], padding="same", activation = tf.nn.sigmoid)
        # 输出层
        output_layer = deconv6
        y_pred_vox = tf.multiply(self.x_source_mix, output_layer)                              #得到输出幅度谱
        Mask = output_layer
        return Mask, y_pred_vox

    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        ckpt = tf.train.latest_checkpoint(file_dir)
        print("ckpt:", ckpt)
        startepo = 0
        if ckpt != None:
            self.saver.restore(self.sess, ckpt)
            ind = ckpt.find("-")
            startepo = int(ckpt[ind + 1:])

        return startepo
    def load_predict(self, file_dir):

        self.saver.restore(self.sess, file_dir)

    #开始训练
    def train(self, x_source_mix, y_source_vox, learning_rate, dropout_rate, istraining):
        #已经训练了多少步
        step = self.sess.run(self.g_step)
        
        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_source_mix: x_source_mix,  self.y_source_vox: y_source_vox,
                         self.learning_rate: learning_rate, self.dropout_rate: dropout_rate, self.training: istraining})
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    #验证
    def validate(self, x_source_mix, y_source_vox, dropout_rate, istraining):

        validate_loss = self.sess.run(self.loss,
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_vox: y_source_vox, 
                        self.dropout_rate: dropout_rate, self.training: istraining})
        return validate_loss

    #测试
    def test(self, x_source_mix, dropout_rate, istraining):

        MASK_vox = self.sess.run([self.MASK_vox],feed_dict = {self.x_source_mix: x_source_mix, 
                                                            self.dropout_rate: dropout_rate, self.training: istraining})

        return MASK_vox

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            summary_op = tf.summary.merge_all()

        return summary_op


#以下为一维卷积神经网络
    """
        属性：
    """
class CNN_1D(object):

    def __init__(self, input_shape, output_shape, tensorboard_directory = './CNN_1D_model/tensorboard', clear_tensorboard = True, loss_type = "MSE"):
        self.loss_type = loss_type    
        self.gamma = 0.001    
        #设置训练步数变量
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')
        # 一维卷积层数
        self.num_layers = 12  
        # 第一层的卷积核数量
        self.num_initial_filters = 24
        # 为了一维卷积的降采样的滤波器大小
        self.filter_size = 15          
        # 升采样的卷积核大小
        self.merge_filter_size = 5
        # Filter size of first convolution in first downsampling block
        self.input_filter_size = 15
        # Filter size of first convolution in first downsampling block
        self.output_filter_size = 1
        # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
        self.upsampling = 'linear'
        # 卷积神经网络的输出类型
        self.output_type = 'difference'
        # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
        self.context = True
        # 卷积方式使用same，或者valide
        self.padding = "valid" if self.context else "same"      
        #是否是单声道
        self.num_channels = 1
        # 输出激活函数
        self.output_activation = 'tanh'
        #设置学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据、人声、背景音乐
        # self.x_source_mix = tf.placeholder(tf.float32, shape=[None, input_shape[1], 1], name='x_source_mix')
        # self.y_source_vox = tf.placeholder(tf.float32, shape=[None, output_shape[1], 1], name='y_source_vox')
        # self.y_source_acc = tf.placeholder(tf.float32, shape=[None, output_shape[1], 1], name='y_source_acc')

        self.x_source_mix = tf.placeholder(tf.float32, shape=input_shape, name='x_source_mix')
        self.y_source_vox = tf.placeholder(tf.float32, shape=output_shape, name='y_source_vox')
        self.y_source_acc = tf.placeholder(tf.float32, shape=output_shape, name='y_source_acc')

        self.training = True

        #初始化神经网络得到输出的人声和伴奏
        self.y_predict_vox, self.y_predict_acc = self.network_init()       
        # 设置损失函数
        self.loss = self.loss_init()
        # 设置优化器
        self.optimizer = self.optimizer_init()
        #创建会话
        self.sess = tf.Session()
        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def Generalized_KL_Divergence(self, y, y_hat):
        safe_log = tf.clip_by_value(y / y_hat, 1e-10, 1e100)
        return tf.reduce_mean(y * tf.log(safe_log) - y + y_hat)

    #该模型损失函数的初始化
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            #求方差
            if self.loss_type == "MSE":
                loss = tf.reduce_mean(
                    tf.square(self.y_source_acc - self.y_predict_acc)
                    + tf.square(self.y_source_vox - self.y_predict_vox), name='loss')

            elif self.loss_type == "DKL":
            # 一般 KL 散度 损失
                loss = tf.add(
                    x = self.Generalized_KL_Divergence(y = self.y_source_acc, y_hat = self.y_predict_acc), 
                    y = self.Generalized_KL_Divergence(y = self.y_source_vox, y_hat = self.y_predict_vox), 
                    name = 'GKL_loss')

            elif self.loss_type == "MSE+SIR":                    
                # 均方差 + SIR 损失
                loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
           
            elif self.loss_type == "DKL+SIR":  
                # 一般 KL 散度 + SIR 损失
                loss = tf.subtract(
                    x = (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_acc) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_vox)), 
                    y = self.gamma * (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_vox) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_acc)), 
                    name = 'GKL_SIR_loss')
            elif self.loss_type == "MSE+SIR+recognize":                    
                # 均方差 + SIR 损失
               loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
           
        return loss
    
    #初始化该模型的优化器，使用Adam优化器
    def optimizer_init(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.g_step)
        return optimizer


    #神经网络初始化
    def network_init(self):    
        # 向下卷积：重复跨步卷积
        enc_outputs = list()
        current_layer = self.x_source_mix
        
        for i in range(self.num_layers):
            current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, activation=tf.nn.leaky_relu, padding=self.padding) # out = in - filter + 1
            enc_outputs.append(current_layer)
            current_layer = current_layer[:,::2,:] # 按因子抽取 2 # out = (in-1)/2 + 1

        current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * self.num_layers),self.filter_size,activation=tf.nn.leaky_relu,padding=self.padding) # One more conv here since we need to compute features after last decimation

        # 这里特征图会是一维的X

        # 向上卷积
        for i in range(self.num_layers):
            #升采样开始
            current_layer = tf.expand_dims(current_layer, axis=1)
            if self.upsampling == 'learned':
                # 通过使用宽度为2的卷积滤波器在两个相邻时间位置之间进行插值，并将响应插入两个相应输入的中间
                current_layer = utils.learned_interpolation_layer(current_layer, self.padding, i)
            else:
                if self.context:
                    current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
                else:
                    current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
            current_layer = tf.squeeze(current_layer, axis=1)
        
            # 升采样结束
            assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
            current_layer = utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
            current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * (self.num_layers - i - 1)), self.merge_filter_size,
                                                activation=tf.nn.leaky_relu,
                                                padding=self.padding)  # out = in - filter + 1

        current_layer = utils.crop_and_concat(self.x_source_mix, current_layer, match_feature_dim=False)

        # 输出层
        # 输出层激活函数选择
        if self.output_activation == "tanh":
            out_activation = tf.tanh
        elif self.output_activation == "linear":
            out_activation = lambda x: utils.AudioClip(x, self.training)

        if self.output_type == "direct":
            y_predict_vox = tf.layers.conv1d(current_layer, self.num_channels, self.output_filter_size, activation=out_activation, padding=self.padding)
            y_predict_acc = tf.layers.conv1d(current_layer, self.num_channels, self.output_filter_size, activation=out_activation, padding=self.padding)
            return self.y_source_vox, self.y_source_acc   


        elif self.output_type == "difference":
            cropped_input = utils.crop(self.x_source_mix, current_layer.get_shape().as_list(), match_feature_dim=False)
            sum_source = 0
            y_predict_vox = tf.layers.conv1d(current_layer, self.num_channels, self.output_filter_size, activation=out_activation, padding=self.padding)
            sum_source = sum_source + y_predict_vox               
            # 做差求出另一个语音
            last_source = utils.crop(cropped_input, sum_source.get_shape().as_list()) - sum_source
            y_predict_acc = utils.AudioClip(last_source, self.training)
            return y_predict_vox, y_predict_acc   


    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        ckpt = tf.train.latest_checkpoint(file_dir)
        print("ckpt:", ckpt)
        startepo = 0
        if ckpt != None:
            self.saver.restore(self.sess, ckpt)
            ind = ckpt.find("-")
            startepo = int(ckpt[ind + 1:])

        return startepo
    def load_predict(self, file_dir):

        self.saver.restore(self.sess, file_dir)


    #开始训练
    def train(self, x_source_mix, y_source_acc, y_source_vox, learning_rate, training):
        #已经训练了多少步
        self.training = training
        step = self.sess.run(self.g_step)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox,
                         self.learning_rate: learning_rate})
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    def validate(self, x_source_mix, y_source_acc, y_source_vox, training):
        self.training = training
        validate_loss = self.sess.run(self.loss,feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox})
        return validate_loss

    #测试
    def test(self, x_source_mix, training):
        self.training = training
        y_predict_vox, y_predict_acc = self.sess.run([self.y_predict_vox, self.y_predict_acc],feed_dict = {self.x_source_mix: x_source_mix})

        return y_predict_vox, y_predict_acc

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            summary_op = tf.summary.merge_all()

        return summary_op

class DRNN_Output_rectified(object):

    def __init__(self, num_features, num_hidden_units=[256,256,256], tensorboard_directory = './RNN_model/tensorboard', clear_tensorboard = True, loss_type = "MSE"):
        self.loss_type = loss_type    
        self.gamma = 0.001       
        #设置特征数量
        self.num_features = num_features
        #循环神经网络有多少层
        self.num_rnn_layer = len(num_hidden_units)
        #有多少个隐藏单元
        self.num_hidden_units = num_hidden_units
        #设置训练步数变量
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

        #设置学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据、人声、背景音乐
        self.x_source_mix = tf.placeholder(tf.float32, shape=[None, None, num_features], name='x_source_mix')
        self.y_source_vox = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_vox')
        self.y_source_acc = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_acc')
        
        #设置随机失活
        self.dropout_rate = tf.placeholder(tf.float32)

        #初始化神经网络得到输出的人声和伴奏
        self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox = self.network_init()
        # 设置损失函数
        self.loss = self.loss_init()
        # 设置优化器
        self.optimizer = self.optimizer_init()
        #创建会话
        self.sess = tf.Session()
        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def Generalized_KL_Divergence(self, y, y_hat):
        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)

    #该模型损失函数的初始化
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            if self.loss_type == "MSE":
                #求方差
                loss = tf.reduce_mean(
                    tf.square(self.y_source_acc - self.y_predict_acc)
                    + tf.square(self.y_source_vox - self.y_predict_vox), name='loss')
            elif self.loss_type == "DKL":
            # 一般 KL 散度 损失
                loss = tf.add(
                    x = self.Generalized_KL_Divergence(y = self.y_source_acc, y_hat = self.y_predict_acc), 
                    y = self.Generalized_KL_Divergence(y = self.y_source_vox, y_hat = self.y_predict_vox), 
                    name = 'GKL_loss')

            elif self.loss_type == "MSE+SIR":                    
                # 均方差 + SIR 损失
                loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
           
            elif self.loss_type == "DKL+SIR":  
                # 一般 KL 散度 + SIR 损失
                loss = tf.subtract(
                    x = (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_acc) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_vox)), 
                    y = self.gamma * (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_vox) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_acc)), 
                    name = 'GKL_SIR_loss')
            
            elif self.loss_type == "MSE+SIR+recognize":
                loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
                   
        return loss
    
    #初始化该模型的优化器，使用Adam优化器
    def optimizer_init(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.g_step)
        return optimizer

    #神经网络初始化
    def network_init(self):
        rnn_layer = []
        #根据num_hidden_units的长度来决定创建几层RNN，每个RNN长度为size
        for size in self.num_hidden_units:
            #使用GRU，同时，加上dropout
            layer_cell = tf.nn.rnn_cell.GRUCell(size)
            layer_cell = tf.contrib.rnn.DropoutWrapper(layer_cell, input_keep_prob=self.dropout_rate)
            rnn_layer.append(layer_cell)

        #创建多层RNN
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_source_mix, dtype = tf.float32)

        #全连接层
        y_dense_music_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,   #有多少个特征有多少个神经元
            activation = tf.nn.sigmoid,     #激活函数为relu
            name = 'y_dense_voice_src')

        Mask_acc = 1-y_dense_music_src
        Mask_vox = y_dense_music_src
        y_pred_acc = Mask_acc * self.x_source_mix
        y_pred_vox = Mask_vox * self.x_source_mix

        return Mask_acc, Mask_vox, y_pred_acc, y_pred_vox

    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        kpt = tf.train.latest_checkpoint(file_dir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            self.saver.restore(self.sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])

        return startepo
    def load_predict(self, file_dir):

        self.saver.restore(self.sess, file_dir)

    #开始训练
    def train(self, x_source_mix, y_source_acc, y_source_vox, learning_rate, dropout_rate):
        #已经训练了多少步
        step = self.sess.run(self.g_step)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox,
                         self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    #验证
    def validate(self, x_source_mix, y_source_acc, y_source_vox, dropout_rate):
        y_music_src_pred, y_voice_src_pred, validate_loss = self.sess.run([self.y_predict_acc, self.y_predict_vox, self.loss],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox, self.dropout_rate: dropout_rate})
        return y_music_src_pred, y_voice_src_pred, validate_loss

    #测试
    def test(self, x_source_mix, dropout_rate):
        MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred = self.sess.run([self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox],
                                         feed_dict = {self.x_source_mix: x_source_mix, self.dropout_rate: dropout_rate})

        return MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            
            # tf.summary.histogram('x_mixed', self.x_mixed)
            # tf.summary.histogram('y_src1', self.y_src1)
            # tf.summary.histogram('y_src2', self.y_src2)
            summary_op = tf.summary.merge_all()

        return summary_op

class DRNN_Output_LSTM(object):

    def __init__(self, num_features, num_hidden_units=[256,256,256], tensorboard_directory = './RNN_model/tensorboard', clear_tensorboard = True, loss_type = "MSE"):
        self.loss_type = loss_type    
        self.gamma = 0.001       
        #设置特征数量
        self.num_features = num_features
        #循环神经网络有多少层
        self.num_rnn_layer = len(num_hidden_units)
        #有多少个隐藏单元
        self.num_hidden_units = num_hidden_units
        #设置训练步数变量
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

        #设置学习率
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #混合了背景音乐和人声的数据、人声、背景音乐
        self.x_source_mix = tf.placeholder(tf.float32, shape=[None, None, num_features], name='x_source_mix')
        self.y_source_vox = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_vox')
        self.y_source_acc = tf.placeholder(tf.float32, shape=[None, None, num_features], name='y_source_acc')
        
        #设置随机失活
        self.dropout_rate = tf.placeholder(tf.float32)

        #初始化神经网络得到输出的人声和伴奏
        self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox = self.network_init()
        # 设置损失函数
        self.loss = self.loss_init()
        # 设置优化器
        self.optimizer = self.optimizer_init()
        #创建会话
        self.sess = tf.Session()
        #需要保存模型，所以获取saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def Generalized_KL_Divergence(self, y, y_hat):
        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)

    #该模型损失函数的初始化
    def loss_init(self):
        with tf.variable_scope('loss') as scope:
            if self.loss_type == "MSE":
                #求方差
                loss = tf.reduce_mean(
                    tf.square(self.y_source_acc - self.y_predict_acc)
                    + tf.square(self.y_source_vox - self.y_predict_vox), name='loss')
            elif self.loss_type == "DKL":
            # 一般 KL 散度 损失
                loss = tf.add(
                    x = self.Generalized_KL_Divergence(y = self.y_source_acc, y_hat = self.y_predict_acc), 
                    y = self.Generalized_KL_Divergence(y = self.y_source_vox, y_hat = self.y_predict_vox), 
                    name = 'GKL_loss')

            elif self.loss_type == "MSE+SIR":                    
                # 均方差 + SIR 损失
                loss = tf.reduce_mean(tf.square(self.y_source_acc - self.y_predict_acc) + tf.square(self.y_source_vox - self.y_predict_vox) - self.gamma * (tf.square(self.y_source_acc - self.y_predict_vox) + tf.square(self.y_source_vox - self.y_predict_acc)), name = 'MSE_SIR_loss')
           
            elif self.loss_type == "DKL+SIR":  
                # 一般 KL 散度 + SIR 损失
                loss = tf.subtract(
                    x = (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_acc) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_vox)), 
                    y = self.gamma * (self.generalized_kl_divergence(y = self.y_source_acc, y_hat = self.y_predict_vox) + self.generalized_kl_divergence(y = self.y_source_vox, y_hat = self.y_predict_acc)), 
                    name = 'GKL_SIR_loss')
        return loss
    
    #初始化该模型的优化器，使用Adam优化器
    def optimizer_init(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.g_step)
        return optimizer

    #神经网络初始化
    def network_init(self):
        rnn_layer = []
        #根据num_hidden_units的长度来决定创建几层RNN，每个RNN长度为size
        for size in self.num_hidden_units:
            #使用GRU，同时，加上dropout
            layer_cell = tf.nn.rnn_cell.LSTMCell(size)
            layer_cell = tf.contrib.rnn.DropoutWrapper(layer_cell, input_keep_prob=self.dropout_rate)
            rnn_layer.append(layer_cell)

        #创建多层RNN
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_source_mix, dtype = tf.float32)

        #全连接层
        y_dense_music_src = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,   #有多少个特征有多少个神经元
            activation = tf.nn.sigmoid,     #激活函数为relu
            name = 'y_dense_voice_src')

        Mask_acc = 1-y_dense_music_src
        Mask_vox = y_dense_music_src
        y_pred_acc = Mask_acc * self.x_source_mix
        y_pred_vox = Mask_vox * self.x_source_mix

        return Mask_acc, Mask_vox, y_pred_acc, y_pred_vox

    #保存模型
    def save(self, directory, filename, global_step):
        #如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)

    # 加载模型，如果没有模型，则初始化所有变量
    def load(self, file_dir):
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 没有模型的话，就重新初始化
        kpt = tf.train.latest_checkpoint(file_dir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            self.saver.restore(self.sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])

        return startepo
    def load_predict(self, file_dir):

        self.saver.restore(self.sess, file_dir)

    #开始训练
    def train(self, x_source_mix, y_source_acc, y_source_vox, learning_rate, dropout_rate):
        #已经训练了多少步
        step = self.sess.run(self.g_step)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox,
                         self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    #验证
    def validate(self, x_source_mix, y_source_acc, y_source_vox, dropout_rate):
        y_music_src_pred, y_voice_src_pred, validate_loss = self.sess.run([self.y_predict_acc, self.y_predict_vox, self.loss],
            feed_dict = {self.x_source_mix: x_source_mix, self.y_source_acc: y_source_acc, self.y_source_vox: y_source_vox, self.dropout_rate: dropout_rate})
        return y_music_src_pred, y_voice_src_pred, validate_loss

    #测试
    def test(self, x_source_mix, dropout_rate):
        MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred = self.sess.run([self.MASK_acc, self.MASK_vox, self.y_predict_acc, self.y_predict_vox],
                                         feed_dict = {self.x_source_mix: x_source_mix, self.dropout_rate: dropout_rate})

        return MASK_acc, MASK_vox, y_music_src_pred, y_voice_src_pred

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            
            # tf.summary.histogram('x_mixed', self.x_mixed)
            # tf.summary.histogram('y_src1', self.y_src1)
            # tf.summary.histogram('y_src2', self.y_src2)
            summary_op = tf.summary.merge_all()

        return summary_op