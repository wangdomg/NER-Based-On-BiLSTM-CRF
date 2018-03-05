#!Python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from model import BLC
import numpy as np
import os, argparse, time, random
from utils import read_data, read_dict, tag2label, init_embedding, get_entity
import sys

if __name__ == '__main__':
    # train/test
    mode = sys.argv[1]

    # 参数
    args = {}
    args['train_data'] = 'data'  # 训练数据路径
    args['test_data'] = 'data'  # 测试数据路径
    args['batch_size'] = 64  # 每一批用来训练的样本数
    args['epoch'] = 10  # 迭代次数
    args['hidden_dim'] = 100  # lstm接受的数据的维度
    args['optimizer'] = 'Adam'  # 优化损失函数的方法
    args['lr'] = 0.001  # 学习率
    args['clip'] = 5.0  # 限定梯度更新的时候的阈值
    args['dropout'] = 0.5  # 保留率
    args['update_embedding'] = True  # 是否要对embedding进行更新，embedding初始化之后，这里设置成更新，就可以更新embedding
    args['embedding_dim'] = 100  # embedding的维度
    args['shuffle'] = True  # 是否每次在把数据送进lstm中训练时都混洗

    # 读取词典，把一个字映射到一个id，这个词典是从训练数据中得到的
    word2id = read_dict(os.path.join('.', args['train_data'], 'word2id.pkl'))

    # 随机初始化embedding
    embeddings = init_embedding(word2id, args['embedding_dim'])

    # 设置模型的输出路径
    model_path = 'BLCM'
    output_path = os.path.join('.', model_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    result_path = os.path.join(output_path, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 训练模型
    if mode == 'train':
        # 读取数据
        train_path = os.path.join('.', args['train_data'], 'train_data')
        test_path = os.path.join('.', args['test_data'], 'test_data')
        train_data = read_data(train_path)
        test_data = read_data(test_path)
        # 创建模型并训练
        model = BLC(batch_size=args['batch_size'], epoch_num=args['epoch'], hidden_dim=args['hidden_dim'], embeddings=embeddings,
                        dropout_keep=args['dropout'], optimizer=args['optimizer'], lr=args['lr'], clip_grad=args['clip'],
                        tag2label=tag2label, vocab=word2id, shuffle=args['shuffle'],
                        model_path=ckpt_prefix, summary_path=summary_path, result_path=result_path, update_embedding=args['update_embedding'])
        model.build_graph()
        model.train(train_data, test_data)
    # 演示模型
    elif mode == 'test':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        model = BLC(batch_size=args['batch_size'], epoch_num=args['epoch'], hidden_dim=args['hidden_dim'],embeddings=embeddings,
                        dropout_keep=args['dropout'], optimizer=args['optimizer'], lr=args['lr'], clip_grad=args['clip'],
                        tag2label=tag2label, vocab=word2id, shuffle=args['shuffle'],
                        model_path=ckpt_file, summary_path=summary_path, result_path=result_path, update_embedding=args['update_embedding'])
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt_file)
            while(1):
                print('输入待识别的句子: ')
                sent = input()
                if sent == '' or sent.isspace():
                    break
                else:
                    sent = list(sent.strip())
                    data = [(sent, ['O'] * len(sent))]
                    tag = model.test(sess, data)
                    ENT = get_entity(tag, sent)
                    print('ENT: {}\n'.format(ENT))