import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from utils import pad_sequences, gen_batch, conlleval

class BLC(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings,dropout_keep, optimizer, lr, clip_grad,
                 tag2label, vocab, shuffle,model_path, summary_path, result_path, update_embedding=True):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle
        self.model_path = model_path
        self.summary_path = summary_path
        self.result_path = result_path
        self.update_embedding = update_embedding

    # 创建图
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer()
        self.biLSTM_layer()
        self.loss_layer()
        self.optimize()
        self.init()
    
    # 得到传递进来的真实的训练样本
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    # 第一层，embedding层，根据传进来的词找到id，再根据id找到其对应的embedding，并返回
    def lookup_layer(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,dtype=tf.float32,trainable=self.update_embedding,name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,ids=self.word_ids,name="word_embeddings")
        # 在进入下一层之前先做一个dropout，防止过拟合
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)

    # 第二层，双向lstm层
    def biLSTM_layer(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=self.word_embeddings,sequence_length=self.sequence_lengths,dtype=tf.float32)
            # 对正反输出的向量直接拼接
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
        # 将lstm输出的向量转到了多个tag上的概率
        # 最终bi-lstm的输出是一个n*m的矩阵p，其中n表示词的个数，m表示tag的个数，其中pij表示词wi映射到tagj的非归一化概率
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",shape=[2 * self.hidden_dim, self.num_tags],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            b = tf.get_variable(name="b",shape=[self.num_tags],initializer=tf.zeros_initializer(),dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    # 第三层，CRF层
    def loss_layer(self):
        # log_likelihood是对数似然函数，transition_params是转移概率矩阵
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,tag_indices=self.labels,sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", self.loss)

    # 确定优化方法
    def optimize(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    # 训练模型
    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                print ('第',epoch+1,'轮训练')
                self.run(sess, train, dev, self.tag2label, epoch, saver)
    
    # 进行一轮训练
    def run(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        # 将所有的训练数据分成多个batch，每次把一个batch送进网络中学习
        batches = gen_batch(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write('batch总数: {}, 当前batch: {}'.format(num_batches, step+1)+'\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            
            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)
        print ('模型验证')
        # 在测试集上验证这一轮训练之后模型的效果
        label_list_dev, seq_len_list_dev = self.dev(sess, dev)
        # 计算指标
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    # 在全体测试集上进行测试
    def dev(self, sess, dev):
        label_list, seq_len_list = [], []
        # 生成多个batch
        for seqs, labels in gen_batch(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict(sess, seqs)  # 预测标注序列
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    # mode为test的时候调用，运行训练好的模型为输入的句子进行命名实体识别
    def test(self, sess, sent):
        label_list = []
        # 生成多个batch
        for seqs, labels in gen_batch(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        # 将数字转换成tag
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def predict(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed(seqs, dropout=1.0)
        logits, transition_params = sess.run([self.logits, self.transition_params],feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    
    # 准备输入的参数
    def get_feed(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list
    
    # 评估模型的效果，调用了一个第三方的perl程序
    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        # 将模型的预测写到这个文件中，每一轮迭代都要写一个
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        # 将评测结果写到这个文件中，每一轮迭代都要写一个
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            print (_)

