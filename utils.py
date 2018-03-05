import sys, pickle, os, random
import numpy as np
import sys

# 将tag转换成数字，是为了训练的时候使用
tag2label = {"O": 0, "B-PER": 1, "I-PER": 2,"B-LOC": 3, "I-LOC": 4,"B-ORG": 5, "I-ORG": 6}

# 读取训练数据和测试数据，返回[([sent_],[tag_])]
def read_data(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data

# 从训练数据中构建词典word2id
def build_vocab(vocab_path, corpus_path, min_count):
    data = read_data(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word) 
    for word in low_freq_words:  # 要过滤掉低频词
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id  # 那些没有在词典中出现的词的id，模型训练好之后，肯定会遇到一些不在词典中的词
    word2id['<PAD>'] = 0

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

# 将句子中的词全部替换成其在词典中的id，在embedding层可以根据这个id得到这个词的embedding
# 对于那些没有出现在词典中的词，均认为其为<UNK>
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:  # 对于没出现在词典中的词，就用<UNK>代替
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

# 读取词典
def read_dict(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

# 随机初始化embedding，返回embedding矩阵，这个embedding可以在训练的更新，不需要强制传入已训练好的词向量，但是用已训练好的词向量肯定更快
def init_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


# 将原始数据进行分批，每一批包含的数据量是batch_size，这里要将原始数据中的词换成其在word2id中的id，要把原始数据中的tag换成数字label
# 现在数据变成[([sent_id], [label])]
def gen_batch(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:  # 是否混洗数据
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels

# 从返回的标注序列中取出所有的实体
def get_entity(tag_seq, char_seq):
    length = len(char_seq)
    ENT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER' or tag == 'B-LOC' or tag == 'B-ORG':
            if 'ent' in locals().keys():  # 变量ent如果存在那么就将其添加到实体中
                ENT.append(ent)
                del ent  # 将变量从命名空间中删除
            ent = char
            if i+1 == length:
                ENT.append(ent)
        if tag == 'I-PER' or tag == 'I-LOC' or tag == 'I-ORG':
            if 'ent' in locals().keys():
                ent += char
            if i+1 == length:
                if 'ent' in locals().keys():
                    ENT.append(ent)
        if tag not in ['I-PER', 'B-PER', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG']:
            if 'ent' in locals().keys():
                ENT.append(ent)
                del ent
            continue
    return ENT

# 调用第三方perl程序进行评测
def conlleval(label_predict, label_path, metric_path):
    eval_perl = "./conlleval.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics