#encoding:utf-8
from collections import  Counter
import tensorflow.contrib.keras as kr
# import keras as kr
import numpy as np
import codecs
import re

from nltk import *
from nltk.corpus import stopwords
from config import *
from sklearn import preprocessing

def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]
    return stopwords
def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation(匹配中文 大小写字母)
    contents,labels=[],[]
    with codecs.open(filename,'r',encoding='gbk') as f:
        for line in f:
            try:
                line=line.rstrip()
                ###############多标签############
                label = []
                labels_content = line.split('\t')
                #工单数据格式：label+"\t"+keywords+"\t"+abstract
                for i in range(len(labels_content[:-1])):
                    if labels_content[i] != '':
                        label.append(labels_content[i])
                labels.append(label)
                #labels_content[-2:]=keywords Abstract
                content = labels_content[-1:]
                # print(content)
                ###########################
                stopWords = set(stopwords.words('english'))
                # stopWords = stopwordslist(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\output\stopwordds')
                # print(stopWords)
                wordsFiltered = []
                for items in content:
                    # print('items:'+items)
                    word_tok_list = items.split(' ')
                    for w in word_tok_list:
                        if w not in stopWords and not isSymbol(w) and not hasNumbers(w) and len(w)>=2:
                            wordsFiltered.append(w.rstrip(';').rstrip('.').rstrip(',').rstrip('."').lstrip('"'))
                contents.append(wordsFiltered)
            except:
                pass
    return labels,contents

def build_vocab(filenames,vocab_dir,vocab_size):
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words, _ =list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')
def read_vocab(vocab_dir):
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    y1 = ['Computer Science', 'Electrical Engineering', 'Psychology', 'Mechanical Engineering', 'Civil Engineering',
          'Medical Sciences', 'Biochemistry']

    y1_to_id=dict(zip(y1,range(len(y1))))

    return y1,y1_to_id
def read_files(filename):
    contents, labels1, labels2 = [], [], []
    i = 0
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                content = line.split(' ')
                # stopWords = set(stopwords.words('english'))
                wordsFiltered = []
                for w in content:
                    if len(w)>=2:
                        wordsFiltered.append(w.rstrip('\n').rstrip('\r'))
                contents.append(wordsFiltered)
                #######################################################
                i=i+1
            except:
                pass
    print(len(contents))
    return contents

def process_file(cont_file,y1_file,y2_file,y3_file, word_to_id,y1_to_id,y2_to_id,y3_to_id, max_length=500,y1_length = 10,y2_length = 10,y3_length=10):

    contents=read_files(cont_file)
    y1 = read_files(y1_file)
    y2 = read_files(y2_file)

    data_id,y1_id,y2_id,y3_id=[],[],[],[]
    y1_id_pad,y2_id_pad,y3_id_pad = [],[],[]
    label_y1 = []
    label_y2 = []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        y1_id_pad.append([word_to_id[x] for x in y1[i] if x in word_to_id])
        y2_id_pad.append([word_to_id[x] for x in y2[i] if x in word_to_id])

        ##############y[i]=['computer','science']转化为y[i]=['computer science']#################################

        str = ""
        for label in y1[i]:
            str = str+ label + " "
        label_y1.append(str.rstrip(' '))
        # label_id.append(label_idd)

        str2 = ""
        for label in y2[i]:
            str2 = str2 + label + " "
        label_y2.append(str2.rstrip(' '))



        y1_id.append(y1_to_id[label_y1[i]])
        y2_id.append(y2_to_id[label_y2[i]])


    cont_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')
    y2_pad = kr.preprocessing.sequence.pad_sequences(y2_id_pad, y2_length, padding='post', truncating='post')

    ##################################
    y1_index = kr.utils.to_categorical(y1_id)
    y2_index = kr.utils.to_categorical(y2_id)

    #####################################

    return cont_pad,y1_index,y2_index,y1_pad,y2_pad

def batch_iter(x,y,batch_size=64):

    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):

    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):

    with np.load(filename) as data:
        return data["embeddings"]
