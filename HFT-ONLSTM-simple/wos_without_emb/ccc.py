# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import pickle as pl
import tensorflow.contrib.keras as kr
from wos_textONlstmL1 import TextONLSTM
import os

predict = 1
print(predict)
print(np.shape(predict))

# The path of the ID set of the predicted first level label
print(os.getcwd())
print(os.path.split(os.getcwd())[0])

pl.dump(predict, open( r'E:\HFT-ONLSTM-master\HFT-ONLSTM-master\dataset\wos\predictlabel\wos_layer1_predict1_2', 'wb'))
print( r'E:\HFT-ONLSTM-master\HFT-ONLSTM-master\dataset\wos\predictlabel\wos_layer1_predict1_2')
print(os.path.split(os.getcwd())[0] + r'\dataset\wos\output\predictlabel\wos_layer1_predict1_2')
pl.dump(predict, open(os.path.split(os.getcwd())[0] + r'\dataset\wos\output\predictlabel\wos_layer1_predict1_2', 'wb'))
print(os.path.split(os.getcwd())[0] + r'\dataset\wos\output\predictlabel\wos_layer1_predict1_2')
pretrained_w2v, word_to_id, _ = pl.load(
    open(r'embeddings\emb_matrix_glove_300', 'rb'))
y1 = ['biochemistry', 'civil', 'computer science', 'electrical', 'mechanical', 'medical', 'psychology']
y1_id_pad = []
label1_id =pl.load(open(r'../dataset\wos\output\predictlabel\wos_layer1_predict1_2','rb'))
for i in label1_id:
    y1_id_pad.append([word_to_id[x] for x in y1[i].split(' ') if x in word_to_id])
y1_length = 2
y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')

#  The path of the set of semantic vectors embedded in matrix mapping for label ID
with open(r'../dataset\wos\output\predictlabel\py1_id_pad_2', 'wb') as f:
    pl.dump(y1_pad, f)
