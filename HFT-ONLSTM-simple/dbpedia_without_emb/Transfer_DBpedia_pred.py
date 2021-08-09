# coding=utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from dbpedia_textONlsLstmL1 import TextONLSTM
import pickle as pl
import tensorflow.contrib.keras as kr
from dbp_process.probabilityonehot import trans_to_onehot

maxlen = 300
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100
#  The path of embedding matrix
pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')
#  Path to the processed dataset
x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'../dataset/dbp/DBP_txt_vector300dim_y1y2y3_10dim_zjp', 'rb'))
x_train,x_test,y1_train,y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)

print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
fileweights = r"../dataset/dbp/output/dbp_l1_weights.hdf5"
model.load_weights(fileweights,True)
loss, accuracy = model.evaluate(x_test,y1_test,verbose=0)
print('\ntest loss',loss,'accuracy',accuracy)



print("level1 evaluation: end up")
