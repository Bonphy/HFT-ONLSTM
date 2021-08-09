# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import pickle as pl
import keras
from wos_textONlstmL1 import TextONLSTM
from dbp_process.probabilityonehot import trans_to_onehot

maxlen = 500
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

#  The path of embedding matrix
pretrained_w2v, _, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
########################################################################################################################
print('Loading data...')

#  Path to the processed dataset
x,y1,y2,y1_pad,y2_pad = pl.load(open(r'../dataset/wos/WOSDATA_txt_vector500dimsy1y2_10dim_zjp', 'rb'))
x_train, x_test, y1_train, y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)
########################################################################################################################
print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

########################################################################################################################
print('Train...')


fileweights = r"../log/wos/wos_weight_l1.hdf5"


########################################################################################################################
model.load_weights(fileweights, by_name = True)
loss, accuracy = model.evaluate(x_test,y1_test,verbose=0)
print('\ntest loss',loss,'accuracy',accuracy)
print('level1 evaluation: end up')
