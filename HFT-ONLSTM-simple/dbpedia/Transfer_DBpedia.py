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
if os.path.exists(fileweights):
    model.load_weights(fileweights)
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
# model.fit(x_train, y1_train,
#           validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping, checkpoint,reduce_lr],
#           # validation_data=(x_test, y1_test),
#           shuffle= True)
#####################################
print('category Embedding')
lossl1, accl1 = model.evaluate(x_test, y1_test, verbose=0)
print('lossl1,',lossl1, ', accl1,', accl1)
pred_l1 = model.predict([x])
print("Save normal label to file")
pl.dump(pred_l1, open(r'../dataset/dbp/predictlabel/dbp_pred_l1', 'wb'))
pred_l1 = np.argmax(pred_l1, axis=1)

semantic_l1 = ['agent', 'device', 'event', 'place', 'species', 'sports season', 'topical concept', 'unit of work', 'work']

pred_semantic_l1 = []


for i in pred_l1:
    pred_semantic_l1.append([word_to_id[x] for x in semantic_l1[i].split(' ') if x in word_to_id])

l1_length = 3
pred_semantic_l1_pad = keras.preprocessing.sequence.pad_sequences(pred_semantic_l1, l1_length, padding='post', truncating='post')
# 存储经过embedding后的label

#######################################
print("Save semantic label to file")
with open(r'../dataset/dbp/predictlabel/dbp_pred_semantic_l1_pad', 'wb') as f:
    pl.dump(pred_semantic_l1_pad, f)

print("level1: end up")
