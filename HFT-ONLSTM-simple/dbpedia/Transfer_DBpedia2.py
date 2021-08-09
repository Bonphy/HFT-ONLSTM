# coding=utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time
import numpy as np
from sklearn.model_selection import train_test_split
from dbpedia_textONlstmL2 import TextONLSTM2
import pickle as pl
from dbp_process.probabilityonehot import trans_to_onehot
import keras
maxlen = 303
# maxlen = 310
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

start_time =time.time()
#  The path of embedding matrix
pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')
#  Path to the processed dataset
x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'../dataset/dbp/DBP_txt_vector300dim_y1y2y3_10dim_zjp', 'rb'))

pred_l1 = pl.load(open(r"../dataset/dbp/predictlabel/dbp_pred_l1", 'rb'))
pred_semantic_l1_pad = pl.load(open(r'../dataset/dbp/predictlabel/dbp_pred_semantic_l1_pad','rb'))
# print(pre_y1_pad[:3])
######
emb_pre_label_x = list(np.column_stack((pred_semantic_l1_pad,x)))
#真实标签310维
# emb_true_label_x = list(np.column_stack((y1_pad,x)))
########################################################################################################################
# x_train,x_test,y1_train,y1_test,y2_train,y2_test,pre_y1_train_pad,pre_y1_test_pad = train_test_split( x, y1, y2, pred_semantic_l1_pad, test_size=0.2, random_state=42)
# x_train,x_test,pre_y1_train_pad,pre_y1_test_pad=train_test_split( x, pred_semantic_l1_pad, test_size=0.2, random_state=42)
x_train, x_test, y1_train, y1_test, y2_train, y2_test, pred_semantic_l1_train, pred_semantic_l1_test,\
    pred_l1_train, pred_l1_test= \
    train_test_split(x, y1, y2, pred_semantic_l1_pad, pred_l1, test_size=0.2, random_state=42)
#预测标签
emb_label_train = list(np.column_stack((pred_semantic_l1_train,x_train)))
emb_label_test = list(np.column_stack((pred_semantic_l1_test,x_test)))
#真实标签
# x_train,x_test,y1_train_pad,y1_test_pad=train_test_split( x, y1_pad, test_size=0.2, random_state=42)
# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))
########################################################################################################################

print('Build model...')
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../dataset/dbp/output/dbp_l1_weights.hdf5",by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
##################################################################
model.summary()
print('Train...')
fileweights = r"../dataset/dbp/output/dbp_l2_weights.hdf5"
if os.path.exists(fileweights):
    model.load_weights(fileweights)
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
##############################
# 当评价指标不在提升时，减少学习率

# model.fit([emb_label_train], y2_train,
#           validation_split=0.1,
#           verbose=2,
#           batch_size=batch_size,
#           epochs=epochs,
#
#           callbacks=[early_stopping, checkpoint,reduce_lr],
#
#           shuffle= True)
#####################################
#####################################

lossl2, accl2 = model.evaluate([emb_label_test], y2_test, verbose=0)
print('lossl2,',lossl2, ', accl2,', accl2)
print('category embedding')

pred_l2 = model.predict([emb_pre_label_x])
print("Save normal label to file")
pl.dump(pred_l2, open(r'../dataset/dbp/predictlabel/dbp_pred_l2', 'wb'))
pred_l2 = np.argmax(pred_l2, axis=1)



semantic_l2 = ['actor', 'amusement park attraction', 'animal', 'artist', 'athlete', 'body of water', 'boxer', 'british royalty',
      'broadcaster', 'building', 'cartoon', 'celestial body', 'cleric', 'clerical administrative region', 'coach', 'comic',
      'comics character', 'company', 'database', 'educational institution', 'engine', 'eukaryote', 'fictional character',
      'flowering plant', 'football leagueseason', 'genre', 'gridiron football player', 'group', 'horse', 'infrastructure',
      'legal case', 'motorcycle rider', 'musical artist', 'musical work', 'natural event', 'natural place', 'olympics',
      'organisation', 'organisation member', 'periodical literature', 'person', 'plant', 'politician', 'presenter',
      'race', 'race track', 'racing driver', 'route of transportation', 'satellite', 'scientist', 'settlement',
      'societal event', 'software', 'song', 'sport facility', 'sports event', 'sports league', 'sports manager',
      'sports team', 'sports team season', 'station', 'stream', 'tournament', 'tower', 'venue', 'volleyball player',
      'winter sport player', 'wrestler', 'writer', 'written work']

pred_semantic_l2 = []


for i in pred_l2:
    pred_semantic_l2.append([word_to_id[x] for x in semantic_l2[i].split(' ') if x in word_to_id])


y2_length = 3
pred_semantic_l2_pad = keras.preprocessing.sequence.pad_sequences(pred_semantic_l2, y2_length, padding='post', truncating='post')
print("Save semantic label to file")
with open(r'../dataset/dbp/predictlabel/dbp_pred_semantic_l2_pad', 'wb') as f:
    pl.dump(pred_semantic_l2_pad, f)
#######################################
# print("Time cost: %.3f seconds...\n" % (time.time() - start_time))
print("level2: end up")
